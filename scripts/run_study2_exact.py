#!/usr/bin/env python
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from _study2_bootstrap import ROOT
from eqmarl_network_study.learning_noise.exact_study import (
    atomic_json, inventory, make_manifest, manifest_matches, read_json, treatment_order,
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def controller_lock(output):
    import fcntl
    with (output/"controller.lock").open("a+") as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("An exact-study controller already holds this directory")
        try:
            yield
        finally:
            fcntl.flock(lock,fcntl.LOCK_UN)


def prepare(config):
    output = ROOT/config["output_root"]
    output.mkdir(parents=True,exist_ok=True)
    expected = make_manifest(ROOT,config)
    path = output/"manifest.json"
    if path.exists():
        if not manifest_matches(ROOT, expected, read_json(path)):
            raise ValueError("Manifest/config/source/code changed; refusing to mix protocols")
    else:
        atomic_json(path,expected)
    return output,expected


def resumable(row):
    path = ROOT/row["path"]
    if not path.exists():
        return True
    record = read_json(path)
    prefix = record.get("checkpoint")
    return bool(prefix and (ROOT/(prefix+".index")).exists()
                and list((ROOT/prefix).parent.glob(Path(prefix).name+".data-*"))
                and "rng_state" in record)


def command(row,config):
    return [sys.executable,str(ROOT/"scripts/run_noise_feasibility.py"),
            "--backend","exact_density","--q",str(row["q"]),"--seed",str(row["seed"]),
            "--episodes",str(config["episodes"]),"--eval-episodes",str(config["evaluation_episodes"]),
            "--chunk-episodes",str(config["chunk_episodes"]),"--log-every",str(config["checkpoint_every"]),
            "--threads",str(config["threads_per_worker"]),"--output-root",config["output_root"],
            "--group","raw","--purpose","exact-noise Study 2 rerun, not hardware error mitigation","--resume"]


def save_status(output,manifest,state,active,failures,error=None):
    rows = inventory(ROOT,manifest)
    quantum = [row for row in rows if row["framework"] == "eqmarl_psi_plus"]
    classical = [row for row in rows if row["framework"] == "sctde"]
    progress = dict(status=state,updated_utc=utc_now(),controller_pid=os.getpid(),
                    quantum_complete=sum(row["status"]=="complete" for row in quantum),
                    quantum_total=len(quantum),classical_complete=sum(row["status"]=="complete" for row in classical),
                    active={key:value[0].pid for key,value in active.items()},failures=failures,records=rows)
    if error:
        progress["error"] = error
    atomic_json(output/"progress.json",progress)
    return rows


def run(config,prepare_only=False):
    output = ROOT/config["output_root"]
    output.mkdir(parents=True,exist_ok=True)
    with controller_lock(output):
        output,manifest = prepare(config)
        old = read_json(output/"progress.json") if (output/"progress.json").exists() else {}
        failures = old.get("failures",{})
        active, failed = {}, {}
        (output/"logs").mkdir(exist_ok=True)
        rows = inventory(ROOT,manifest)
        if any(row["status"]=="invalid" for row in rows):
            raise ValueError("Invalid result in inventory: "+str([row for row in rows if row["status"]=="invalid"]))
        if prepare_only:
            save_status(output,manifest,"prepared",active,failures)
            print(dict(Counter(row["origin"] for row in rows)),flush=True)
            return
        queue = sorted([row for row in rows if row["origin"]=="new_exact" and row["status"]!="complete"],
                       key=lambda row:treatment_order(row,config))
        for row in queue:
            if failures.get(row["id"],0)>=config["max_failures_per_job"] or not resumable(row):
                failed[row["id"]] = "Retry limit reached or no usable checkpoint; needs inspection"
        queue = [row for row in queue if row["id"] not in failed]
        state, last_saved = "running",0.
        try:
            while queue or active:
                while queue and len(active)<config["workers"]:
                    row = queue.pop(0)
                    handle = (output/"logs"/(row["id"]+".log")).open("ab",buffering=0)
                    env = dict(os.environ,TF_CPP_MIN_LOG_LEVEL="3",PYTHONUNBUFFERED="1",
                               TF_NUM_INTRAOP_THREADS=str(config["threads_per_worker"]),
                               TF_NUM_INTEROP_THREADS="1",OMP_NUM_THREADS=str(config["threads_per_worker"]))
                    try:
                        process = subprocess.Popen(command(row,config),cwd=str(ROOT),env=env,
                                                   stdout=handle,stderr=subprocess.STDOUT)
                    except BaseException:
                        handle.close()
                        raise
                    active[row["id"]] = (process,handle,row)
                    print(json_line(event="chunk_started",treatment=row["id"],pid=process.pid),flush=True)
                for key,(process,handle,row) in list(active.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    handle.close()
                    del active[key]
                    current = next(item for item in inventory(ROOT,manifest) if item["id"]==key)
                    print(json_line(event="chunk_finished",treatment=key,exit_code=code,
                                    status=current["status"],episodes=current["episodes_completed"]),flush=True)
                    if current["status"]=="complete" and code==0:
                        continue
                    if current["status"]=="paused_checkpoint" and code==0 and resumable(current):
                        queue.insert(0,current)
                    else:
                        failures[key] = failures.get(key,0)+1
                        if current["status"]!="invalid" and resumable(current) and failures[key]<config["max_failures_per_job"]:
                            queue.insert(0,current)
                        else:
                            failed[key] = current.get("error","Child exit %s; see treatment log"%code)
                    last_saved = 0.
                if time.monotonic()-last_saved>=15:
                    save_status(output,manifest,state,active,failures,str(failed) if failed else None)
                    last_saved = time.monotonic()
                if queue or active:
                    time.sleep(2)
            rows = inventory(ROOT,manifest)
            if failed or any(row["status"]!="complete" for row in rows):
                raise RuntimeError("Incomplete treatments: "+str(failed))

            prepare(config)
            save_status(output,manifest,"analyzing",active,failures)
            subprocess.run([sys.executable,str(ROOT/"scripts/analyze_study2_exact.py"),
                            "--manifest",str(output/"manifest.json")],cwd=str(ROOT),check=True)
            state = "complete"
            save_status(output,manifest,state,active,failures)
        except BaseException as exc:
            for process,handle,row in active.values():
                process.terminate()
            for process,handle,row in active.values():
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                handle.close()
            active.clear()
            state = "interrupted" if isinstance(exc,KeyboardInterrupt) else "failed"
            save_status(output,manifest,state,active,failures,repr(exc))
            raise


def json_line(**data):
    import json
    return json.dumps(dict(at=utc_now(),**data))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/study2_exact.json")
    parser.add_argument("--prepare-only",action="store_true")
    args = parser.parse_args()
    def interrupt(signum,frame):
        raise KeyboardInterrupt("Controller received signal %s"%signum)
    signal.signal(signal.SIGTERM,interrupt)
    run(read_json(ROOT/args.config),args.prepare_only)
