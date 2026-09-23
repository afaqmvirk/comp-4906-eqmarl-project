#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER_TREE = PROJECT_ROOT / "upstream" / "eqmarl-paper-iclr2025"
RESULT_ROOT = PROJECT_ROOT / "results" / "paper_reference"
EXPECTED_COMMIT = "62917c087f7bda38338df7a738692169d90f307c"
PAPER_CROSSINGS = {
    "eqmarl": {"20": 568, "25": 2332},
    "sctde": {"20": 1640, "25": 2615},
}
EXPERIMENTS = {
    "eqmarl": {
        "config": "coingame_maa2c_mdp_eqmarl_psi+.yml",
        "output": "coingame_maa2c_mdp_eqmarl_psi+",
    },
    "sctde": {
        "config": "coingame_maa2c_mdp_sctde.yml",
        "output": "coingame_maa2c_mdp_sctde",
    },
}
PACKAGE_NAMES = (
    "cirq",
    "gymnasium",
    "minigrid",
    "numpy",
    "pandas",
    "qutip",
    "scipy",
    "sympy",
    "tensorflow",
    "tensorflow-quantum",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    executable = "git"
    tree = str(PAPER_TREE)

    if sys.platform.startswith("linux") and tree.startswith("/mnt/"):
        path_parts = PAPER_TREE.parts
        drive = path_parts[2].upper()
        tree = drive + ":/" + "/".join(path_parts[3:])
        executable = "git.exe"
    return subprocess.check_output(
        [executable, "-C", tree, *args], text=True
    ).strip()


def session_dirs(parent: Path) -> Set[Path]:
    if not parent.exists():
        return set()
    return {entry.resolve() for entry in parent.iterdir() if entry.is_dir()}


def generated_smoke_config(framework: str, episodes: int) -> Path:
    source = PAPER_TREE / "experiments" / EXPERIMENTS[framework]["config"]
    text = source.read_text(encoding="utf-8")
    text, episode_count = re.subn(
        r"(?m)^(\s*n_episodes:\s*)3000\s*$",
        rf"\g<1>{episodes}",
        text,
        count=1,
    )
    original_root = f"../experiment_output/{EXPERIMENTS[framework]['output']}"
    smoke_root = str(RESULT_ROOT / "upstream_smoke" / framework)
    text, root_count = text.replace(original_root, smoke_root, 1), text.count(original_root)
    if episode_count != 1 or root_count != 1:
        raise RuntimeError("Could not make the narrowly modified smoke-test config")
    target = RESULT_ROOT / "generated_configs" / f"smoke_{framework}_{episodes}ep.yml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def crossing(curve: np.ndarray, target: float) -> Optional[int]:
    locations = np.flatnonzero(curve >= target)
    return int(locations[0]) if locations.size else None


def summarize_session(session: Path, expected_rounds: int) -> Dict[str, object]:
    def round_number(path: Path) -> int:
        match = re.fullmatch(r"metrics-(\d+)\.json", path.name)
        if not match:
            raise ValueError(path.name)
        return int(match.group(1))

    metric_files = sorted(session.glob("metrics-*.json"), key=round_number)
    if len(metric_files) != expected_rounds:
        raise RuntimeError(
            f"Expected {expected_rounds} metric files in {session}, found {len(metric_files)}"
        )
    histories: List[np.ndarray] = []
    for path in metric_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        history = np.asarray(payload["metrics"]["undiscounted_reward"], dtype=float)
        histories.append(history)
    lengths = {history.size for history in histories}
    if len(lengths) != 1:
        raise RuntimeError(f"Metric histories have unequal lengths: {sorted(lengths)}")
    data = np.stack(histories)
    mean_curve = np.mean(data, axis=0)
    paper_curve = pd.DataFrame(mean_curve).rolling(10).mean().to_numpy().ravel()
    per_run = {
        str(target): [
            crossing(
                pd.DataFrame(history).rolling(10).mean().to_numpy().ravel(),
                float(target),
            )
            for history in data
        ]
        for target in (20, 25)
    }
    return {
        "metric": "metrics.undiscounted_reward",
        "rounds": int(data.shape[0]),
        "episodes_per_round": int(data.shape[1]),
        "paper_matched_definition": (
            "mean across rounds at each zero-based episode index, then pandas "
            "rolling(window=10).mean(), then first value >= target"
        ),
        "paper_matched_crossing_index": {
            str(target): crossing(paper_curve, float(target)) for target in (20, 25)
        },
        "per_run_rolling10_crossing_index": per_run,
        "final_100_episode_mean_by_round": [
            float(np.mean(history[-100:])) for history in data
        ],
        "files": [
            {"name": path.name, "sha256": sha256_file(path)} for path in metric_files
        ],
    }


def package_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    for package in PACKAGE_NAMES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def build_manifest(mode: str, frameworks: Sequence[str], rounds: int) -> Dict[str, object]:
    runner = PAPER_TREE / "scripts" / "experiment_runner.py"
    configs = {
        name: PAPER_TREE / "experiments" / EXPERIMENTS[name]["config"]
        for name in frameworks
    }
    return {
        "schema_version": 1,
        "status": "running",
        "mode": mode,
        "started_utc": utc_now(),
        "controller_pid": os.getpid(),
        "frameworks": list(frameworks),
        "rounds": rounds,
        "paper_fidelity": {
            "upstream_repository": "https://github.com/news-vt/eqmarl",
            "tag": "paper-iclr2025",
            "expected_commit": EXPECTED_COMMIT,
            "actual_commit": git("rev-parse", "HEAD"),
            "upstream_worktree_clean_at_start": git("status", "--porcelain") == "",
            "training_runner_modified": False,
            "full_configs_modified": False,
            "seeds_set": False,
            "seed_note": (
                "The upstream runner does not accept or record seeds. Repeated rounds "
                "therefore reproduce the paper protocol, but not its exact RNG states."
            ),
        },
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "packages": package_versions(),
            "dependency_lock": str(PROJECT_ROOT / "requirements-paper-2024.lock"),
            "dependency_lock_sha256": sha256_file(
                PROJECT_ROOT / "requirements-paper-2024.lock"
            ),
        },
        "inputs": {
            "runner": {"path": str(runner), "sha256": sha256_file(runner)},
            "configs": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in configs.items()
            },
        },
        "paper_reported_crossing_indices": PAPER_CROSSINGS,
        "runs": {},
    }


def copy_artifacts(source: Path, mode: str, framework: str) -> Path:
    destination = RESULT_ROOT / "raw" / mode / framework / source.name
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    return destination


def run_one(
    framework: str,
    mode: str,
    rounds: int,
    smoke_episodes: int,
    manifest: Dict[str, object],
    manifest_path: Path,
) -> None:
    spec = EXPERIMENTS[framework]
    if mode == "smoke":
        config = generated_smoke_config(framework, smoke_episodes)
        output_parent = RESULT_ROOT / "upstream_smoke" / framework
    else:
        config = PAPER_TREE / "experiments" / spec["config"]
        output_parent = PAPER_TREE / "experiment_output" / spec["output"]

    before = session_dirs(output_parent)
    command = [
        sys.executable,
        "experiment_runner.py",
        str(config),
        "-r",
        str(rounds),
    ]
    log_path = RESULT_ROOT / "logs" / f"{mode}_{framework}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    run_record: Dict[str, object] = {
        "status": "running",
        "started_utc": utc_now(),
        "command": command,
        "cwd": str(PAPER_TREE / "scripts"),
        "config": str(config),
        "log": str(log_path),
    }
    manifest["runs"][framework] = run_record
    write_json(manifest_path, manifest)

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PAPER_TREE)
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        log.write("Command: " + " ".join(command) + "\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=PAPER_TREE / "scripts",
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    elapsed = time.monotonic() - start
    run_record.update(
        {
            "ended_utc": utc_now(),
            "elapsed_seconds": elapsed,
            "exit_code": result.returncode,
        }
    )
    if result.returncode != 0:
        run_record["status"] = "failed"
        write_json(manifest_path, manifest)
        raise subprocess.CalledProcessError(result.returncode, command)

    new_sessions = session_dirs(output_parent) - before
    if len(new_sessions) != 1:
        run_record["status"] = "failed_artifact_discovery"
        run_record["new_session_candidates"] = [str(path) for path in new_sessions]
        write_json(manifest_path, manifest)
        raise RuntimeError(f"Expected one new session directory, found {len(new_sessions)}")
    source_session = next(iter(new_sessions))
    copied_session = copy_artifacts(source_session, mode, framework)
    summary = summarize_session(copied_session, rounds)
    run_record.update(
        {
            "status": "complete",
            "upstream_session": str(source_session),
            "copied_session": str(copied_session),
            "summary": summary,
        }
    )
    write_json(manifest_path, manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument(
        "--framework", choices=("eqmarl", "sctde", "both"), default="both"
    )
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--smoke-episodes", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frameworks = ["sctde", "eqmarl"] if args.framework == "both" else [args.framework]
    rounds = args.rounds if args.rounds is not None else (10 if args.mode == "full" else 1)
    if rounds < 1 or args.smoke_episodes < 1:
        raise ValueError("Round and episode counts must be positive")
    if not PAPER_TREE.is_dir():
        raise FileNotFoundError(f"Paper worktree is missing: {PAPER_TREE}")
    actual_commit = git("rev-parse", "HEAD")
    if actual_commit != EXPECTED_COMMIT:
        raise RuntimeError(f"Paper worktree is at {actual_commit}, expected {EXPECTED_COMMIT}")

    manifest_path = RESULT_ROOT / f"{args.mode}_manifest.json"
    lock_path = RESULT_ROOT / f"{args.mode}.lock"
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f"A {args.mode} run lock already exists: {lock_path}") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as lock:
        lock.write(str(os.getpid()) + "\n")

    manifest = build_manifest(args.mode, frameworks, rounds)
    write_json(manifest_path, manifest)
    try:
        for framework in frameworks:
            run_one(
                framework,
                args.mode,
                rounds,
                args.smoke_episodes,
                manifest,
                manifest_path,
            )
        manifest["status"] = "complete"
        manifest["ended_utc"] = utc_now()
        if args.mode == "full" and set(frameworks) == set(EXPERIMENTS):
            observed = {
                name: manifest["runs"][name]["summary"][
                    "paper_matched_crossing_index"
                ]
                for name in frameworks
            }
            manifest["comparison_to_paper"] = {
                name: {
                    target: {
                        "paper": PAPER_CROSSINGS[name][target],
                        "reproduction": observed[name][target],
                        "difference": (
                            None
                            if observed[name][target] is None
                            else observed[name][target] - PAPER_CROSSINGS[name][target]
                        ),
                    }
                    for target in ("20", "25")
                }
                for name in frameworks
            }
        write_json(manifest_path, manifest)
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["ended_utc"] = utc_now()
        manifest["error"] = f"{type(error).__name__}: {error}"
        write_json(manifest_path, manifest)
        raise
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
