import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import yaml

COMMIT = "6b1b2e3817f661da81aad5e7188a362380dbdd1d"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding="utf-8")
    temporary.replace(path)


def validate_record(record, framework, q, seed, episodes=3000, require_complete=True):
    if record.get("framework") != framework or record.get("seed") != seed:
        raise ValueError("Framework/seed mismatch")
    if record.get("fresh_training") is not True:
        raise ValueError("Not a fresh-trained model")
    applied = record.get("q",record.get("q_depolarizing_applied"))
    if applied is None or abs(float(applied)-q)>1e-12:
        raise ValueError("Noise strength mismatch")
    if record.get("versions",{}).get("eqmarl_upstream_commit") != COMMIT:
        raise ValueError("Upstream revision mismatch")
    if framework == "eqmarl_psi_plus" and record.get("backend") != "exact_density":
        raise ValueError("Quantum record is not exact-density training")
    if record.get("episodes_requested",record.get("n_episodes_requested")) != episodes:
        raise ValueError("Training horizon mismatch")
    done = record.get("episodes_completed",record.get("n_episodes",0))
    if not isinstance(done,int) or not 0 <= done <= episodes:
        raise ValueError("Invalid completed episode count")
    if require_complete:
        if record.get("status") != "complete" or done != episodes:
            raise ValueError("Training is incomplete")
        values = np.asarray(record["trajectory"]["undiscounted_reward"],float)
        if len(values) != episodes or not np.isfinite(values).all():
            raise ValueError("Invalid reward trajectory")
        for values in record["trajectory"].values():
            if len(values) != episodes or not np.isfinite(np.asarray(values,float)).all():
                raise ValueError("Invalid diagnostic trajectory")
        if framework == "eqmarl_psi_plus":
            evaluation = np.asarray(record.get("evaluation_scores",[]),float)
            if evaluation.shape != (100,) or not np.isfinite(evaluation).all():
                raise ValueError("Missing/invalid fresh evaluation")
    for key, expected in [("steps_per_episode",50),("gamma",.99),("entropy_coefficient",.001)]:
        if key in record and record[key] != expected:
            raise ValueError("Incompatible %s"%key)
    return int(done)


def make_manifest(root, config):
    root = Path(root)
    if config["episodes"] != 3000 or config["evaluation_episodes"] != 100 or config["targets"] != [20,25]:
        raise ValueError("This protocol requires 3000 training episodes, 100 evaluations and targets 20/25")
    if any(config[key] < 1 for key in ["workers","threads_per_worker","chunk_episodes","checkpoint_every","max_failures_per_job"]):
        raise ValueError("Execution counts must be positive")
    grid = yaml.safe_load((root/config["source_grid"]).read_text(encoding="utf-8"))["treatments"]
    reused = {}
    for relative in config["existing_exact"]:
        record = read_json(root/relative)
        key = (record["framework"],float(record["q"]),int(record["seed"]))
        validate_record(record,*key,episodes=config["episodes"])
        if not (root/relative).with_name(Path(relative).stem+"_weights.npz").exists():
            raise ValueError("Missing reusable model weights")
        if key in reused:
            raise ValueError("Duplicate existing treatment")
        reused[key] = relative
    records, seen = [], set()
    for item in grid:
        framework,q,seed = item["framework"],float(item["q_depolarizing"]),int(item["seed"])
        key = framework,q,seed
        if key in seen:
            raise ValueError("Duplicate grid treatment")
        seen.add(key)
        if framework == "sctde":
            relative = "results/study2/raw/sctde_q0_seed%d.json"%seed
            origin = "existing_classical"
        elif framework == "eqmarl_psi_plus":
            relative = reused.get(key,"%s/raw/exact_density_q%g_r1_seed%d.json"%(config["output_root"],q,seed))
            origin = "existing_exact" if key in reused else "new_exact"
        else:
            raise ValueError("Unsupported original-grid framework")
        entry = dict(id="%s_q%g_seed%d"%key,framework=framework,q=q,seed=seed,origin=origin,path=relative)
        if origin != "new_exact":
            validate_record(read_json(root/relative),framework,q,seed,config["episodes"])
            entry["source_sha256"] = sha256(root/relative)
        records.append(entry)
    if set(reused)-seen:
        raise ValueError("A reusable result is outside the original grid")
    code_paths = ["scripts/run_noise_feasibility.py", "scripts/run_study2_exact.py",
                  "scripts/analyze_study2_exact.py",
                  "src/eqmarl_network_study/learning_noise/exact_study.py",
                  "src/eqmarl_network_study/learning_noise/exact_critic.py",
                  "src/eqmarl_network_study/learning_noise/training.py",
                  "src/eqmarl_network_study/learning_noise/noisy_critic.py"]
    return dict(schema=1,config=config,source_grid_sha256=sha256(root/config["source_grid"]),
                code_sha256={path:sha256(root/path) for path in code_paths},records=records)


def manifest_matches(root, current, recorded):
    if current == recorded:
        return True
    path = Path(root)/"results/study2_exact/source_cleanup.json"
    if not path.exists():
        return False
    changes = read_json(path)["files"]

    def matches(filename, before, after):
        change = changes.get(filename, {})
        return before == after or (
            change.get("before_sha256") == before
            and change.get("after_sha256") == after
        )

    old_code, new_code = recorded["code_sha256"], current["code_sha256"]
    if old_code.keys() != new_code.keys():
        return False
    if not all(matches(name, old_code[name], new_code[name]) for name in old_code):
        return False
    if not matches(recorded["config"]["source_grid"],
                   recorded["source_grid_sha256"], current["source_grid_sha256"]):
        return False
    adjusted = dict(current, code_sha256=old_code,
                    source_grid_sha256=recorded["source_grid_sha256"])
    return adjusted == recorded


def inventory(root, manifest):
    rows = []
    for entry in manifest["records"]:
        row = dict(entry,episodes_completed=0,status="pending")
        path = Path(root)/entry["path"]
        if path.exists():
            try:
                if "source_sha256" in entry and sha256(path) != entry["source_sha256"]:
                    raise ValueError("A reused source changed after manifest creation")
                record = read_json(path)
                row["episodes_completed"] = validate_record(record,entry["framework"],entry["q"],entry["seed"],manifest["config"]["episodes"],require_complete=False)
                row["status"] = record.get("status","invalid")
                if row["status"] == "complete":
                    validate_record(record,entry["framework"],entry["q"],entry["seed"],manifest["config"]["episodes"])
                    if entry["framework"] == "eqmarl_psi_plus" and not path.with_name(path.stem+"_weights.npz").exists():
                        raise ValueError("Missing final quantum weights")
            except Exception as exc:
                row.update(status="invalid",error=str(exc))
        rows.append(row)
    return rows


def treatment_order(entry, config):
    priority = config["priority_q"]
    if entry["q"] in priority[:2]:
        return (0,entry["seed"],priority.index(entry["q"]))
    return (1,priority.index(entry["q"]) if entry["q"] in priority else 100+entry["q"],entry["seed"])


def comparable_metrics(record, window, horizon=3000):
    values = np.asarray(record["trajectory"]["undiscounted_reward"],float)
    if len(values) != horizon or not np.isfinite(values).all() or not 1<=window<=horizon:
        raise ValueError("Invalid full-length trajectory or window")

    import pandas as pd
    smoothed = pd.Series(values).rolling(window,min_periods=window).mean().to_numpy()
    result = dict(final100=float(values[-100:].mean()),mean_reward=float(values.mean()),window=window)
    for target in [20,25]:
        reached = np.flatnonzero(smoothed>=target)
        episode = int(reached[0])+1 if len(reached) else None
        result["crossing_%d"%target] = episode
        result["reached_%d"%target] = episode is not None
        result["restricted_%d"%target] = episode if episode is not None else horizon
    return result


def paired_summary(left, right):
    seeds = sorted(set(left) & set(right))
    if not seeds:
        raise ValueError("No matched seeds")
    differences = np.array([left[s]-right[s] for s in seeds],float)
    if not np.isfinite(differences).all():
        raise ValueError("Nonfinite paired values")
    mean = float(differences.mean())
    if len(seeds) > 1:
        bootstrap = np.random.default_rng(731).choice(differences,(10000,len(seeds)),replace=True).mean(axis=1)
        interval = np.quantile(bootstrap,[.025,.975]).tolist()
        statistics = [abs(np.mean(differences*np.asarray(signs)))
                      for signs in itertools.product((-1,1),repeat=len(seeds))]
        p_value = float(np.mean(np.asarray(statistics)>=abs(mean)-1e-12))
    else:
        interval, p_value = None, None
    return dict(seeds=seeds,n=len(seeds),differences=differences.tolist(),
                mean_left_minus_right=mean,bootstrap_ci95=interval,
                exact_sign_flip_p_two_sided=p_value,
                positive=int(np.sum(differences>0)),negative=int(np.sum(differences<0)),
                ties=int(np.sum(differences==0)))
