#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from _study2_bootstrap import ROOT
from eqmarl_network_study.learning_noise.training import (
    DISTRIBUTED_QUANTUM_FRAMEWORKS,
    run_fresh_training,
)


def treatment_grid(config):
    if "treatments" in config:
        for treatment in config["treatments"]:
            yield (
                treatment["framework"],
                float(treatment.get("q_depolarizing", 0.0)),
                int(treatment["seed"]),
            )
        return
    for framework in config["frameworks"]:
        q_values = (
            config["q_depolarizing"]
            if framework in DISTRIBUTED_QUANTUM_FRAMEWORKS
            else [0.0]
        )
        for seed in config["learning_seeds"]:
            for q_value in q_values:
                yield framework, float(q_value), int(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/study2_pilot.yaml")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--framework", action="append")
    parser.add_argument("--q", action="append", type=float)
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--show-progress", action="store_true")
    args = parser.parse_args()

    with (ROOT / args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if args.framework or args.q or args.seed:
        config.pop("treatments", None)
        if args.framework:
            config["frameworks"] = args.framework
        if args.q:
            config["q_depolarizing"] = args.q
        if args.seed:
            config["learning_seeds"] = args.seed
    episodes = args.episodes or int(config["n_episodes"])
    noise = config["noise"]
    treatments = list(treatment_grid(config))
    if args.max_runs is not None:
        treatments = treatments[: args.max_runs]

    completed = []
    for index, (framework, q_value, seed) in enumerate(treatments, start=1):
        print(
            "[%d/%d] framework=%s q=%g seed=%d episodes=%d"
            % (index, len(treatments), framework, q_value, seed, episodes),
            flush=True,
        )
        try:
            result = run_fresh_training(
                framework=framework,
                q_depolarizing=q_value,
                seed=seed,
                n_episodes=episodes,
                noise_repetitions=int(noise["trajectories_per_expectation"]),
                noise_bank_size=int(noise["trajectory_bank_size"]),
                noise_backend=noise["backend"],
                output_root=ROOT / "results" / "study2" / "raw",
                overwrite=args.overwrite,
                quiet=not args.show_progress,
            )
        except Exception as exc:
            completed.append(
                {
                    "framework": framework,
                    "q_depolarizing": q_value,
                    "seed": seed,
                    "status": "failed",
                    "exception": repr(exc),
                }
            )
            print("treatment failed; continuing grid: %r" % exc, flush=True)
            continue
        completed.append(
            {
                "framework": framework,
                "q_depolarizing": q_value,
                "seed": seed,
                "status": result["status"],
                "runtime_seconds": result["runtime_seconds"],
            }
        )
    print(json.dumps(completed, indent=2))


if __name__ == "__main__":
    main()
