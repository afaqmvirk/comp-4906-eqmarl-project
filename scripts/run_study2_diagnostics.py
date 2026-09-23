#!/usr/bin/env python

from __future__ import annotations

import argparse
import json

import yaml

from _study2_bootstrap import ROOT
from eqmarl_network_study.learning_noise.diagnostics import run_critic_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/study2_pilot.yaml")
    parser.add_argument("--observations", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--trajectories", type=int)
    parser.add_argument("--bank-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=4242)
    args = parser.parse_args()
    with (ROOT / args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    diagnostic = config["diagnostics"]
    result = run_critic_diagnostics(
        q_values=config["q_depolarizing"],
        output_root=ROOT / "results" / "study2",
        observation_count=args.observations or int(diagnostic["observations"]),
        noise_samples=args.samples or int(diagnostic["noise_samples"]),
        trajectories_per_expectation=(
            args.trajectories or int(diagnostic["trajectories_per_expectation"])
        ),
        trajectory_bank_size=args.bank_size,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
