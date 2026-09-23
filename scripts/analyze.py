#!/usr/bin/env python
from __future__ import annotations

import argparse

from _bootstrap import ROOT

from eqmarl_network_study.analysis.plots import make_all_figures
from eqmarl_network_study.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--curves", default="results/raw/pilot_learning_curves.csv")
    parser.add_argument("--sweep", default="results/raw/pilot_sweep.csv")
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    make_all_figures(
        ROOT / args.curves,
        ROOT / args.sweep,
        ROOT / "results/figures",
        config.learning.target_reward,
    )
    print("regenerated six figure pairs in results/figures")


if __name__ == "__main__":
    main()
