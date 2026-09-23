#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ROOT

from eqmarl_network_study.config import load_config
from eqmarl_network_study.experiments import run_sweep, write_sweep_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sweep.yaml")
    parser.add_argument("--output", default="sweep.csv")
    args = parser.parse_args()
    frame = run_sweep(load_config(ROOT / args.config), ROOT, output_name=args.output)
    summary_name = f"{Path(args.output).stem}_summary.json"
    summary = write_sweep_summary(frame, ROOT, output_name=summary_name)
    print(f"wrote {len(frame):,} raw rows to results/raw/{args.output}")
    print(
        f"summarized {summary['unique_parameter_points']:,} parameter points in "
        f"results/summaries/{summary_name}"
    )


if __name__ == "__main__":
    main()
