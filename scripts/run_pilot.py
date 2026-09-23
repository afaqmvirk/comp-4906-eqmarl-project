#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import numpy as np

from _bootstrap import ROOT

from eqmarl_network_study.config import load_config
from eqmarl_network_study.experiments import run_pilot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    args = parser.parse_args()
    summary = run_pilot(load_config(ROOT / args.config), ROOT)
    print(
        json.dumps(
            summary,
            indent=2,
            default=lambda obj: obj.item() if isinstance(obj, np.generic) else str(obj),
        )
    )


if __name__ == "__main__":
    main()
