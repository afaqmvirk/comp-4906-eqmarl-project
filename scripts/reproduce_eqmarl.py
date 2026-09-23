#!/usr/bin/env python

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys

from _bootstrap import ROOT

from eqmarl_network_study.config import load_config
from eqmarl_network_study.experiments import reproduce_ideal


def native_audit() -> dict:
    supported_python = sys.version_info[:2] == (3, 9)
    modules = {
        name: importlib.util.find_spec(name) is not None
        for name in ("tensorflow", "tensorflow_quantum", "cirq", "gymnasium")
    }
    ready = supported_python and all(modules.values())
    result = {
        "status": "ready" if ready else "blocked",
        "attempted": True,
        "training_started": False,
        "python": platform.python_version(),
        "upstream_required_python": ">=3.9,<3.10",
        "required_modules_present": modules,
        "reason": None if ready else (
            "The official stack pins Python 3.9, TensorFlow 2.7.0, and "
            "TensorFlow Quantum 0.7.2; this interpreter is not a compatible native "
            "training environment. Use the documented Python 3.9 environment or "
            "the committed official trajectories."
        ),
    }
    path = ROOT / "results/summaries/native_reproduction_attempt.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ideal.yaml")
    parser.add_argument("--mode", choices=("stored", "native-audit"), default="stored")
    args = parser.parse_args()
    if args.mode == "native-audit":
        result = native_audit()
    else:
        result = reproduce_ideal(load_config(ROOT / args.config), ROOT)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
