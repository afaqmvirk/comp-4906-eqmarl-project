from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def metric_files(official_repo: str | Path, experiment: str) -> list[Path]:
    root = Path(official_repo) / "experiment_output" / experiment
    paths = sorted(root.glob("*/metrics-*.json"), key=lambda p: (p.parent.name, p.name))
    if not paths:
        raise FileNotFoundError(f"no official metric files found under {root}")
    return paths


def load_official_metric(
    official_repo: str | Path,
    experiment: str,
    metric: str,
    max_seeds: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    paths = metric_files(official_repo, experiment)
    if max_seeds is not None:
        paths = paths[:max_seeds]
    runs: list[list[float]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if metric in payload.get("metrics", {}):
            values = payload["metrics"][metric]
        elif metric == "reward":
            values = payload["reward"]
        else:
            raise KeyError(f"metric {metric!r} absent from {path}")
        runs.append(values)
    lengths = {len(x) for x in runs}
    if len(lengths) != 1:
        raise ValueError(f"inconsistent trajectory lengths: {lengths}")
    return np.asarray(runs, dtype=float), [str(p) for p in paths]


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if window <= 0:
        raise ValueError("window must be positive")
    out = np.full(values.shape, np.nan, dtype=float)
    if len(values) >= window:
        out[window - 1 :] = np.convolve(values, np.ones(window) / window, mode="valid")
    return out


def aggregate_runs(runs: np.ndarray, window: int) -> dict[str, np.ndarray]:
    mean = np.mean(runs, axis=0)
    std = np.std(runs, axis=0)
    return {"mean": mean, "std": std, "rolling_mean": rolling_mean(mean, window)}


def first_target_iteration(values: np.ndarray, target: float) -> int | None:
    indices = np.flatnonzero(np.asarray(values) >= target)
    return int(indices[0]) if len(indices) else None
