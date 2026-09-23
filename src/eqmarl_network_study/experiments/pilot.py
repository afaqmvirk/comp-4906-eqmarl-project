from __future__ import annotations

import json
import math
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..accounting import ResourceAccounting
from ..analysis.plots import make_all_figures
from ..analysis.trajectories import (
    aggregate_runs,
    first_target_iteration,
    load_official_metric,
)
from ..config import StudyConfig
from ..network import NetworkSimulator


FRAMEWORKS = ("eqmarl", "sctde")


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            indent=2,
            allow_nan=False,
            default=lambda obj: obj.item() if isinstance(obj, np.generic) else str(obj),
        )
        handle.write("\n")


def _resolve_repo(config: StudyConfig, root: Path) -> Path:
    repo = Path(config.learning.official_repo)
    return repo if repo.is_absolute() else root / repo


def load_learning_curves(config: StudyConfig, root: Path, max_seeds: int | None):
    repo = _resolve_repo(config, root)
    output = {}
    source_files = {}
    for framework, experiment in (
        ("eqmarl", config.learning.quantum_experiment),
        ("sctde", config.learning.classical_experiment),
    ):
        runs, files = load_official_metric(
            repo, experiment, config.learning.metric, max_seeds=max_seeds
        )
        output[framework] = aggregate_runs(runs, config.learning.smoothing_window)
        source_files[framework] = files
    return output, source_files


def reproduce_ideal(config: StudyConfig, root: Path) -> dict:
    curves, files = load_learning_curves(config, root, max_seeds=None)
    rows = []
    crossings = {}
    for framework, data in curves.items():
        crossing = first_target_iteration(data["rolling_mean"], config.learning.target_reward)
        crossings[framework] = crossing
        for iteration in range(len(data["mean"])):
            rows.append(
                {
                    "framework": framework,
                    "iteration": iteration,
                    "reward_mean": data["mean"][iteration],
                    "reward_std": data["std"][iteration],
                    "reward_rolling": data["rolling_mean"][iteration],
                }
            )
    raw_path = root / "results/raw/ideal_learning_curves.csv"
    pd.DataFrame(rows).to_csv(raw_path, index=False)
    summary = {
        "status": "reproduced_from_official_committed_trajectories",
        "not_a_fresh_training_run": True,
        "target_reward": config.learning.target_reward,
        "smoothing_window": config.learning.smoothing_window,
        "crossing_iteration_zero_based": crossings,
        "number_of_runs": {name: len(paths) for name, paths in files.items()},
        "source_files": files,
        "official_repository_commit": "6b1b2e3817f661da81aad5e7188a362380dbdd1d",
    }
    _json_dump(root / "results/summaries/ideal_reproduction.json", summary)
    return summary


def _cumulative_curves(
    config: StudyConfig, learning: dict, root: Path
) -> pd.DataFrame:
    records: list[dict] = []
    iterations = len(learning["eqmarl"]["mean"])
    for seed in config.network_seeds:
        simulator = NetworkSimulator(config.protocol, config.network)
        rng = np.random.default_rng(seed)
        cumulative = {name: ResourceAccounting() for name in FRAMEWORKS}
        for iteration in range(iterations):
            q = simulator.simulate_quantum(
                config.learning.steps_per_iteration, rng, training_iterations=1
            )
            c = simulator.simulate_classical(
                config.learning.steps_per_iteration, training_iterations=1
            )
            cumulative["eqmarl"] += q
            cumulative["sctde"] += c
            for framework in FRAMEWORKS:
                records.append(
                    {
                        "network_seed": seed,
                        "framework": framework,
                        "iteration": iteration,
                        "reward_mean": learning[framework]["mean"][iteration],
                        "reward_std": learning[framework]["std"][iteration],
                        "reward_rolling": learning[framework]["rolling_mean"][iteration],
                        "simulated_wall_time_s": cumulative[framework].simulated_wall_time_s,
                    }
                )
    frame = pd.DataFrame.from_records(records)
    frame.to_csv(root / "results/raw/pilot_learning_curves.csv", index=False)
    return frame


def _axis_values(spec: Any) -> list[float | int]:
    if isinstance(spec, list):
        return spec
    if "values" in spec:
        return spec["values"]
    if "logspace" in spec:
        start, stop, count = spec["logspace"]
        return np.logspace(float(start), float(stop), int(count)).tolist()
    if "linspace" in spec:
        start, stop, count = spec["linspace"]
        return np.linspace(float(start), float(stop), int(count)).tolist()
    raise ValueError(f"unsupported axis specification: {spec}")


NETWORK_AXIS_FIELDS = {
    "entanglement_rate_hz",
    "transmission_success_probability",
    "base_latency_s",
    "distance_km",
    "memory_t2_s",
    "quantum_channel_capacity",
    "entanglement_parallelism",
}


def run_sweep(
    config: StudyConfig,
    root: Path,
    learning: dict | None = None,
    output_name: str = "pilot_sweep.csv",
) -> pd.DataFrame:
    if learning is None:
        learning, _ = load_learning_curves(
            config, root, max_seeds=config.learning.trajectory_seeds
        )
    crossings = {
        name: first_target_iteration(data["rolling_mean"], config.learning.target_reward)
        for name, data in learning.items()
    }
    if any(value is None for value in crossings.values()):
        raise RuntimeError(f"target not reached by all frameworks: {crossings}")
    q_steps = (crossings["eqmarl"] + 1) * config.learning.steps_per_iteration
    c_steps = (crossings["sctde"] + 1) * config.learning.steps_per_iteration

    axes = config.sweep.get("axes", {})
    if not axes:
        axes = {
            "entanglement_rate_hz": [config.network.entanglement_rate_hz],
            "transmission_success_probability": [
                config.network.transmission_success_probability
            ],
        }
    unknown = set(axes) - NETWORK_AXIS_FIELDS - {"agents"}
    if unknown:
        raise ValueError(f"unknown sweep axes: {sorted(unknown)}")
    axis_names = list(axes)
    axis_values = [_axis_values(axes[name]) for name in axis_names]
    rows: list[dict] = []
    for values in product(*axis_values):
        changes = dict(zip(axis_names, values))
        protocol_changes = {}
        if "agents" in changes:
            protocol_changes["agents"] = int(changes.pop("agents"))
        if "quantum_channel_capacity" in changes:
            changes["quantum_channel_capacity"] = int(changes["quantum_channel_capacity"])
        if "entanglement_parallelism" in changes:
            changes["entanglement_parallelism"] = int(changes["entanglement_parallelism"])
        protocol = replace(config.protocol, **protocol_changes)
        network = replace(config.network, **changes)
        simulator = NetworkSimulator(protocol, network)
        classical = simulator.simulate_classical(c_steps, crossings["sctde"] + 1)
        for seed in config.network_seeds:
            quantum = simulator.simulate_quantum(
                q_steps, np.random.default_rng(seed), crossings["eqmarl"] + 1
            )
            row = {
                "network_seed": seed,
                "quantum_target_iteration": crossings["eqmarl"],
                "classical_target_iteration": crossings["sctde"],
                "quantum_target_steps": q_steps,
                "classical_target_steps": c_steps,
                "quantum_time_s": quantum.simulated_wall_time_s,
                "classical_time_s": classical.simulated_wall_time_s,
                "speedup_classical_over_quantum": (
                    classical.simulated_wall_time_s / quantum.simulated_wall_time_s
                    if quantum.simulated_wall_time_s > 0
                    else math.inf
                ),
                "quantum_favorable": quantum.usable
                and quantum.simulated_wall_time_s < classical.simulated_wall_time_s,
                "agents": protocol.agents,
                "entanglement_rate_hz": network.entanglement_rate_hz,
                "p_success": network.transmission_success_probability,
                "base_latency_s": network.base_latency_s,
                "distance_km": network.distance_km,
                "memory_t2_s": network.memory_t2_s,
                "quantum_channel_capacity": network.quantum_channel_capacity,
            }
            for key, value in quantum.to_dict().items():
                row[f"quantum_{key}"] = value
            for key, value in classical.to_dict().items():
                row[f"classical_{key}"] = value
            rows.append(row)
    output = pd.DataFrame.from_records(rows)
    path = root / "results/raw" / output_name
    output.to_csv(path, index=False)
    return output


def write_sweep_summary(
    sweep: pd.DataFrame, root: Path, output_name: str = "sweep_summary.json"
) -> dict:
    usable = sweep["quantum_usable"].astype(bool)
    favorable = sweep["quantum_favorable"].astype(bool)
    parameter_columns = [
        "entanglement_rate_hz",
        "p_success",
        "distance_km",
        "memory_t2_s",
        "agents",
        "quantum_channel_capacity",
    ]

    by_axis: dict[str, list[dict]] = {}
    for axis in parameter_columns:
        entries = []
        for value, frame in sweep.groupby(axis, sort=True, dropna=False):
            axis_usable = frame["quantum_usable"].astype(bool)
            axis_favorable = frame["quantum_favorable"].astype(bool)
            entries.append(
                {
                    "value": float(value),
                    "seeded_rows": int(len(frame)),
                    "usable_fraction": float(axis_usable.mean()),
                    "quantum_favorable_fraction_all": float(axis_favorable.mean()),
                    "quantum_favorable_fraction_usable": (
                        float(axis_favorable[axis_usable].mean())
                        if axis_usable.any()
                        else None
                    ),
                }
            )
        by_axis[axis] = entries

    unique_points = sweep[parameter_columns].drop_duplicates()
    summary = {
        "status": "completed",
        "seeded_rows": int(len(sweep)),
        "unique_parameter_points": int(len(unique_points)),
        "network_seed_count": int(sweep["network_seed"].nunique()),
        "usable_fraction": float(usable.mean()),
        "quantum_favorable_fraction_all": float(favorable.mean()),
        "quantum_favorable_fraction_usable": (
            float(favorable[usable].mean()) if usable.any() else None
        ),
        "fractions_are_grid_summaries_not_probabilities": True,
        "by_axis": by_axis,
        "break_even_by_slice": _break_even_summary(sweep),
    }
    _json_dump(root / "results/summaries" / output_name, summary)
    return summary


def _break_even_summary(sweep: pd.DataFrame) -> list[dict]:
    result = []
    group_cols = [
        col
        for col in ("p_success", "agents", "distance_km", "memory_t2_s", "quantum_channel_capacity")
        if col in sweep.columns
    ]
    for keys, frame in sweep.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        grouped = frame.groupby("entanglement_rate_hz", as_index=False).mean(numeric_only=True)
        favorable = grouped[
            (grouped.quantum_usable >= 0.5)
            & (grouped.quantum_time_s < grouped.classical_time_s)
        ]

        sample = grouped.iloc[-1]
        attempts = sample["quantum_entangled_states_generated"]
        no_generation = sample.quantum_time_s - sample.quantum_entanglement_generation_time_s
        denominator = sample.classical_time_s - no_generation
        analytical = (
            attempts / denominator
            if sample.quantum_usable >= 0.5 and denominator > 0
            else None
        )
        result.append(
            {
                **dict(zip(group_cols, keys)),
                "minimum_favorable_sampled_rate_hz": (
                    float(favorable.entanglement_rate_hz.min()) if len(favorable) else None
                ),
                "estimated_continuous_break_even_rate_hz": (
                    float(analytical) if analytical is not None else None
                ),
            }
        )
    return result


def run_pilot(config: StudyConfig, root: Path) -> dict:
    learning, files = load_learning_curves(
        config, root, max_seeds=config.learning.trajectory_seeds
    )
    curves = _cumulative_curves(config, learning, root)
    sweep = run_sweep(config, root, learning, output_name="pilot_sweep.csv")
    crossings = {
        name: first_target_iteration(data["rolling_mean"], config.learning.target_reward)
        for name, data in learning.items()
    }
    default_rows = []
    for seed in config.network_seeds:
        q = NetworkSimulator(config.protocol, config.network).simulate_quantum(
            (crossings["eqmarl"] + 1) * config.learning.steps_per_iteration,
            np.random.default_rng(seed),
            crossings["eqmarl"] + 1,
        )
        c = NetworkSimulator(config.protocol, config.network).simulate_classical(
            (crossings["sctde"] + 1) * config.learning.steps_per_iteration,
            crossings["sctde"] + 1,
        )
        default_rows.append((q, c))

    def mean_accounting(position: int) -> dict:
        records = [pair[position].to_dict() for pair in default_rows]
        return {
            key: (
                float(np.mean([record[key] for record in records]))
                if key != "usable"
                else bool(all(record[key] for record in records))
            )
            for key in records[0]
        }

    summary = {
        "status": "completed",
        "learning_trajectory_source": "official committed eQMARL raw metrics",
        "learning_trajectories_are_reused_not_retrained": True,
        "trajectory_seed_count": config.learning.trajectory_seeds,
        "network_seeds": list(config.network_seeds),
        "target_reward": config.learning.target_reward,
        "crossing_iteration_zero_based": crossings,
        "default_network": asdict(config.network),
        "default_mean_time_to_target_s": {
            "eqmarl": float(np.mean([q.simulated_wall_time_s for q, _ in default_rows])),
            "sctde": float(np.mean([c.simulated_wall_time_s for _, c in default_rows])),
        },
        "default_mean_resource_accounting": {
            "eqmarl": mean_accounting(0),
            "sctde": mean_accounting(1),
        },
        "default_quantum_favorable_fraction": float(
            np.mean([q.simulated_wall_time_s < c.simulated_wall_time_s for q, c in default_rows])
        ),
        "break_even_by_slice": _break_even_summary(sweep),
        "official_source_files": files,
        "raw_outputs": [
            "results/raw/pilot_learning_curves.csv",
            "results/raw/pilot_sweep.csv",
        ],
    }
    _json_dump(root / "results/summaries/pilot_summary.json", summary)
    make_all_figures(
        root / "results/raw/pilot_learning_curves.csv",
        root / "results/raw/pilot_sweep.csv",
        root / "results/figures",
        config.learning.target_reward,
    )
    return summary
