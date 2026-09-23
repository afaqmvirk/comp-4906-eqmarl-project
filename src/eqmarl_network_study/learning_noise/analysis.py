from __future__ import annotations

import itertools
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd
import yaml

from eqmarl_network_study.config import load_config
from eqmarl_network_study.learning_noise.channels import (
    pauli_trajectory_bank_statistics,
)
from eqmarl_network_study.network import NetworkSimulator


def rolling_mean(values: Sequence[float], window: int) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    return series.rolling(window=window, min_periods=window).mean().to_numpy()


def crossing(values: Sequence[float], target: float, window: int) -> Optional[int]:
    smoothed = rolling_mean(values, window)
    indices = np.flatnonzero(smoothed >= float(target))
    return int(indices[0]) if len(indices) else None


def _bootstrap_ci(
    values: Sequence[float], confidence: float = 0.95, seed: int = 731
) -> Tuple[Optional[float], Optional[float]]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return None, None
    if len(array) == 1:
        value = float(array[0])
        return value, value
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(10000, len(array)), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(samples, [alpha, 1.0 - alpha])
    return float(low), float(high)


def _wilson_ci(successes: int, total: int) -> Tuple[Optional[float], Optional[float]]:
    if total <= 0:
        return None, None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return float(center - half), float(center + half)


def _paired_sign_flip(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    n = len(differences)
    if n == 0:
        return float("nan")
    observed = abs(float(np.mean(differences)))
    if n <= 20:
        statistics = [
            abs(float(np.mean(differences * np.asarray(signs))))
            for signs in itertools.product((-1.0, 1.0), repeat=n)
        ]
    else:
        rng = np.random.default_rng(811)
        signs = rng.choice((-1.0, 1.0), size=(100000, n))
        statistics = np.abs(np.mean(signs * differences[None, :], axis=1))
    return float(np.mean(np.asarray(statistics) >= observed - 1e-15))


def _paired_comparison(
    left: pd.DataFrame, right: pd.DataFrame, metric: str
) -> Dict[str, Any]:
    paired = left[["seed", metric]].merge(
        right[["seed", metric]], on="seed", suffixes=("_left", "_right")
    ).dropna()
    difference = (
        paired[metric + "_left"].to_numpy(dtype=float)
        - paired[metric + "_right"].to_numpy(dtype=float)
    )
    std = float(np.std(difference, ddof=1)) if len(difference) > 1 else float("nan")
    return {
        "metric": metric,
        "paired_seeds": paired["seed"].astype(int).tolist(),
        "n_pairs": int(len(paired)),
        "mean_difference_left_minus_right": float(np.mean(difference))
        if len(difference)
        else None,
        "cohen_dz": float(np.mean(difference) / std)
        if len(difference) > 1 and std > 0
        else None,
        "sign_flip_p_two_sided": _paired_sign_flip(difference)
        if len(difference)
        else None,
        "difference_ci95": _bootstrap_ci(difference),
    }


def load_raw_runs(raw_root: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    complete: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for path in sorted(Path(raw_root).glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                run = json.load(handle)
        except Exception as exc:
            failures.append({"file": str(path), "reason": repr(exc)})
            continue
        if (
            run.get("status") == "complete"
            and run.get("fresh_training") is True
            and run.get("stored_trajectory_used") is False
        ):
            run["_file"] = str(path)
            complete.append(run)
        else:
            if run.get("status") != "complete":
                reason = run.get("status", "unknown")
            elif run.get("fresh_training") is not True:
                reason = "fresh_training_not_true"
            else:
                reason = "stored_trajectory_used_not_false"
            failures.append(
                {"file": str(path), "reason": reason}
            )
    return complete, failures


def summarize_runs(
    runs: Sequence[Dict[str, Any]], smoothing_window: int = 10, final_window: int = 100
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    run_rows: List[Dict[str, Any]] = []
    trajectory_rows: List[Dict[str, Any]] = []
    for run in runs:
        scores = np.asarray(run["trajectory"]["undiscounted_reward"], dtype=float)
        if len(scores) != int(run["n_episodes"]):
            raise ValueError("trajectory length mismatch in %s" % run["_file"])
        smooth = rolling_mean(scores, smoothing_window)
        c20 = crossing(scores, 20.0, smoothing_window)
        c25 = crossing(scores, 25.0, smoothing_window)
        row = {
            "framework": run["framework"],
            "q_depolarizing": float(run["q_depolarizing_applied"]),
            "q_requested": float(run["q_depolarizing_requested"]),
            "seed": int(run["seed"]),
            "n_episodes": int(run["n_episodes"]),
            "runtime_seconds": float(run["runtime_seconds"]),
            "first_episode_score": float(scores[0]),
            "final_episode_score": float(scores[-1]),
            "mean_final_100": float(np.mean(scores[-final_window:])),
            "auc": float(
                np.trapezoid(scores, dx=1.0)
                if hasattr(np, "trapezoid")
                else np.trapz(scores, dx=1.0)
            ),
            "crossing_20": c20,
            "crossing_25": c25,

            "restricted_time_to_20": c20 if c20 is not None else int(len(scores)),
            "reached_20": c20 is not None,
            "reached_25": c25 is not None,
            "censored_20": c20 is None,
            "censored_25": c25 is None,
            "critic_loss_final": float(run["trajectory"]["critic_loss"][-1]),
            "actor_loss_final": float(run["trajectory"]["actor_loss"][-1]),
            "value_mean_final": float(run["trajectory"]["value_mean"][-1]),
            "value_std_final": float(run["trajectory"]["value_std"][-1]),
            "actor_gradient_norm_final": float(
                run["trajectory"]["actor_gradient_norm"][-1]
            ),
            "critic_gradient_norm_final": float(
                run["trajectory"]["critic_gradient_norm"][-1]
            ),
        }
        if (
            row["q_depolarizing"] > 0
            and run.get("noise_backend") == "pauli_trajectory_adjoint"
        ):
            bank = pauli_trajectory_bank_statistics(
                row["q_depolarizing"],
                int(run.get("noise_bank_size", 256)),
                row["seed"],
            )
            row["realized_bank_error_rate"] = bank["realized_nonidentity_rate"]
        else:
            row["realized_bank_error_rate"] = 0.0
        run_rows.append(row)
        for episode, score in enumerate(scores):
            trajectory_rows.append(
                {
                    "framework": row["framework"],
                    "q_depolarizing": row["q_depolarizing"],
                    "seed": row["seed"],
                    "episode": episode,
                    "score": float(score),
                    "score_rolling_10": float(smooth[episode])
                    if np.isfinite(smooth[episode])
                    else np.nan,
                    "critic_loss": float(run["trajectory"]["critic_loss"][episode]),
                    "actor_loss": float(run["trajectory"]["actor_loss"][episode]),
                    "value_mean": float(run["trajectory"]["value_mean"][episode]),
                    "value_std": float(run["trajectory"]["value_std"][episode]),
                    "actor_gradient_norm": float(
                        run["trajectory"]["actor_gradient_norm"][episode]
                    ),
                    "critic_gradient_norm": float(
                        run["trajectory"]["critic_gradient_norm"][episode]
                    ),
                }
            )
    return pd.DataFrame(run_rows), pd.DataFrame(trajectory_rows)


def aggregate_summary(run_frame: pd.DataFrame) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for (framework, q_value), group in run_frame.groupby(
        ["framework", "q_depolarizing"], sort=True
    ):
        record: Dict[str, Any] = {
            "framework": framework,
            "q_depolarizing": float(q_value),
            "n_seeds": int(len(group)),
            "seeds": group["seed"].astype(int).tolist(),
            "fraction_reached_20": float(group["reached_20"].mean()),
            "fraction_reached_25": float(group["reached_25"].mean()),
            "realized_bank_error_rate_mean": float(
                group["realized_bank_error_rate"].mean()
            ),
            "realized_bank_error_rate_min": float(
                group["realized_bank_error_rate"].min()
            ),
            "realized_bank_error_rate_max": float(
                group["realized_bank_error_rate"].max()
            ),
        }
        for target in (20, 25):
            successes = int(group["reached_%d" % target].sum())
            record["fraction_reached_%d_ci95_wilson" % target] = list(
                _wilson_ci(successes, len(group))
            )
        for metric in (
            "final_episode_score",
            "mean_final_100",
            "auc",
            "restricted_time_to_20",
            "runtime_seconds",
            "critic_loss_final",
            "actor_loss_final",
            "value_mean_final",
            "value_std_final",
            "actor_gradient_norm_final",
            "critic_gradient_norm_final",
        ):
            values = group[metric].to_numpy(dtype=float)
            low, high = _bootstrap_ci(values)
            record[metric + "_mean"] = float(np.mean(values))
            record[metric + "_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else None
            )
            record[metric + "_ci95"] = [low, high]
        record["restricted_time_to_20_median"] = float(
            group["restricted_time_to_20"].median()
        )
        for target in (20, 25):
            successful = group.loc[group["reached_%d" % target], "crossing_%d" % target]
            record["crossing_%d_successful_mean" % target] = (
                float(successful.mean()) if len(successful) else None
            )
            record["crossing_%d_successful_median" % target] = (
                float(successful.median()) if len(successful) else None
            )
            record["crossing_%d_successful_mean_ci95" % target] = list(
                _bootstrap_ci(successful.to_numpy(dtype=float))
            )
        best = group.loc[group["mean_final_100"].idxmax()]
        worst = group.loc[group["mean_final_100"].idxmin()]
        record["best_seed"] = int(best["seed"])
        record["worst_seed"] = int(worst["seed"])
        output.append(record)
    return output


def statistical_comparisons(run_frame: pd.DataFrame) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    eq = run_frame[run_frame["framework"] == "eqmarl_psi_plus"]
    q0 = eq[eq["q_depolarizing"] == 0.0]
    sctde = run_frame[run_frame["framework"] == "sctde"]
    for q_value, treatment in eq.groupby("q_depolarizing", sort=True):
        for comparison_name, reference in (
            ("eqmarl_q0", q0),
            ("sctde", sctde),
        ):
            if reference.empty:
                continue
            for metric in (
                "mean_final_100",
                "auc",
                "restricted_time_to_20",
                "crossing_20",
            ):
                record = _paired_comparison(treatment, reference, metric)
                record.update(
                    {
                        "treatment": "eqmarl_psi_plus",
                        "q_depolarizing": float(q_value),
                        "reference": comparison_name,
                        "censoring_handling": (
                            "non-reaching seeds retain their explicit censored flag and "
                            "are assigned the common training horizon only for this "
                            "restricted-time comparison"
                            if metric == "restricted_time_to_20"
                            else "successful paired seeds only; exploratory"
                            if metric == "crossing_20"
                            else "not applicable"
                        ),
                    }
                )
                output.append(record)
    return output


def zero_noise_reproduction_comparison(
    trajectories: pd.DataFrame, root: Path
) -> Dict[str, Any]:
    stored_path = root / "results" / "raw" / "ideal_learning_curves.csv"
    if not stored_path.exists():
        return {"status": "official_reanalysis_missing"}
    stored = pd.read_csv(stored_path)
    records: Dict[str, Any] = {}
    name_map = {"eqmarl_psi_plus": "eqmarl", "sctde": "sctde"}
    for fresh_name, stored_name in name_map.items():
        fresh = trajectories[
            (trajectories["framework"] == fresh_name)
            & (trajectories["q_depolarizing"] == 0.0)
        ]
        official = stored[stored["framework"] == stored_name]
        if fresh.empty or official.empty:
            continue
        fresh_mean = fresh.groupby("episode")["score"].mean().to_numpy()
        official_rolling = official.sort_values("iteration")[
            "reward_rolling"
        ].to_numpy()
        framework_record: Dict[str, Any] = {
            "fresh_seed_count": int(fresh["seed"].nunique()),
            "fresh_final_100_mean_curve": float(np.mean(fresh_mean[-100:])),
            "official_final_100_mean_curve": float(
                official.sort_values("iteration")["reward_mean"].tail(100).mean()
            ),
        }
        for target in (20, 25):
            framework_record["fresh_crossing_%d" % target] = crossing(
                fresh_mean, target, 10
            )
            official_indices = np.flatnonzero(official_rolling >= target)
            framework_record["official_crossing_%d" % target] = (
                int(official_indices[0]) if len(official_indices) else None
            )
        records[fresh_name] = framework_record
    eq = records.get("eqmarl_psi_plus", {})
    classical = records.get("sctde", {})
    ordering = None
    if eq.get("fresh_crossing_20") is not None and classical.get(
        "fresh_crossing_20"
    ) is not None:
        ordering = eq["fresh_crossing_20"] < classical["fresh_crossing_20"]
    return {
        "status": "fresh_vs_official_compared",
        "qualitative_score20_ordering_matches_official": ordering,
        "frameworks": records,
        "exact_crossing_reproduction_not_expected": True,
    }


def learning_threshold(run_frame: pd.DataFrame) -> Dict[str, Any]:
    eq = run_frame[run_frame["framework"] == "eqmarl_psi_plus"]
    sctde = run_frame[run_frame["framework"] == "sctde"]
    final_performance_margin = 2.0
    definition = (
        "first tested q where the paired median horizon-restricted time to score "
        "20 is not earlier than sCTDE, or the paired score-20 success fraction "
        "is lower, or median final-100 performance is at least two CoinGame "
        "score points below sCTDE. At "
        "least five paired seeds identify a pilot candidate; the "
        "threshold is confirmed only when the paired two-sided sign-flip p-value "
        "is <=0.05 in the worse direction or Wilson 95% success intervals do not "
        "overlap."
    )
    reference_seeds = set(sctde["seed"])
    adequately_sampled = [
        (float(q_value), group)
        for q_value, group in eq.groupby("q_depolarizing", sort=True)
        if len(set(group["seed"]) & reference_seeds) >= 5
    ]
    if not adequately_sampled:
        return {
            "status": "unresolved_underpowered",
            "q_learning_star": None,
            "paired_seed_count_by_q": {
                str(float(q_value)): len(set(group["seed"]) & reference_seeds)
                for q_value, group in eq.groupby("q_depolarizing", sort=True)
            },
            "definition": definition,
            "final_100_practical_margin_points": final_performance_margin,
        }
    candidates: List[Dict[str, Any]] = []
    for q_value, group in adequately_sampled:
        common = sorted(set(group["seed"]) & reference_seeds)
        treatment = group[group["seed"].isin(common)]
        reference = sctde[sctde["seed"].isin(common)]
        treatment_median = float(treatment["restricted_time_to_20"].median())
        reference_median = float(reference["restricted_time_to_20"].median())
        treatment_successes = int(treatment["reached_20"].sum())
        reference_successes = int(reference["reached_20"].sum())
        treatment_fraction = treatment_successes / len(treatment)
        reference_fraction = reference_successes / len(reference)
        treatment_final_median = float(treatment["mean_final_100"].median())
        reference_final_median = float(reference["mean_final_100"].median())
        candidate = (
            treatment_median >= reference_median
            or treatment_fraction < reference_fraction
            or treatment_final_median
            <= reference_final_median - final_performance_margin
        )
        if not candidate:
            continue
        comparison = _paired_comparison(
            treatment, reference, "restricted_time_to_20"
        )
        final_comparison = _paired_comparison(
            treatment, reference, "mean_final_100"
        )
        treatment_wilson = _wilson_ci(treatment_successes, len(treatment))
        reference_wilson = _wilson_ci(reference_successes, len(reference))
        confirmed_by_time = (
            comparison["mean_difference_left_minus_right"] is not None
            and comparison["mean_difference_left_minus_right"] >= 0
            and comparison["sign_flip_p_two_sided"] is not None
            and comparison["sign_flip_p_two_sided"] <= 0.05
        )
        confirmed_by_success = treatment_wilson[1] < reference_wilson[0]
        confirmed_by_final = (
            final_comparison["mean_difference_left_minus_right"] is not None
            and final_comparison["mean_difference_left_minus_right"]
            <= -final_performance_margin
            and final_comparison["sign_flip_p_two_sided"] is not None
            and final_comparison["sign_flip_p_two_sided"] <= 0.05
        )
        candidate_record = {
            "q_depolarizing": q_value,
            "paired_seed_count": len(common),
            "eqmarl_restricted_median": treatment_median,
            "sctde_restricted_median": reference_median,
            "eqmarl_success_fraction": treatment_fraction,
            "sctde_success_fraction": reference_fraction,
            "eqmarl_final_100_median": treatment_final_median,
            "sctde_final_100_median": reference_final_median,
            "restricted_time_sign_flip_p_two_sided": comparison[
                "sign_flip_p_two_sided"
            ],
            "confirmed_by_time": bool(confirmed_by_time),
            "confirmed_by_success": bool(confirmed_by_success),
            "final_100_sign_flip_p_two_sided": final_comparison[
                "sign_flip_p_two_sided"
            ],
            "confirmed_by_final": bool(confirmed_by_final),
            "final_100_practical_margin_points": final_performance_margin,
        }
        candidates.append(candidate_record)
        if confirmed_by_time or confirmed_by_success or confirmed_by_final:
            return {
                "status": "bracketed_at_tested_value",
                "q_learning_star": q_value,
                "candidate_q_learning_star": q_value,
                "evidence": candidate_record,
                "definition": definition,
                "final_100_practical_margin_points": final_performance_margin,
            }
    if candidates:
        return {
            "status": "candidate_but_unresolved_underpowered",
            "q_learning_star": None,
            "candidate_q_learning_star": candidates[0]["q_depolarizing"],
            "candidates": candidates,
            "definition": definition,
            "final_100_practical_margin_points": final_performance_margin,
        }
    return {
        "status": "not_observed_in_tested_range",
        "q_learning_star": None,
        "adequately_sampled_q_values": [value for value, _ in adequately_sampled],
        "definition": definition,
        "final_100_practical_margin_points": final_performance_margin,
    }


def retains_learning_advantage(
    quantum_restricted_time: float,
    classical_restricted_time: float,
    quantum_success_fraction: float,
    classical_success_fraction: float,
    quantum_final_100_median: float,
    classical_final_100_median: float,
    final_performance_margin: float = 2.0,
) -> bool:

    return bool(
        quantum_success_fraction >= classical_success_fraction
        and quantum_restricted_time < classical_restricted_time
        and quantum_final_100_median
        >= classical_final_100_median - final_performance_margin
    )


def restricted_index_to_episode_count(value: float, horizon: int) -> int:

    return min(int(value) + 1, int(horizon))


def pilot_completeness(run_frame: pd.DataFrame, root: Path) -> Dict[str, Any]:

    config_path = Path(root) / "configs" / "study2_pilot.yaml"
    if not config_path.exists():
        return {"status": "pilot_config_missing"}
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    planned_rows = config.get("treatments")
    if not planned_rows:
        return {"status": "explicit_treatment_manifest_missing"}
    planned = {
        (
            row["framework"],
            float(row.get("q_depolarizing", 0.0)),
            int(row["seed"]),
        )
        for row in planned_rows
    }
    completed = {
        (row.framework, float(row.q_depolarizing), int(row.seed))
        for row in run_frame.itertuples()
    }
    missing = sorted(planned - completed, key=lambda item: (item[0], item[1], item[2]))
    extra = sorted(completed - planned, key=lambda item: (item[0], item[1], item[2]))

    def records(items: Iterable[Tuple[str, float, int]]) -> List[Dict[str, Any]]:
        return [
            {"framework": framework, "q_depolarizing": q_value, "seed": seed}
            for framework, q_value, seed in items
        ]

    return {
        "status": "complete" if not missing else "incomplete",
        "planned_treatments": len(planned),
        "completed_planned_treatments": len(planned & completed),
        "missing_treatments": records(missing),
        "additional_complete_treatments": records(extra),
    }


def common_seed_pairing_check(run_frame: pd.DataFrame) -> Dict[str, Any]:

    quantum = run_frame[run_frame["framework"] == "eqmarl_psi_plus"]
    by_seed = []
    for seed, group in quantum.groupby("seed", sort=True):
        scores = sorted({float(value) for value in group["first_episode_score"]})
        by_seed.append(
            {
                "seed": int(seed),
                "treatment_count": int(len(group)),
                "first_episode_scores": scores,
                "match": len(scores) == 1,
            }
        )
    return {
        "all_quantum_q_treatments_match_within_seed": all(
            record["match"] for record in by_seed
        ),
        "note": (
            "episode 1 precedes any learned-parameter update; later actions are "
            "expected to diverge after noise changes training"
        ),
        "by_seed": by_seed,
    }


def _save_figure(fig: plt.Figure, root: Path, stem: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    fig.savefig(root / (stem + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(root / (stem + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def _set_q_axis(ax: plt.Axes, values: Sequence[float], axis: str = "x") -> None:
    unique = sorted({float(value) for value in values})
    if not unique:
        return
    setter = ax.set_xscale if axis == "x" else ax.set_yscale
    ticks = ax.set_xticks if axis == "x" else ax.set_yticks
    labels = ax.set_xticklabels if axis == "x" else ax.set_yticklabels
    setter("symlog", linthresh=0.001, linscale=0.8)
    ticks(unique)
    if axis == "x":
        labels(
            ["%g" % value for value in unique],
            rotation=30,
            ha="right",
        )
    else:
        labels(["%g" % value for value in unique])


def _mean_curve(group: pd.DataFrame) -> pd.DataFrame:
    return group.groupby("episode", as_index=False)["score_rolling_10"].agg(
        mean="mean", std="std", count="count"
    )


def _plot_curve(
    ax: plt.Axes,
    group: pd.DataFrame,
    label: str,
    color: Optional[str] = None,
    linewidth: float = 1.6,
) -> None:
    curve = _mean_curve(group)
    line = ax.plot(
        curve["episode"],
        curve["mean"],
        label=label,
        color=color,
        linewidth=linewidth,
    )[0]
    if int(curve["count"].max()) > 1:
        half_width = 1.96 * curve["std"].fillna(0.0) / np.sqrt(curve["count"])
        ax.fill_between(
            curve["episode"],
            curve["mean"] - half_width,
            curve["mean"] + half_width,
            color=line.get_color(),
            alpha=0.16,
            linewidth=0,
            label="95% normal CI" if label.endswith("Ψ+") else None,
        )


def make_figures(
    run_frame: pd.DataFrame,
    trajectories: pd.DataFrame,
    root: Path,
) -> List[str]:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure_root = root / "results" / "study2" / "figures"
    made: List[str] = []

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for framework, label, color in (
        ("eqmarl_psi_plus", "fresh eQMARL-Ψ+", "#006BA4"),
        ("sctde", "fresh sCTDE", "#FF800E"),
    ):
        group = trajectories[
            (trajectories["framework"] == framework)
            & (trajectories["q_depolarizing"] == 0.0)
        ]
        if group.empty:
            continue
        _plot_curve(
            ax,
            group,
            label="%s (n=%d)" % (label, group["seed"].nunique()),
            color=color,
        )
    stored_path = root / "results" / "raw" / "ideal_learning_curves.csv"
    if stored_path.exists():
        stored = pd.read_csv(stored_path)
        for framework, label, color in (
            ("eqmarl", "official 10-run eQMARL", "#006BA4"),
            ("sctde", "official 10-run sCTDE", "#FF800E"),
        ):
            subset = stored[stored["framework"] == framework]
            ax.plot(
                subset["iteration"],
                subset["reward_rolling"],
                linestyle="--",
                alpha=0.65,
                color=color,
                label=label,
            )
    ax.axhline(20, color="black", linewidth=0.8, linestyle=":")
    ax.set(xlabel="episode", ylabel="CoinGame score", title="Fresh zero-noise training")
    ax.legend(fontsize=8, ncol=2)
    _save_figure(fig, figure_root, "01_fresh_zero_noise_learning_curves")
    made.append("01_fresh_zero_noise_learning_curves")

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    eq = trajectories[trajectories["framework"] == "eqmarl_psi_plus"]
    for q_value, group in eq.groupby("q_depolarizing", sort=True):
        _plot_curve(
            ax,
            group,
            label="q=%g (n=%d)" % (q_value, group["seed"].nunique()),
        )
    ax.axhline(20, color="black", linewidth=0.8, linestyle=":")
    ax.set(xlabel="episode", ylabel="CoinGame score", title="Fresh eQMARL learning under critic-network noise")
    ax.legend(fontsize=8, ncol=2)
    _save_figure(fig, figure_root, "02_eqmarl_learning_curves_all_q")
    made.append("02_eqmarl_learning_curves_all_q")

    eq_runs = run_frame[run_frame["framework"] == "eqmarl_psi_plus"]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    plot = eq_runs.groupby("q_depolarizing")["crossing_20"].median()
    ax.plot(
        plot.index,
        plot.values,
        marker="o",
        label="eQMARL-Ψ+ median (successful seeds)",
    )
    reached = eq_runs[eq_runs["reached_20"]]
    ax.scatter(
        reached["q_depolarizing"],
        reached["crossing_20"],
        s=18,
        alpha=0.35,
        label="successful seeds",
    )
    censored = eq_runs[~eq_runs["reached_20"]]
    if not censored.empty:
        ax.scatter(
            censored["q_depolarizing"],
            censored["n_episodes"],
            s=34,
            marker="x",
            color="#B33A3A",
            label="censored at training horizon",
        )
    sctde = run_frame[run_frame["framework"] == "sctde"]
    if not sctde.empty:
        ax.axhline(
            sctde["crossing_20"].median(),
            color="#FF800E",
            label="sCTDE median (successful seeds)",
        )
    ax.set(
        xlabel="q per qubit per leg",
        ylabel="score-20 crossing episode",
        title="Convergence crossing versus critic noise",
    )
    _set_q_axis(ax, plot.index)
    ax.legend(fontsize=8)
    _save_figure(fig, figure_root, "03_score20_crossing_vs_q")
    made.append("03_score20_crossing_vs_q")

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    grouped = eq_runs.groupby("q_depolarizing")["mean_final_100"]
    means = grouped.mean()
    intervals = {
        float(q_value): _bootstrap_ci(group.to_numpy(dtype=float))
        for q_value, group in grouped
    }
    lower = [means.loc[q] - intervals[float(q)][0] for q in means.index]
    upper = [intervals[float(q)][1] - means.loc[q] for q in means.index]
    ax.errorbar(
        means.index,
        means.values,
        yerr=[lower, upper],
        marker="o",
        capsize=3,
        label="eQMARL mean (bootstrap 95% CI)",
    )
    for q_value, value in means.items():
        ax.annotate(
            "n=%d" % len(grouped.get_group(q_value)),
            (q_value, value),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=7,
        )
    if not sctde.empty:
        ax.axhline(sctde["mean_final_100"].mean(), color="#FF800E", label="sCTDE")
    ax.legend()
    ax.set(xlabel="q per qubit per leg", ylabel="mean score, final 100 episodes", title="Final learning performance")
    _set_q_axis(ax, means.index)
    _save_figure(fig, figure_root, "04_final_score_vs_q")
    made.append("04_final_score_vs_q")

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    fractions = eq_runs.groupby("q_depolarizing")["reached_20"].mean()
    intervals = [
        _wilson_ci(
            int(eq_runs[eq_runs["q_depolarizing"] == q_value]["reached_20"].sum()),
            int((eq_runs["q_depolarizing"] == q_value).sum()),
        )
        for q_value in fractions.index
    ]
    lower = [value - interval[0] for value, interval in zip(fractions, intervals)]
    upper = [interval[1] - value for value, interval in zip(fractions, intervals)]
    ax.errorbar(
        fractions.index,
        fractions.values,
        yerr=[lower, upper],
        marker="o",
        capsize=3,
        label="eQMARL-Ψ+ (Wilson 95% CI)",
    )
    if not sctde.empty:
        ax.axhline(
            sctde["reached_20"].mean(),
            color="#FF800E",
            label="sCTDE",
        )
    ax.set_ylim(-0.03, 1.03)
    ax.set(xlabel="q per qubit per leg", ylabel="fraction reaching score 20", title="Target-reaching fraction")
    _set_q_axis(ax, fractions.index)
    ax.legend()
    _save_figure(fig, figure_root, "05_fraction_reaching_target_vs_q")
    made.append("05_fraction_reaching_target_vs_q")

    diagnostic_path = root / "results" / "study2" / "raw" / "critic_diagnostics.csv"
    if diagnostic_path.exists():
        diagnostic = pd.read_csv(diagnostic_path)
        fig, (value_ax, gradient_ax) = plt.subplots(1, 2, figsize=(10.4, 4.2))
        for label, group in diagnostic.groupby("entanglement"):
            per_sample = group.groupby(
                ["q_depolarizing", "sample"], as_index=False
            )["absolute_error"].mean()
            grouped = per_sample.groupby("q_depolarizing")["absolute_error"]
            values = grouped.mean()
            half_width = 1.96 * grouped.std().fillna(0.0) / np.sqrt(grouped.count())
            value_ax.errorbar(
                values.index,
                values.values,
                yerr=half_width.values,
                marker="o",
                capsize=3,
                label={
                    "psi_plus": "Ψ+",
                    "no_entanglement": "no entanglement",
                }.get(label, label.replace("_", " ")),
            )
        value_ax.set(
            xlabel="q per qubit per leg",
            ylabel="mean absolute value error",
            title="Value error (mean ± 95% CI)",
        )
        _set_q_axis(value_ax, diagnostic["q_depolarizing"].unique())
        value_ax.legend()

        diagnostic_summary_path = (
            root / "results" / "study2" / "summaries" / "critic_diagnostics.json"
        )
        if diagnostic_summary_path.exists():
            with diagnostic_summary_path.open("r", encoding="utf-8") as handle:
                conditions = pd.DataFrame(json.load(handle)["conditions"])
            for label, group in conditions.groupby("entanglement"):
                gradient_ax.plot(
                    group["q_depolarizing"],
                    group["gradient_relative_l2_deviation"],
                    marker="o",
                    label={
                        "psi_plus": "Ψ+",
                        "no_entanglement": "no entanglement",
                    }.get(label, label.replace("_", " ")),
                )
            gradient_ax.set(
                xlabel="q per qubit per leg",
                ylabel="relative L2 gradient deviation",
                title="Value-gradient sensitivity",
            )
            _set_q_axis(gradient_ax, conditions["q_depolarizing"].unique())
            gradient_ax.legend()
        else:
            gradient_ax.axis("off")
        fig.suptitle("Untrained critic sensitivity diagnostic")
        _save_figure(fig, figure_root, "06_critic_value_error_vs_q")
        made.append("06_critic_value_error_vs_q")

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for framework, group in trajectories.groupby("framework"):
        if framework not in ("eqmarl_psi_plus", "sctde"):
            continue
        if framework == "eqmarl_psi_plus":
            group = group[group["q_depolarizing"].isin(sorted(group["q_depolarizing"].unique()))]
            for q_value, subgroup in group.groupby("q_depolarizing"):
                _plot_curve(
                    ax,
                    subgroup,
                    label="q=%g (n=%d)" % (q_value, subgroup["seed"].nunique()),
                )
        else:
            _plot_curve(ax, group, label="sCTDE", color="black", linewidth=2.2)
    ax.axhline(20, color="grey", linewidth=0.8, linestyle=":")
    ax.set(xlabel="episode", ylabel="CoinGame score", title="eQMARL versus sCTDE convergence")
    ax.legend(fontsize=8, ncol=2)
    _save_figure(fig, figure_root, "07_eqmarl_vs_sctde_convergence")
    made.append("07_eqmarl_vs_sctde_convergence")
    return made


def combine_with_study1(run_frame: pd.DataFrame, root: Path) -> Optional[pd.DataFrame]:
    eq = run_frame[run_frame["framework"] == "eqmarl_psi_plus"]
    classical = run_frame[run_frame["framework"] == "sctde"]
    if eq.empty or classical.empty:
        return None
    config = load_config(root / "configs" / "pilot.yaml")
    classical_restricted = float(classical["restricted_time_to_20"].median())
    classical_success = float(classical["reached_20"].mean())
    classical_final_median = float(classical["mean_final_100"].median())
    classical_successful_crossing = classical.loc[
        classical["reached_20"], "crossing_20"
    ].median()
    classical_episodes = restricted_index_to_episode_count(
        classical_restricted, int(classical["n_episodes"].min())
    )
    classical_steps = int(classical_episodes * config.learning.steps_per_iteration)
    classical_time = NetworkSimulator(config.protocol, config.network).simulate_classical(
        classical_steps
    ).simulated_wall_time_s
    rates = np.geomspace(10.0, 100000.0, 81)
    rows = []
    for q_value, group in eq.groupby("q_depolarizing", sort=True):
        quantum_restricted = float(group["restricted_time_to_20"].median())
        quantum_success = float(group["reached_20"].mean())
        quantum_final_median = float(group["mean_final_100"].median())
        quantum_successful_crossing = group.loc[
            group["reached_20"], "crossing_20"
        ].median()
        quantum_episodes = restricted_index_to_episode_count(
            quantum_restricted, int(group["n_episodes"].min())
        )
        quantum_steps = int(quantum_episodes * config.learning.steps_per_iteration)
        learning_advantage = retains_learning_advantage(
            quantum_restricted_time=quantum_restricted,
            classical_restricted_time=classical_restricted,
            quantum_success_fraction=quantum_success,
            classical_success_fraction=classical_success,
            quantum_final_100_median=quantum_final_median,
            classical_final_100_median=classical_final_median,
        )
        for rate in rates:
            network = replace(config.network, entanglement_rate_hz=float(rate))
            times = []
            usable = True
            for seed in config.network_seeds:
                outcome = NetworkSimulator(config.protocol, network).simulate_quantum(
                    quantum_steps, np.random.default_rng(seed)
                )
                usable = usable and outcome.usable
                times.append(outcome.simulated_wall_time_s)
            quantum_time = float(np.mean(times)) if usable else float("inf")
            if not usable:
                regime = 4
            elif not learning_advantage:
                regime = 3
            elif quantum_time < classical_time:
                regime = 1
            else:
                regime = 2
            rows.append(
                {
                    "q_depolarizing": float(q_value),
                    "entanglement_rate_hz": float(rate),
                    "p_success_per_qubit_per_leg": float(
                        config.network.transmission_success_probability
                    ),
                    "distance_km": float(config.network.distance_km),
                    "memory_t2_s": float(config.network.memory_t2_s),
                    "eqmarl_crossing_20_successful_median": (
                        float(quantum_successful_crossing)
                        if np.isfinite(quantum_successful_crossing)
                        else np.nan
                    ),
                    "sctde_crossing_20_successful_median": (
                        float(classical_successful_crossing)
                        if np.isfinite(classical_successful_crossing)
                        else np.nan
                    ),
                    "eqmarl_restricted_time_to_20_median": quantum_restricted,
                    "sctde_restricted_time_to_20_median": classical_restricted,
                    "eqmarl_fraction_reached_20": quantum_success,
                    "sctde_fraction_reached_20": classical_success,
                    "eqmarl_final_100_median": quantum_final_median,
                    "sctde_final_100_median": classical_final_median,
                    "final_100_practical_margin_points": 2.0,
                    "eqmarl_learning_seed_count": int(group["seed"].nunique()),
                    "sctde_learning_seed_count": int(classical["seed"].nunique()),
                    "eqmarl_training_steps": quantum_steps,
                    "sctde_training_steps": classical_steps,
                    "eqmarl_wall_time_s": quantum_time,
                    "sctde_wall_time_s": classical_time,
                    "regime": regime,
                }
            )
    frame = pd.DataFrame(rows)
    output = root / "results" / "study2" / "summaries" / "combined_regime_map.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    pivot = frame.pivot(
        index="q_depolarizing", columns="entanglement_rate_hz", values="regime"
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    cmap = ListedColormap(["#2E8B57", "#E6AB02", "#D95F02", "#777777"])
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)
    mesh = ax.pcolormesh(
        pivot.columns,
        pivot.index,
        pivot.to_numpy(),
        cmap=cmap,
        norm=norm,
        shading="nearest",
    )
    ax.set_xscale("log")
    _set_q_axis(ax, pivot.index, axis="y")
    q_seed_counts = eq.groupby("q_depolarizing")["seed"].nunique()
    ax.set_yticklabels(
        ["%g (n=%d)" % (q, q_seed_counts.loc[q]) for q in pivot.index]
    )
    ax.set(
        xlabel="usable-at-source entanglement rate (states/s)",
        ylabel="q per qubit per leg",
        title=(
            "Combined learning and wall-clock regimes\n"
            "p_success=%.2f per qubit/leg, distance=%g km, T2=%g s"
            % (
                config.network.transmission_success_probability,
                config.network.distance_km,
                config.network.memory_t2_s,
            )
        ),
    )
    colorbar = fig.colorbar(mesh, ax=ax, ticks=[1, 2, 3, 4])
    colorbar.ax.set_yticklabels(
        ["learns + faster", "learns, overhead loses", "learning edge gone", "infeasible"]
    )
    _save_figure(fig, root / "results" / "study2" / "figures", "08_combined_regime_map")
    return frame


def analyze_study2(root: Path, minimum_episodes: int = 1) -> Dict[str, Any]:
    root = Path(root)
    runs, failures = load_raw_runs(root / "results" / "study2" / "raw")
    runs = [run for run in runs if int(run["n_episodes"]) >= int(minimum_episodes)]
    if not runs:
        raise RuntimeError("no complete Study 2 runs satisfy the episode filter")
    run_frame, trajectories = summarize_runs(runs)
    summary_root = root / "results" / "study2" / "summaries"
    raw_root = root / "results" / "study2" / "raw"
    summary_root.mkdir(parents=True, exist_ok=True)
    run_frame.to_csv(summary_root / "per_run_metrics.csv", index=False)
    trajectories.to_csv(raw_root / "learning_trajectories.csv", index=False)
    aggregate = aggregate_summary(run_frame)
    comparisons = statistical_comparisons(run_frame)
    zero_noise = zero_noise_reproduction_comparison(trajectories, root)
    threshold = learning_threshold(run_frame)
    completeness = pilot_completeness(run_frame, root)
    pairing = common_seed_pairing_check(run_frame)
    figures = make_figures(run_frame, trajectories, root)
    combined = combine_with_study1(run_frame, root)
    if combined is not None:
        figures.append("08_combined_regime_map")
    result = {
        "status": "complete",
        "minimum_episodes_filter": int(minimum_episodes),
        "fresh_runs_analyzed": int(len(run_frame)),
        "total_recorded_training_runtime_seconds": float(
            run_frame["runtime_seconds"].sum()
        ),
        "failed_or_invalid_file_count": int(len(failures)),
        "failed_or_invalid_files": failures,
        "pilot_completeness": completeness,
        "common_seed_pairing": pairing,
        "aggregate": aggregate,
        "paired_comparisons": comparisons,
        "zero_noise_reproduction": zero_noise,
        "q_learning_star": threshold,
        "figures": figures,
        "combined_study1_study2": combined is not None,
    }
    with (summary_root / "study2_analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return result
