from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


COLORS = {"eqmarl": "#2f6f9f", "sctde": "#469d62"}


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _style(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.22, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def plot_reward_iteration(curves: pd.DataFrame, output_dir: Path, target: float) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for framework, frame in curves.groupby("framework"):
        one = frame.drop_duplicates("iteration").sort_values("iteration")
        x, y, sd = one["iteration"], one["reward_rolling"], one["reward_std"]
        ax.plot(x, y, label=framework, color=COLORS[framework], lw=1.6)
        ax.fill_between(x, y - sd, y + sd, color=COLORS[framework], alpha=0.16, linewidth=0)
    ax.axhline(target, color="#666666", ls="--", lw=1, label=f"target = {target:g}")
    ax.set(xlabel="Training iteration (episode)", ylabel="CoinGame score", title="Ideal learning curves")
    ax.legend(frameon=False, ncol=3)
    _style(ax)
    _save(fig, output_dir, "01_reward_vs_iteration")


def plot_reward_time(curves: pd.DataFrame, output_dir: Path, target: float) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    grouped = curves.groupby(["framework", "iteration"], as_index=False).agg(
        reward_rolling=("reward_rolling", "first"),
        time_s=("simulated_wall_time_s", "mean"),
    )
    for framework, frame in grouped.groupby("framework"):
        ax.plot(
            frame["time_s"], frame["reward_rolling"], label=framework,
            color=COLORS[framework], lw=1.6,
        )
    ax.axhline(target, color="#666666", ls="--", lw=1)
    ax.set(xlabel="Simulated wall-clock time (s)", ylabel="CoinGame score", title="Learning with network cost")
    ax.legend(frameon=False)
    _style(ax)
    _save(fig, output_dir, "02_reward_vs_simulated_time")


def _summary(sweep: pd.DataFrame) -> pd.DataFrame:
    return sweep.groupby(["entanglement_rate_hz", "p_success"], as_index=False).agg(
        quantum_time_s=("quantum_time_s", "mean"),
        classical_time_s=("classical_time_s", "mean"),
        states_generated=("quantum_entangled_states_generated", "mean"),
        retries=("quantum_retries", "mean"),
        failed_transmissions=("quantum_failed_quantum_transmissions", "mean"),
    )


def plot_time_vs_rate(sweep: pd.DataFrame, output_dir: Path) -> None:
    data = _summary(sweep)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    choices = sorted(data.p_success.unique())
    for p in choices:
        part = data[data.p_success == p]
        ax.plot(part.entanglement_rate_hz, part.quantum_time_s, marker="o", ms=3, lw=1.2, label=f"eQMARL p={p:g}")
    ax.axhline(data.classical_time_s.iloc[0], color=COLORS["sctde"], ls="--", label="sCTDE")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(xlabel="Entanglement generation rate (states/s)", ylabel="Simulated time to target (s)", title="Time-to-target versus entanglement rate")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    _style(ax)
    _save(fig, output_dir, "03_time_to_target_vs_entanglement_rate")


def plot_time_vs_loss(sweep: pd.DataFrame, output_dir: Path) -> None:
    data = _summary(sweep)
    rate = float(np.median(sorted(data.entanglement_rate_hz.unique())))
    rate = min(data.entanglement_rate_hz.unique(), key=lambda x: abs(x - rate))
    part = data[data.entanglement_rate_hz == rate].sort_values("p_success")
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(1 - part.p_success, part.quantum_time_s, marker="o", color=COLORS["eqmarl"], label=f"eQMARL, R={rate:g}/s")
    ax.axhline(part.classical_time_s.iloc[0], color=COLORS["sctde"], ls="--", label="sCTDE")
    ax.set_yscale("log")
    ax.set(xlabel="Per-leg quantum transmission loss (1-p)", ylabel="Simulated time to target (s)", title="Time-to-target versus loss")
    ax.legend(frameon=False)
    _style(ax)
    _save(fig, output_dir, "04_time_to_target_vs_loss")


def plot_resources(sweep: pd.DataFrame, output_dir: Path) -> None:
    data = _summary(sweep)
    rate = max(data.entanglement_rate_hz.unique())
    part = data[data.entanglement_rate_hz == rate].sort_values("p_success")
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(part.p_success, part.states_generated, marker="o", label="states generated", color="#7b4ab5")
    ax.plot(part.p_success, part.retries, marker="s", label="resource retries", color="#d87b2f")
    ax.plot(part.p_success, part.failed_transmissions, marker="^", label="failed qubit legs", color="#b83f55")

    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylim(bottom=0)
    ax.set(xlabel="Per-leg success probability", ylabel="Count to target", title="Quantum resources versus network quality")
    ax.legend(frameon=False)
    _style(ax)
    _save(fig, output_dir, "05_quantum_resources_vs_network_quality")


def plot_regime_map(sweep: pd.DataFrame, output_dir: Path) -> None:
    data = _summary(sweep)
    data["speedup"] = data.classical_time_s / data.quantum_time_s
    pivot = data.pivot(index="entanglement_rate_hz", columns="p_success", values="speedup")
    x = pivot.columns.to_numpy(float)
    y = pivot.index.to_numpy(float)
    z = pivot.to_numpy(float)
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    max_abs = max(abs(np.nanmin(np.log10(z))), abs(np.nanmax(np.log10(z))), 0.05)
    mesh = ax.pcolormesh(x, y, np.log10(z), shading="nearest", cmap="RdBu", norm=TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs))
    if np.nanmin(z) <= 1 <= np.nanmax(z):
        ax.contour(x, y, z, levels=[1.0], colors="black", linewidths=1.4)
    ax.set_yscale("log")
    ax.set(xlabel="Per-leg quantum success probability", ylabel="Entanglement generation rate (states/s)", title="Preliminary break-even regime map")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("log10 speedup = log10(T_classical / T_quantum)")
    _save(fig, output_dir, "06_break_even_regime_map")


def make_all_figures(curves_csv: Path, sweep_csv: Path, output_dir: Path, target: float) -> None:
    curves = pd.read_csv(curves_csv)
    sweep = pd.read_csv(sweep_csv)
    plot_reward_iteration(curves, output_dir, target)
    plot_reward_time(curves, output_dir, target)
    plot_time_vs_rate(sweep, output_dir)
    plot_time_vs_loss(sweep, output_dir)
    plot_resources(sweep, output_dir)
    plot_regime_map(sweep, output_dir)
