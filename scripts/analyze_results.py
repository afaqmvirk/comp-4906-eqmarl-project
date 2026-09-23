#!/usr/bin/env python

import argparse
import itertools
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from _bootstrap import ROOT
from eqmarl_network_study.learning_noise.exact_study import (
    comparable_metrics, inventory, make_manifest, manifest_matches, read_json, sha256,
)

UPSTREAM = {
    "upstream/eqmarl": "6b1b2e3817f661da81aad5e7188a362380dbdd1d",
    "upstream/eqmarl-paper-iclr2025": "62917c087f7bda38338df7a738692169d90f307c",
}


def paper_crossing(trajectories, target):
    values = np.asarray(trajectories, dtype=float)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError("Expected finite run-by-episode trajectories")
    curve = pd.Series(values.mean(axis=0)).rolling(10).mean().to_numpy()
    reached = np.flatnonzero(curve >= target)
    return int(reached[0]) if len(reached) else None


def paired_curve_test(quantum, classical, target):
    quantum, classical = np.asarray(quantum), np.asarray(classical)
    if quantum.shape != classical.shape or quantum.ndim != 2:
        raise ValueError("Matched run-by-episode arrays required")
    horizon = quantum.shape[1]

    def restricted(array):
        crossing = paper_crossing(array, target)
        return horizon if crossing is None else crossing + 1

    observed = restricted(quantum) - restricted(classical)
    statistics = []
    censored = 0
    for swap in itertools.product((False, True), repeat=len(quantum)):
        mask = np.asarray(swap)[:, None]
        left = np.where(mask, classical, quantum)
        right = np.where(mask, quantum, classical)
        left_cross = paper_crossing(left, target)
        right_cross = paper_crossing(right, target)
        censored += int(left_cross is None or right_cross is None)
        left_count = horizon if left_cross is None else left_cross + 1
        right_count = horizon if right_cross is None else right_cross + 1
        statistics.append(left_count - right_count)
    return {
        "p_two_sided": float(np.mean(np.abs(statistics) >= abs(observed))),
        "permutations": len(statistics),
        "permutations_with_non_crossing": censored,
    }


def verify_upstream(root):
    for directory, expected in UPSTREAM.items():
        command = ["git", "-C", str(root / directory)]
        actual = subprocess.check_output(command + ["rev-parse", "HEAD"], text=True).strip()
        if actual != expected:
            raise ValueError("Unexpected upstream revision: " + directory)
        subprocess.run(command + ["diff", "--exit-code", "HEAD", "--"], check=True,
                       stdout=subprocess.DEVNULL)


def analyze(root):
    root = Path(root)
    verify_upstream(root)
    saved = read_json(root / "results/study2_exact/summary.json")
    manifest = read_json(root / "results/study2_exact/manifest.json")
    if not manifest_matches(root, make_manifest(root, manifest["config"]), manifest):
        raise ValueError("Training implementation, input hashes, or configuration changed")
    for filename, expected in saved["source_sha256"].items():
        if sha256(root / filename) != expected:
            raise ValueError("Raw result hash mismatch: " + filename)
    if any(row["status"] != "complete" for row in inventory(root, manifest)):
        raise ValueError("Incomplete exact-study results or missing final weights")

    grouped = {}
    for entry in manifest["records"]:
        record = read_json(root / entry["path"])
        for window in (10, 100):
            expected = next(row for row in saved["windows"][str(window)]["runs"]
                            if row["id"] == entry["id"])
            actual = comparable_metrics(record, window)
            if any(expected[key] != value for key, value in actual.items()):
                raise ValueError("Saved per-run metric mismatch: " + entry["id"])
        key = "classical" if entry["framework"] == "sctde" else format(entry["q"], "g")
        grouped.setdefault(key, {})[entry["seed"]] = record["trajectory"]["undiscounted_reward"]
    seeds = [101, 202, 303, 404, 505]
    arrays = {key: np.asarray([group[s] for s in seeds], float)
              for key, group in grouped.items() if sorted(group) == seeds}
    classical = arrays["classical"]
    rows = []
    for key in ("classical", "0", "0.01", "0.03", "0.1"):
        row = {"condition": key, "runs": 5,
               "final100": float(arrays[key][:, -100:].mean())}
        for target in (20, 25):
            crossing = paper_crossing(arrays[key], target)
            reference = paper_crossing(classical, target)
            row[f"reward{target}_index"] = crossing
            if key != "classical":
                row[f"reward{target}_savings_percent"] = (
                    None if crossing is None or reference is None
                    else 100 * (reference - crossing) / (reference + 1)
                )
                row[f"reward{target}_test"] = paired_curve_test(arrays[key], classical, target)
        rows.append(row)
    previous = read_json(root / "results/study2_exact/paper_method_comparison.json")
    for old, new in zip(previous["rows"], rows):
        for field in ("condition", "runs", "reward20_index", "reward25_index"):
            if old[field] != new[field]:
                raise ValueError("Paper-method comparison changed")

    provenance = read_json(root / "results/paper_reference/provenance.json")
    paper = {}
    for framework, entry in provenance["runs"].items():
        trajectories = []
        for source in entry["summary"]["files"]:
            path = root / entry["directory"] / source["name"]
            if sha256(path) != source["sha256"]:
                raise ValueError("Paper reproduction raw hash mismatch: " + str(path))
            trajectories.append(read_json(path)["metrics"]["undiscounted_reward"])
        values = np.asarray(trajectories, dtype=float)
        if values.shape != (10, 3000) or not np.isfinite(values).all():
            raise ValueError("Expected ten complete paper reproduction rounds")
        crossings = {str(target): paper_crossing(values, target) for target in (20, 25)}
        if crossings != entry["summary"]["paper_matched_crossing_index"]:
            raise ValueError("Paper reproduction crossing mismatch")
        paper[framework] = {"runs": 10, "crossings": crossings,
                            "final100": float(values[:, -100:].mean())}
    pilot = read_json(root / "results/summaries/pilot_summary.json")
    sweep = read_json(root / "results/summaries/sweep_summary.json")
    return {
        "method": previous["method"],
        "statistical_note": "Exploratory paired whole-curve label swaps; not predeclared; no multiplicity adjustment.",
        "exact_noise": rows,
        "fresh_paper_reproduction": paper,
        "network": {
            "mean_seconds": pilot["default_mean_time_to_target_s"],
            "parameter_combinations": sweep["unique_parameter_points"],
            "seeded_rows": sweep["seeded_rows"],
            "favorable_fraction_of_usable_grid": sweep["quantum_favorable_fraction_usable"],
        },
    }, arrays


def plot(arrays, target):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for ax, key, color in zip(axes.flat, ("0", "0.01", "0.03", "0.1"),
                              ("#2563eb", "#08916c", "#b16b06", "#b43871")):
        for condition, label, shade in (("classical", "Classical sCTDE", "#555555"),
                                         (key, "Exact-noisy eQMARL", color)):
            smooth = pd.DataFrame(arrays[condition].T).rolling(10).mean()
            mean, spread = smooth.mean(axis=1), smooth.std(axis=1, ddof=0)
            ax.plot(mean.index, mean, color=shade, lw=1.5, label=label)
            ax.fill_between(mean.index, mean-spread, mean+spread, color=shade, alpha=.12)
        for reward in (20, 25):
            ax.axhline(reward, color="0.6", ls="--", lw=.7)
        ax.set_title("q = " + key)
        ax.grid(alpha=.15)
    axes[0, 0].legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("Training episode (zero-based)")
    for ax in axes[:, 0]:
        ax.set_ylabel("10-episode mean team reward")
    fig.suptitle("Paper-style learning curves: five seeds per condition")
    fig.tight_layout()
    fig.savefig(target, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Verify without writing files")
    mode.add_argument("--write", action="store_true", help="Rebuild the summary and plot")
    args = parser.parse_args()
    result, arrays = analyze(ROOT)
    output = ROOT / "results/review"
    if args.write:
        output.mkdir(parents=True, exist_ok=True)
        (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        plot(arrays, output / "learning_curves.png")
    else:
        if read_json(output / "summary.json") != result:
            raise ValueError("Review summary differs from recomputed results")
    print("Verified 28 study records, 20 reproduction runs, and source revisions.")


if __name__ == "__main__":
    main()
