import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from eqmarl_network_study.learning_noise.analysis import (
    common_seed_pairing_check,
    crossing,
    learning_threshold,
    load_raw_runs,
    retains_learning_advantage,
    restricted_index_to_episode_count,
    summarize_runs,
)
from eqmarl_network_study.learning_noise.channels import (
    cirq_depolarizing_pauli_probabilities,
    depolarizing_bloch_shrinkage,
    pauli_trajectory_bank_statistics,
    validate_q_depolarizing,
)


def test_depolarizing_parameter_is_distinct_and_validated():
    assert cirq_depolarizing_pauli_probabilities(0.3) == pytest.approx(
        (0.7, 0.1, 0.1, 0.1)
    )
    assert depolarizing_bloch_shrinkage(0.3) == pytest.approx(0.6)
    with pytest.raises(ValueError):
        validate_q_depolarizing(-0.01)
    with pytest.raises(ValueError):
        validate_q_depolarizing(1.01)


def test_censored_crossing_is_explicit_none():
    assert crossing([0.0] * 30, target=20.0, window=10) is None
    assert crossing([0.0] * 9 + [20.0] * 10, target=20.0, window=10) == 18


def test_raw_json_can_represent_censored_value(tmp_path):
    value = {"crossing_20": None, "censored_20": True}
    path = tmp_path / "run.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8")) == value


def test_pauli_bank_statistics_are_seeded_and_close_to_requested_rate():
    first = pauli_trajectory_bank_statistics(0.03, 4096, 17)
    second = pauli_trajectory_bank_statistics(0.03, 4096, 17)
    assert first == second
    assert first["total_channel_locations"] == 4096 * 16
    assert first["realized_nonidentity_rate"] == pytest.approx(0.03, abs=0.002)


def test_run_summary_preserves_non_crossing_as_censored():
    episodes = 30
    run = {
        "_file": "synthetic.json",
        "framework": "eqmarl_psi_plus",
        "seed": 1,
        "n_episodes": episodes,
        "runtime_seconds": 1.0,
        "q_depolarizing_applied": 0.1,
        "q_depolarizing_requested": 0.1,
        "noise_backend": "pauli_trajectory_adjoint",
        "noise_bank_size": 32,
        "trajectory": {
            "undiscounted_reward": [0.0] * episodes,
            "critic_loss": [1.0] * episodes,
            "actor_loss": [1.0] * episodes,
            "value_mean": [0.0] * episodes,
            "value_std": [0.0] * episodes,
            "actor_gradient_norm": [0.0] * episodes,
            "critic_gradient_norm": [0.0] * episodes,
        },
    }
    summary, trajectories = summarize_runs([run])
    assert bool(summary.loc[0, "censored_20"])
    assert summary.loc[0, "crossing_20"] is None
    assert summary.loc[0, "restricted_time_to_20"] == episodes
    assert summary.loc[0, "first_episode_score"] == 0.0
    assert len(trajectories) == episodes


def test_five_seed_threshold_candidate_is_not_overclaimed():
    rows = []
    for seed in range(5):
        rows.append(
            {
                "framework": "sctde",
                "seed": seed,
                "q_depolarizing": 0.0,
                "restricted_time_to_20": 1000,
                "reached_20": True,
                "mean_final_100": 25.0,
            }
        )
        rows.append(
            {
                "framework": "eqmarl_psi_plus",
                "seed": seed,
                "q_depolarizing": 0.1,
                "restricted_time_to_20": 3000,
                "reached_20": True,
                "mean_final_100": 25.0,
            }
        )
    result = learning_threshold(pd.DataFrame(rows))
    assert result["status"] == "candidate_but_unresolved_underpowered"
    assert result["q_learning_star"] is None
    assert result["candidate_q_learning_star"] == 0.1


def test_executed_pilot_config_encodes_the_unbalanced_priority_grid():
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs" / "study2_pilot.yaml").read_text())
    treatments = config["treatments"]
    quantum = [row for row in treatments if row["framework"] == "eqmarl_psi_plus"]
    classical = [row for row in treatments if row["framework"] == "sctde"]

    assert len(treatments) == 28
    assert len(classical) == 5
    counts = {
        float(q): sum(float(row["q_depolarizing"]) == float(q) for row in quantum)
        for q in config["q_depolarizing"]
    }
    assert counts == {0.0: 5, 0.001: 1, 0.003: 1, 0.01: 5, 0.03: 5, 0.05: 1, 0.1: 5}
    assert config["diagnostics"] == {
        "observations": 16,
        "noise_samples": 16,
        "trajectories_per_expectation": 8,
    }


def test_combined_map_rejects_transient_or_unreliable_learning_advantage():
    common = {
        "quantum_restricted_time": 600,
        "classical_restricted_time": 1700,
        "quantum_success_fraction": 1.0,
        "classical_success_fraction": 1.0,
        "quantum_final_100_median": 24.0,
        "classical_final_100_median": 25.0,
    }
    assert retains_learning_advantage(**common)
    assert not retains_learning_advantage(
        **{**common, "quantum_final_100_median": 10.0}
    )
    assert not retains_learning_advantage(
        **{**common, "quantum_success_fraction": 0.8}
    )


def test_common_seed_pairing_check_detects_a_mismatched_initial_episode():
    frame = pd.DataFrame(
        [
            {
                "framework": "eqmarl_psi_plus",
                "seed": 1,
                "first_episode_score": -2.0,
            },
            {
                "framework": "eqmarl_psi_plus",
                "seed": 1,
                "first_episode_score": -1.0,
            },
        ]
    )
    assert not common_seed_pairing_check(frame)[
        "all_quantum_q_treatments_match_within_seed"
    ]


def test_restricted_crossing_conversion_caps_a_censored_run_at_the_horizon():
    assert restricted_index_to_episode_count(467, 3000) == 468
    assert restricted_index_to_episode_count(3000, 3000) == 3000


def test_analysis_rejects_a_stored_trajectory_labeled_as_fresh(tmp_path):
    path = tmp_path / "shortcut.json"
    path.write_text(
        json.dumps(
            {
                "status": "complete",
                "fresh_training": True,
                "stored_trajectory_used": True,
            }
        ),
        encoding="utf-8",
    )
    accepted, rejected = load_raw_runs(tmp_path)
    assert accepted == []
    assert rejected[0]["reason"] == "stored_trajectory_used_not_false"
