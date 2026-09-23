from pathlib import Path

import numpy as np

from eqmarl_network_study.analysis.trajectories import (
    aggregate_runs,
    first_target_iteration,
    load_official_metric,
    rolling_mean,
)


ROOT = Path(__file__).resolve().parents[1]


def test_rolling_mean_matches_trailing_window_semantics():
    result = rolling_mean(np.arange(6, dtype=float), 3)
    assert np.isnan(result[:2]).all()
    assert np.allclose(result[2:], [1, 2, 3, 4])


def test_official_ideal_crossings_match_published_readme():
    repo = ROOT / "upstream/eqmarl"
    q, _ = load_official_metric(
        repo, "coingame_maa2c_mdp_eqmarl_psi+", "undiscounted_reward"
    )
    c, _ = load_official_metric(
        repo, "coingame_maa2c_mdp_sctde", "undiscounted_reward"
    )
    assert q.shape == (10, 3000)
    assert c.shape == (10, 3000)
    assert first_target_iteration(aggregate_runs(q, 10)["rolling_mean"], 20) == 568
    assert first_target_iteration(aggregate_runs(c, 10)["rolling_mean"], 20) == 1640
