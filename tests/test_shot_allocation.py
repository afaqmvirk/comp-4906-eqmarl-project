import numpy as np
import pytest
from eqmarl_network_study.learning_noise.shot_allocation import parity_sample, adaptive_counts, compare_allocations


def test_parity_samples_are_unbiased_and_have_expected_variance():
    samples = parity_sample(np.full(100000,.3),40,np.random.default_rng(19))
    assert abs(samples.mean()-.3) < .002
    assert abs(samples.var()-(1-.3**2)/40) < .0004


def test_parity_handles_zero_shots_and_invalid_values():
    np.testing.assert_array_equal(parity_sample([1,-1,0],[3,7,0],np.random.default_rng(1)),[1,-1,0])
    with pytest.raises(ValueError):
        parity_sample([2],[1],np.random.default_rng(1))


def test_adaptive_budget_cap_and_terminal_no_next_measurement():
    counts = adaptive_counts(np.zeros((2,2)), np.zeros(2), np.array([.99,0.]),100.)
    assert counts[0,0] == 8192
    assert counts[1,1] == 0
    assert np.all((counts == 0) | ((counts >= 64) & (counts <= 8192)))


def test_comparison_is_reproducible_and_counts_pilot():
    e = np.array([[.1,.2],[-.1,.3],[.5,.5]])
    kwargs = dict(expectations=e,rewards=[0,1,-1],discounts=[.99,.99,0],scale=2,trials=20,seed=12)
    one = compare_allocations(**kwargs)
    assert one == compare_allocations(**kwargs)
    assert one["mean_total_shots_per_transition"] >= (5/3)*128
    assert one["settings"]["pilot_included_in_budget"]


def test_uncertainty_aware_margin_never_requests_fewer_shots():
    rng = np.random.default_rng(10)
    pilot = rng.uniform(-1,1,(200,2))
    rewards, discounts = np.zeros(200), np.full(200,.99)
    raw = adaptive_counts(pilot,rewards,discounts,10.)
    cautious = adaptive_counts(pilot,rewards,discounts,10.,allocation_margin="lower_confidence")
    assert np.all(cautious >= raw)
