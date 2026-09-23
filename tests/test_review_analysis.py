import importlib.util
from pathlib import Path
import sys

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("review_analysis", SCRIPTS / "analyze_results.py")
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


def test_averaged_curve_crossing_is_not_average_individual_crossing():
    early = np.r_[np.zeros(10), np.full(10, 40.), np.zeros(20)]
    late = np.r_[np.zeros(30), np.full(10, 40.)]
    individual = [review.paper_crossing([row], 20) for row in (early, late)]
    combined = review.paper_crossing([early, late], 20)
    assert combined == 19
    assert combined != np.mean(individual)


def test_paired_curve_test_uses_training_pairs_and_handles_censoring():
    classical = np.tile(np.arange(30), (5, 1))
    quantum = classical + 5
    result = review.paired_curve_test(quantum, classical, 20)
    assert result["permutations"] == 32
    assert result["p_two_sided"] == .0625
    censored = np.zeros((5, 30))
    assert review.paper_crossing(censored, 20) is None
    result = review.paired_curve_test(np.full((5, 30), 25.), censored, 20)
    assert result["permutations_with_non_crossing"] == 32


def test_identical_trajectories_have_no_difference():
    values = np.tile(np.arange(30), (5, 1))
    assert review.paired_curve_test(values, values, 20)["p_two_sided"] == 1.
