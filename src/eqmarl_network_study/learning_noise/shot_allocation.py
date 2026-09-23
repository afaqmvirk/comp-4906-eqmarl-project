import numpy as np


def parity_sample(mean, shots, rng):
    mean, shots = np.broadcast_arrays(np.asarray(mean, float), np.asarray(shots, int))
    if np.any(shots < 0) or np.any(np.abs(mean) > 1 + 1e-6):
        raise ValueError("Invalid parity mean or shot count")
    count = rng.binomial(shots, (1 + np.clip(mean, -1, 1)) / 2)
    return np.divide(2 * count, shots, out=np.zeros(mean.shape), where=shots > 0) - (shots > 0)


def td_advantage(expectations, rewards, discounts, scale):
    values = scale * (1 + expectations) / 2
    return rewards + discounts * values[..., 1] - values[..., 0]


def adaptive_counts(pilot, rewards, discounts, scale, pilot_shots=64,
                    minimum=64, maximum=8192, margin_floor=.02, z=1.96,
                    allocation_margin="raw"):
    if not 0 < minimum <= maximum or pilot_shots < 2 or margin_floor <= 0:
        raise ValueError("Invalid allocation settings")
    delta = td_advantage(pilot, rewards, discounts, scale)
    variance = np.maximum(1 - pilot**2, 0) * pilot_shots / (pilot_shots - 1)
    numerator = (scale/2)**2 * (variance[..., 0] + discounts**2 * variance[..., 1])
    margin = np.abs(delta)
    if allocation_margin == "lower_confidence":
        margin = margin - z*np.sqrt(numerator/pilot_shots)
    elif allocation_margin != "raw":
        raise ValueError("Unknown allocation margin")
    requested = np.ceil(z*z*numerator / np.maximum(margin, margin_floor)**2)
    count = np.clip(requested, minimum, maximum).astype(int)
    active = np.stack([np.ones_like(discounts, bool), discounts != 0], axis=-1)
    return count[..., None] * active


def compare_allocations(expectations, rewards, discounts, scale, trials=500, seed=8319,
                        pilot_shots=64, minimum=64, maximum=8192, margin_floor=.02,
                        allocation_margin="raw"):
    if trials < 2:
        raise ValueError("At least two Monte Carlo trials required")
    e = np.asarray(expectations, float)
    rewards, discounts = np.asarray(rewards, float), np.asarray(discounts, float)
    if len(rewards) == 0 or e.shape != (len(rewards), 2) or discounts.shape != rewards.shape:
        raise ValueError("Expected N pairs of current/next expectations")
    if not np.isfinite(scale) or not np.isfinite(rewards).all() or not np.isfinite(discounts).all():
        raise ValueError("Non-finite scale, rewards, or discounts")
    active = np.stack([np.ones_like(discounts, bool), discounts != 0], -1)
    true_delta = td_advantage(e, rewards, discounts, scale)
    rng = np.random.default_rng(seed)
    records = {name: [] for name in ["adaptive", "fixed"]}
    costs, cap_fraction = [], []
    signed_errors = {name: np.zeros(len(e)) for name in records}
    for trial in range(trials):
        pilot_counts = pilot_shots * active
        pilot = parity_sample(e, pilot_counts, rng)
        counts = adaptive_counts(pilot, rewards, discounts, scale, pilot_shots,
                                 minimum, maximum, margin_floor, allocation_margin=allocation_margin)
        total = int(np.sum(counts + pilot_counts))

        per_readout, remainder = divmod(total, int(active.sum()))
        fixed = per_readout * active.astype(int)
        indices = np.flatnonzero(active)
        if remainder:
            fixed.flat[rng.choice(indices, size=remainder, replace=False)] += 1
        assert fixed.sum() == total
        costs.append(total / len(e))
        cap_fraction.append(float(np.mean(counts[:, 0] == maximum)))
        for name, allocation in [("adaptive", counts), ("fixed", fixed)]:
            estimate = td_advantage(parity_sample(e, allocation, rng), rewards, discounts, scale)
            error = estimate - true_delta
            wrong = np.sign(estimate) != np.sign(true_delta)
            significant = np.abs(true_delta) >= margin_floor
            signed_errors[name] += error
            records[name].append(dict(mse=float(np.mean(error**2)),
                sign_error=float(np.mean(wrong)),
                sign_error_margin_ge_floor=float(np.mean(wrong[significant])) if significant.any() else None,
                advantage_weighted_sign_error=float(np.sum(np.abs(true_delta)*wrong) / np.sum(np.abs(true_delta))) if np.any(true_delta) else None))
    summary = {}
    for name, rows in records.items():
        summary[name] = {}
        for key in rows[0]:
            if rows[0][key] is not None:
                samples = np.array([row[key] for row in rows])
                summary[name][key] = dict(mean=float(samples.mean()), monte_carlo_se=float(samples.std(ddof=1)/np.sqrt(trials)))
        summary[name]["rmse"] = float(np.sqrt(summary[name]["mse"]["mean"]))
        summary[name]["mean_bias"] = float(np.mean(signed_errors[name]/trials))
    return dict(scope="frozen-critic measurement simulation, no policy or gradient updates",
        trials=trials, seed=seed, transitions=len(e), scale=float(scale),
        settings=dict(pilot_shots_per_active_value=pilot_shots, min_final_shots=minimum,
            max_final_shots=maximum, margin_floor=margin_floor, z=1.96,
            allocation_margin=allocation_margin,
            independent_pilot_and_final=True, pilot_included_in_budget=True),
        mean_total_shots_per_transition=float(np.mean(costs)),
        mean_fraction_hitting_cap=float(np.mean(cap_fraction)),
        expectation_min=float(e.min()), expectation_max=float(e.max()),
        median_absolute_true_advantage=float(np.median(np.abs(true_delta))),
        fraction_margin_ge_floor=float(np.mean(np.abs(true_delta)>=margin_floor)),
        results=summary,
        limitations=["Repeated measurements of one frozen critic are not independent training seeds.",
            "No finite-shot parameter-shift gradients or noisy training are simulated here.",
            "Allocation uses a normal-approximation heuristic, not a confidence guarantee."])
