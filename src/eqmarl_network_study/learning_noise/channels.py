from __future__ import annotations

from typing import Tuple

import numpy as np


NETWORK_LEGS: Tuple[str, str] = ("server_to_agent", "agent_to_server")


def validate_q_depolarizing(q_depolarizing: float) -> float:

    q = float(q_depolarizing)
    if not 0.0 <= q <= 1.0:
        raise ValueError("q_depolarizing must be in [0, 1]")
    return q


def cirq_depolarizing_pauli_probabilities(
    q_depolarizing: float,
) -> Tuple[float, float, float, float]:

    q = validate_q_depolarizing(q_depolarizing)
    return (1.0 - q, q / 3.0, q / 3.0, q / 3.0)


def depolarizing_bloch_shrinkage(q_depolarizing: float) -> float:

    q = validate_q_depolarizing(q_depolarizing)
    return 1.0 - 4.0 * q / 3.0


def pauli_trajectory_bank_statistics(
    q_depolarizing: float,
    bank_size: int,
    seed: int,
    channel_locations_per_trajectory: int = 16,
):

    q = validate_q_depolarizing(q_depolarizing)
    if bank_size <= 0 or channel_locations_per_trajectory <= 0:
        raise ValueError("bank and trajectory sizes must be positive")
    rng = np.random.default_rng(int(seed))
    counts = np.zeros(4, dtype=int)
    probabilities = [1.0 - q, q / 3.0, q / 3.0, q / 3.0]
    for _ in range(int(bank_size) * int(channel_locations_per_trajectory)):
        counts[int(rng.choice(4, p=probabilities))] += 1
    total = int(counts.sum())
    return {
        "identity": int(counts[0]),
        "x": int(counts[1]),
        "y": int(counts[2]),
        "z": int(counts[3]),
        "total_channel_locations": total,
        "realized_nonidentity_rate": float(counts[1:].sum() / total),
    }
