from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

import eqmarl

from .noisy_critic import generate_noisy_CoinGame2_critic_quantum_partite_mdp
from .training import _json_safe, seed_everything, software_versions


def collect_representative_observations(count: int, seed: int) -> np.ndarray:

    seed_everything(seed)
    rng = np.random.default_rng(seed)
    env = eqmarl.environments.coin_game.vector_coin_game_make(
        {"domain_name": "CoinGame-2", "gamma": 0.99, "time_limit": 50}
    )
    observations = []

    for _ in range(int(count)):
        states, _ = env.reset()
        rollout_depth = int(rng.integers(0, 50))
        for _ in range(rollout_depth):
            actions = rng.integers(0, 4, size=2).tolist()
            states, _, done, _, _ = env.step(actions)
            if bool(np.any(done)):
                break
        observations.append(np.asarray(states, dtype=np.float32))
    return np.asarray(observations, dtype=np.float32)


def _gradient_vector(model: keras.Model, observations: np.ndarray) -> np.ndarray:
    with tf.GradientTape() as tape:
        values = model(tf.convert_to_tensor(observations))
        objective = tf.reduce_mean(values)
    gradients = tape.gradient(objective, model.trainable_variables)
    return np.concatenate(
        [
            np.ravel(gradient.numpy())
            for gradient in gradients
            if gradient is not None
        ]
    )


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def run_critic_diagnostics(
    q_values: Sequence[float],
    output_root: Path,
    observation_count: int = 24,
    noise_samples: int = 32,
    trajectories_per_expectation: int = 64,
    trajectory_bank_size: int = 1024,
    seed: int = 4242,
) -> Dict[str, Any]:

    start = time.perf_counter()
    output_root = Path(output_root)
    raw_root = output_root / "raw"
    summary_root = output_root / "summaries"
    raw_root.mkdir(parents=True, exist_ok=True)
    summary_root.mkdir(parents=True, exist_ok=True)
    observations = collect_representative_observations(observation_count, seed)
    np.savez_compressed(raw_root / "representative_observations.npz", observations=observations)

    rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    for entanglement_label, input_entanglement in (
        ("psi_plus", True),
        ("no_entanglement", False),
    ):
        keras.backend.clear_session()
        seed_everything(seed)
        ideal = generate_noisy_CoinGame2_critic_quantum_partite_mdp(
            n_agents=2,
            n_layers=5,
            q_depolarizing=0.0,
            noise_repetitions=1,
            noise_bank_size=1,
            noise_seed=seed,
            input_entanglement=input_entanglement,
            input_entanglement_type="psi+",
            name="diagnostic-ideal-%s" % entanglement_label,
        )
        ideal.build((None, 2, 36))
        ideal_values = ideal(observations).numpy().reshape(-1)
        ideal_gradient = _gradient_vector(ideal, observations)

        for q_value in [float(value) for value in q_values]:
            keras.backend.clear_session()
            seed_everything(seed)
            noisy = generate_noisy_CoinGame2_critic_quantum_partite_mdp(
                n_agents=2,
                n_layers=5,
                q_depolarizing=q_value,
                noise_repetitions=trajectories_per_expectation,
                noise_bank_size=trajectory_bank_size,
                noise_seed=seed,
                input_entanglement=input_entanglement,
                input_entanglement_type="psi+",
                name="diagnostic-noisy-%s" % entanglement_label,
            )
            noisy.build((None, 2, 36))
            noisy.set_weights(ideal.get_weights())

            seed_everything(seed)

            samples = []
            for sample_index in range(int(noise_samples)):
                values = noisy(observations).numpy().reshape(-1)
                samples.append(values)
                for observation_index, (ideal_value, noisy_value) in enumerate(
                    zip(ideal_values, values)
                ):
                    rows.append(
                        {
                            "entanglement": entanglement_label,
                            "q_depolarizing": q_value,
                            "sample": sample_index,
                            "observation": observation_index,
                            "value_ideal": float(ideal_value),
                            "value_noisy": float(noisy_value),
                            "error": float(noisy_value - ideal_value),
                            "absolute_error": float(abs(noisy_value - ideal_value)),
                        }
                    )
            sample_array = np.asarray(samples)
            mean_noisy = np.mean(sample_array, axis=0)
            error = sample_array - ideal_values[None, :]
            noisy_gradient = _gradient_vector(noisy, observations)
            gradient_delta = noisy_gradient - ideal_gradient
            gradient_denominator = float(np.linalg.norm(ideal_gradient))
            summaries.append(
                {
                    "entanglement": entanglement_label,
                    "q_depolarizing": q_value,
                    "bias": float(np.mean(error)),
                    "variance": float(np.var(sample_array, axis=0, ddof=1).mean())
                    if noise_samples > 1
                    else 0.0,
                    "mean_absolute_error": float(np.mean(np.abs(error))),
                    "correlation_with_ideal": _correlation(ideal_values, mean_noisy),
                    "gradient_l2_deviation": float(np.linalg.norm(gradient_delta)),
                    "gradient_relative_l2_deviation": (
                        float(np.linalg.norm(gradient_delta) / gradient_denominator)
                        if gradient_denominator > 0
                        else float("nan")
                    ),
                    "observations": int(observation_count),
                    "noise_samples": int(noise_samples),
                    "trajectories_per_expectation": int(
                        trajectories_per_expectation
                    ),
                }
            )

    raw_path = raw_root / "critic_diagnostics.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "status": "complete",
        "runtime_seconds": time.perf_counter() - start,
        "fresh_circuit_evaluations": True,
        "stored_learning_trajectories_used": False,
        "seed": int(seed),
        "critic_parameter_state": "common seeded initialization (untrained)",
        "q_definition": "per constituent critic qubit per network leg",
        "network_noise_legs": ["server_to_agent", "agent_to_server"],
        "versions": software_versions(),
        "conditions": summaries,
        "raw_file": str(raw_path),
        "observation_file": str(raw_root / "representative_observations.npz"),
    }
    with (summary_root / "critic_diagnostics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(_json_safe(result), handle, indent=2, allow_nan=False)
        handle.write("\n")
    return result
