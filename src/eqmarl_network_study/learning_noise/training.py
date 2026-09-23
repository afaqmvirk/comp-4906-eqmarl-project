from __future__ import annotations

import json
import math
import os
import platform
import random
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cirq
import gymnasium
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_quantum as tfq

import eqmarl
from eqmarl.algorithms import MAA2C

from .channels import validate_q_depolarizing
from .noisy_critic import generate_noisy_CoinGame2_critic_quantum_partite_mdp


FRAMEWORKS = ("eqmarl_psi_plus", "eqmarl_noentanglement", "sctde", "qfctde")
DISTRIBUTED_QUANTUM_FRAMEWORKS = ("eqmarl_psi_plus", "eqmarl_noentanglement")


def seed_everything(seed: int) -> None:

    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _quantum_optimizers() -> List[keras.optimizers.Optimizer]:
    return [
        keras.optimizers.Adam(learning_rate=1.0e-2),
        keras.optimizers.Adam(learning_rate=1.0e-1),
        keras.optimizers.Adam(learning_rate=1.0e-1),
    ]


def build_training_components(
    framework: str,
    q_depolarizing: float,
    noise_repetitions: int,
    seed: int,
    noise_bank_size: int = 256,
    noise_backend: str = "pauli_trajectory_adjoint",
) -> Tuple[Any, keras.Model, keras.Model, Any, Any, float]:

    if framework not in FRAMEWORKS:
        raise ValueError("unsupported framework %r" % framework)
    requested_q = validate_q_depolarizing(q_depolarizing)
    applied_q = requested_q if framework in DISTRIBUTED_QUANTUM_FRAMEWORKS else 0.0

    keras.backend.clear_session()
    seed_everything(seed)
    env = eqmarl.environments.coin_game.vector_coin_game_make(
        {"domain_name": "CoinGame-2", "gamma": 0.99, "time_limit": 50}
    )

    if framework == "sctde":
        actor = eqmarl.models.generate_model_CoinGame2_actor_classical_shared_mdp(
            n_actions=4, units=[12], name="actor-classical-shared"
        )
        critic = eqmarl.models.generate_model_CoinGame2_critic_classical_joint_mdp(
            n_agents=2, units=[12], name="critic-classical-joint"
        )
        actor_optimizer = keras.optimizers.Adam(learning_rate=1.0e-3)
        critic_optimizer = keras.optimizers.Adam(learning_rate=1.0e-3)
    else:
        actor = eqmarl.models.generate_model_CoinGame2_actor_quantum_shared_mdp(
            n_layers=5,
            squash_activation="arctan",
            name="actor-quantum-shared",
        )
        actor_optimizer = _quantum_optimizers()
        if framework == "qfctde":
            critic = eqmarl.models.generate_model_CoinGame2_critic_quantum_central_mdp(
                n_agents=2,
                n_layers=5,
                squash_activation="arctan",
                name="critic-quantum-joint",
            )
        else:
            critic = generate_noisy_CoinGame2_critic_quantum_partite_mdp(
                n_agents=2,
                n_layers=5,
                q_depolarizing=applied_q,
                noise_repetitions=noise_repetitions,
                noise_bank_size=noise_bank_size,
                noise_seed=seed,
                noise_backend=noise_backend,
                squash_activation="arctan",
                input_entanglement=(framework == "eqmarl_psi_plus"),
                input_entanglement_type="psi+",
                name="critic-quantum-joint",
            )
        critic_optimizer = _quantum_optimizers()

    actor.build((None, 36))
    critic.build((None, 2, 36))

    seed_everything(seed)
    return env, actor, critic, actor_optimizer, critic_optimizer, applied_q


class InstrumentedMAA2C(MAA2C):

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.actor_loss_history: List[float] = []
        self.critic_loss_history: List[float] = []
        self.value_mean_history: List[float] = []
        self.value_std_history: List[float] = []
        self.actor_gradient_norm_history: List[float] = []
        self.critic_gradient_norm_history: List[float] = []

    def update(self, batch: List[Any]) -> None:

        packed = {
            key: [asdict(item)[key] for item in batch]
            for key in asdict(batch[0]).keys()
        }
        states_batched = tf.convert_to_tensor(np.array(packed["states"]))
        actions_batched = tf.convert_to_tensor(np.array(packed["actions"]))
        rewards_batched = tf.convert_to_tensor(
            np.array(packed["rewards"], dtype="float32")
        )
        next_states_batched = tf.convert_to_tensor(np.array(packed["next_states"]))
        dones_batched = tf.convert_to_tensor(
            np.array(packed["dones"], dtype="float32")
        )
        rewards_batched = tf.reduce_sum(rewards_batched, axis=-1, keepdims=True)
        dones_batched = tf.cast(
            tf.reduce_sum(dones_batched, axis=-1, keepdims=True) > 0.0, "float32"
        )
        huber_loss = tf.keras.losses.Huber(reduction=keras.losses.Reduction.SUM)

        with tf.GradientTape() as tape_critic, tf.GradientTape() as tape_actor:
            tape_critic.watch(self.model_critic.trainable_variables)
            tape_actor.watch(self.model_actor.trainable_variables)
            joint_state_values = self.model_critic(states_batched)
            joint_next_state_values = self.model_critic(next_states_batched)

            agents_action_probs = tf.TensorArray(
                dtype=tf.float32, size=0, dynamic_size=True
            )
            agents_action_probs_log = tf.TensorArray(
                dtype=tf.float32, size=0, dynamic_size=True
            )
            action_probs = None
            for agent_index in range(self.n_envs):
                action_probs = self.model_actor(states_batched[:, agent_index])
                agents_action_probs = agents_action_probs.write(
                    agent_index, action_probs
                )
                indices = np.array(
                    [
                        (index, action)
                        for index, action in enumerate(
                            actions_batched[:, agent_index]
                        )
                    ]
                )
                chosen = tf.gather_nd(action_probs, indices)
                agents_action_probs_log = agents_action_probs_log.write(
                    agent_index, tf.math.log(chosen)
                )
            agents_action_probs_log = tf.transpose(
                agents_action_probs_log.stack()
            )
            agents_action_probs = tf.transpose(
                agents_action_probs.stack(), [1, 0, 2]
            )

            q_values = (
                rewards_batched
                + (1.0 - dones_batched) * self.gamma * joint_next_state_values
            )
            advantage = q_values - joint_state_values

            entropy = -tf.reduce_sum(
                action_probs * tf.math.log(action_probs), axis=-1
            )
            entropy = tf.reduce_mean(entropy)
            actor_loss = (
                tf.reduce_mean(-agents_action_probs_log * advantage)
                + self.alpha * entropy
            )
            critic_loss = huber_loss(joint_state_values, q_values)

        grads_actor = tape_actor.gradient(
            actor_loss, self.model_actor.trainable_variables
        )
        grads_critic = tape_critic.gradient(
            critic_loss, self.model_critic.trainable_variables
        )
        if isinstance(self.optimizer_actor, (list, tuple)):
            for index, optimizer in enumerate(self.optimizer_actor):
                optimizer.apply_gradients(
                    [(grads_actor[index], self.model_actor.trainable_variables[index])]
                )
        else:
            self.optimizer_actor.apply_gradients(
                zip(grads_actor, self.model_actor.trainable_variables)
            )
        if isinstance(self.optimizer_critic, (list, tuple)):
            for index, optimizer in enumerate(self.optimizer_critic):
                optimizer.apply_gradients(
                    [(grads_critic[index], self.model_critic.trainable_variables[index])]
                )
        else:
            self.optimizer_critic.apply_gradients(
                zip(grads_critic, self.model_critic.trainable_variables)
            )

        actor_grads = [gradient for gradient in grads_actor if gradient is not None]
        critic_grads = [gradient for gradient in grads_critic if gradient is not None]
        self.actor_loss_history.append(float(actor_loss.numpy()))
        self.critic_loss_history.append(float(critic_loss.numpy()))
        self.value_mean_history.append(float(tf.reduce_mean(joint_state_values).numpy()))
        self.value_std_history.append(
            float(tf.math.reduce_std(joint_state_values).numpy())
        )
        self.actor_gradient_norm_history.append(
            float(tf.linalg.global_norm(actor_grads).numpy())
        )
        self.critic_gradient_norm_history.append(
            float(tf.linalg.global_norm(critic_grads).numpy())
        )


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def q_slug(q_depolarizing: float) -> str:
    return ("%.6f" % float(q_depolarizing)).rstrip("0").rstrip(".").replace(".", "p")


def raw_result_path(
    output_root: Path, framework: str, q_depolarizing: float, seed: int
) -> Path:
    return output_root / (
        "%s_q%s_seed%d.json" % (framework, q_slug(q_depolarizing), int(seed))
    )


def software_versions() -> Dict[str, str]:
    return {
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "tensorflow_quantum": tfq.__version__,
        "cirq": cirq.__version__,
        "gymnasium": gymnasium.__version__,
        "numpy": np.__version__,
        "eqmarl_upstream_commit": "6b1b2e3817f661da81aad5e7188a362380dbdd1d",
    }


def run_fresh_training(
    framework: str,
    q_depolarizing: float,
    seed: int,
    n_episodes: int,
    noise_repetitions: int,
    output_root: Path,
    noise_bank_size: int = 256,
    noise_backend: str = "pauli_trajectory_adjoint",
    overwrite: bool = False,
    quiet: bool = True,
) -> Dict[str, Any]:

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    path = raw_result_path(output_root, framework, q_depolarizing, seed)
    if path.exists() and not overwrite:
        with path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        expected_applied_q = (
            float(q_depolarizing)
            if framework in DISTRIBUTED_QUANTUM_FRAMEWORKS
            else 0.0
        )
        noise_compatible = expected_applied_q == 0.0 or (
            existing.get("noise_repetitions") == int(noise_repetitions)
            and existing.get("noise_bank_size") == int(noise_bank_size)
            and existing.get("noise_backend") == noise_backend
        )
        reusable = (
            existing.get("status") == "complete"
            and existing.get("n_episodes") == int(n_episodes)
            and noise_compatible
        )
        if reusable:
            return existing

    start = time.perf_counter()
    try:
        env, actor, critic, actor_optimizer, critic_optimizer, applied_q = (
            build_training_components(
                framework=framework,
                q_depolarizing=q_depolarizing,
                noise_repetitions=noise_repetitions,
                seed=seed,
                noise_bank_size=noise_bank_size,
                noise_backend=noise_backend,
            )
        )
        algorithm = InstrumentedMAA2C(
            env=env,
            model_actor=actor,
            model_critic=critic,
            optimizer_actor=actor_optimizer,
            optimizer_critic=critic_optimizer,
            gamma=0.99,
            alpha=0.001,
            episode_metrics_callback=(
                eqmarl.environments.coin_game.episode_metrics_callback
            ),
        )
        reward_history, metrics_history = algorithm.train(
            n_episodes=int(n_episodes),
            callbacks=[],
            tqdm_kwargs={"disable": bool(quiet)},
        )
    except Exception as exc:
        failure = {
            "status": "failed",
            "fresh_training": True,
            "stored_trajectory_used": False,
            "framework": framework,
            "seed": int(seed),
            "q_depolarizing_requested": float(q_depolarizing),
            "q_depolarizing_applied": (
                float(q_depolarizing)
                if framework in DISTRIBUTED_QUANTUM_FRAMEWORKS
                else 0.0
            ),
            "noise_repetitions": int(noise_repetitions),
            "noise_bank_size": int(noise_bank_size),
            "noise_backend": noise_backend,
            "n_episodes_requested": int(n_episodes),
            "steps_per_episode": 50,
            "runtime_seconds_before_failure": time.perf_counter() - start,
            "exception": repr(exc),
            "traceback": traceback.format_exc(),
            "versions": software_versions(),
        }
        failure_path = path.with_name(path.stem + ".failure.json")
        with failure_path.open("w", encoding="utf-8") as handle:
            json.dump(failure, handle, indent=2, allow_nan=False)
            handle.write("\n")
        raise
    elapsed = time.perf_counter() - start
    trajectory = {
        "reward": reward_history,
        **metrics_history,
        "actor_loss": algorithm.actor_loss_history,
        "critic_loss": algorithm.critic_loss_history,
        "value_mean": algorithm.value_mean_history,
        "value_std": algorithm.value_std_history,
        "actor_gradient_norm": algorithm.actor_gradient_norm_history,
        "critic_gradient_norm": algorithm.critic_gradient_norm_history,
    }
    nonfinite_fields = [
        name
        for name, values in trajectory.items()
        if not np.isfinite(np.asarray(values, dtype=float)).all()
    ]
    completed_episodes = len(trajectory.get("undiscounted_reward", []))
    if nonfinite_fields:
        run_status = "diverged_nonfinite"
    elif completed_episodes != int(n_episodes):
        run_status = "incomplete_interrupted"
    else:
        run_status = "complete"
    result: Dict[str, Any] = {
        "status": run_status,
        "fresh_training": True,
        "stored_trajectory_used": False,
        "framework": framework,
        "seed": int(seed),
        "q_depolarizing_requested": float(q_depolarizing),
        "q_depolarizing_applied": float(applied_q),
        "q_definition": "per constituent critic qubit per network leg",
        "network_noise_legs": (
            ["server_to_agent", "agent_to_server"] if applied_q > 0 else []
        ),
        "noise_repetitions": int(noise_repetitions),
        "noise_bank_size": int(noise_bank_size),
        "noise_backend": noise_backend,
        "n_episodes": int(completed_episodes),
        "n_episodes_requested": int(n_episodes),
        "steps_per_episode": 50,
        "environment": "CoinGame-2 fully observed MDP",
        "agents": 2,
        "qubits_per_agent": 4 if framework != "sctde" else 0,
        "critic_vqc_blocks": 5 if framework != "sctde" else None,
        "gamma": 0.99,
        "entropy_coefficient": 0.001,
        "runtime_seconds": elapsed,
        "versions": software_versions(),
        "nonfinite_fields": nonfinite_fields,
        "trajectory": trajectory,
    }
    safe = _json_safe(result)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(safe, handle, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)
    return safe
