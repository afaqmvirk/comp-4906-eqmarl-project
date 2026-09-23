from pathlib import Path

import numpy as np
import pytest


tf = pytest.importorskip("tensorflow")
tfq = pytest.importorskip("tensorflow_quantum")
cirq = pytest.importorskip("cirq")
eqmarl = pytest.importorskip("eqmarl")

from eqmarl_network_study.learning_noise.noisy_critic import (
    NetworkNoisyPartiteVariationalEncodingPQC,
    _sample_depolarizing_pauli_trajectory,
    generate_network_noisy_partite_circuit,
    generate_noisy_CoinGame2_critic_quantum_partite_mdp,
    network_noise_operation_count,
)
from eqmarl_network_study.learning_noise.training import (
    build_training_components,
    run_fresh_training,
    seed_everything,
)


def _observations():
    values = np.zeros((4, 2, 36), dtype=np.float32)
    for batch in range(4):
        values[batch, 0, batch] = 1.0
        values[batch, 1, 8 - batch] = 1.0
        values[batch, 0, 18 + batch] = 1.0
        values[batch, 1, 27 + batch] = 1.0
    return values


def test_q_zero_reproduces_official_critic_numerically():
    seed_everything(11)
    official = eqmarl.models.generate_model_CoinGame2_critic_quantum_partite_mdp(
        n_agents=2,
        n_layers=5,
        input_entanglement_type="psi+",
        name="official",
    )
    official.build((None, 2, 36))
    custom = generate_noisy_CoinGame2_critic_quantum_partite_mdp(
        n_agents=2,
        n_layers=5,
        q_depolarizing=0.0,
        input_entanglement_type="psi+",
        name="custom",
    )
    custom.build((None, 2, 36))
    custom.set_weights(official.get_weights())
    assert np.allclose(official(_observations()), custom(_observations()), atol=1e-7)


def test_q_positive_changes_actual_critic_expectations():
    seed_everything(12)
    ideal = generate_noisy_CoinGame2_critic_quantum_partite_mdp(
        2, 5, q_depolarizing=0.0, name="ideal"
    )
    ideal.build((None, 2, 36))
    noisy = generate_noisy_CoinGame2_critic_quantum_partite_mdp(
        2,
        5,
        q_depolarizing=0.2,
        noise_repetitions=16,
        noise_bank_size=256,
        noise_seed=12,
        name="noisy",
    )
    noisy.build((None, 2, 36))
    noisy.set_weights(ideal.get_weights())
    assert not np.allclose(ideal(_observations()), noisy(_observations()), atol=1e-4)


def test_increasing_q_increases_deviation_in_controlled_circuit():
    qubit = cirq.LineQubit(0)
    observable = cirq.X(qubit)
    simulator = cirq.DensityMatrixSimulator()
    deviations = []
    for q_value in (0.01, 0.10, 0.30):
        circuit = cirq.Circuit(
            cirq.H(qubit),
            cirq.depolarize(q_value)(qubit),
            cirq.depolarize(q_value)(qubit),
        )
        expectation = simulator.simulate_expectation_values(circuit, [observable])[0]
        deviations.append(abs(1.0 - float(np.real(expectation))))
    assert deviations[0] < deviations[1] < deviations[2]


def test_pauli_trajectory_unravelling_matches_depolarizing_channel_mean():
    qubit = cirq.LineQubit(0)
    channel_circuit = cirq.Circuit(
        cirq.H(qubit),
        cirq.depolarize(0.2)(qubit),
        cirq.depolarize(0.2)(qubit),
    )
    observable = cirq.X(qubit)
    expected = cirq.DensityMatrixSimulator().simulate_expectation_values(
        channel_circuit, [observable]
    )[0]
    rng = np.random.default_rng(123)
    simulator = cirq.Simulator()
    sampled = [
        simulator.simulate_expectation_values(
            _sample_depolarizing_pauli_trajectory(channel_circuit, rng),
            [observable],
        )[0]
        for _ in range(1024)
    ]
    assert np.mean(np.real(sampled)) == pytest.approx(float(np.real(expected)), abs=0.06)


def test_noise_is_applied_once_per_qubit_after_each_network_leg():
    model = generate_noisy_CoinGame2_critic_quantum_partite_mdp(
        2, 5, q_depolarizing=0.1, noise_bank_size=32, name="leg-count"
    )
    model.build((None, 2, 36))
    assert network_noise_operation_count(model) == 2 * 2 * 4
    layer = next(
        item
        for item in model.layers
        if isinstance(item, NetworkNoisyPartiteVariationalEncodingPQC)
    )
    moments = []
    for moment_index, moment in enumerate(layer.circuit):
        count = sum(
            isinstance(getattr(operation, "gate", None), cirq.DepolarizingChannel)
            for operation in moment.operations
        )
        if count:
            moments.append((moment_index, count))
    assert moments == [(moments[0][0], 8), (moments[1][0], 8)]
    assert moments[0][0] > 0
    assert moments[0][0] < moments[1][0]
    assert moments[1][0] == len(layer.circuit) - 1


def test_actor_initialization_is_unchanged_when_only_critic_noise_changes():
    env_ideal, actor_ideal, critic_ideal, _, _, _ = build_training_components(
        "eqmarl_psi_plus", 0.0, 1, seed=22, noise_bank_size=16
    )
    ideal_weights = [value.copy() for value in actor_ideal.get_weights()]
    ideal_critic_weights = [value.copy() for value in critic_ideal.get_weights()]
    initial_state_ideal, _ = env_ideal.reset()
    env_noisy, actor_noisy, critic_noisy, _, _, applied_q = build_training_components(
        "eqmarl_psi_plus", 0.2, 1, seed=22, noise_bank_size=16
    )
    initial_state_noisy, _ = env_noisy.reset()
    assert applied_q == 0.2
    assert np.array_equal(initial_state_ideal, initial_state_noisy)
    assert all(
        np.array_equal(left, right)
        for left, right in zip(ideal_weights, actor_noisy.get_weights())
    )
    assert all(
        np.array_equal(left, right)
        for left, right in zip(ideal_critic_weights, critic_noisy.get_weights())
    )


def test_noisy_critic_sampling_does_not_consume_actor_random_stream():
    sequences = []
    for q_value in (0.0, 0.2):
        _, actor, critic, _, _, _ = build_training_components(
            "eqmarl_psi_plus", q_value, 2, seed=23, noise_bank_size=32
        )
        probabilities = actor(_observations()[:, 0])
        first_tf = tf.random.categorical(tf.math.log(probabilities), 1).numpy()
        first_numpy = np.random.choice(
            probabilities.shape[-1], p=np.asarray(probabilities[0])
        )
        critic(_observations())
        second_tf = tf.random.categorical(tf.math.log(probabilities), 1).numpy()
        second_numpy = np.random.choice(
            probabilities.shape[-1], p=np.asarray(probabilities[0])
        )
        sequences.append((first_tf, second_tf, first_numpy, second_numpy))
    assert np.array_equal(sequences[0][0], sequences[1][0])
    assert np.array_equal(sequences[0][1], sequences[1][1])
    assert sequences[0][2:] == sequences[1][2:]


@pytest.mark.parametrize("framework", ["sctde", "qfctde"])
def test_non_distributed_baselines_do_not_receive_q(framework):
    _, _, critic_a, _, _, applied_a = build_training_components(
        framework, 0.0, 1, seed=31, noise_bank_size=8
    )
    weights_a = [value.copy() for value in critic_a.get_weights()]
    _, _, critic_b, _, _, applied_b = build_training_components(
        framework, 0.3, 1, seed=31, noise_bank_size=8
    )
    assert applied_a == applied_b == 0.0
    assert all(
        np.array_equal(left, right)
        for left, right in zip(weights_a, critic_b.get_weights())
    )


def test_q_zero_seed_is_reproducible():
    _, _, critic_a, _, _, _ = build_training_components(
        "eqmarl_psi_plus", 0.0, 1, seed=41, noise_bank_size=8
    )
    output_a = critic_a(_observations()).numpy()
    _, _, critic_b, _, _, _ = build_training_components(
        "eqmarl_psi_plus", 0.0, 1, seed=41, noise_bank_size=8
    )
    output_b = critic_b(_observations()).numpy()
    assert np.array_equal(output_a, output_b)


def test_noisy_trajectory_seed_is_reproducible_for_first_evaluation():
    _, _, critic_a, _, _, _ = build_training_components(
        "eqmarl_psi_plus", 0.1, 4, seed=42, noise_bank_size=64
    )
    output_a = critic_a(_observations()).numpy()
    _, _, critic_b, _, _, _ = build_training_components(
        "eqmarl_psi_plus", 0.1, 4, seed=42, noise_bank_size=64
    )
    output_b = critic_b(_observations()).numpy()
    assert np.array_equal(output_a, output_b)


def test_fresh_raw_result_serializes_complete_trajectory(tmp_path):
    result = run_fresh_training(
        "sctde",
        q_depolarizing=0.8,
        seed=51,
        n_episodes=2,
        noise_repetitions=1,
        noise_bank_size=8,
        output_root=Path(tmp_path),
        overwrite=True,
    )
    assert result["q_depolarizing_applied"] == 0.0
    assert len(result["trajectory"]["undiscounted_reward"]) == 2
    assert list(Path(tmp_path).glob("*.json"))
