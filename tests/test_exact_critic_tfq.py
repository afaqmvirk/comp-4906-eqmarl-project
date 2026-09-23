import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
cirq = pytest.importorskip("cirq")
pytest.importorskip("tensorflow_quantum")
from eqmarl_network_study.learning_noise.noisy_critic import generate_noisy_CoinGame2_critic_quantum_partite_mdp


def make_model(q, backend="exact_density", entangled=True, layers=2):
    tf.random.set_seed(619)
    return generate_noisy_CoinGame2_critic_quantum_partite_mdp(2, layers, q_depolarizing=q, noise_backend=backend, input_entanglement_type="psi+", input_entanglement=entangled)


@pytest.mark.parametrize("q", [0., .03, .1, .75, 1.])
@pytest.mark.parametrize("entangled", [True, False])
def test_exact_matches_full_density(q, entangled):
    model = make_model(q, entangled=entangled)
    x = tf.constant(np.random.default_rng(901).uniform(-1,1,(2,2,36)), tf.float32)
    encoded = model.layers[0](x)
    layer = model.layers[1]
    actual = layer(encoded).numpy()[:,0]
    for row in range(2):
        enc = np.arctan(layer.w_enc.numpy() * encoded.numpy()[row,:,None,:,:])
        values = np.concatenate([layer.w_var.numpy().ravel(), enc.ravel()])
        resolver = dict(zip(layer.symbols, values))
        circuit = cirq.resolve_parameters(layer.circuit, resolver)
        result = cirq.DensityMatrixSimulator(dtype=np.complex128).simulate(circuit, qubit_order=sorted(circuit.all_qubits()))
        parity = np.array([(-1)**bin(i).count("1") for i in range(256)])
        reference = np.real(np.dot(np.diag(result.final_density_matrix), parity))
        np.testing.assert_allclose(actual[row], reference, atol=3e-6)


@pytest.mark.parametrize("layers", [2, 5])
def test_exact_gradients_match_tfq_at_zero_noise(layers):
    exact = make_model(0., layers=layers)
    reference = make_model(0., "pauli_trajectory_adjoint", layers=layers)
    reference.set_weights(exact.get_weights())
    x = tf.constant(np.random.default_rng(902).uniform(-1,1,(3,2,36)), tf.float32)
    gradients = []
    outputs = []
    for model in [exact, reference]:
        with tf.GradientTape() as tape:
            output = model(x)
            loss = tf.reduce_sum(output**2)
        outputs.append(output.numpy())
        gradients.append(tape.gradient(loss, model.trainable_variables))
    np.testing.assert_allclose(outputs[0], outputs[1], atol=3e-6)
    for left, right in zip(*gradients):
        np.testing.assert_allclose(left.numpy(), right.numpy(), atol=2e-5)


def test_exact_noisy_gradient_matches_full_density_difference():
    model = make_model(.1)
    x = tf.constant(np.random.default_rng(903).uniform(-1,1,(1,2,36)), tf.float32)
    layer = model.layers[1]
    with tf.GradientTape() as tape:
        out = tf.reduce_sum(layer(model.layers[0](x)))
    gradient = tape.gradient(out, layer.w_var).numpy()
    encoded = model.layers[0](x).numpy()[0]
    enc = np.arctan(layer.w_enc.numpy() * encoded[:,None,:,:])
    weights = layer.w_var.numpy()
    for index in [(0,0,0,0), (1,1,2,1), (1,2,3,2)]:
        measured = []
        for sign in [-1, 1]:
            changed = weights.copy()
            changed[index] += sign * .001
            resolver = dict(zip(layer.symbols, np.concatenate([changed.ravel(), enc.ravel()])))
            circuit = cirq.resolve_parameters(layer.circuit, resolver)
            result = cirq.DensityMatrixSimulator(dtype=np.complex128).simulate(circuit, qubit_order=sorted(circuit.all_qubits()))
            parity = np.array([(-1)**bin(i).count("1") for i in range(256)])
            measured.append(np.real(np.dot(np.diag(result.final_density_matrix), parity)))
        np.testing.assert_allclose(gradient[index], (measured[1]-measured[0])/.002, atol=2e-5)
