import cirq
import numpy as np
import tensorflow as tf

from eqmarl.ops import entangle_agents_psi_plus
from .channels import validate_q_depolarizing


def batch_kron(a, b):
    product = tf.einsum("...ij,...kl->...ikjl", a, b)
    shape = tf.shape(a)
    return tf.reshape(product, tf.concat([shape[:-2], [shape[-2] * tf.shape(b)[-2], shape[-1] * tf.shape(b)[-1]]], axis=0))


def rotation_layer(angles):
    half = angles / 2.0
    c = tf.cast(tf.cos(half), tf.complex64)
    s = tf.cast(tf.sin(half), tf.complex64)
    rx = tf.stack([tf.stack([c[..., 0], -1j * s[..., 0]], -1), tf.stack([-1j * s[..., 0], c[..., 0]], -1)], -2)
    ry = tf.stack([tf.stack([c[..., 1], -s[..., 1]], -1), tf.stack([s[..., 1], c[..., 1]], -1)], -2)
    zero = tf.zeros_like(c[..., 2])
    rz = tf.stack([tf.stack([c[..., 2] - 1j*s[..., 2], zero], -1), tf.stack([zero, c[..., 2] + 1j*s[..., 2]], -1)], -2)
    gates = tf.matmul(rz, tf.matmul(ry, rx))
    result = gates[..., 0, :, :]
    for index in range(1, 4):
        result = batch_kron(result, gates[..., index, :, :])
    return result


class ExactTwoPartyExpectation:

    def __init__(self, q, n_layers, input_entanglement=True):
        self.q = validate_q_depolarizing(q)
        self.n_layers = int(n_layers)
        qubits = cirq.LineQubit.range(8)
        circuit = cirq.Circuit(entangle_agents_psi_plus(qubits, 4, 2) if input_entanglement else cirq.I.on_each(*qubits))
        circuit.append(cirq.depolarize(self.q).on_each(*qubits))
        rho = cirq.DensityMatrixSimulator(dtype=np.complex128).simulate(circuit, qubit_order=qubits).final_density_matrix

        self.kernel = tf.constant(rho.reshape(16,16,16,16).transpose(2,0,3,1).reshape(256,256), tf.complex64)
        parity = np.array([(-1)**bin(index).count("1") for index in range(16)], dtype=np.complex64)
        self.parity = tf.constant(parity)
        cz = np.ones(16, dtype=np.complex64)
        for index in range(16):
            bits = [(index >> (3-i)) & 1 for i in range(4)]
            cz[index] = (-1)**sum(bits[i]*bits[(i+1)%4] for i in range(4))
        self.cz = tf.constant(cz)
        self.return_factor = tf.constant((1 - 4*self.q/3)**8, tf.float32)

    def __call__(self, angles_var, angles_enc):
        unitary = rotation_layer(angles_var[:, :, 0])
        for layer in range(self.n_layers):
            if layer:
                unitary = tf.matmul(rotation_layer(angles_var[:, :, layer]), unitary)
            unitary = self.cz[None,None,:,None] * unitary
            unitary = tf.matmul(rotation_layer(angles_enc[:, :, layer]), unitary)
        unitary = tf.matmul(rotation_layer(angles_var[:, :, self.n_layers]), unitary)
        observable = tf.matmul(unitary, self.parity[None,None,:,None] * unitary, adjoint_a=True)
        a = tf.reshape(observable[:, 0], (-1, 256))
        b = tf.reshape(observable[:, 1], (-1, 256))
        values = tf.math.real(tf.reduce_sum(tf.matmul(a, self.kernel) * b, axis=-1))
        return values[:, None] * self.return_factor
