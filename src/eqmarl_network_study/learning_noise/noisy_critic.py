from __future__ import annotations

import functools as ft
from typing import Any, List, Optional, Sequence, Tuple, Type

import cirq
import numpy as np
import sympy
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_quantum as tfq

from eqmarl.layers import RescaleWeighted
from eqmarl.models import map_CoinGame2_obs_to_encoded_vector
from eqmarl.ops import (
    EncodingLayer,
    ParameterizedOperationGate,
    ParameterizedRotationLayer_RxRyRz,
    VariationalRotationLayer,
    circular_entangling_layer,
    entangle_agents_phi_minus,
    entangle_agents_phi_plus,
    entangle_agents_psi_minus,
    entangle_agents_psi_plus,
)

from .channels import NETWORK_LEGS, validate_q_depolarizing


_ENTANGLERS = {
    "phi+": entangle_agents_phi_plus,
    "phi-": entangle_agents_phi_minus,
    "psi+": entangle_agents_psi_plus,
    "psi-": entangle_agents_psi_minus,
}


def _decompose_operations(operations: Sequence[Any]) -> List[cirq.Operation]:
    return [
        decomposed
        for operation in cirq.ops.flatten_to_ops(operations)
        for decomposed in cirq.ops.flatten_to_ops(cirq.decompose(operation))
    ]


def generate_network_noisy_partite_circuit(
    qubits: Sequence[cirq.Qid],
    n_parts: int,
    d_qubits: int,
    n_layers: int,
    q_depolarizing: float = 0.0,
    decompose: bool = True,
    variational_layer_cls: Type[ParameterizedOperationGate] = VariationalRotationLayer,
    encoding_layer_cls: Type[ParameterizedOperationGate] = EncodingLayer,
    input_entanglement: bool = True,
    input_entanglement_type: str = "phi+",
) -> Tuple[cirq.Circuit, Tuple[np.ndarray, np.ndarray]]:

    q = validate_q_depolarizing(q_depolarizing)
    if len(qubits) != n_parts * d_qubits:
        raise ValueError("len(qubits) must equal n_parts * d_qubits")
    if n_parts <= 0 or d_qubits <= 0 or n_layers <= 0:
        raise ValueError("n_parts, d_qubits, and n_layers must be positive")

    shape_var = variational_layer_cls.get_shape(d_qubits)
    shape_enc = encoding_layer_cls.get_shape(d_qubits)
    theta_var = sympy.symbols(
        "var^{(0:%d)}(0:%d)_" % (n_parts, n_layers + 1)
        + "_".join("(0:%d)" % x for x in shape_var)
    )
    theta_var = np.asarray(theta_var).reshape((n_parts, n_layers + 1, *shape_var))
    theta_enc = sympy.symbols(
        "enc^{(0:%d)}(0:%d)_" % (n_parts, n_layers)
        + "_".join("(0:%d)" % x for x in shape_enc)
    )
    theta_enc = np.asarray(theta_enc).reshape((n_parts, n_layers, *shape_enc))

    preparation_ops: List[Any] = []
    if input_entanglement:
        try:
            entangler = _ENTANGLERS[input_entanglement_type]
        except KeyError as exc:
            raise ValueError(
                "unsupported input entanglement type %s" % input_entanglement_type
            ) from exc
        preparation_ops.extend(entangler(list(qubits), d_qubits, n_parts))

    local_ops: List[Any] = []
    for part_index in range(n_parts):
        first = part_index * d_qubits
        local_qubits = qubits[first : first + d_qubits]
        for layer_index in range(n_layers):
            local_ops.append(
                variational_layer_cls(
                    theta_var[part_index, layer_index],
                    name="%d-v%d" % (part_index, layer_index),
                )(*local_qubits)
            )
            local_ops.append(circular_entangling_layer(local_qubits))
            local_ops.append(
                encoding_layer_cls(
                    theta_enc[part_index, layer_index],
                    name="%d-e%d" % (part_index, layer_index),
                )(*local_qubits)
            )
        local_ops.append(
            variational_layer_cls(
                theta_var[part_index, n_layers], name="v%d" % n_layers
            )(*local_qubits)
        )

    if decompose:
        preparation_ops = _decompose_operations(preparation_ops)
        local_ops = _decompose_operations(local_ops)

    if q == 0.0:

        return cirq.Circuit(list(preparation_ops) + list(local_ops)), (
            theta_var,
            theta_enc,
        )

    circuit = cirq.Circuit(preparation_ops)
    circuit.append(
        cirq.Moment(cirq.depolarize(q).on_each(*qubits)),
        strategy=cirq.InsertStrategy.NEW,
    )
    circuit.append(local_ops)
    circuit.append(
        cirq.Moment(cirq.depolarize(q).on_each(*qubits)),
        strategy=cirq.InsertStrategy.NEW,
    )
    return circuit, (theta_var, theta_enc)


class NetworkNoisyPartiteVariationalEncodingPQC(keras.layers.Layer):

    def __init__(
        self,
        qubits: Sequence[cirq.Qid],
        n_parts: int,
        d_qubits: int,
        n_layers: int,
        observables: Sequence[Any],
        q_depolarizing: float = 0.0,
        noise_repetitions: int = 32,
        noise_bank_size: int = 256,
        noise_seed: int = 0,
        noise_backend: str = "pauli_trajectory_adjoint",
        name: Optional[str] = None,
        squash_activation: str = "linear",
        variational_layer_cls: Type[ParameterizedOperationGate] = VariationalRotationLayer,
        encoding_layer_cls: Type[ParameterizedOperationGate] = EncodingLayer,
        trainable_w_enc: bool = True,
        input_entanglement: bool = True,
        input_entanglement_type: str = "phi+",
    ):
        super().__init__(name=name or self.__class__.__name__)
        self.n_layers = n_layers
        self.n_qubits = len(qubits)
        self.n_parts = n_parts
        self.d_qubits = d_qubits
        self.q_depolarizing = validate_q_depolarizing(q_depolarizing)
        self.noise_repetitions = int(noise_repetitions)
        self.noise_bank_size = int(noise_bank_size)
        self.noise_seed = int(noise_seed)
        self.noise_backend = str(noise_backend)
        self.network_noise_locations = NETWORK_LEGS if self.q_depolarizing > 0 else ()
        self.squash_activation = squash_activation
        if self.noise_repetitions <= 0:
            raise ValueError("noise_repetitions must be positive")
        if self.noise_bank_size <= 0:
            raise ValueError("noise_bank_size must be positive")
        if self.noise_backend not in (
            "pauli_trajectory_adjoint",
            "tfq_noisy_parameter_shift",
            "exact_density",
        ):
            raise ValueError("unsupported noise_backend %r" % self.noise_backend)

        circuit, (symbols_var, symbols_enc) = generate_network_noisy_partite_circuit(
            qubits=qubits,
            n_parts=n_parts,
            d_qubits=d_qubits,
            n_layers=n_layers,
            q_depolarizing=self.q_depolarizing,
            decompose=True,
            variational_layer_cls=variational_layer_cls,
            encoding_layer_cls=encoding_layer_cls,
            input_entanglement=input_entanglement,
            input_entanglement_type=input_entanglement_type,
        )
        self.circuit = circuit
        self.w_var = tf.Variable(
            initial_value=tf.random_uniform_initializer(minval=0.0, maxval=np.pi)(
                shape=symbols_var.shape, dtype="float32"
            ),
            trainable=True,
            name="w_var",
        )
        self.w_enc = tf.Variable(
            initial_value=tf.ones(shape=symbols_enc.shape, dtype="float32"),
            trainable=trainable_w_enc,
            name="w_enc",
        )
        self.symbols = [
            str(symbol)
            for symbol in np.concatenate((symbols_var.flatten(), symbols_enc.flatten()))
        ]
        self.sorted_symbols = sorted(self.symbols)
        self.symbols_idx = tf.constant(
            [self.symbols.index(symbol) for symbol in self.sorted_symbols]
        )
        self.sorted_symbols_tensor = tf.convert_to_tensor(self.sorted_symbols)
        self.empty_circuit_tensor = tfq.convert_to_tensor([cirq.Circuit()])
        self.observables = list(observables)

        if self.noise_backend == "exact_density":
            if self.n_parts != 2 or self.d_qubits != 4 or input_entanglement_type != "psi+":
                raise ValueError("exact_density requires two four-qubit psi+ branches")
            if variational_layer_cls is not ParameterizedRotationLayer_RxRyRz or encoding_layer_cls is not ParameterizedRotationLayer_RxRyRz:
                raise ValueError("exact_density requires RxRyRz variational and encoding layers")
            expected_observable = ft.reduce(lambda a, b: a*b, [cirq.Z(q) for q in qubits])
            if list(observables) != [expected_observable]:
                raise ValueError("exact_density requires the global Z**8 observable")
            from .exact_critic import ExactTwoPartyExpectation
            self.exact_computation = ExactTwoPartyExpectation(self.q_depolarizing, n_layers, input_entanglement)
        elif self.q_depolarizing == 0.0:
            self.computation_layer = tfq.layers.ControlledPQC(circuit, observables)
        elif self.noise_backend == "pauli_trajectory_adjoint":
            rng = np.random.default_rng(self.noise_seed)
            bank = [
                _sample_depolarizing_pauli_trajectory(circuit, rng)
                for _ in range(self.noise_bank_size)
            ]
            self.noise_circuit_bank = tfq.convert_to_tensor(bank)

            object.__setattr__(
                self,
                "_noise_generator",
                tf.random.Generator.from_seed(self.noise_seed),
            )
            self.computation_layer = tfq.layers.Expectation(
                backend="noiseless",
                differentiator=tfq.differentiators.Adjoint(),
            )
        else:
            self.computation_layer = tfq.layers.NoisyControlledPQC(
                circuit,
                observables,
                repetitions=self.noise_repetitions,
                sample_based=False,
                differentiator=tfq.differentiators.ParameterShift(),
            )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        batch_size = tf.gather(tf.shape(inputs), 0)
        batched_circuits = tf.repeat(self.empty_circuit_tensor, repeats=batch_size)
        angles_var = tf.reshape(
            tf.tile(
                self.w_var,
                multiples=[batch_size, *([1] * (len(self.w_var.shape) - 1))],
            ),
            shape=(-1, *self.w_var.shape),
        )
        angles_enc = tf.einsum("plqf,bpqf->bplqf", self.w_enc, inputs)
        if self.squash_activation in ("arctan", "atan"):
            angles_enc = tf.math.atan(angles_enc)
        else:
            angles_enc = keras.layers.Activation(self.squash_activation)(angles_enc)
        if self.noise_backend == "exact_density":
            return self.exact_computation(angles_var, angles_enc)
        joined_angles = tf.concat(
            [
                tf.reshape(angles_var, (batch_size, -1)),
                tf.reshape(angles_enc, (batch_size, -1)),
            ],
            axis=1,
        )
        joined_angles = tf.gather(joined_angles, self.symbols_idx, axis=1)
        if self.q_depolarizing == 0.0:
            return self.computation_layer([batched_circuits, joined_angles])
        if self.noise_backend == "tfq_noisy_parameter_shift":
            return self.computation_layer([batched_circuits, joined_angles])

        repetitions = self.noise_repetitions
        sample_count = batch_size * repetitions
        indices = self._noise_generator.uniform(
            shape=tf.reshape(sample_count, (1,)),
            minval=0,
            maxval=self.noise_bank_size,
            dtype=tf.int32,
        )
        sampled_circuits = tf.gather(self.noise_circuit_bank, indices)
        repeated_angles = tf.repeat(joined_angles, repeats=repetitions, axis=0)
        trajectory_values = self.computation_layer(
            sampled_circuits,
            symbol_names=self.sorted_symbols_tensor,
            symbol_values=repeated_angles,
            operators=self.observables,
        )
        return tf.reduce_mean(
            tf.reshape(
                trajectory_values,
                (batch_size, repetitions, len(self.observables)),
            ),
            axis=1,
        )


def generate_noisy_CoinGame2_critic_quantum_partite_mdp(
    n_agents: int,
    n_layers: int,
    q_depolarizing: float = 0.0,
    noise_repetitions: int = 32,
    noise_bank_size: int = 256,
    noise_seed: int = 0,
    noise_backend: str = "pauli_trajectory_adjoint",
    squash_activation: str = "arctan",
    beta: float = 1.0,
    name: Optional[str] = None,
    input_entanglement: bool = True,
    input_entanglement_type: str = "phi+",
) -> keras.Model:

    obs_shape = (4, 3, 3)
    d_qubits = 4
    qubits = cirq.LineQubit.range(n_agents * d_qubits)
    observables = [ft.reduce(lambda x, y: x * y, [cirq.Z(q) for q in qubits])]
    qlayer = NetworkNoisyPartiteVariationalEncodingPQC(
        qubits=qubits,
        n_parts=n_agents,
        d_qubits=d_qubits,
        n_layers=n_layers,
        observables=observables,
        q_depolarizing=q_depolarizing,
        noise_repetitions=noise_repetitions,
        noise_bank_size=noise_bank_size,
        noise_seed=noise_seed,
        noise_backend=noise_backend,
        squash_activation=squash_activation,
        encoding_layer_cls=ParameterizedRotationLayer_RxRyRz,
        input_entanglement=input_entanglement,
        input_entanglement_type=input_entanglement_type,
    )
    input_size = ft.reduce(lambda x, y: x * y, obs_shape)
    return keras.Sequential(
        [
            keras.Input(
                shape=(n_agents, input_size),
                dtype=tf.dtypes.float32,
                name="input",
            ),
            keras.Sequential(
                [
                    keras.layers.Reshape((n_agents, *obs_shape)),
                    keras.layers.Lambda(map_CoinGame2_obs_to_encoded_vector),
                ],
                name="input-preprocess",
            ),
            qlayer,
            keras.Sequential(
                [
                    RescaleWeighted(len(observables)),
                    keras.layers.Lambda(lambda x: x * beta),
                ],
                name="observables-value",
            ),
        ],
        name=name,
    )


def _sample_depolarizing_pauli_trajectory(
    channel_circuit: cirq.Circuit, rng: np.random.Generator
) -> cirq.Circuit:

    sampled_operations: List[cirq.Operation] = []
    for operation in channel_circuit.all_operations():
        gate = getattr(operation, "gate", None)
        if not isinstance(gate, cirq.DepolarizingChannel):
            sampled_operations.append(operation)
            continue
        q = float(gate.p)
        choice = int(rng.choice(4, p=[1.0 - q, q / 3.0, q / 3.0, q / 3.0]))
        if choice == 1:
            sampled_operations.append(cirq.X(operation.qubits[0]))
        elif choice == 2:
            sampled_operations.append(cirq.Y(operation.qubits[0]))
        elif choice == 3:
            sampled_operations.append(cirq.Z(operation.qubits[0]))
    return cirq.Circuit(sampled_operations)


def network_noise_operation_count(model: keras.Model) -> int:

    qlayer = next(
        layer
        for layer in model.layers
        if isinstance(layer, NetworkNoisyPartiteVariationalEncodingPQC)
    )
    return sum(
        1
        for operation in qlayer.circuit.all_operations()
        if isinstance(getattr(operation, "gate", None), cirq.DepolarizingChannel)
    )
