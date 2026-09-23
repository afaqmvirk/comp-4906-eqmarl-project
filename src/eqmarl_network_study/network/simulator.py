from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from ..accounting import ResourceAccounting
from ..config import NetworkConfig, ProtocolConfig
from .classical_link import ClassicalLink
from .entanglement import EntanglementSource, expected_attempts_per_resource
from .memory import QuantumMemory
from .quantum_link import QuantumLink


def _conditional_failed_legs(
    rng: np.random.Generator, failed_stages: int, legs: int, p_success: float
) -> int:
    if failed_stages == 0:
        return 0
    q = 1.0 - p_success
    weights = np.array(
        [math.comb(legs, k) * q**k * p_success ** (legs - k) for k in range(1, legs + 1)],
        dtype=float,
    )
    weights /= weights.sum()
    counts = rng.multinomial(failed_stages, weights)
    return int(np.dot(np.arange(1, legs + 1), counts))


class NetworkSimulator:

    def __init__(self, protocol: ProtocolConfig, network: NetworkConfig):
        self.protocol = protocol
        self.network = network
        self.source = EntanglementSource(
            network.entanglement_rate_hz, network.entanglement_parallelism
        )
        self.quantum_link = QuantumLink(
            p_success=network.transmission_success_probability,
            base_latency_s=network.base_latency_s,
            distance_km=network.distance_km,
            fiber_speed_m_s=network.fiber_speed_m_s,
            fixed_processing_latency_s=network.fixed_quantum_processing_latency_s,
            capacity_qubits=network.quantum_channel_capacity,
        )
        self.classical_link = ClassicalLink(
            bandwidth_bps=network.classical_bandwidth_bps,
            base_latency_s=network.base_latency_s,
            distance_km=network.distance_km,
            fiber_speed_m_s=network.fiber_speed_m_s,
            fixed_processing_latency_s=network.classical_fixed_processing_latency_s,
            parallelism=network.classical_parallelism,
        )
        self.memory = QuantumMemory(
            network.memory_initial_fidelity,
            network.memory_t2_s,
            network.memory_min_fidelity,
        )

    @property
    def memory_age_per_attempt_s(self) -> float:
        waves = math.ceil(self.protocol.agents / self.network.quantum_channel_capacity)

        return (
            2 * waves * self.quantum_link.one_way_latency_s
            + 2 * self.classical_link.one_way_latency_s
            + self.network.local_quantum_processing_s
        )

    def analytical_expected_attempts(self) -> float:
        return expected_attempts_per_resource(
            self.network.transmission_success_probability, self.protocol.agents
        )

    def simulate_quantum(
        self, environment_steps: int, rng: np.random.Generator, training_iterations: int = 0
    ) -> ResourceAccounting:
        p = self.network.transmission_success_probability
        n = self.protocol.agents
        logical_states = (
            environment_steps
            * self.protocol.observations_per_step
            * self.protocol.qubits_per_agent
        )
        out = ResourceAccounting(
            environment_steps=environment_steps,
            training_iterations=training_iterations,
            compute_time_s=environment_steps
            * (self.network.base_compute_step_s + self.network.local_quantum_processing_s),
        )
        if logical_states == 0:
            return out.finalize()

        if not self.memory.is_usable(self.memory_age_per_attempt_s):
            out.usable = False
            out.entangled_states_generated = logical_states
            out.entangled_pairs_generated = logical_states * (n - 1)
            out.states_discarded_memory_expiration = logical_states
            out.entanglement_generation_time_s = self.source.generation_time(logical_states)
            return out.finalize()

        p_logical = self.quantum_link.logical_success_probability(n)
        failed_attempts = int(rng.negative_binomial(logical_states, p_logical))
        attempts = logical_states + failed_attempts

        p_outbound_failure_given_failure = 1.0 / (1.0 + p**n)
        outbound_failed = int(
            rng.binomial(failed_attempts, p_outbound_failure_given_failure)
        )
        inbound_failed = failed_attempts - outbound_failed
        inbound_stages = logical_states + inbound_failed

        failed_legs = _conditional_failed_legs(rng, outbound_failed, n, p)
        failed_legs += _conditional_failed_legs(rng, inbound_failed, n, p)
        outbound_qubits = attempts * n
        inbound_qubits = inbound_stages * n
        transmitted_qubits = outbound_qubits + inbound_qubits

        out.entangled_states_generated = attempts
        out.entangled_pairs_generated = attempts * (n - 1)
        out.entangled_resources_successfully_used = logical_states
        out.qubits_transmitted = transmitted_qubits
        out.failed_quantum_transmissions = failed_legs
        out.retries = failed_attempts
        out.entanglement_generation_time_s = self.source.generation_time(attempts)
        out.quantum_transmission_time_s = self.quantum_link.service_time(
            outbound_qubits
        ) + self.quantum_link.service_time(inbound_qubits)
        out.quantum_link_busy_time_s = out.quantum_transmission_time_s

        ack_bytes = transmitted_qubits * self.protocol.quantum_ack_bytes
        ack_time = self.classical_link.transfer_time(ack_bytes, transmitted_qubits)
        reward_bytes_step = n * self.protocol.reward_bytes
        gradient_bytes_step = (
            n
            * self.protocol.observations_per_step
            * self.protocol.eqmarl_gradient_floats_per_observation
            * self.protocol.float_bytes
        )
        protocol_bytes = environment_steps * (reward_bytes_step + gradient_bytes_step)
        protocol_time = environment_steps * (
            self.classical_link.transfer_time(reward_bytes_step, n)
            + self.classical_link.transfer_time(gradient_bytes_step, n)
        )
        out.classical_bytes_transmitted = ack_bytes + protocol_bytes
        out.classical_communication_time_s = ack_time + protocol_time
        return out.finalize()

    def simulate_classical(
        self, environment_steps: int, training_iterations: int = 0
    ) -> ResourceAccounting:
        n = self.protocol.agents
        obs = self.protocol.observations_per_step
        upload_step = n * (
            obs
            * self.protocol.classical_activation_floats_per_observation
            * self.protocol.float_bytes
            + self.protocol.reward_bytes
        )
        download_step = (
            n
            * obs
            * self.protocol.classical_gradient_floats_per_observation
            * self.protocol.float_bytes
        )
        communication_step = self.classical_link.transfer_time(
            upload_step, n
        ) + self.classical_link.transfer_time(download_step, n)
        out = ResourceAccounting(
            environment_steps=environment_steps,
            training_iterations=training_iterations,
            compute_time_s=environment_steps * self.network.base_compute_step_s,
            classical_communication_time_s=environment_steps * communication_step,
            classical_bytes_transmitted=environment_steps * (upload_step + download_step),
        )
        return out.finalize()

    def with_network(self, **changes) -> "NetworkSimulator":
        return NetworkSimulator(self.protocol, replace(self.network, **changes))
