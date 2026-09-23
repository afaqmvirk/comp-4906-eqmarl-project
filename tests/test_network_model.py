from dataclasses import replace
import math

import numpy as np
import pytest

from eqmarl_network_study.config import NetworkConfig, ProtocolConfig
from eqmarl_network_study.network import (
    EntanglementSource,
    NetworkSimulator,
    QuantumLink,
    QuantumMemory,
)
from eqmarl_network_study.network.entanglement import expected_attempts_per_resource


@pytest.fixture
def protocol():
    return ProtocolConfig(agents=2, qubits_per_agent=4)


@pytest.fixture
def network():
    return NetworkConfig(
        entanglement_rate_hz=1_000,
        transmission_success_probability=0.9,
        base_latency_s=0.0001,
        distance_km=10,
        memory_t2_s=100,
    )


def test_infinite_entanglement_rate_removes_generation_delay():
    assert EntanglementSource(float("inf")).generation_time(10**9) == 0


def test_success_one_has_no_retries_or_failures(protocol, network):
    sim = NetworkSimulator(
        protocol, replace(network, transmission_success_probability=1.0)
    )
    out = sim.simulate_quantum(100, np.random.default_rng(1))
    assert out.retries == 0
    assert out.failed_quantum_transmissions == 0
    assert out.entangled_states_generated == 100 * 2 * 4


def test_lower_success_never_reduces_analytical_retry_cost():
    probabilities = [1.0, 0.95, 0.8, 0.5]
    attempts = [expected_attempts_per_resource(p, agents=2) for p in probabilities]
    assert attempts == sorted(attempts)
    assert expected_attempts_per_resource(0.5, 2) == pytest.approx(16.0)


def test_joint_condition_is_not_naive_one_over_p():
    p = 0.8
    assert expected_attempts_per_resource(p, 2) == pytest.approx(1 / p**4)
    assert expected_attempts_per_resource(p, 2) != pytest.approx(1 / p)


def test_simulated_average_matches_joint_analytical_expectation(protocol, network):
    net = replace(network, transmission_success_probability=0.8)
    sim = NetworkSimulator(protocol, net)
    logical = 200_000 * protocol.observations_per_step * protocol.qubits_per_agent
    out = sim.simulate_quantum(200_000, np.random.default_rng(5))
    measured = out.entangled_states_generated / logical
    assert measured == pytest.approx(sim.analytical_expected_attempts(), rel=0.005)


def test_zero_distance_removes_only_propagation_delay():
    link = QuantumLink(1.0, 0.002, 0, capacity_qubits=1)
    assert link.propagation_latency_s == 0
    assert link.one_way_latency_s == pytest.approx(0.002)


def test_distance_increases_propagation_at_five_microseconds_per_km():
    link = QuantumLink(1.0, 0, 20, fiber_speed_m_s=2e8, capacity_qubits=1)
    assert link.propagation_latency_s == pytest.approx(100e-6)


def test_very_long_memory_has_no_expiration(protocol, network):
    sim = NetworkSimulator(protocol, replace(network, memory_t2_s=float("inf")))
    out = sim.simulate_quantum(100, np.random.default_rng(3))
    assert out.usable
    assert out.states_discarded_memory_expiration == 0
    assert QuantumMemory(0.99, float("inf"), 0.98).fidelity(1e12) == 0.99


def test_short_memory_marks_resources_unusable(protocol, network):
    sim = NetworkSimulator(
        protocol,
        replace(network, memory_t2_s=1e-9, memory_min_fidelity=0.98),
    )
    out = sim.simulate_quantum(10, np.random.default_rng(3))
    assert not out.usable
    assert out.states_discarded_memory_expiration > 0


def test_poor_quantum_conditions_make_wall_clock_large(protocol, network):
    good = NetworkSimulator(
        protocol,
        replace(
            network,
            entanglement_rate_hz=1e6,
            transmission_success_probability=1.0,
        ),
    ).simulate_quantum(100, np.random.default_rng(7))
    poor = NetworkSimulator(
        protocol,
        replace(
            network,
            entanglement_rate_hz=1.0,
            transmission_success_probability=0.5,
        ),
    ).simulate_quantum(100, np.random.default_rng(7))
    assert poor.simulated_wall_time_s > 1000 * good.simulated_wall_time_s


def test_all_network_costs_off_recovers_ideal_step_accounting(protocol):
    network = NetworkConfig(
        entanglement_rate_hz=float("inf"),
        transmission_success_probability=1.0,
        base_latency_s=0,
        distance_km=0,
        local_quantum_processing_s=0,
        classical_bandwidth_bps=float("inf"),
        memory_t2_s=float("inf"),
        base_compute_step_s=0.0125,
    )
    out = NetworkSimulator(protocol, network).simulate_quantum(
        80, np.random.default_rng(0), training_iterations=4
    )
    assert out.simulated_wall_time_s == pytest.approx(1.0)
    assert out.environment_steps == 80
    assert out.training_iterations == 4


def test_classical_byte_accounting_is_unit_consistent(protocol, network):
    steps = 7
    out = NetworkSimulator(protocol, network).simulate_classical(steps)
    per_step = protocol.agents * (
        protocol.observations_per_step
        * protocol.classical_activation_floats_per_observation
        * protocol.float_bytes
        + protocol.reward_bytes
    ) + (
        protocol.agents
        * protocol.observations_per_step
        * protocol.classical_gradient_floats_per_observation
        * protocol.float_bytes
    )
    assert out.classical_bytes_transmitted == steps * per_step
    assert out.qubits_transmitted == 0
    assert isinstance(out.classical_bytes_transmitted, int)


def test_quantum_and_classical_units_are_not_combined(protocol, network):
    out = NetworkSimulator(protocol, network).simulate_quantum(
        5, np.random.default_rng(9)
    )
    assert out.classical_bytes_transmitted > 0
    assert out.qubits_transmitted > 0
    assert "total_communication" not in out.to_dict()
    assert 0 <= out.quantum_link_utilization <= 1
