from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class QuantumLink:
    p_success: float
    base_latency_s: float
    distance_km: float
    fiber_speed_m_s: float = 2.0e8
    fixed_processing_latency_s: float = 0.0
    capacity_qubits: int = 1

    @property
    def propagation_latency_s(self) -> float:
        return self.distance_km * 1_000.0 / self.fiber_speed_m_s

    @property
    def one_way_latency_s(self) -> float:
        return (
            self.base_latency_s
            + self.propagation_latency_s
            + self.fixed_processing_latency_s
        )

    def service_time(self, qubit_transmissions: int) -> float:
        if qubit_transmissions <= 0:
            return 0.0
        waves = math.ceil(qubit_transmissions / self.capacity_qubits)
        return waves * self.one_way_latency_s

    def logical_success_probability(self, agents: int) -> float:
        return self.p_success ** (2 * agents)
