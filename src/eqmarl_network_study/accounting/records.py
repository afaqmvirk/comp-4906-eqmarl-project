from __future__ import annotations

from dataclasses import asdict, dataclass, fields


@dataclass
class ResourceAccounting:
    environment_steps: int = 0
    training_iterations: int = 0
    simulated_wall_time_s: float = 0.0
    compute_time_s: float = 0.0
    entanglement_generation_time_s: float = 0.0
    quantum_transmission_time_s: float = 0.0
    classical_communication_time_s: float = 0.0
    classical_bytes_transmitted: int = 0
    qubits_transmitted: int = 0
    entangled_states_generated: int = 0
    entangled_pairs_generated: int = 0
    entangled_resources_successfully_used: int = 0
    failed_quantum_transmissions: int = 0
    retries: int = 0
    states_discarded_memory_expiration: int = 0
    quantum_link_busy_time_s: float = 0.0
    quantum_link_utilization: float = 0.0
    usable: bool = True

    def finalize(self) -> "ResourceAccounting":
        self.simulated_wall_time_s = (
            self.compute_time_s
            + self.entanglement_generation_time_s
            + self.quantum_transmission_time_s
            + self.classical_communication_time_s
        )
        if self.simulated_wall_time_s > 0 and self.usable:
            self.quantum_link_utilization = min(
                1.0, self.quantum_link_busy_time_s / self.simulated_wall_time_s
            )
        return self

    def __iadd__(self, other: "ResourceAccounting") -> "ResourceAccounting":
        for item in fields(self):
            if item.name in {"quantum_link_utilization", "usable"}:
                continue
            setattr(self, item.name, getattr(self, item.name) + getattr(other, item.name))
        self.usable = self.usable and other.usable
        if self.simulated_wall_time_s > 0 and self.usable:
            self.quantum_link_utilization = min(
                1.0, self.quantum_link_busy_time_s / self.simulated_wall_time_s
            )
        else:
            self.quantum_link_utilization = 0.0
        return self

    def to_dict(self) -> dict:
        return asdict(self)
