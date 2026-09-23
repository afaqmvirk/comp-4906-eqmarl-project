from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


def _number(value: Any) -> Any:
    if isinstance(value, str):
        if value.lower() in {"inf", ".inf", "infinity"}:
            return float("inf")
        try:
            return float(value)
        except ValueError:
            pass
    return value


@dataclass(frozen=True)
class ProtocolConfig:
    agents: int = 2
    qubits_per_agent: int = 4
    observations_per_step: int = 2
    float_bytes: int = 4
    reward_bytes: int = 4
    eqmarl_gradient_floats_per_observation: int = 1
    classical_activation_floats_per_observation: int = 12
    classical_gradient_floats_per_observation: int = 12
    quantum_ack_bytes: int = 32

    def __post_init__(self) -> None:
        for name in ("agents", "qubits_per_agent", "observations_per_step", "float_bytes"):
            if getattr(self, name) <= 0:
                raise ValueError(f"protocol.{name} must be positive")


@dataclass(frozen=True)
class NetworkConfig:
    entanglement_rate_hz: float = 1_000.0
    entanglement_parallelism: int = 1
    transmission_success_probability: float = 0.95
    base_latency_s: float = 0.00025
    distance_km: float = 10.0
    fiber_speed_m_s: float = 2.0e8
    fixed_quantum_processing_latency_s: float = 0.0
    local_quantum_processing_s: float = 0.0005
    quantum_channel_capacity: int = 8
    classical_bandwidth_bps: float = 1.0e9
    classical_parallelism: int = 2
    classical_fixed_processing_latency_s: float = 0.0
    memory_initial_fidelity: float = 0.99
    memory_t2_s: float = 1.0
    memory_min_fidelity: float = 0.80
    base_compute_step_s: float = 0.010

    def __post_init__(self) -> None:
        if self.entanglement_rate_hz <= 0:
            raise ValueError("network.entanglement_rate_hz must be positive")
        if self.entanglement_parallelism <= 0:
            raise ValueError("network.entanglement_parallelism must be positive")
        if not 0 < self.transmission_success_probability <= 1:
            raise ValueError("network.transmission_success_probability must be in (0, 1]")
        if self.distance_km < 0 or self.base_latency_s < 0:
            raise ValueError("distance and latency cannot be negative")
        if self.fiber_speed_m_s <= 0:
            raise ValueError("network.fiber_speed_m_s must be positive")
        if self.quantum_channel_capacity <= 0 or self.classical_parallelism <= 0:
            raise ValueError("channel capacities must be positive")
        if self.classical_bandwidth_bps <= 0:
            raise ValueError("network.classical_bandwidth_bps must be positive")
        if self.memory_t2_s <= 0:
            raise ValueError("network.memory_t2_s must be positive")
        if not 0 < self.memory_initial_fidelity <= 1:
            raise ValueError("initial fidelity must be in (0, 1]")
        if not 0 < self.memory_min_fidelity <= 1:
            raise ValueError("minimum fidelity must be in (0, 1]")


@dataclass(frozen=True)
class LearningConfig:
    metric: str = "undiscounted_reward"
    target_reward: float = 20.0
    steps_per_iteration: int = 50
    smoothing_window: int = 10
    trajectory_seeds: int = 3
    quantum_experiment: str = "coingame_maa2c_mdp_eqmarl_psi+"
    classical_experiment: str = "coingame_maa2c_mdp_sctde"
    official_repo: str = "upstream/eqmarl"


@dataclass(frozen=True)
class StudyConfig:
    name: str = "eqmarl-network-study"
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    network_seeds: tuple[int, ...] = (101, 202, 303)
    sweep: dict[str, Any] = field(default_factory=dict)


def _construct(cls: type, values: dict[str, Any] | None):
    values = values or {}
    names = {f.name for f in fields(cls)}
    unknown = set(values) - names
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**{key: _number(value) for key, value in values.items()})


def load_config(path: str | Path) -> StudyConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return StudyConfig(
        name=raw.get("name", path.stem),
        protocol=_construct(ProtocolConfig, raw.get("protocol")),
        network=_construct(NetworkConfig, raw.get("network")),
        learning=_construct(LearningConfig, raw.get("learning")),
        network_seeds=tuple(int(x) for x in raw.get("network_seeds", (101, 202, 303))),
        sweep=raw.get("sweep", {}),
    )
