from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class EventKind(str, Enum):
    GENERATE = "generate_entanglement"
    QUANTUM_SEND = "quantum_send"
    HERALD = "classical_herald"
    PROCESS = "local_quantum_process"
    MEASURE = "joint_measurement"
    DISCARD = "memory_discard"


@dataclass(order=True, frozen=True)
class Event:
    time_s: float
    kind: EventKind = field(compare=False)
    resource_id: int | None = field(default=None, compare=False)
    agent_id: int | None = field(default=None, compare=False)
    metadata: dict = field(default_factory=dict, compare=False)
