from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ClassicalLink:
    bandwidth_bps: float
    base_latency_s: float
    distance_km: float
    fiber_speed_m_s: float = 2.0e8
    fixed_processing_latency_s: float = 0.0
    parallelism: int = 1

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

    def transfer_time(self, byte_count: int, message_count: int) -> float:
        if byte_count < 0 or message_count < 0:
            raise ValueError("counts cannot be negative")
        if byte_count == 0 and message_count == 0:
            return 0.0
        serialization = 0.0 if math.isinf(self.bandwidth_bps) else 8 * byte_count / self.bandwidth_bps
        latency = math.ceil(message_count / self.parallelism) * self.one_way_latency_s
        return latency + serialization
