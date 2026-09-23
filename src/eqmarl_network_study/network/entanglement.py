from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class EntanglementSource:
    rate_hz: float
    parallelism: int = 1

    def generation_time(self, states: int) -> float:
        if states < 0:
            raise ValueError("states cannot be negative")
        if math.isinf(self.rate_hz) or states == 0:
            return 0.0
        if self.rate_hz <= 0 or self.parallelism <= 0:
            raise ValueError("rate and parallelism must be positive")
        return states / (self.rate_hz * self.parallelism)


def expected_attempts_per_resource(p_leg: float, agents: int) -> float:
    if not 0 < p_leg <= 1:
        raise ValueError("p_leg must be in (0, 1]")
    if agents <= 0:
        raise ValueError("agents must be positive")
    return 1.0 / (p_leg ** (2 * agents))
