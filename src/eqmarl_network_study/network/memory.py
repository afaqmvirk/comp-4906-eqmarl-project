from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class QuantumMemory:
    initial_fidelity: float
    t2_s: float
    minimum_fidelity: float

    def fidelity(self, age_s: float) -> float:
        if age_s < 0:
            raise ValueError("age cannot be negative")
        if math.isinf(self.t2_s):
            return self.initial_fidelity
        return self.initial_fidelity * math.exp(-age_s / self.t2_s)

    def is_usable(self, age_s: float) -> bool:
        return self.fidelity(age_s) >= self.minimum_fidelity

    @property
    def maximum_usable_age_s(self) -> float:
        if self.minimum_fidelity > self.initial_fidelity:
            return 0.0
        if math.isinf(self.t2_s):
            return float("inf")
        return self.t2_s * math.log(self.initial_fidelity / self.minimum_fidelity)
