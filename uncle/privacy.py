"""The shared experiment budget and unchanged analytic Gaussian calibration."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.special import ndtr


@dataclass(frozen=True)
class Privacy:
    epsilon: float
    delta: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.epsilon) or not 0 < self.epsilon <= 1:
            raise ValueError("the shared methods require 0 < epsilon <= 1")
        if not np.isfinite(self.delta) or not 0 < self.delta < 1:
            raise ValueError("delta must lie strictly between zero and one")

    def gaussian_std(self, radius: float) -> float:
        """Balle--Wang calibration for an L2 sensitivity/radius, not a variance."""
        if not np.isfinite(radius) or radius < 0:
            raise ValueError("radius must be finite and nonnegative")
        if radius == 0:
            return 0.0
        epsilon, delta = self.epsilon, self.delta

        def case_a(s: float) -> float:
            return float(ndtr(np.sqrt(epsilon * s))
                         - np.exp(epsilon) * ndtr(-np.sqrt(epsilon) * np.sqrt(s + 2)))

        def case_b(s: float) -> float:
            return float(ndtr(-np.sqrt(epsilon * s))
                         - np.exp(epsilon) * ndtr(-np.sqrt(epsilon) * np.sqrt(s + 2)))

        threshold = case_a(0.0)
        if delta == threshold:
            alpha = 1.0
        else:
            use_a = delta > threshold
            lower, upper = 0.0, 1.0
            # These are the two branches of the published scalar calibration,
            # not interchangeable privacy mechanisms or pluggable callbacks.
            while (case_a(upper) < delta) if use_a else (case_b(upper) > delta):
                lower, upper = upper, 2 * upper
            while upper - lower > 1e-12 * max(1.0, upper):
                middle = (lower + upper) / 2
                value = case_a(middle) if use_a else case_b(middle)
                if (value > delta) == use_a:
                    upper = middle
                else:
                    lower = middle
            s = (lower + upper) / 2
            alpha = np.sqrt(1 + s / 2) + (-1 if use_a else 1) * np.sqrt(s / 2)
        return float(alpha * radius / np.sqrt(2 * epsilon))

