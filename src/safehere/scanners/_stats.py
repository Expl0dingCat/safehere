"""rolling statistics using Welford's online algorithm."""

import math
from typing import Dict, List, Optional


class WelfordAccumulator:
    """online mean/variance tracker with a windowed buffer."""

    __slots__ = ("count", "_mean", "_m2", "_min", "_max",
                 "_window_values", "_window_size")

    def __init__(self, window_size=100):
        # type: (int) -> None
        self.count = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._min = float("inf")
        self._max = float("-inf")
        self._window_size = window_size
        self._window_values = []  # type: List[float]

    def update(self, value):
        # type: (float) -> None
        """record a new observation."""
        self.count += 1
        delta = value - self._mean
        self._mean += delta / self.count
        delta2 = value - self._mean
        self._m2 += delta * delta2
        if value < self._min:
            self._min = value
        if value > self._max:
            self._max = value
        self._window_values.append(value)
        if len(self._window_values) > self._window_size:
            self._window_values.pop(0)

    @property
    def mean(self):
        # type: () -> float
        return self._mean if self.count > 0 else 0.0

    @property
    def variance(self):
        # type: () -> float
        # Bessel's correction: divide by (n-1) for sample variance
        return self._m2 / (self.count - 1) if self.count > 1 else 0.0

    @property
    def stddev(self):
        # type: () -> float
        return math.sqrt(self.variance)

    def z_score(self, value):
        # type: (float) -> float
        """z-score against all-time statistics."""
        if self.count < 2 or self.stddev < 1e-9:
            return 0.0
        return (value - self.mean) / self.stddev

    @property
    def window_mean(self):
        # type: () -> float
        if not self._window_values:
            return 0.0
        return sum(self._window_values) / len(self._window_values)

    @property
    def window_stddev(self):
        # type: () -> float
        n = len(self._window_values)
        if n < 2:
            return 0.0
        m = self.window_mean
        var = sum((x - m) ** 2 for x in self._window_values) / (n - 1)
        return math.sqrt(var)

    def window_z_score(self, value):
        # type: (float) -> float
        """z-score against the rolling window."""
        sd = self.window_stddev
        if len(self._window_values) < 2 or sd < 1e-9:
            return 0.0
        return (value - self.window_mean) / sd

    def to_dict(self):
        # type: () -> dict
        """serialize for persistence."""
        return {
            "count": self.count,
            "mean": self._mean,
            "m2": self._m2,
            "min": self._min,
            "max": self._max,
            "window_values": list(self._window_values),
            "window_size": self._window_size,
        }

    @classmethod
    def from_dict(cls, d):
        # type: (dict) -> WelfordAccumulator
        """deserialize from persistence."""
        acc = cls(window_size=d.get("window_size", 100))
        acc.count = d["count"]
        acc._mean = d["mean"]
        acc._m2 = d["m2"]
        acc._min = d["min"]
        acc._max = d["max"]
        acc._window_values = list(d.get("window_values", []))
        return acc
