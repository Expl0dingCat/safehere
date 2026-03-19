"""statistical anomaly detector for tool outputs."""

import json
import math
import threading
from collections import Counter
from typing import Any, Dict, List, Optional

from .._types import DetectorKind, Finding, Severity
from ._base import BaseScanner
from ._stats import WelfordAccumulator


_ENTROPY_SAMPLE_SIZE = 8192

# global baseline defaults for cold-start hardening: if a tool has fewer than
# cold_start_count samples, outputs exceeding 3x these values trigger a WARN
_GLOBAL_DEFAULT_LENGTH = 500.0
_GLOBAL_DEFAULT_ENTROPY = 5.0
_COLD_START_MULTIPLIER = 3.0


def _shannon_entropy(text):
    # type: (str) -> float
    """shannon entropy in bits per character, sampling for long texts."""
    if not text:
        return 0.0
    if len(text) > _ENTROPY_SAMPLE_SIZE:
        # sample head, middle, tail without overlap
        chunk = _ENTROPY_SAMPLE_SIZE // 3
        mid_start = (len(text) - chunk) // 2
        text = text[:chunk] + text[mid_start:mid_start + chunk] + text[-chunk:]
    counts = Counter(text)
    length = len(text)
    return -sum(
        (c / length) * math.log2(c / length)
        for c in counts.values()
    )


def _natural_language_ratio(text):
    # type: (str) -> float
    """fraction of space-separated tokens that are word-like."""
    if not text:
        return 0.0
    sample = text[:4096] if len(text) > 4096 else text
    tokens = sample.split()
    if not tokens:
        return 0.0
    word_like = sum(
        1 for t in tokens
        if t.isalpha() or (t.rstrip(".,!?;:") and t.rstrip(".,!?;:").isalpha())
    )
    return word_like / len(tokens)


class AnomalyScanner(BaseScanner):
    """tracks rolling statistics per tool and flags sudden deviations."""

    DEFAULT_WINDOW_SIZE = 100
    DEFAULT_Z_THRESHOLD = 3.0
    DEFAULT_COLD_START_COUNT = 5
    DEFAULT_NL_RATIO_JUMP = 0.3

    @property
    def name(self):
        # type: () -> str
        return "anomaly"

    def __init__(self, window_size=None, z_threshold=None,
                 cold_start_count=None, nl_ratio_jump_threshold=None,
                 persistence_path=None):
        # type: (Optional[int], Optional[float], Optional[int], Optional[float], Optional[str]) -> None
        self._window_size = window_size or self.DEFAULT_WINDOW_SIZE
        self._z_threshold = z_threshold or self.DEFAULT_Z_THRESHOLD
        self._cold_start = cold_start_count or self.DEFAULT_COLD_START_COUNT
        self._nl_jump = nl_ratio_jump_threshold or self.DEFAULT_NL_RATIO_JUMP
        self._persistence_path = persistence_path

        self._stats = {}  # type: Dict[str, Dict[str, WelfordAccumulator]]
        self._locks = {}  # type: Dict[str, threading.Lock]

        if persistence_path:
            self._load()

    def _get_lock(self, tool_name):
        # type: (str) -> threading.Lock
        # setdefault is atomic under CPython GIL; for free-threaded Python
        # the worst case is two Lock objects created and one discarded
        return self._locks.setdefault(tool_name, threading.Lock())

    def _get_tool_stats(self, tool_name):
        # type: (str) -> Dict[str, WelfordAccumulator]
        if tool_name not in self._stats:
            self._stats[tool_name] = {
                "length": WelfordAccumulator(self._window_size),
                "entropy": WelfordAccumulator(self._window_size),
                "nl_ratio": WelfordAccumulator(self._window_size),
            }
        return self._stats[tool_name]

    def scan(self, tool_name, output_text, output_structured=None):
        # type: (str, str, Optional[Dict[str, Any]]) -> List[Finding]
        findings = []  # type: List[Finding]

        lock = self._get_lock(tool_name)
        with lock:
            stats = self._get_tool_stats(tool_name)

            length = float(len(output_text))
            entropy = _shannon_entropy(output_text)
            nl_ratio = _natural_language_ratio(output_text)

            in_cold_start = stats["length"].count < self._cold_start

            if in_cold_start:
                # cold-start hardening: use conservative global defaults
                if length > _GLOBAL_DEFAULT_LENGTH * _COLD_START_MULTIPLIER:
                    findings.append(Finding(
                        scanner_name=self.name,
                        rule_id="ANOM-COLDSTART-LENGTH-001",
                        severity=Severity.MEDIUM,
                        confidence=0.50,
                        description="Cold-start: output length {:.0f} exceeds {:.0f} (3x global default)".format(
                            length, _GLOBAL_DEFAULT_LENGTH * _COLD_START_MULTIPLIER,
                        ),
                        location="$",
                    ))
                if entropy > _GLOBAL_DEFAULT_ENTROPY * _COLD_START_MULTIPLIER:
                    findings.append(Finding(
                        scanner_name=self.name,
                        rule_id="ANOM-COLDSTART-ENTROPY-001",
                        severity=Severity.MEDIUM,
                        confidence=0.45,
                        description="Cold-start: entropy {:.2f} exceeds {:.2f} (3x global default)".format(
                            entropy, _GLOBAL_DEFAULT_ENTROPY * _COLD_START_MULTIPLIER,
                        ),
                        location="$",
                    ))

            if not in_cold_start:
                length_z = stats["length"].window_z_score(length)
                if abs(length_z) > self._z_threshold:
                    direction = "longer" if length_z > 0 else "shorter"
                    if abs(length_z) > self._z_threshold * 2:
                        sev = Severity.HIGH
                        conf = min(0.85, 0.60 + (abs(length_z) - self._z_threshold) * 0.05)
                    else:
                        sev = Severity.MEDIUM
                        conf = min(0.80, 0.50 + (abs(length_z) - self._z_threshold) * 0.05)
                    findings.append(Finding(
                        scanner_name=self.name,
                        rule_id="ANOM-LENGTH-001",
                        severity=sev,
                        confidence=conf,
                        description="Output length {} is {:.1f} sigma {} than rolling mean ({:.0f})".format(
                            int(length), abs(length_z), direction,
                            stats["length"].window_mean,
                        ),
                        location="$",
                    ))

                entropy_z = stats["entropy"].window_z_score(entropy)
                if abs(entropy_z) > self._z_threshold:
                    direction = "higher" if entropy_z > 0 else "lower"
                    conf = min(0.80, 0.45 + (abs(entropy_z) - self._z_threshold) * 0.05)
                    findings.append(Finding(
                        scanner_name=self.name,
                        rule_id="ANOM-ENTROPY-001",
                        severity=Severity.MEDIUM,
                        confidence=conf,
                        description="Entropy {:.2f} bits is {:.1f} sigma {} than rolling mean ({:.2f})".format(
                            entropy, abs(entropy_z), direction,
                            stats["entropy"].window_mean,
                        ),
                        location="$",
                    ))

                nl_z = stats["nl_ratio"].window_z_score(nl_ratio)
                nl_jump = nl_ratio - stats["nl_ratio"].window_mean
                if nl_jump > self._nl_jump and nl_z > self._z_threshold:
                    conf = min(0.80, 0.55 + nl_jump)
                    findings.append(Finding(
                        scanner_name=self.name,
                        rule_id="ANOM-NL-RATIO-001",
                        severity=Severity.HIGH,
                        confidence=conf,
                        description="Natural language ratio {:.2f} jumped by {:.2f} over rolling mean ({:.2f})".format(
                            nl_ratio, nl_jump, stats["nl_ratio"].window_mean,
                        ),
                        location="$",
                    ))

            stats["length"].update(length)
            stats["entropy"].update(entropy)
            stats["nl_ratio"].update(nl_ratio)

            if self._persistence_path and stats["length"].count % 10 == 0:
                self._save()

        return findings

    def reset(self):
        # type: () -> None
        self._stats.clear()

    def _save(self):
        # type: () -> None
        data = {}
        for tool_name, tool_stats in self._stats.items():
            data[tool_name] = {k: v.to_dict() for k, v in tool_stats.items()}
        try:
            with open(self._persistence_path, "w") as f:
                json.dump(data, f)
        except OSError:
            pass

    def _load(self):
        # type: () -> None
        try:
            with open(self._persistence_path, "r") as f:
                data = json.load(f)
            for tool_name, tool_stats in data.items():
                self._stats[tool_name] = {
                    k: WelfordAccumulator.from_dict(v)
                    for k, v in tool_stats.items()
                }
        except (OSError, json.JSONDecodeError, KeyError):
            pass
