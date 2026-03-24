"""core type definitions for safehere."""

import dataclasses
import enum
from typing import Any, Dict, List, Optional


class Severity(enum.IntEnum):
    """threat severity level, ordered for comparison."""

    NONE = 0
    LOW = 10
    MEDIUM = 20
    HIGH = 30
    CRITICAL = 40


class Action(enum.Enum):
    """policy action when a threat is detected."""

    ALLOW = "allow"
    LOG = "log"
    WARN = "warn"
    BLOCK = "block"
    QUARANTINE = "quarantine"


class DetectorKind(enum.Enum):
    """identifies which detection layer produced a finding."""

    PATTERN = "pattern"
    SCHEMA_DRIFT = "schema_drift"
    ANOMALY = "anomaly"
    HEURISTIC = "heuristic"
    POLYGLOT = "polyglot"
    SEMANTIC = "semantic"


@dataclasses.dataclass(frozen=True)
class Finding:
    """a single detection from a scanner. frozen to prevent mutation."""

    scanner_name: str
    rule_id: str
    severity: Severity
    confidence: float
    description: str
    matched_text: str = ""
    location: str = ""


@dataclasses.dataclass
class ScanResult:
    """aggregated result from the scanner pipeline for one tool output."""

    tool_name: str
    findings: List[Finding] = dataclasses.field(default_factory=list)
    max_severity: Severity = Severity.NONE
    combined_score: float = 0.0
    action: Action = Action.ALLOW
    scan_time_ms: float = 0.0

    def is_blocked(self):
        # type: () -> bool
        return self.action in (Action.BLOCK, Action.QUARANTINE)


@dataclasses.dataclass
class ToolPolicy:
    """per-tool policy overrides."""

    tool_name: str
    thresholds: Optional[Dict[Action, float]] = None
    expected_schema: Optional[Any] = None
    schema_strict: bool = False


DEFAULT_THRESHOLDS = {
    Action.LOG: 0.20,
    Action.WARN: 0.45,
    Action.BLOCK: 0.70,
    Action.QUARANTINE: 0.90,
}


@dataclasses.dataclass
class GuardConfig:
    """top-level configuration for ToolGuard."""

    default_thresholds: Dict[Action, float] = dataclasses.field(
        default_factory=lambda: dict(DEFAULT_THRESHOLDS)
    )
    tool_policies: Dict[str, ToolPolicy] = dataclasses.field(default_factory=dict)
    audit_log_path: Optional[str] = None
    audit_log_max_text_len: int = 500
    block_message: str = (
        "[Content blocked by safehere: potential prompt injection detected]"
    )
    enabled: bool = True
