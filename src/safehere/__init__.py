"""safehere - runtime tool-output scanning for Cohere agents."""

__version__ = "1.0.0"
__author__ = "SafeHere Contributors"

from .guard import ToolGuard
from ._types import (
    Action,
    Severity,
    DetectorKind,
    Finding,
    ScanResult,
    ToolPolicy,
    GuardConfig,
)
from .scanners import BaseScanner, ScannerPipeline
from .scanners.pattern import PatternScanner
from .scanners.schema import SchemaDriftScanner, ANY
from .scanners.anomaly import AnomalyScanner
from .scanners.heuristic import HeuristicScanner
from .scanners.semantic import SemanticScanner
from .audit import AuditLogger
from .scoring import ScoringEngine
from .exceptions import (
    SafeHereError,
    QuarantineError,
    ConfigurationError,
    ScanError,
)

__all__ = [
    "ToolGuard",
    "Action",
    "Severity",
    "DetectorKind",
    "Finding",
    "ScanResult",
    "ToolPolicy",
    "GuardConfig",
    "BaseScanner",
    "ScannerPipeline",
    "PatternScanner",
    "SchemaDriftScanner",
    "AnomalyScanner",
    "HeuristicScanner",
    "SemanticScanner",
    "ANY",
    "AuditLogger",
    "ScoringEngine",
    "SafeHereError",
    "QuarantineError",
    "ConfigurationError",
    "ScanError",
]
