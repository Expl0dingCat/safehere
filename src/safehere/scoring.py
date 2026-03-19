"""scoring engine -- combines findings from all scanners into a final action."""

from typing import Dict, List, Optional

from ._types import Action, DetectorKind, Finding, Severity, ScanResult


DEFAULT_WEIGHTS = {
    DetectorKind.PATTERN: 0.40,
    DetectorKind.SCHEMA_DRIFT: 0.20,
    DetectorKind.ANOMALY: 0.15,
    DetectorKind.HEURISTIC: 0.25,
}

DEFAULT_THRESHOLDS = {
    Action.LOG: 0.20,
    Action.WARN: 0.45,
    Action.BLOCK: 0.70,
    Action.QUARANTINE: 0.90,
}

_SCANNER_TO_KIND = {
    "pattern": DetectorKind.PATTERN,
    "schema_drift": DetectorKind.SCHEMA_DRIFT,
    "anomaly": DetectorKind.ANOMALY,
    "heuristic": DetectorKind.HEURISTIC,
}


class ScoringEngine:
    """evaluates aggregated findings into a scored action decision."""

    def __init__(self, weights=None, thresholds=None, per_tool_thresholds=None):
        # type: (Optional[Dict[DetectorKind, float]], Optional[Dict[Action, float]], Optional[Dict[str, Dict[Action, float]]]) -> None
        self._weights = weights or dict(DEFAULT_WEIGHTS)
        self._thresholds = thresholds or dict(DEFAULT_THRESHOLDS)
        self._per_tool = per_tool_thresholds or {}

    def evaluate(self, tool_name, findings, scan_time_ms=0.0):
        # type: (str, List[Finding], float) -> ScanResult
        if not findings:
            return ScanResult(
                tool_name=tool_name,
                findings=[],
                max_severity=Severity.NONE,
                combined_score=0.0,
                action=Action.ALLOW,
                scan_time_ms=scan_time_ms,
            )

        detector_scores = {}  # type: Dict[DetectorKind, float]
        for kind in DetectorKind:
            kind_findings = [
                f for f in findings
                if _SCANNER_TO_KIND.get(f.scanner_name) == kind
            ]
            if not kind_findings:
                detector_scores[kind] = 0.0
                continue

            primary = max(
                (f.severity.value / Severity.CRITICAL.value) * f.confidence
                for f in kind_findings
            )
            breadth_bonus = min(0.15, (len(kind_findings) - 1) * 0.03)
            detector_scores[kind] = min(1.0, primary + breadth_bonus)

        combined = sum(
            detector_scores.get(kind, 0.0) * weight
            for kind, weight in self._weights.items()
        )

        # corroboration: independent detectors agreeing is stronger evidence
        # than any single detector. if pattern + heuristic both fire, the
        # chance of coincidental FP drops multiplicatively.
        active = sum(1 for s in detector_scores.values() if s > 0.1)
        if active >= 3:
            combined = min(1.0, combined * 1.30)
        elif active >= 2:
            combined = min(1.0, combined * 1.15)

        # single-detector floor: without this, a detector weighted at 0.25
        # can never reach BLOCK (0.70) alone even at score 1.0. the
        # heuristic scanner catches attacks no other layer sees — its
        # score shouldn't be suppressed by silence from other layers.
        max_detector = max(detector_scores.values())
        if max_detector >= 0.65:
            combined = max(combined, 0.75)
        elif max_detector >= 0.50:
            # if multiple detectors are active, the corroboration is
            # sufficient evidence to push toward BLOCK
            if active >= 2:
                combined = max(combined, 0.70)
            else:
                combined = max(combined, 0.50)

        max_severity = max(f.severity for f in findings)

        thresholds = self._per_tool.get(tool_name, self._thresholds)
        action = Action.ALLOW
        for policy_action in [Action.QUARANTINE, Action.BLOCK, Action.WARN, Action.LOG]:
            threshold = thresholds.get(policy_action, 1.1)
            if combined >= threshold:
                action = policy_action
                break

        # CRITICAL + high confidence always forces a block
        critical_high = any(
            f.severity == Severity.CRITICAL and f.confidence >= 0.90
            for f in findings
        )
        if critical_high and action not in (Action.BLOCK, Action.QUARANTINE):
            action = Action.BLOCK
            combined = max(combined, 0.90)

        return ScanResult(
            tool_name=tool_name,
            findings=findings,
            max_severity=max_severity,
            combined_score=round(combined, 4),
            action=action,
            scan_time_ms=scan_time_ms,
        )
