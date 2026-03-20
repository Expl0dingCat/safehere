"""pattern-based prompt injection detector."""

import re
from typing import Any, Dict, List, Optional, Tuple  # noqa: F401 -- used in type comments

from .._types import DetectorKind, Finding, Severity  # noqa: F401 -- DetectorKind used in type comments
from ._base import BaseScanner
from ._normalize import (
    extract_encoded_payloads,
    has_suspicious_unicode,
    normalize_unicode,
)
from ._patterns_db import PatternEntry, get_compiled_patterns, quick_reject


# injections need to land where the model attends (beginning or end),
# so we only scan head + tail of large texts
_MAX_SCAN_LEN = 8192
_HEAD_SIZE = 6144
_TAIL_SIZE = 2048


def _truncate_for_scan(text):
    # type: (str) -> str
    """truncate long text to a scan window (head + tail)."""
    if len(text) <= _MAX_SCAN_LEN:
        return text
    return text[:_HEAD_SIZE] + "\n...\n" + text[-_TAIL_SIZE:]


class PatternScanner(BaseScanner):
    """detects known prompt injection patterns using regex.

    confidence is reduced for matches inside quoted/code regions to
    limit false positives.
    """

    @property
    def name(self):
        # type: () -> str
        return "pattern"

    def __init__(self, disabled_rules=None, max_scan_len=None):
        # type: (Optional[set], Optional[int]) -> None
        self._disabled = disabled_rules or set()
        self._extra_patterns = []  # type: List[Tuple[re.Pattern, PatternEntry]]
        self._max_scan_len = max_scan_len or _MAX_SCAN_LEN

    def add_pattern(self, rule_id, pattern, severity, description,
                    base_confidence=0.80, case_sensitive=False,
                    context_reducer=0.0):
        # type: (str, str, Severity, str, float, bool, float) -> None
        """add a custom pattern to the scanner."""
        entry = PatternEntry(
            rule_id=rule_id,
            category="custom",
            severity=severity,
            base_confidence=base_confidence,
            pattern=pattern,
            description=description,
            case_sensitive=case_sensitive,
            context_reducer=context_reducer,
        )
        flags = 0 if case_sensitive else re.IGNORECASE
        flags |= re.MULTILINE
        self._extra_patterns.append((re.compile(pattern, flags), entry))

    def scan(self, tool_name, output_text, output_structured=None):
        # type: (str, str, Optional[Dict[str, Any]]) -> List[Finding]
        findings = []  # type: List[Finding]

        if has_suspicious_unicode(output_text):
            findings.append(Finding(
                scanner_name=self.name,
                rule_id="PAT-UNICODE-001",
                severity=Severity.MEDIUM,
                confidence=0.60,
                description="Text contains zero-width or bidirectional control characters",
                location="raw_text",
            ))

        scan_text = _truncate_for_scan(output_text)
        text_lower = scan_text.lower()
        skip_main = quick_reject(text_lower) and not self._extra_patterns

        if not skip_main:
            normalized = normalize_unicode(scan_text)
            quoted_regions = _find_quoted_regions(normalized)
            self._match_patterns(normalized, findings, quoted_regions, "normalized")

        # always check encoded payloads -- injections can hide in base64/hex
        for enc_type, decoded in extract_encoded_payloads(scan_text):
            decoded_norm = normalize_unicode(decoded)
            decoded_regions = _find_quoted_regions(decoded_norm)
            self._match_patterns(
                decoded_norm, findings, decoded_regions,
                "decoded_{}".format(enc_type),
                confidence_penalty=0.10,
            )

        return findings

    def _match_patterns(self, text, findings, quoted_regions, source,
                        confidence_penalty=0.0):
        # type: (str, List[Finding], List[Tuple[int, int]], str, float) -> None
        all_patterns = list(get_compiled_patterns()) + self._extra_patterns

        for compiled, entry in all_patterns:
            if entry.rule_id in self._disabled:
                continue

            # one match per pattern is enough -- breadth bonus caps at 0.15
            match = compiled.search(text)
            if match is None:
                continue

            confidence = entry.base_confidence - confidence_penalty

            if _in_quoted_region(match.start(), match.end(), quoted_regions):
                confidence -= entry.context_reducer

            confidence = max(0.0, min(1.0, confidence))

            matched = match.group()
            findings.append(Finding(
                scanner_name=self.name,
                rule_id=entry.rule_id,
                severity=entry.severity,
                confidence=confidence,
                description=entry.description,
                matched_text=matched[:200],
                location="{}@offset:{}".format(source, match.start()),
            ))


def _find_quoted_regions(text):
    # type: (str) -> List[Tuple[int, int]]
    """find regions inside backtick code blocks and double-quoted strings."""
    regions = []  # type: List[Tuple[int, int]]
    has_backtick = "`" in text
    has_quote = '"' in text
    if not has_backtick and not has_quote:
        return regions
    if has_backtick:
        for m in re.finditer(r"```[\s\S]*?```", text):
            regions.append((m.start(), m.end()))
        for m in re.finditer(r"`[^`]+`", text):
            regions.append((m.start(), m.end()))
    if has_quote:
        # 4+ chars to avoid matching short JSON values
        for m in re.finditer(r'"[^"]{4,}"', text):
            regions.append((m.start(), m.end()))
    return regions


def _in_quoted_region(start, end, regions):
    # type: (int, int, List[Tuple[int, int]]) -> bool
    """check if a match span falls entirely within a quoted region."""
    for rs, re_ in regions:
        if rs <= start and end <= re_:
            return True
    return False
