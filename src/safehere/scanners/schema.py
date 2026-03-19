"""schema drift detector for tool outputs."""

import json
from typing import Any, Dict, List, Optional

from .._types import DetectorKind, Finding, Severity
from ._base import BaseScanner


ANY = type("ANY", (), {"__repr__": lambda self: "ANY"})()


class SchemaDriftScanner(BaseScanner):
    """detects when tool outputs deviate from registered or baseline schemas.

    tools without a registered schema get their first response auto-baselined.
    """

    @property
    def name(self):
        # type: () -> str
        return "schema_drift"

    def __init__(self):
        # type: () -> None
        self._registered = {}  # type: Dict[str, Any]
        self._baselines = {}   # type: Dict[str, Any]
        self._strict = {}      # type: Dict[str, bool]

    def register_schema(self, tool_name, schema, strict=False):
        # type: (str, Any, bool) -> None
        """register expected output schema for a tool."""
        self._registered[tool_name] = schema
        self._strict[tool_name] = strict

    def reset(self):
        # type: () -> None
        self._baselines.clear()

    def scan(self, tool_name, output_text, output_structured=None):
        # type: (str, str, Optional[Dict[str, Any]]) -> List[Finding]
        findings = []  # type: List[Finding]

        parsed = output_structured
        if parsed is None:
            parse_text = output_text if len(output_text) <= 32768 else output_text[:32768]
            try:
                parsed = json.loads(parse_text)
            except (json.JSONDecodeError, ValueError):
                if tool_name in self._registered:
                    findings.append(Finding(
                        scanner_name=self.name,
                        rule_id="SCHEMA-FORMAT-001",
                        severity=Severity.HIGH,
                        confidence=0.85,
                        description="Tool returned free text but structured output was expected",
                        matched_text=output_text[:200],
                        location="$",
                    ))
                return findings

        schema = self._registered.get(tool_name)
        if schema is None:
            if tool_name not in self._baselines:
                self._baselines[tool_name] = _infer_schema(parsed)
                return findings
            schema = self._baselines[tool_name]
            is_baseline = True
        else:
            is_baseline = False

        strict = self._strict.get(tool_name, False)
        _check_shape(parsed, schema, "$", findings, strict, is_baseline)

        _scan_string_fields(parsed, "$", findings)

        return findings


_INJECTION_MARKERS = [
    "ignore previous", "you are now", "new instructions",
    "system:", "<important>", "[inst]", "<<sys>>",
    "ignore all prior", "disregard",
]


def _check_shape(value, schema, path, findings, strict, is_baseline):
    # type: (Any, Any, str, List[Finding], bool, bool) -> None
    """recursively compare value against schema."""
    if schema is ANY:
        return

    if isinstance(schema, type):
        if not isinstance(value, schema):
            sev = Severity.HIGH if not is_baseline else Severity.MEDIUM
            conf = 0.80 if not is_baseline else 0.60
            findings.append(Finding(
                scanner_name="schema_drift",
                rule_id="SCHEMA-TYPE-001",
                severity=sev,
                confidence=conf,
                description="Expected {}, got {}".format(
                    schema.__name__, type(value).__name__
                ),
                matched_text=str(value)[:200],
                location=path,
            ))
        return

    if isinstance(schema, (set, frozenset)):
        if not any(isinstance(value, t) for t in schema):
            findings.append(Finding(
                scanner_name="schema_drift",
                rule_id="SCHEMA-TYPE-002",
                severity=Severity.MEDIUM,
                confidence=0.70,
                description="Expected one of {}, got {}".format(
                    [t.__name__ for t in schema], type(value).__name__
                ),
                matched_text=str(value)[:200],
                location=path,
            ))
        return

    if isinstance(schema, dict):
        if not isinstance(value, dict):
            findings.append(Finding(
                scanner_name="schema_drift",
                rule_id="SCHEMA-SHAPE-001",
                severity=Severity.HIGH,
                confidence=0.85,
                description="Expected object, got {}".format(type(value).__name__),
                matched_text=str(value)[:200],
                location=path,
            ))
            return

        for key, sub_schema in schema.items():
            if key not in value:
                if strict:
                    findings.append(Finding(
                        scanner_name="schema_drift",
                        rule_id="SCHEMA-MISSING-001",
                        severity=Severity.LOW,
                        confidence=0.50,
                        description="Expected key '{}' missing".format(key),
                        location="{}.{}".format(path, key),
                    ))
            else:
                _check_shape(
                    value[key], sub_schema,
                    "{}.{}".format(path, key),
                    findings, strict, is_baseline,
                )

        if strict:
            extra_keys = set(value.keys()) - set(schema.keys())
            for key in sorted(extra_keys):
                findings.append(Finding(
                    scanner_name="schema_drift",
                    rule_id="SCHEMA-EXTRA-001",
                    severity=Severity.LOW,
                    confidence=0.40,
                    description="Unexpected key '{}'".format(key),
                    location="{}.{}".format(path, key),
                ))
        return

    if isinstance(schema, list) and len(schema) == 1:
        if not isinstance(value, list):
            findings.append(Finding(
                scanner_name="schema_drift",
                rule_id="SCHEMA-SHAPE-002",
                severity=Severity.HIGH,
                confidence=0.85,
                description="Expected array, got {}".format(type(value).__name__),
                matched_text=str(value)[:200],
                location=path,
            ))
            return
        item_schema = schema[0]
        for i, item in enumerate(value):
            _check_shape(
                item, item_schema,
                "{}[{}]".format(path, i),
                findings, strict, is_baseline,
            )
        return


def _scan_string_fields(value, path, findings, _depth=0):
    # type: (Any, str, List[Finding], int) -> None
    """flag long string fields containing injection markers."""
    if _depth > 10:
        return
    if isinstance(value, str):
        if len(value) > 200:
            lower = value.lower()
            for marker in _INJECTION_MARKERS:
                if marker in lower:
                    findings.append(Finding(
                        scanner_name="schema_drift",
                        rule_id="SCHEMA-SUSPICIOUS-STRING-001",
                        severity=Severity.MEDIUM,
                        confidence=0.65,
                        description="Long string field contains injection marker: '{}'".format(marker),
                        matched_text=value[:200],
                        location=path,
                    ))
                    break
    elif isinstance(value, dict):
        for k, v in list(value.items())[:50]:
            _scan_string_fields(v, "{}.{}".format(path, k), findings, _depth + 1)
    elif isinstance(value, list):
        for i, item in enumerate(value[:50]):
            _scan_string_fields(item, "{}[{}]".format(path, i), findings, _depth + 1)


def _infer_schema(value):
    # type: (Any) -> Any
    """infer a schema from a concrete value for auto-baseline."""
    if isinstance(value, dict):
        return {k: _infer_schema(v) for k, v in value.items()}
    elif isinstance(value, list):
        if not value:
            return [ANY]
        return [_infer_schema(value[0])]
    elif isinstance(value, bool):
        return bool  # bool is a subclass of int, so check first
    elif isinstance(value, int):
        return int
    elif isinstance(value, float):
        return float
    elif isinstance(value, str):
        return str
    elif value is None:
        return ANY
    return ANY
