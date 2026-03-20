"""ToolGuard -- main public entry point for safehere."""

import json
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union  # noqa: F401 -- used in type comments

from ._cohere_loop import run_tool_loop_async, run_tool_loop_sync
from ._extract import (  # noqa: F401 -- ExtractedOutput used in type comments
    ExtractedOutput,
    extract_auto,
    extract_v1_tool_results,
    extract_v2_messages,
)
from ._types import Action, Finding, GuardConfig, ScanResult, ToolPolicy  # noqa: F401 -- used in type comments
from .audit import AuditLogger
from .exceptions import ConfigurationError, QuarantineError, ScanError  # noqa: F401
from .scanners import BaseScanner, ScannerPipeline  # noqa: F401 -- used in type comments
from .scanners.anomaly import AnomalyScanner
from .scanners.heuristic import HeuristicScanner
from .scanners.pattern import PatternScanner
from .scanners.schema import SchemaDriftScanner
from .scanners.semantic import SemanticScanner
from .scoring import ScoringEngine


class ToolGuard:
    """runtime tool-output scanner for Cohere agents.

    three API tiers: ``run()`` wraps the full loop, ``scan_tool_results()``
    scans a batch, and ``scan_output()`` scans a single output.
    """

    def __init__(self, config=None, scanners=None, audit_log_path=None,
                 on_finding=None):
        # type: (Optional[GuardConfig], Optional[List[BaseScanner]], Optional[str], Optional[Callable[[ScanResult], None]]) -> None  # noqa: E501
        self._config = config or GuardConfig()
        self._on_finding = on_finding

        if audit_log_path is not None:
            self._config.audit_log_path = audit_log_path

        if scanners is not None:
            self._pipeline = ScannerPipeline(scanners)
        else:
            self._pipeline = ScannerPipeline([
                PatternScanner(),
                SchemaDriftScanner(),
                AnomalyScanner(),
                HeuristicScanner(),
                SemanticScanner(),
            ])

        for policy in self._config.tool_policies.values():
            if policy.expected_schema is not None:
                self.register_schema(
                    policy.tool_name,
                    policy.expected_schema,
                    strict=policy.schema_strict,
                )

        per_tool_thresholds = {}
        for name, policy in self._config.tool_policies.items():
            if policy.thresholds:
                per_tool_thresholds[name] = policy.thresholds

        self._scoring = ScoringEngine(
            thresholds=self._config.default_thresholds,
            per_tool_thresholds=per_tool_thresholds,
        )

        self._audit = AuditLogger(
            log_path=self._config.audit_log_path,
            max_text_len=self._config.audit_log_max_text_len,
        )

        self._last_extracted = []  # type: List[ExtractedOutput]

    def run(self, client, tool_executors, max_turns=10, **chat_kwargs):
        # type: (Any, Dict[str, Callable], int, Any) -> Any
        """synchronous managed tool-use loop with scanning."""
        if not self._config.enabled:
            return client.chat(**chat_kwargs)
        return run_tool_loop_sync(
            client, self, tool_executors,
            max_turns=max_turns, **chat_kwargs,
        )

    async def arun(self, client, tool_executors, max_turns=10, **chat_kwargs):
        # type: (Any, Dict[str, Callable], int, Any) -> Any
        """async managed tool-use loop for ``cohere.AsyncClient``."""
        if not self._config.enabled:
            return await client.chat(**chat_kwargs)
        return await run_tool_loop_async(
            client, self, tool_executors,
            max_turns=max_turns, **chat_kwargs,
        )

    def scan_tool_results(self, tool_results, api_version="auto"):
        # type: (Any, str) -> List[ScanResult]
        """scan assembled tool results before passing to ``chat()``."""
        if not self._config.enabled:
            return []

        if api_version == "auto":
            extracted = extract_auto(tool_results)
        elif api_version == "v1":
            extracted = extract_v1_tool_results(tool_results)
        elif api_version == "v2":
            extracted = extract_v2_messages(tool_results)
        else:
            raise ConfigurationError(
                "api_version must be 'v1', 'v2', or 'auto', got: {}".format(
                    api_version
                )
            )

        self._last_extracted = extracted
        results = []  # type: List[ScanResult]

        for eo in extracted:
            result = self._scan_single(eo.tool_name, eo.text, eo.structured)
            results.append(result)

        return results

    def check(self, tool_name, output):
        # type: (str, Union[str, Dict[str, Any]]) -> Tuple[bool, Union[str, Dict[str, Any]]]
        """check a tool output and return (safe, sanitized_output).

        the simplest API surface -- returns a boolean and the output to use.
        if safe is True, sanitized_output is the original output unchanged.
        if safe is False, sanitized_output is the configured block message.

        note: this writes to the audit log and updates anomaly baselines.
        """
        if output is None:
            return True, output
        if isinstance(output, dict):
            text = json.dumps(output, default=str, ensure_ascii=False)
            structured = output
        elif isinstance(output, str):
            text = output
            try:
                structured = json.loads(output)
            except (json.JSONDecodeError, ValueError):
                structured = None
        else:
            text = str(output)
            structured = None

        result = self._scan_single(tool_name, text, structured)

        if result.is_blocked():
            return False, self._config.block_message
        return True, output

    def scan_output(self, tool_name, output):
        # type: (str, Union[str, Dict[str, Any]]) -> List[Finding]
        """scan a single tool output. returns raw findings without scoring."""
        if isinstance(output, dict):
            text = json.dumps(output, default=str, ensure_ascii=False)
            structured = output
        else:
            text = output
            try:
                structured = json.loads(output)
            except (json.JSONDecodeError, ValueError):
                structured = None

        return self._pipeline.scan_all(tool_name, text, structured)

    def register_schema(self, tool_name, schema, strict=False):
        # type: (str, Any, bool) -> None
        """register expected output schema for a tool."""
        schema_scanner = self._find_scanner(SchemaDriftScanner)
        if schema_scanner is None:
            raise ConfigurationError(
                "Cannot register schema: no SchemaDriftScanner in pipeline"
            )
        schema_scanner.register_schema(tool_name, schema, strict=strict)

    def register_schemas(self, schemas):
        # type: (Dict[str, Any]) -> None
        """bulk register schemas. keys are tool names, values are schemas."""
        for tool_name, schema in schemas.items():
            self.register_schema(tool_name, schema)

    def add_scanner(self, scanner):
        # type: (BaseScanner) -> None
        """add a custom scanner to the pipeline."""
        self._pipeline.add_scanner(scanner)

    def remove_scanner(self, name):
        # type: (str) -> None
        """remove a scanner by name."""
        self._pipeline.remove_scanner(name)

    def reset(self):
        # type: () -> None
        """reset all scanner state for a new conversation."""
        self._pipeline.reset()
        self._last_extracted = []

    def close(self):
        # type: () -> None
        """flush audit log and release resources."""
        self._audit.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _scan_single(self, tool_name, text, structured):
        # type: (str, str, Optional[Dict[str, Any]]) -> ScanResult
        start = time.monotonic()

        try:
            findings = self._pipeline.scan_all(tool_name, text, structured)
        except Exception as e:
            raise ScanError("Scanner failed on tool '{}': {}".format(tool_name, e)) from e

        elapsed_ms = (time.monotonic() - start) * 1000
        result = self._scoring.evaluate(tool_name, findings, elapsed_ms)

        if result.findings:
            self._audit.log(result)

        if result.findings and self._on_finding:
            self._on_finding(result)

        return result

    def _find_scanner(self, scanner_type):
        # type: (type) -> Optional[Any]
        for s in self._pipeline.scanners:
            if isinstance(s, scanner_type):
                return s
        return None
