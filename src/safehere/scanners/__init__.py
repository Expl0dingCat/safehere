"""scanner pipeline and built-in scanners."""

from typing import Any, Dict, List, Optional  # noqa: F401 -- used in type comments

from .._types import Finding  # noqa: F401 -- used in type comments
from ._base import BaseScanner


class ScannerPipeline:
    """composes multiple scanners and runs them in sequence."""

    def __init__(self, scanners=None):
        # type: (Optional[List[BaseScanner]]) -> None
        self._scanners = list(scanners) if scanners else []

    def add_scanner(self, scanner):
        # type: (BaseScanner) -> None
        self._scanners.append(scanner)

    def remove_scanner(self, name):
        # type: (str) -> None
        self._scanners = [s for s in self._scanners if s.name != name]

    def scan_all(self, tool_name, output_text, output_structured=None):
        # type: (str, str, Optional[Dict[str, Any]]) -> List[Finding]
        findings = []  # type: List[Finding]
        for scanner in self._scanners:
            findings.extend(scanner.scan(tool_name, output_text, output_structured))
        return findings

    def reset(self):
        # type: () -> None
        for scanner in self._scanners:
            scanner.reset()

    @property
    def scanners(self):
        # type: () -> List[BaseScanner]
        return list(self._scanners)


__all__ = ["BaseScanner", "ScannerPipeline"]
