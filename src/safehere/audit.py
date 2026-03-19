"""structured audit logger for scan decisions."""

import datetime
import json
import logging
import threading
from typing import Any, Dict, List, Optional

from ._types import ScanResult

logger = logging.getLogger("safehere.audit")


class AuditLogger:
    """writes structured JSON-lines audit records for scan decisions."""

    def __init__(self, log_path=None, max_text_len=500):
        # type: (Optional[str], int) -> None
        self._log_path = log_path
        self._max_text_len = max_text_len
        self._lock = threading.Lock()
        self._file = None  # type: Optional[Any]

    def log(self, scan_result, context=None):
        # type: (ScanResult, Optional[Dict[str, Any]]) -> None
        """write a single audit record."""
        record = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "tool_name": scan_result.tool_name,
            "action": scan_result.action.value,
            "combined_score": scan_result.combined_score,
            "max_severity": scan_result.max_severity.name,
            "scan_time_ms": round(scan_result.scan_time_ms, 3),
            "findings": [
                {
                    "scanner": f.scanner_name,
                    "rule_id": f.rule_id,
                    "severity": f.severity.name,
                    "confidence": round(f.confidence, 3),
                    "description": f.description,
                    "matched_text": f.matched_text[:self._max_text_len],
                    "location": f.location,
                }
                for f in scan_result.findings
            ],
        }
        if context:
            record["context"] = context

        line = json.dumps(record, ensure_ascii=False)

        with self._lock:
            if self._log_path:
                self._write_to_file(line)
            else:
                logger.info(line)

    def _write_to_file(self, line):
        # type: (str) -> None
        if self._file is None:
            self._file = open(self._log_path, "a", encoding="utf-8")
        self._file.write(line + "\n")
        self._file.flush()

    def close(self):
        # type: () -> None
        """flush and close the log file."""
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None
