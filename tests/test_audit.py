"""Tests for safehere.audit.AuditLogger."""

import json
import logging

from safehere._types import Action, Finding, ScanResult, Severity
from safehere.audit import AuditLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scan_result(tool_name="weather", findings=None, action=Action.ALLOW,
                      combined_score=0.0, max_severity=Severity.NONE,
                      scan_time_ms=1.5):
    return ScanResult(
        tool_name=tool_name,
        findings=findings or [],
        max_severity=max_severity,
        combined_score=combined_score,
        action=action,
        scan_time_ms=scan_time_ms,
    )


def _make_finding(matched_text="suspicious text", severity=Severity.HIGH):
    return Finding(
        scanner_name="pattern",
        rule_id="PAT-TEST-001",
        severity=severity,
        confidence=0.85,
        description="Test finding",
        matched_text=matched_text,
        location="normalized@offset:0",
    )


# ---------------------------------------------------------------------------
# 1. File-based logging writes valid JSONL
# ---------------------------------------------------------------------------

class TestFileBasedLogging:
    def test_writes_valid_jsonl(self, tmp_path):
        log_file = str(tmp_path / "audit.jsonl")
        logger = AuditLogger(log_path=log_file)

        finding = _make_finding()
        result = _make_scan_result(
            findings=[finding],
            action=Action.BLOCK,
            combined_score=0.85,
            max_severity=Severity.HIGH,
        )
        logger.log(result)
        logger.log(result)
        logger.close()

        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.read().strip().split("\n")

        assert len(lines) == 2
        for line in lines:
            record = json.loads(line)  # Must parse without error
            assert isinstance(record, dict)

    def test_appends_to_existing_file(self, tmp_path):
        log_file = str(tmp_path / "audit.jsonl")
        result = _make_scan_result(findings=[_make_finding()])

        logger1 = AuditLogger(log_path=log_file)
        logger1.log(result)
        logger1.close()

        logger2 = AuditLogger(log_path=log_file)
        logger2.log(result)
        logger2.close()

        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.read().strip().split("\n")
        assert len(lines) == 2


# ---------------------------------------------------------------------------
# 2. Each record has required fields
# ---------------------------------------------------------------------------

class TestRecordFields:
    def test_required_fields_present(self, tmp_path):
        log_file = str(tmp_path / "audit.jsonl")
        logger = AuditLogger(log_path=log_file)

        finding = _make_finding()
        result = _make_scan_result(
            tool_name="search",
            findings=[finding],
            action=Action.WARN,
            combined_score=0.55,
            max_severity=Severity.HIGH,
            scan_time_ms=2.345,
        )
        logger.log(result)
        logger.close()

        with open(log_file, "r", encoding="utf-8") as f:
            record = json.loads(f.readline())

        assert "timestamp" in record
        assert "+00:00" in record["timestamp"] or record["timestamp"].endswith("Z")
        assert record["tool_name"] == "search"
        assert record["action"] == "warn"
        assert record["combined_score"] == 0.55
        assert record["max_severity"] == "HIGH"
        assert record["scan_time_ms"] == 2.345
        assert isinstance(record["findings"], list)
        assert len(record["findings"]) == 1

        f_rec = record["findings"][0]
        assert f_rec["scanner"] == "pattern"
        assert f_rec["rule_id"] == "PAT-TEST-001"
        assert f_rec["severity"] == "HIGH"
        assert f_rec["confidence"] == 0.85
        assert f_rec["description"] == "Test finding"
        assert f_rec["matched_text"] == "suspicious text"
        assert f_rec["location"] == "normalized@offset:0"

    def test_context_included_when_provided(self, tmp_path):
        log_file = str(tmp_path / "audit.jsonl")
        logger = AuditLogger(log_path=log_file)

        result = _make_scan_result(findings=[_make_finding()])
        logger.log(result, context={"session_id": "abc-123"})
        logger.close()

        with open(log_file, "r", encoding="utf-8") as f:
            record = json.loads(f.readline())

        assert record["context"] == {"session_id": "abc-123"}


# ---------------------------------------------------------------------------
# 3. matched_text is truncated to max_text_len
# ---------------------------------------------------------------------------

class TestMatchedTextTruncation:
    def test_truncation_at_default_500(self, tmp_path):
        log_file = str(tmp_path / "audit.jsonl")
        logger = AuditLogger(log_path=log_file, max_text_len=500)

        long_text = "x" * 1000
        finding = _make_finding(matched_text=long_text)
        result = _make_scan_result(findings=[finding])
        logger.log(result)
        logger.close()

        with open(log_file, "r", encoding="utf-8") as f:
            record = json.loads(f.readline())

        assert len(record["findings"][0]["matched_text"]) == 500

    def test_truncation_custom_length(self, tmp_path):
        log_file = str(tmp_path / "audit.jsonl")
        logger = AuditLogger(log_path=log_file, max_text_len=50)

        long_text = "a" * 200
        finding = _make_finding(matched_text=long_text)
        result = _make_scan_result(findings=[finding])
        logger.log(result)
        logger.close()

        with open(log_file, "r", encoding="utf-8") as f:
            record = json.loads(f.readline())

        assert len(record["findings"][0]["matched_text"]) == 50

    def test_short_text_not_truncated(self, tmp_path):
        log_file = str(tmp_path / "audit.jsonl")
        logger = AuditLogger(log_path=log_file, max_text_len=500)

        finding = _make_finding(matched_text="short")
        result = _make_scan_result(findings=[finding])
        logger.log(result)
        logger.close()

        with open(log_file, "r", encoding="utf-8") as f:
            record = json.loads(f.readline())

        assert record["findings"][0]["matched_text"] == "short"


# ---------------------------------------------------------------------------
# 4. close() flushes and closes file
# ---------------------------------------------------------------------------

class TestClose:
    def test_close_flushes_and_closes(self, tmp_path):
        log_file = str(tmp_path / "audit.jsonl")
        logger = AuditLogger(log_path=log_file)

        result = _make_scan_result(findings=[_make_finding()])
        logger.log(result)

        # Before close, file handle should be open
        assert logger._file is not None
        assert not logger._file.closed

        logger.close()

        # After close, file handle should be None
        assert logger._file is None

        # File should still have content
        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
        assert len(content) > 0

    def test_close_idempotent(self, tmp_path):
        log_file = str(tmp_path / "audit.jsonl")
        logger = AuditLogger(log_path=log_file)

        result = _make_scan_result(findings=[_make_finding()])
        logger.log(result)
        logger.close()
        logger.close()  # Should not raise


# ---------------------------------------------------------------------------
# 5. Logger-based mode (no file path) logs to safehere.audit logger
# ---------------------------------------------------------------------------

class TestLoggerBasedMode:
    def test_logs_to_python_logger(self, caplog):
        logger = AuditLogger()  # No log_path -> logger mode

        finding = _make_finding()
        result = _make_scan_result(
            findings=[finding],
            action=Action.BLOCK,
            combined_score=0.85,
        )

        with caplog.at_level(logging.INFO, logger="safehere.audit"):
            logger.log(result)

        assert len(caplog.records) == 1
        record_data = json.loads(caplog.records[0].message)
        assert record_data["tool_name"] == "weather"
        assert record_data["action"] == "block"

    def test_no_file_created_in_logger_mode(self, tmp_path):
        logger = AuditLogger()  # No log_path
        result = _make_scan_result(findings=[_make_finding()])
        logger.log(result)
        logger.close()

        # No files should have been created
        assert logger._file is None
