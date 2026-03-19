"""Tests for safehere._extract."""

import json

import pytest

from safehere._extract import (
    ExtractedOutput,
    detect_api_version,
    extract_auto,
    extract_v1_tool_results,
    extract_v2_messages,
    replace_v1_tool_result,
    replace_v2_message_content,
)
from safehere.exceptions import ConfigurationError


# ---------------------------------------------------------------------------
# Helpers: mock objects (same as conftest.py but defined locally for import)
# ---------------------------------------------------------------------------

class MockToolCall:
    """Mimics a V1 Cohere ToolCall."""
    def __init__(self, name, parameters=None):
        self.name = name
        self.parameters = parameters or {}


class MockToolResult:
    """Mimics a V1 Cohere ToolResult."""
    def __init__(self, name, outputs):
        self.call = MockToolCall(name)
        self.outputs = outputs

class MockV2ToolMessage:
    """Mimics a V2 Cohere tool message object (attribute-based, not dict)."""
    def __init__(self, role, tool_call_id, content):
        self.role = role
        self.tool_call_id = tool_call_id
        self.content = content


# ---------------------------------------------------------------------------
# 1. V1 extraction reads tool_name from .call.name and outputs from .outputs
# ---------------------------------------------------------------------------

class TestV1Extraction:
    def test_single_tool_result(self):
        tr = MockToolResult("weather", [{"temp": 72, "humidity": 45}])
        extracted = extract_v1_tool_results([tr])

        assert len(extracted) == 1
        assert extracted[0].tool_name == "weather"
        assert extracted[0].source_index == 0
        assert extracted[0].structured == {"temp": 72, "humidity": 45}
        assert "72" in extracted[0].text

    def test_multiple_tool_results(self):
        results = [
            MockToolResult("weather", [{"temp": 72}]),
            MockToolResult("search", [{"title": "Result"}]),
        ]
        extracted = extract_v1_tool_results(results)

        assert len(extracted) == 2
        assert extracted[0].tool_name == "weather"
        assert extracted[1].tool_name == "search"
        assert extracted[0].source_index == 0
        assert extracted[1].source_index == 1

    def test_multiple_outputs_per_tool(self):
        tr = MockToolResult("multi", [{"a": 1}, {"b": 2}])
        extracted = extract_v1_tool_results([tr])

        assert len(extracted) == 2
        assert all(e.tool_name == "multi" for e in extracted)
        assert all(e.source_index == 0 for e in extracted)

    def test_string_output(self):
        tr = MockToolResult("echo", ["hello world"])
        extracted = extract_v1_tool_results([tr])

        assert len(extracted) == 1
        assert extracted[0].text == "hello world"
        assert extracted[0].structured is None


# ---------------------------------------------------------------------------
# 2. V2 dict extraction reads role=tool messages
# ---------------------------------------------------------------------------

class TestV2DictExtraction:
    def test_single_v2_message(self):
        messages = [
            {"role": "tool", "tool_call_id": "tc1", "content": '{"temp": 72}'},
        ]
        extracted = extract_v2_messages(messages)

        assert len(extracted) == 1
        assert extracted[0].tool_name == "tc1"
        assert extracted[0].structured == {"temp": 72}
        assert extracted[0].source_index == 0

    def test_multiple_v2_messages(self):
        messages = [
            {"role": "tool", "tool_call_id": "tc1", "content": '{"temp": 72}'},
            {"role": "tool", "tool_call_id": "tc2", "content": '{"title": "Result"}'},
        ]
        extracted = extract_v2_messages(messages)

        assert len(extracted) == 2
        assert extracted[0].tool_name == "tc1"
        assert extracted[1].tool_name == "tc2"

    def test_non_tool_messages_skipped(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "tool", "tool_call_id": "tc1", "content": '{"data": 1}'},
            {"role": "assistant", "content": "Hi"},
        ]
        extracted = extract_v2_messages(messages)

        assert len(extracted) == 1
        assert extracted[0].tool_name == "tc1"
        assert extracted[0].source_index == 1

    def test_plain_text_content(self):
        messages = [
            {"role": "tool", "tool_call_id": "tc1", "content": "plain text response"},
        ]
        extracted = extract_v2_messages(messages)

        assert len(extracted) == 1
        assert extracted[0].text == "plain text response"
        assert extracted[0].structured is None


# ---------------------------------------------------------------------------
# 3. V2 object extraction reads .tool_call_id attributes
# ---------------------------------------------------------------------------

class TestV2ObjectExtraction:
    def test_object_style_message(self):
        msg = MockV2ToolMessage("tool", "tc1", '{"temp": 72}')
        extracted = extract_v2_messages([msg])

        assert len(extracted) == 1
        assert extracted[0].tool_name == "tc1"
        assert extracted[0].structured == {"temp": 72}

    def test_object_non_tool_skipped(self):
        msgs = [
            MockV2ToolMessage("user", None, "hello"),
            MockV2ToolMessage("tool", "tc1", '{"data": 1}'),
        ]
        extracted = extract_v2_messages(msgs)

        assert len(extracted) == 1
        assert extracted[0].tool_name == "tc1"


# ---------------------------------------------------------------------------
# 4. Auto-detection correctly identifies V1 vs V2
# ---------------------------------------------------------------------------

class TestAutoDetection:
    def test_detects_v1(self):
        results = [MockToolResult("weather", [{"temp": 72}])]
        assert detect_api_version(results) == "v1"

    def test_detects_v2_dict(self):
        messages = [{"role": "tool", "tool_call_id": "tc1", "content": "data"}]
        assert detect_api_version(messages) == "v2"

    def test_detects_v2_object(self):
        messages = [MockV2ToolMessage("tool", "tc1", "data")]
        assert detect_api_version(messages) == "v2"

    def test_extract_auto_v1(self):
        results = [MockToolResult("weather", [{"temp": 72}])]
        extracted = extract_auto(results)
        assert len(extracted) == 1
        assert extracted[0].tool_name == "weather"

    def test_extract_auto_v2(self):
        messages = [{"role": "tool", "tool_call_id": "tc1", "content": '{"temp": 72}'}]
        extracted = extract_auto(messages)
        assert len(extracted) == 1
        assert extracted[0].tool_name == "tc1"

    def test_unknown_format_raises(self):
        with pytest.raises(ConfigurationError, match="Cannot determine"):
            detect_api_version([{"some": "random", "dict": "value"}])


# ---------------------------------------------------------------------------
# 5. replace_v1_tool_result returns new list with replaced output
# ---------------------------------------------------------------------------

class TestReplaceV1:
    def test_replaces_correct_index(self):
        results = [
            MockToolResult("weather", [{"temp": 72}]),
            MockToolResult("search", [{"title": "Result"}]),
        ]
        new_results = replace_v1_tool_result(results, 1, "[BLOCKED]")

        # Original unchanged
        assert results[1].outputs == [{"title": "Result"}]

        # New list has replacement at index 1
        assert len(new_results) == 2
        assert new_results[0] is results[0]  # Unchanged
        assert new_results[1].outputs == [{"result": "[BLOCKED]"}]
        assert getattr(new_results[1].call, "name", None) == "search"

    def test_original_list_not_mutated(self):
        results = [MockToolResult("weather", [{"temp": 72}])]
        new_results = replace_v1_tool_result(results, 0, "replaced")

        assert results[0].outputs == [{"temp": 72}]
        assert new_results[0].outputs == [{"result": "replaced"}]
        assert new_results is not results


# ---------------------------------------------------------------------------
# 6. replace_v2_message_content returns new list with replaced content
# ---------------------------------------------------------------------------

class TestReplaceV2:
    def test_replaces_dict_message(self):
        messages = [
            {"role": "tool", "tool_call_id": "tc1", "content": '{"temp": 72}'},
            {"role": "tool", "tool_call_id": "tc2", "content": "original"},
        ]
        new_messages = replace_v2_message_content(messages, 1, "[BLOCKED]")

        # Original unchanged
        assert messages[1]["content"] == "original"

        # New list has replacement
        assert len(new_messages) == 2
        assert new_messages[0] is messages[0]
        assert new_messages[1]["content"] == "[BLOCKED]"
        assert new_messages[1]["role"] == "tool"
        assert new_messages[1]["tool_call_id"] == "tc2"

    def test_replaces_object_message(self):
        msg = MockV2ToolMessage("tool", "tc1", '{"temp": 72}')
        messages = [msg]
        new_messages = replace_v2_message_content(messages, 0, "[BLOCKED]")

        # Original unchanged
        assert msg.content == '{"temp": 72}'

        # New item is a dict (not the original object)
        assert isinstance(new_messages[0], dict)
        assert new_messages[0]["content"] == "[BLOCKED]"
        assert new_messages[0]["role"] == "tool"
        assert new_messages[0]["tool_call_id"] == "tc1"

    def test_original_list_not_mutated(self):
        messages = [
            {"role": "tool", "tool_call_id": "tc1", "content": "data"},
        ]
        new_messages = replace_v2_message_content(messages, 0, "replaced")
        assert messages[0]["content"] == "data"
        assert new_messages is not messages


# ---------------------------------------------------------------------------
# 7. Empty tool_results raises ConfigurationError
# ---------------------------------------------------------------------------

class TestEmptyToolResults:
    def test_empty_list_raises(self):
        with pytest.raises(ConfigurationError, match="Empty tool_results"):
            detect_api_version([])

    def test_extract_auto_empty_raises(self):
        with pytest.raises(ConfigurationError, match="Empty tool_results"):
            extract_auto([])

    def test_none_like_empty_raises(self):
        """Passing None or falsy value raises ConfigurationError."""
        with pytest.raises(ConfigurationError):
            detect_api_version(None)
