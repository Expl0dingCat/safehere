"""Cohere V1/V2 tool-result adapter.

duck-typed extraction so safehere stays loosely coupled to the SDK.
"""

import dataclasses
import json
from typing import Any, Dict, List, Optional  # noqa: F401 -- used in type comments

from .exceptions import ConfigurationError


@dataclasses.dataclass
class ExtractedOutput:
    """normalized representation of one tool output."""

    tool_name: str
    text: str
    structured: Optional[Dict[str, Any]]
    source_index: int  # position in the original list


def detect_api_version(tool_results):
    # type: (Any) -> str
    """auto-detect whether tool_results are V1 or V2 format."""
    if not tool_results:
        raise ConfigurationError("Empty tool_results list")

    first = tool_results[0]

    if hasattr(first, "call"):
        return "v1"

    if isinstance(first, dict) and first.get("role") == "tool":
        return "v2"

    if hasattr(first, "tool_call_id"):
        return "v2"

    raise ConfigurationError(
        "Cannot determine Cohere API version from tool_results. "
        "Expected V1 ToolResult objects or V2 tool messages."
    )


def extract_v1_tool_results(tool_results):
    # type: (List[Any]) -> List[ExtractedOutput]
    """extract text and structured data from V1 tool_results."""
    extracted = []  # type: List[ExtractedOutput]
    for i, tr in enumerate(tool_results):
        tool_name = getattr(getattr(tr, "call", None), "name", "unknown")
        outputs = getattr(tr, "outputs", [])

        for output in outputs:
            if isinstance(output, dict):
                text = json.dumps(output, default=str, ensure_ascii=False)
                structured = output
            elif isinstance(output, str):
                text = output
                structured = None
            else:
                text = str(output)
                structured = None

            extracted.append(ExtractedOutput(
                tool_name=tool_name,
                text=text,
                structured=structured,
                source_index=i,
            ))

    return extracted


def build_tool_call_map(messages):
    # type: (List[Any]) -> Dict[str, str]
    """build a map from tool_call_id -> function_name from assistant messages.

    scans the message list for assistant messages containing tool_calls
    and extracts the id->name mapping. this is needed because V2 tool
    result messages only carry the call ID, not the function name.
    """
    id_to_name = {}  # type: Dict[str, str]
    for msg in messages:
        # dict form
        if isinstance(msg, dict):
            role = msg.get("role", "")
            if role != "assistant":
                continue
            tool_calls = msg.get("tool_calls", []) or []
        else:
            role = getattr(msg, "role", "")
            if role != "assistant":
                continue
            tool_calls = getattr(msg, "tool_calls", None) or []
            # V2 response wraps in .message
            if not tool_calls and hasattr(msg, "message"):
                tool_calls = getattr(msg.message, "tool_calls", None) or []

        for tc in tool_calls:
            tc_id = ""
            tc_name = "unknown"
            if isinstance(tc, dict):
                tc_id = tc.get("id", "")
                func = tc.get("function", {})
                tc_name = func.get("name", "unknown") if isinstance(func, dict) else str(func)
            else:
                tc_id = getattr(tc, "id", "")
                func = getattr(tc, "function", None)
                if func:
                    tc_name = getattr(func, "name", None) or "unknown"
                else:
                    tc_name = getattr(tc, "name", "unknown")
            if tc_id:
                id_to_name[tc_id] = tc_name
    return id_to_name


def extract_v2_messages(messages, tool_call_map=None):
    # type: (List[Any], Optional[Dict[str, str]]) -> List[ExtractedOutput]
    """extract text and structured data from V2 tool messages.

    if tool_call_map is provided, it maps tool_call_id -> function_name
    so per-tool policies and schemas work correctly. without it, falls
    back to the call ID (which breaks per-tool config).
    """
    if tool_call_map is None:
        tool_call_map = build_tool_call_map(messages)

    extracted = []  # type: List[ExtractedOutput]
    for i, msg in enumerate(messages):
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
            tool_call_id = msg.get("tool_call_id", "")
        else:
            role = getattr(msg, "role", "")
            content = getattr(msg, "content", "")
            tool_call_id = getattr(msg, "tool_call_id", "")

        if role != "tool":
            continue

        # resolve call ID to function name via the map
        call_id_str = str(tool_call_id) if tool_call_id else ""
        tool_name = tool_call_map.get(call_id_str, call_id_str or "unknown")

        if isinstance(content, str):
            try:
                structured = json.loads(content)
            except (json.JSONDecodeError, ValueError):
                structured = None
            extracted.append(ExtractedOutput(
                tool_name=tool_name,
                text=content,
                structured=structured,
                source_index=i,
            ))
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "document":
                        doc_data = block.get("document", {}).get("data", "")
                        text = json.dumps(doc_data, default=str) if isinstance(doc_data, dict) else str(doc_data)
                        structured = doc_data if isinstance(doc_data, dict) else None
                    elif block.get("type") == "text":
                        text = block.get("text", "")
                        structured = None
                    else:
                        text = json.dumps(block, default=str)
                        structured = block
                else:
                    text = str(block)
                    structured = None

                extracted.append(ExtractedOutput(
                    tool_name=tool_name,
                    text=text,
                    structured=structured,
                    source_index=i,
                ))

    return extracted


def extract_auto(tool_results):
    # type: (Any) -> List[ExtractedOutput]
    """auto-detect format and extract tool outputs."""
    version = detect_api_version(tool_results)
    if version == "v1":
        return extract_v1_tool_results(tool_results)
    return extract_v2_messages(tool_results)


def replace_v1_tool_result(tool_results, source_index, replacement_text):
    # type: (List[Any], int, str) -> List[Any]
    """return a new V1 tool_results list with one output replaced."""
    new_results = list(tool_results)
    original = tool_results[source_index]

    class _ReplacedToolResult:
        def __init__(self, call, outputs):
            self.call = call
            self.outputs = outputs

    new_results[source_index] = _ReplacedToolResult(
        call=getattr(original, "call", None),
        outputs=[{"result": replacement_text}],
    )
    return new_results


def replace_v2_message_content(messages, source_index, replacement_text):
    # type: (List[Any], int, str) -> List[Any]
    """return a new V2 messages list with one tool message content replaced."""
    new_messages = list(messages)
    original = messages[source_index]

    if isinstance(original, dict):
        replaced = dict(original)
        replaced["content"] = replacement_text
    else:
        replaced = {
            "role": "tool",
            "tool_call_id": getattr(original, "tool_call_id", ""),
            "content": replacement_text,
        }

    new_messages[source_index] = replaced
    return new_messages
