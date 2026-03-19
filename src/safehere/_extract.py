"""Cohere V1/V2 tool-result adapter.

duck-typed extraction so safehere stays loosely coupled to the SDK.
"""

import dataclasses
import json
from typing import Any, Dict, List, Optional

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


def extract_v2_messages(messages):
    # type: (List[Any]) -> List[ExtractedOutput]
    """extract text and structured data from V2 tool messages."""
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

        tool_name = str(tool_call_id) if tool_call_id else "unknown"

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
