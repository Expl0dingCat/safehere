"""managed Cohere tool-use loop runners with integrated scanning."""

import inspect
import json
from typing import Any, Callable, Dict, List  # noqa: F401 -- used in type comments

from ._extract import (
    replace_v1_tool_result,
    replace_v2_message_content,
)
from ._types import Action
from .exceptions import QuarantineError


def _execute_tool_calls_v1(tool_calls, tool_executors):
    # type: (List[Any], Dict[str, Callable]) -> List[Any]
    """execute V1 tool calls and build tool_results list."""
    results = []
    for tc in tool_calls:
        name = tc.name
        params = tc.parameters if hasattr(tc, "parameters") else {}
        if name not in tool_executors:
            output = {"error": "Unknown tool: {}".format(name)}
        else:
            try:
                output = tool_executors[name](**params)
            except Exception as e:
                output = {"error": str(e)}
        if not isinstance(output, (dict, list)):
            output = {"result": str(output)}

        class _ToolResult:
            def __init__(self, call, outputs):
                self.call = call
                self.outputs = outputs

        results.append(_ToolResult(call=tc, outputs=[output]))
    return results


def _parse_v2_tool_call(tc):
    # type: (Any) -> tuple
    """extract (name, id, params) from a V2 tool call object."""
    name = getattr(tc, "function", None)
    if name is None:
        name = getattr(tc, "name", "unknown")
    else:
        name = getattr(name, "name", str(name))

    tc_id = getattr(tc, "id", "")
    params = {}
    func = getattr(tc, "function", None)
    if func and hasattr(func, "arguments"):
        args = func.arguments
        if isinstance(args, str):
            try:
                params = json.loads(args)
            except (json.JSONDecodeError, ValueError):
                params = {}
        elif isinstance(args, dict):
            params = args
    elif hasattr(tc, "parameters"):
        params = tc.parameters or {}

    return name, tc_id, params


def _execute_tool_calls_v2(tool_calls, tool_executors):
    # type: (List[Any], Dict[str, Callable]) -> List[Dict[str, Any]]
    """execute V2 tool calls (sync) and build tool messages list."""
    messages = []
    for tc in tool_calls:
        name, tc_id, params = _parse_v2_tool_call(tc)

        if name not in tool_executors:
            content = json.dumps({"error": "Unknown tool: {}".format(name)})
        else:
            try:
                result = tool_executors[name](**params)
                content = json.dumps(result, default=str) if not isinstance(result, str) else result
            except Exception as e:
                content = json.dumps({"error": str(e)})

        messages.append({
            "role": "tool",
            "tool_call_id": tc_id,
            "content": content,
        })
    return messages


async def _execute_tool_calls_v2_async(tool_calls, tool_executors):
    # type: (List[Any], Dict[str, Callable]) -> List[Dict[str, Any]]
    """execute V2 tool calls (async-aware) and build tool messages list."""
    messages = []
    for tc in tool_calls:
        name, tc_id, params = _parse_v2_tool_call(tc)

        if name not in tool_executors:
            content = json.dumps({"error": "Unknown tool: {}".format(name)})
        else:
            try:
                result = tool_executors[name](**params)
                # await if the executor returned a coroutine
                if inspect.isawaitable(result):
                    result = await result
                content = json.dumps(result, default=str) if not isinstance(result, str) else result
            except Exception as e:
                content = json.dumps({"error": str(e)})

        messages.append({
            "role": "tool",
            "tool_call_id": tc_id,
            "content": content,
        })
    return messages


def run_tool_loop_sync(client, guard, tool_executors, max_turns=10, **chat_kwargs):
    # type: (Any, Any, Dict[str, Callable], int, Any) -> Any
    """run a synchronous Cohere tool-use loop with scanning."""
    is_v2 = "messages" in chat_kwargs

    response = client.chat(**chat_kwargs)

    for _ in range(max_turns):
        if is_v2:
            tc = getattr(getattr(response, "message", None), "tool_calls", None)
        else:
            tc = getattr(response, "tool_calls", None)

        if not tc:
            break

        if is_v2:
            tool_messages = _execute_tool_calls_v2(tc, tool_executors)

            scan_results = guard.scan_tool_results(tool_messages, api_version="v2")

            for sr in scan_results:
                if sr.action == Action.QUARANTINE:
                    raise QuarantineError(sr, sr.tool_name)

            for sr in scan_results:
                if sr.action == Action.BLOCK:
                    for eo in guard._last_extracted:
                        if eo.tool_name == sr.tool_name:
                            tool_messages = replace_v2_message_content(
                                tool_messages, eo.source_index,
                                guard._config.block_message,
                            )

            messages = list(chat_kwargs.get("messages", []))
            assistant_msg = getattr(response, "message", response)
            messages.append(assistant_msg)
            messages.extend(tool_messages)
            chat_kwargs["messages"] = messages

        else:
            tool_results = _execute_tool_calls_v1(tc, tool_executors)

            scan_results = guard.scan_tool_results(tool_results, api_version="v1")

            for sr in scan_results:
                if sr.action == Action.QUARANTINE:
                    raise QuarantineError(sr, sr.tool_name)

            for sr in scan_results:
                if sr.action == Action.BLOCK:
                    for eo in guard._last_extracted:
                        if eo.tool_name == sr.tool_name:
                            tool_results = replace_v1_tool_result(
                                tool_results, eo.source_index,
                                guard._config.block_message,
                            )

            chat_kwargs["tool_results"] = tool_results
            if hasattr(response, "chat_history"):
                chat_kwargs["chat_history"] = response.chat_history

        response = client.chat(**chat_kwargs)

    return response


async def run_tool_loop_async(client, guard, tool_executors, max_turns=10, **chat_kwargs):
    # type: (Any, Any, Dict[str, Callable], int, Any) -> Any
    """async version of ``run_tool_loop_sync``."""
    is_v2 = "messages" in chat_kwargs

    response = await client.chat(**chat_kwargs)

    for _ in range(max_turns):
        if is_v2:
            tc = getattr(getattr(response, "message", None), "tool_calls", None)
        else:
            tc = getattr(response, "tool_calls", None)

        if not tc:
            break

        if is_v2:
            tool_messages = await _execute_tool_calls_v2_async(tc, tool_executors)
            scan_results = guard.scan_tool_results(tool_messages, api_version="v2")

            for sr in scan_results:
                if sr.action == Action.QUARANTINE:
                    raise QuarantineError(sr, sr.tool_name)

            for sr in scan_results:
                if sr.action == Action.BLOCK:
                    for eo in guard._last_extracted:
                        if eo.tool_name == sr.tool_name:
                            tool_messages = replace_v2_message_content(
                                tool_messages, eo.source_index,
                                guard._config.block_message,
                            )

            messages = list(chat_kwargs.get("messages", []))
            assistant_msg = getattr(response, "message", response)
            messages.append(assistant_msg)
            messages.extend(tool_messages)
            chat_kwargs["messages"] = messages
        else:
            tool_results = _execute_tool_calls_v1(tc, tool_executors)
            scan_results = guard.scan_tool_results(tool_results, api_version="v1")

            for sr in scan_results:
                if sr.action == Action.QUARANTINE:
                    raise QuarantineError(sr, sr.tool_name)

            for sr in scan_results:
                if sr.action == Action.BLOCK:
                    for eo in guard._last_extracted:
                        if eo.tool_name == sr.tool_name:
                            tool_results = replace_v1_tool_result(
                                tool_results, eo.source_index,
                                guard._config.block_message,
                            )

            chat_kwargs["tool_results"] = tool_results
            if hasattr(response, "chat_history"):
                chat_kwargs["chat_history"] = response.chat_history

        response = await client.chat(**chat_kwargs)

    return response
