"""Format-specific overhead calculation."""

import json
import re
from dataclasses import dataclass
from typing import Any

from src.tokens.counter import count_tokens


@dataclass
class FormatOverhead:
    """Token overhead analysis for a format."""

    format_type: str
    total_tokens: int
    content_tokens: int  # Tokens for actual tool name and arguments
    overhead_tokens: int  # Tokens for format structure

    @property
    def overhead_ratio(self) -> float:
        """Ratio of overhead to content tokens."""
        if self.content_tokens == 0:
            return float("inf")
        return self.overhead_tokens / self.content_tokens

    @property
    def efficiency(self) -> float:
        """Efficiency as percentage of tokens that are content."""
        if self.total_tokens == 0:
            return 0.0
        return self.content_tokens / self.total_tokens


def calculate_format_overhead(
    tool_call_text: str,
    format_type: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> FormatOverhead:
    """Calculate token overhead for a specific format."""
    total_tokens = count_tokens(tool_call_text)

    # Calculate content tokens (tool name + argument values)
    content_parts = [tool_name]
    content_parts.extend(_flatten_values(arguments))
    content_text = " ".join(str(p) for p in content_parts)
    content_tokens = count_tokens(content_text)

    overhead_tokens = total_tokens - content_tokens

    return FormatOverhead(
        format_type=format_type,
        total_tokens=total_tokens,
        content_tokens=content_tokens,
        overhead_tokens=overhead_tokens,
    )


def _flatten_values(obj: Any) -> list[Any]:
    """Flatten nested structure to list of leaf values."""
    values = []

    if isinstance(obj, dict):
        for v in obj.values():
            values.extend(_flatten_values(v))
    elif isinstance(obj, list):
        for item in obj:
            values.extend(_flatten_values(item))
    else:
        values.append(obj)

    return values


def calculate_json_overhead(tool_name: str, arguments: dict[str, Any]) -> FormatOverhead:
    """Calculate overhead for JSON format."""
    # Generate minimal JSON
    tool_call = {"tool": tool_name, "parameters": arguments}
    json_text = json.dumps(tool_call)

    return calculate_format_overhead(json_text, "json", tool_name, arguments)


def calculate_mcp_overhead(tool_name: str, arguments: dict[str, Any]) -> FormatOverhead:
    """Calculate overhead for MCP format."""
    # Generate MCP JSON-RPC
    tool_call = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    }
    json_text = json.dumps(tool_call)

    return calculate_format_overhead(json_text, "mcp", tool_name, arguments)


def calculate_xml_overhead(tool_name: str, arguments: dict[str, Any]) -> FormatOverhead:
    """Calculate overhead for XML format."""
    # Generate XML
    params_xml = _dict_to_xml(arguments)
    xml_text = f"<tool_call><name>{tool_name}</name><parameters>{params_xml}</parameters></tool_call>"

    return calculate_format_overhead(xml_text, "xml", tool_name, arguments)


def _dict_to_xml(obj: Any) -> str:
    """Convert dict/list to XML string."""
    if isinstance(obj, dict):
        parts = []
        for k, v in obj.items():
            inner = _dict_to_xml(v)
            parts.append(f"<{k}>{inner}</{k}>")
        return "".join(parts)
    elif isinstance(obj, list):
        parts = [f"<item>{_dict_to_xml(item)}</item>" for item in obj]
        return "".join(parts)
    else:
        return str(obj)


def calculate_function_sig_overhead(tool_name: str, arguments: dict[str, Any]) -> FormatOverhead:
    """Calculate overhead for function signature format."""
    # Generate function call
    args_str = ", ".join(f"{k}={repr(v)}" for k, v in arguments.items())
    func_text = f"{tool_name}({args_str})"

    return calculate_format_overhead(func_text, "function_sig", tool_name, arguments)


def calculate_unstructured_overhead(tool_name: str, arguments: dict[str, Any]) -> FormatOverhead:
    """Calculate overhead for unstructured format."""
    # Generate unstructured text
    params_text = "\n".join(f"- {k}: {v}" for k, v in arguments.items())
    text = f"TOOL_CALL: {tool_name}\nPARAMETERS:\n{params_text}\nEND_TOOL_CALL"

    return calculate_format_overhead(text, "unstructured", tool_name, arguments)


def compare_format_overhead(
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, FormatOverhead]:
    """Compare overhead across all formats for a given tool call."""
    return {
        "json": calculate_json_overhead(tool_name, arguments),
        "mcp": calculate_mcp_overhead(tool_name, arguments),
        "xml": calculate_xml_overhead(tool_name, arguments),
        "function_sig": calculate_function_sig_overhead(tool_name, arguments),
        "unstructured": calculate_unstructured_overhead(tool_name, arguments),
    }
