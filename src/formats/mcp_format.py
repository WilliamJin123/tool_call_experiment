"""MCP (Model Context Protocol) format for tool calls - strict parsing."""

import json
import re
from typing import Any

from src.formats.base import BaseFormat, ToolCall
from src.tools.base import Tool


class MCPFormat(BaseFormat):
    """
    MCP format handler with strict validation.

    Expected format:
    {
      "jsonrpc": "2.0",
      "method": "tools/call",
      "params": {
        "name": "tool_name",
        "arguments": {
          "param1": "value1"
        }
      }
    }
    """

    @property
    def name(self) -> str:
        return "mcp"

    def generate_system_prompt(self, tools: list[Tool]) -> str:
        tools_text = "\n\n".join(self.format_tool_for_prompt(t) for t in tools)

        return f"""You are a helpful assistant with access to the following tools:

{tools_text}

When you need to use a tool, you MUST respond with a JSON-RPC 2.0 formatted request using EXACTLY this structure:
```json
{{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {{
    "name": "tool_name",
    "arguments": {{
      "param1": "value1",
      "param2": "value2"
    }}
  }}
}}
```

CRITICAL REQUIREMENTS:
- The "jsonrpc" field MUST be exactly "2.0"
- The "method" field MUST be exactly "tools/call"
- The "params" object MUST contain "name" and "arguments"
- Use exact parameter names from the tool definitions

You may call multiple tools by providing multiple JSON-RPC objects.
Only include the JSON-RPC when you want to call a tool. For regular responses, just reply normally."""

    def parse_response(self, response: str) -> list[ToolCall]:
        """Parse MCP tool calls from response with strict validation."""
        tool_calls = []

        # Find JSON blocks in code fences
        code_block_pattern = r"```(?:json)?\s*(\{[^`]*\})\s*```"
        matches = re.findall(code_block_pattern, response, re.DOTALL)

        # Also try bare JSON objects
        if not matches:
            matches = self._extract_json_objects(response)

        for match in matches:
            try:
                parsed = json.loads(match)
                if self._is_valid_mcp(parsed):
                    tool_call = ToolCall(
                        name=parsed["params"]["name"],
                        arguments=parsed["params"].get("arguments", {}),
                        raw_text=match,
                    )
                    tool_calls.append(tool_call)
            except (json.JSONDecodeError, KeyError, TypeError):
                continue

        return tool_calls

    def _is_valid_mcp(self, obj: Any) -> bool:
        """Strictly validate MCP format."""
        if not isinstance(obj, dict):
            return False

        # Check required fields
        if obj.get("jsonrpc") != "2.0":
            return False

        if obj.get("method") != "tools/call":
            return False

        params = obj.get("params")
        if not isinstance(params, dict):
            return False

        if "name" not in params:
            return False

        if not isinstance(params.get("arguments", {}), dict):
            return False

        return True

    def _extract_json_objects(self, text: str) -> list[str]:
        """Extract potential JSON objects from text using brace matching."""
        objects = []
        depth = 0
        start = -1

        for i, char in enumerate(text):
            if char == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0 and start != -1:
                    obj_text = text[start : i + 1]
                    # Only include if it looks like MCP
                    if "jsonrpc" in obj_text and "tools/call" in obj_text:
                        objects.append(obj_text)
                    start = -1

        return objects

    def validate_call(self, call: ToolCall, tools: list[Tool]) -> tuple[bool, str]:
        """Validate tool call with additional MCP-specific checks."""
        # First do base validation
        is_valid, error = super().validate_call(call, tools)
        if not is_valid:
            return is_valid, error

        # MCP-specific validation is already done in parsing
        return True, ""
