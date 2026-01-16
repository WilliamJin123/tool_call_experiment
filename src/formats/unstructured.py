"""Unstructured/natural language format for tool calls."""

import re
from typing import Any

from src.formats.base import BaseFormat, ToolCall
from src.tools.base import Tool


class UnstructuredFormat(BaseFormat):
    """
    Unstructured format handler using natural language patterns.

    Expected format:
    Call the tool "tool_name" with the following parameters:
    - param1: value1
    - param2: value2

    Or variations like:
    I'll use tool_name with param1=value1, param2=value2
    """

    @property
    def name(self) -> str:
        return "unstructured"

    def generate_system_prompt(self, tools: list[Tool]) -> str:
        tools_text = "\n\n".join(self.format_tool_for_prompt(t) for t in tools)

        return f"""You are a helpful assistant with access to the following tools:

{tools_text}

When you need to use a tool, describe your tool call in a structured natural language format:

TOOL_CALL: tool_name
PARAMETERS:
- param1: value1
- param2: value2
END_TOOL_CALL

IMPORTANT:
- Start tool calls with "TOOL_CALL:" followed by the tool name
- List parameters with "PARAMETERS:" followed by "- name: value" lines
- End with "END_TOOL_CALL"
- For list values, use comma-separated values or JSON array notation
- For dict/object values, use JSON object notation
- For boolean values, use "true" or "false"

You may call multiple tools by including multiple TOOL_CALL blocks.
Only use this format when you want to call a tool. For regular responses, just reply normally."""

    def parse_response(self, response: str) -> list[ToolCall]:
        """Parse natural language tool calls from response."""
        tool_calls = []

        # Primary pattern: TOOL_CALL: ... END_TOOL_CALL blocks
        block_pattern = r"TOOL_CALL:\s*(\w+)\s*PARAMETERS:\s*(.*?)\s*END_TOOL_CALL"
        matches = re.findall(block_pattern, response, re.DOTALL | re.IGNORECASE)

        for tool_name, params_text in matches:
            arguments = self._parse_parameter_list(params_text)
            tool_calls.append(
                ToolCall(
                    name=tool_name.strip(),
                    arguments=arguments,
                    raw_text=f"TOOL_CALL: {tool_name}\nPARAMETERS:\n{params_text}\nEND_TOOL_CALL",
                )
            )

        # Fallback patterns if no primary matches
        if not tool_calls:
            tool_calls = self._parse_fallback_patterns(response)

        return tool_calls

    def _parse_parameter_list(self, params_text: str) -> dict[str, Any]:
        """Parse parameters from a list format, including multi-line values."""
        arguments = {}

        # Pattern: - param_name: value (value can span multiple lines)
        # Use [\s\S] to match any character including newlines
        # Look ahead for next parameter line, empty line, or end of string
        param_pattern = r"-\s*(\w+)\s*:\s*([\s\S]+?)(?=\n\s*-\s*\w+\s*:|\n\n|\Z)"
        matches = re.findall(param_pattern, params_text)

        for param_name, value in matches:
            arguments[param_name.strip()] = self._parse_value(value.strip())

        return arguments

    def _parse_value(self, value_str: str) -> Any:
        """Parse a string value into appropriate Python type."""
        value_str = value_str.strip()

        # Remove surrounding quotes if present
        if (value_str.startswith('"') and value_str.endswith('"')) or (
            value_str.startswith("'") and value_str.endswith("'")
        ):
            return value_str[1:-1]

        # Try JSON for arrays and objects
        if value_str.startswith("[") or value_str.startswith("{"):
            try:
                import json

                return json.loads(value_str)
            except Exception:
                pass

        # Try boolean
        if value_str.lower() == "true":
            return True
        if value_str.lower() == "false":
            return False
        if value_str.lower() in ("null", "none"):
            return None

        # Try integer
        try:
            return int(value_str)
        except ValueError:
            pass

        # Try float
        try:
            return float(value_str)
        except ValueError:
            pass

        # Return as string
        return value_str

    def _parse_fallback_patterns(self, response: str) -> list[ToolCall]:
        """Try alternative patterns for tool calls."""
        tool_calls = []

        # Pattern: "Call tool_name with ..."
        call_pattern = r'(?:call|use|invoke|execute)\s+(?:the\s+)?(?:tool\s+)?["\']?(\w+)["\']?\s+(?:with|using)?\s*(?:the\s+following\s+)?(?:parameters?)?[:\s]*(.+?)(?=\n\n|\Z|(?:call|use|invoke))'
        matches = re.findall(call_pattern, response, re.DOTALL | re.IGNORECASE)

        for tool_name, params_text in matches:
            arguments = self._parse_inline_params(params_text)
            if arguments or not params_text.strip():
                tool_calls.append(
                    ToolCall(
                        name=tool_name.strip(),
                        arguments=arguments,
                        raw_text=f"call {tool_name} with {params_text}",
                    )
                )

        # Pattern: "tool_name(param1=value1, param2=value2)"
        if not tool_calls:
            func_pattern = r"(\w+)\s*\(\s*([^)]*)\s*\)"
            matches = re.findall(func_pattern, response)

            for tool_name, params_text in matches:
                arguments = self._parse_function_params(params_text)
                tool_calls.append(
                    ToolCall(
                        name=tool_name.strip(),
                        arguments=arguments,
                        raw_text=f"{tool_name}({params_text})",
                    )
                )

        return tool_calls

    def _parse_inline_params(self, params_text: str) -> dict[str, Any]:
        """Parse inline parameter format (param1=value1, param2=value2)."""
        arguments = {}

        # Try comma-separated key=value pairs
        param_pattern = r"(\w+)\s*[=:]\s*([^,\n]+)"
        matches = re.findall(param_pattern, params_text)

        for param_name, value in matches:
            arguments[param_name.strip()] = self._parse_value(value.strip())

        # Also try bullet point format
        if not arguments:
            arguments = self._parse_parameter_list(params_text)

        return arguments

    def _parse_function_params(self, params_text: str) -> dict[str, Any]:
        """Parse function-style parameters (param1=value1, param2=value2)."""
        arguments = {}

        if not params_text.strip():
            return arguments

        # Split by comma, but be careful with nested structures
        parts = self._smart_split(params_text, ",")

        for part in parts:
            if "=" in part:
                key, value = part.split("=", 1)
                arguments[key.strip()] = self._parse_value(value.strip())

        return arguments

    def _smart_split(self, text: str, delimiter: str) -> list[str]:
        """Split text by delimiter, respecting nested brackets and quotes."""
        parts = []
        current = []
        depth = 0
        in_string = False
        string_char = None

        for char in text:
            if char in "\"'" and not in_string:
                in_string = True
                string_char = char
            elif char == string_char and in_string:
                in_string = False
                string_char = None
            elif char in "([{" and not in_string:
                depth += 1
            elif char in ")]}" and not in_string:
                depth -= 1
            elif char == delimiter and depth == 0 and not in_string:
                parts.append("".join(current))
                current = []
                continue

            current.append(char)

        if current:
            parts.append("".join(current))

        return parts
