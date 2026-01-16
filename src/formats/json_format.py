"""JSON format for tool calls - lenient parsing."""

import json
import re
from typing import Any

from src.formats.base import BaseFormat, ToolCall
from src.tools.base import Tool


class JSONFormat(BaseFormat):
    """
    JSON format handler with lenient parsing.

    Expected format:
    {
      "tool": "tool_name",
      "parameters": {
        "param1": "value1",
        "param2": "value2"
      }
    }
    """

    @property
    def name(self) -> str:
        return "json"

    def generate_system_prompt(self, tools: list[Tool]) -> str:
        tools_text = "\n\n".join(self.format_tool_for_prompt(t) for t in tools)

        return f"""You are a helpful assistant with access to the following tools:

{tools_text}

When you need to use a tool, respond with a JSON object in this exact format:
```json
{{
  "tool": "tool_name",
  "parameters": {{
    "param1": "value1",
    "param2": "value2"
  }}
}}
```

You may call multiple tools by providing multiple JSON objects.
Only include the JSON when you want to call a tool. For regular responses, just reply normally.
Always use the exact parameter names as specified in the tool definitions."""

    def parse_response(self, response: str) -> list[ToolCall]:
        """Parse JSON tool calls from response with lenient matching."""
        tool_calls = []

        # Try to find JSON blocks in code fences first
        code_block_pattern = r"```(?:json)?\s*(\{[^`]*\})\s*```"
        matches = re.findall(code_block_pattern, response, re.DOTALL)

        # Also try to find bare JSON objects
        if not matches:
            # Match JSON objects that look like tool calls
            bare_json_pattern = r'\{[^{}]*"tool"[^{}]*\}'
            matches = re.findall(bare_json_pattern, response, re.DOTALL)

        # If still no matches, try a more aggressive pattern for nested JSON
        if not matches:
            # Find all potential JSON objects
            potential_json = self._extract_json_objects(response)
            matches = potential_json

        for match in matches:
            try:
                parsed = self._lenient_json_parse(match)
                if isinstance(parsed, dict) and "tool" in parsed:
                    tool_call = ToolCall(
                        name=parsed["tool"],
                        arguments=parsed.get("parameters", parsed.get("arguments", {})),
                        raw_text=match,
                    )
                    tool_calls.append(tool_call)
            except (json.JSONDecodeError, ValueError):
                continue

        return tool_calls

    def _extract_json_objects(self, text: str) -> list[str]:
        """Extract potential JSON objects from text using brace matching.

        Properly handles braces inside quoted strings.
        """
        objects = []
        depth = 0
        start = -1
        in_string = False
        escape = False

        for i, char in enumerate(text):
            if escape:
                escape = False
                continue
            if char == "\\" and in_string:
                escape = True
                continue
            if char == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0 and start != -1:
                    objects.append(text[start : i + 1])
                    start = -1

        return objects

    def _lenient_json_parse(self, text: str) -> Any:
        """Parse JSON with some leniency for common model mistakes."""
        # First try standard parsing
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try fixing common issues
        fixed = text

        # Remove trailing commas before } or ]
        fixed = re.sub(r",\s*([}\]])", r"\1", fixed)

        # Try to fix single quotes (but be careful with apostrophes)
        # Only replace single quotes that look like JSON delimiters
        fixed = re.sub(r"'(\w+)':", r'"\1":', fixed)  # Keys
        fixed = re.sub(r":\s*'([^']*)'", r': "\1"', fixed)  # String values

        # Fix unquoted keys
        fixed = re.sub(r"(\{|,)\s*(\w+)\s*:", r'\1"\2":', fixed)

        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            raise ValueError(f"Could not parse JSON: {text}")
