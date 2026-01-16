"""XML format for tool calls."""

import json
import re
import xml.etree.ElementTree as ET
from typing import Any

from src.formats.base import BaseFormat, ToolCall
from src.tools.base import Tool


class XMLFormat(BaseFormat):
    """
    XML format handler.

    Expected format:
    <tool_call>
      <name>tool_name</name>
      <parameters>
        <param1>value1</param1>
        <param2>value2</param2>
      </parameters>
    </tool_call>
    """

    @property
    def name(self) -> str:
        return "xml"

    def generate_system_prompt(self, tools: list[Tool]) -> str:
        tools_text = "\n\n".join(self.format_tool_for_prompt(t) for t in tools)

        return f"""You are a helpful assistant with access to the following tools:

{tools_text}

When you need to use a tool, respond with an XML structure in this exact format:
```xml
<tool_call>
  <name>tool_name</name>
  <parameters>
    <param1>value1</param1>
    <param2>value2</param2>
  </parameters>
</tool_call>
```

IMPORTANT:
- Use exact tag names: <tool_call>, <name>, <parameters>
- Parameter tags should match the exact parameter names from tool definitions
- For list values, use multiple tags with the same name or JSON array syntax
- For dict/object values, use nested tags or JSON object syntax
- For boolean values, use "true" or "false" (lowercase)

You may call multiple tools by providing multiple <tool_call> blocks.
Only include XML when you want to call a tool. For regular responses, just reply normally."""

    def parse_response(self, response: str) -> list[ToolCall]:
        """Parse XML tool calls from response."""
        tool_calls = []

        # Find all tool_call blocks
        pattern = r"<tool_call>(.*?)</tool_call>"
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)

        for match in matches:
            try:
                # Wrap in root for parsing
                xml_text = f"<tool_call>{match}</tool_call>"
                tool_call = self._parse_xml_tool_call(xml_text)
                if tool_call:
                    tool_calls.append(tool_call)
            except Exception:
                continue

        # Also try to find tool calls in code blocks
        code_block_pattern = r"```(?:xml)?\s*(<tool_call>.*?</tool_call>)\s*```"
        code_matches = re.findall(code_block_pattern, response, re.DOTALL | re.IGNORECASE)

        for match in code_matches:
            try:
                tool_call = self._parse_xml_tool_call(match)
                if tool_call and tool_call not in tool_calls:
                    tool_calls.append(tool_call)
            except Exception:
                continue

        return tool_calls

    def _parse_xml_tool_call(self, xml_text: str) -> ToolCall | None:
        """Parse a single XML tool call."""
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            # Try to fix common XML issues
            fixed = self._fix_xml(xml_text)
            try:
                root = ET.fromstring(fixed)
            except ET.ParseError:
                return None

        # Extract tool name
        name_elem = root.find("name")
        if name_elem is None or not name_elem.text:
            return None

        tool_name = name_elem.text.strip()

        # Extract parameters
        params_elem = root.find("parameters")
        arguments = {}

        if params_elem is not None:
            arguments = self._parse_parameters(params_elem)

        return ToolCall(
            name=tool_name,
            arguments=arguments,
            raw_text=xml_text,
        )

    def _parse_parameters(self, params_elem: ET.Element) -> dict[str, Any]:
        """Parse parameter elements into a dictionary."""
        arguments = {}

        for child in params_elem:
            param_name = child.tag
            value = self._parse_value(child)
            arguments[param_name] = value

        return arguments

    def _parse_value(self, elem: ET.Element) -> Any:
        """Parse an element's value, handling various types."""
        # Check if it has children (nested structure)
        if len(elem) > 0:
            # Check if all children have the same tag (array)
            tags = [child.tag for child in elem]
            if len(set(tags)) == 1:
                # Array of same-tagged elements
                return [self._parse_value(child) for child in elem]
            else:
                # Object with different keys
                return {child.tag: self._parse_value(child) for child in elem}

        # Leaf node - parse the text value
        text = elem.text
        if text is None:
            return None

        text = text.strip()

        # Try to parse as JSON (for embedded arrays/objects)
        if text.startswith("[") or text.startswith("{"):
            try:
                return json.loads(text)
            except Exception:
                pass

        # Try boolean
        if text.lower() == "true":
            return True
        if text.lower() == "false":
            return False

        # Try integer
        try:
            return int(text)
        except ValueError:
            pass

        # Try float
        try:
            return float(text)
        except ValueError:
            pass

        # Return as string
        return text

    def _fix_xml(self, xml_text: str) -> str:
        """Try to fix common XML issues."""
        fixed = xml_text

        # Escape unescaped ampersands
        fixed = re.sub(r"&(?!(?:amp|lt|gt|quot|apos);)", "&amp;", fixed)

        # Fix unclosed tags (basic)
        # This is a simple heuristic and may not work for all cases

        return fixed
