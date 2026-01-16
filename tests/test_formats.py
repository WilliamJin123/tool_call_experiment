"""Tests for format implementations."""

import pytest

from src.formats import JSONFormat, MCPFormat, XMLFormat, UnstructuredFormat, FunctionSigFormat
from src.formats.base import ToolCall
from src.tools import get_all_tools


class TestJSONFormat:
    """Tests for JSON format parsing."""

    def setup_method(self):
        self.format = JSONFormat()
        self.tools = get_all_tools()

    def test_parse_basic_json(self):
        """Test parsing a basic JSON tool call."""
        response = '''Here's the weather:
```json
{
  "tool": "get_weather",
  "parameters": {
    "city": "Tokyo"
  }
}
```'''
        calls = self.format.parse_response(response)
        assert len(calls) == 1
        assert calls[0].name == "get_weather"
        assert calls[0].arguments == {"city": "Tokyo"}

    def test_parse_without_code_block(self):
        """Test parsing JSON without code fences."""
        response = '''I'll get the weather: {"tool": "get_weather", "parameters": {"city": "Paris"}}'''
        calls = self.format.parse_response(response)
        assert len(calls) == 1
        assert calls[0].name == "get_weather"

    def test_parse_multiple_calls(self):
        """Test parsing multiple tool calls."""
        response = '''
```json
{"tool": "get_weather", "parameters": {"city": "Tokyo"}}
```
And also:
```json
{"tool": "get_weather", "parameters": {"city": "Paris"}}
```'''
        calls = self.format.parse_response(response)
        assert len(calls) == 2

    def test_validate_valid_call(self):
        """Test validating a correct tool call."""
        call = ToolCall(name="get_weather", arguments={"city": "Tokyo"})
        is_valid, error = self.format.validate_call(call, self.tools)
        assert is_valid
        assert error == ""

    def test_validate_missing_required_param(self):
        """Test validating a call with missing required parameter."""
        call = ToolCall(name="get_weather", arguments={})
        is_valid, error = self.format.validate_call(call, self.tools)
        assert not is_valid
        assert "Missing required parameter" in error

    def test_validate_unknown_tool(self):
        """Test validating a call with unknown tool."""
        call = ToolCall(name="unknown_tool", arguments={})
        is_valid, error = self.format.validate_call(call, self.tools)
        assert not is_valid
        assert "Unknown tool" in error


class TestMCPFormat:
    """Tests for MCP format parsing."""

    def setup_method(self):
        self.format = MCPFormat()
        self.tools = get_all_tools()

    def test_parse_valid_mcp(self):
        """Test parsing valid MCP format."""
        response = '''
```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "get_weather",
    "arguments": {
      "city": "London"
    }
  }
}
```'''
        calls = self.format.parse_response(response)
        assert len(calls) == 1
        assert calls[0].name == "get_weather"
        assert calls[0].arguments == {"city": "London"}

    def test_reject_invalid_jsonrpc(self):
        """Test that invalid jsonrpc version is rejected."""
        response = '''{"jsonrpc": "1.0", "method": "tools/call", "params": {"name": "get_weather", "arguments": {}}}'''
        calls = self.format.parse_response(response)
        assert len(calls) == 0

    def test_reject_invalid_method(self):
        """Test that invalid method is rejected."""
        response = '''{"jsonrpc": "2.0", "method": "wrong/method", "params": {"name": "get_weather", "arguments": {}}}'''
        calls = self.format.parse_response(response)
        assert len(calls) == 0


class TestXMLFormat:
    """Tests for XML format parsing."""

    def setup_method(self):
        self.format = XMLFormat()
        self.tools = get_all_tools()

    def test_parse_basic_xml(self):
        """Test parsing basic XML tool call."""
        response = '''
<tool_call>
  <name>get_weather</name>
  <parameters>
    <city>Berlin</city>
  </parameters>
</tool_call>'''
        calls = self.format.parse_response(response)
        assert len(calls) == 1
        assert calls[0].name == "get_weather"
        assert calls[0].arguments == {"city": "Berlin"}

    def test_parse_xml_in_code_block(self):
        """Test parsing XML in code block."""
        response = '''```xml
<tool_call>
  <name>translate</name>
  <parameters>
    <text>Hello</text>
    <target_language>es</target_language>
  </parameters>
</tool_call>
```'''
        calls = self.format.parse_response(response)
        assert len(calls) == 1
        assert calls[0].name == "translate"

    def test_parse_numeric_values(self):
        """Test parsing numeric values in XML."""
        response = '''
<tool_call>
  <name>search_files</name>
  <parameters>
    <query>import</query>
    <file_type>py</file_type>
    <max_results>5</max_results>
  </parameters>
</tool_call>'''
        calls = self.format.parse_response(response)
        assert len(calls) == 1
        assert calls[0].arguments["max_results"] == 5


class TestUnstructuredFormat:
    """Tests for unstructured format parsing."""

    def setup_method(self):
        self.format = UnstructuredFormat()
        self.tools = get_all_tools()

    def test_parse_structured_format(self):
        """Test parsing the structured TOOL_CALL format."""
        response = '''I'll help you with that.

TOOL_CALL: get_weather
PARAMETERS:
- city: Tokyo
END_TOOL_CALL'''
        calls = self.format.parse_response(response)
        assert len(calls) == 1
        assert calls[0].name == "get_weather"
        assert calls[0].arguments == {"city": "Tokyo"}

    def test_parse_multiple_params(self):
        """Test parsing multiple parameters."""
        response = '''
TOOL_CALL: send_email
PARAMETERS:
- to: test@example.com
- subject: Hello
- body: This is a test
END_TOOL_CALL'''
        calls = self.format.parse_response(response)
        assert len(calls) == 1
        assert calls[0].arguments["to"] == "test@example.com"
        assert calls[0].arguments["subject"] == "Hello"


class TestFunctionSigFormat:
    """Tests for function signature format parsing."""

    def setup_method(self):
        self.format = FunctionSigFormat()
        self.tools = get_all_tools()

    def test_parse_basic_function(self):
        """Test parsing basic function call."""
        response = '''I'll get the weather:
get_weather(city="Tokyo")'''
        calls = self.format.parse_response(response)
        assert len(calls) == 1
        assert calls[0].name == "get_weather"
        assert calls[0].arguments == {"city": "Tokyo"}

    def test_parse_multiple_args(self):
        """Test parsing function with multiple arguments."""
        response = 'translate(text="Hello", target_language="es")'
        calls = self.format.parse_response(response)
        assert len(calls) == 1
        assert calls[0].arguments["text"] == "Hello"
        assert calls[0].arguments["target_language"] == "es"

    def test_parse_in_code_block(self):
        """Test parsing function in code block."""
        response = '''```python
get_weather(city="Paris")
```'''
        calls = self.format.parse_response(response)
        assert len(calls) == 1

    def test_parse_numeric_args(self):
        """Test parsing numeric arguments."""
        response = 'search_files(query="test", file_type="py", max_results=20)'
        calls = self.format.parse_response(response)
        assert len(calls) == 1
        assert calls[0].arguments["max_results"] == 20

    def test_parse_boolean_args(self):
        """Test parsing boolean arguments."""
        response = 'weird_params(kebab_case_param="test", _underscore=True)'
        calls = self.format.parse_response(response)
        assert len(calls) >= 1


class TestSystemPromptGeneration:
    """Tests for system prompt generation."""

    def setup_method(self):
        self.tools = get_all_tools()

    def test_json_prompt_contains_tools(self):
        """Test that JSON prompt contains tool definitions."""
        format_handler = JSONFormat()
        prompt = format_handler.generate_system_prompt(self.tools)

        assert "get_weather" in prompt
        assert "city" in prompt
        assert "JSON" in prompt or "json" in prompt

    def test_mcp_prompt_contains_jsonrpc(self):
        """Test that MCP prompt mentions JSON-RPC."""
        format_handler = MCPFormat()
        prompt = format_handler.generate_system_prompt(self.tools)

        assert "jsonrpc" in prompt.lower()
        assert "2.0" in prompt
        assert "tools/call" in prompt

    def test_xml_prompt_contains_tags(self):
        """Test that XML prompt mentions XML tags."""
        format_handler = XMLFormat()
        prompt = format_handler.generate_system_prompt(self.tools)

        assert "<tool_call>" in prompt
        assert "<name>" in prompt
        assert "<parameters>" in prompt

    def test_function_sig_prompt_contains_def(self):
        """Test that function sig prompt shows function definitions."""
        format_handler = FunctionSigFormat()
        prompt = format_handler.generate_system_prompt(self.tools)

        assert "def " in prompt
        assert "get_weather" in prompt
