"""Tests for tool definitions."""

import pytest

from src.tools import (
    Tool,
    Parameter,
    get_all_tools,
    get_tools_by_complexity,
    SIMPLE_TOOLS,
    MEDIUM_TOOLS,
    COMPLEX_TOOLS,
    EDGE_CASE_TOOLS,
)


class TestParameter:
    """Tests for Parameter dataclass."""

    def test_required_parameter(self):
        """Test creating a required parameter."""
        param = Parameter(name="city", type="str", required=True)
        assert param.name == "city"
        assert param.type == "str"
        assert param.required is True
        assert param.default is None

    def test_optional_parameter_with_default(self):
        """Test that providing default makes parameter optional."""
        param = Parameter(name="limit", type="int", default=10)
        assert param.required is False
        assert param.default == 10

    def test_optional_parameter_explicit(self):
        """Test explicitly optional parameter."""
        param = Parameter(name="name", type="str", required=False)
        assert param.required is False


class TestTool:
    """Tests for Tool dataclass."""

    def test_create_simple_tool(self):
        """Test creating a simple tool."""
        tool = Tool(
            name="test_tool",
            description="A test tool",
            parameters=[
                Parameter(name="input", type="str", required=True),
            ],
        )
        assert tool.name == "test_tool"
        assert len(tool.parameters) == 1

    def test_get_required_params(self):
        """Test getting required parameters."""
        tool = Tool(
            name="test",
            description="Test",
            parameters=[
                Parameter(name="required_param", type="str", required=True),
                Parameter(name="optional_param", type="str", default="default"),
            ],
        )
        required = tool.get_required_params()
        assert len(required) == 1
        assert required[0].name == "required_param"

    def test_get_optional_params(self):
        """Test getting optional parameters."""
        tool = Tool(
            name="test",
            description="Test",
            parameters=[
                Parameter(name="required_param", type="str", required=True),
                Parameter(name="optional_param", type="str", default="default"),
            ],
        )
        optional = tool.get_optional_params()
        assert len(optional) == 1
        assert optional[0].name == "optional_param"

    def test_to_json_schema(self):
        """Test converting tool to JSON schema."""
        tool = Tool(
            name="get_weather",
            description="Get weather for a city",
            parameters=[
                Parameter(name="city", type="str", description="City name", required=True),
                Parameter(name="units", type="str", default="celsius"),
            ],
        )
        schema = tool.to_json_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "get_weather"
        assert "city" in schema["function"]["parameters"]["properties"]
        assert "city" in schema["function"]["parameters"]["required"]
        assert "units" not in schema["function"]["parameters"]["required"]


class TestToolCollections:
    """Tests for tool collections."""

    def test_simple_tools_exist(self):
        """Test that simple tools are defined."""
        assert len(SIMPLE_TOOLS) > 0
        tool_names = [t.name for t in SIMPLE_TOOLS]
        assert "get_weather" in tool_names
        assert "get_time" in tool_names

    def test_medium_tools_exist(self):
        """Test that medium tools are defined."""
        assert len(MEDIUM_TOOLS) > 0
        tool_names = [t.name for t in MEDIUM_TOOLS]
        assert "search_files" in tool_names
        assert "send_email" in tool_names

    def test_complex_tools_exist(self):
        """Test that complex tools are defined."""
        assert len(COMPLEX_TOOLS) > 0
        tool_names = [t.name for t in COMPLEX_TOOLS]
        assert "create_chart" in tool_names
        assert "batch_process" in tool_names

    def test_edge_case_tools_exist(self):
        """Test that edge case tools are defined."""
        assert len(EDGE_CASE_TOOLS) > 0
        tool_names = [t.name for t in EDGE_CASE_TOOLS]
        assert "weird_params" in tool_names
        assert "unicode_tool" in tool_names
        assert "empty_tool" in tool_names

    def test_get_all_tools(self):
        """Test getting all tools."""
        all_tools = get_all_tools()
        expected_count = len(SIMPLE_TOOLS) + len(MEDIUM_TOOLS) + len(COMPLEX_TOOLS) + len(EDGE_CASE_TOOLS)
        assert len(all_tools) == expected_count

    def test_get_tools_by_complexity(self):
        """Test getting tools by complexity level."""
        simple = get_tools_by_complexity("simple")
        assert simple == SIMPLE_TOOLS

        medium = get_tools_by_complexity("medium")
        assert medium == MEDIUM_TOOLS

        complex_tools = get_tools_by_complexity("complex")
        assert complex_tools == COMPLEX_TOOLS

        edge = get_tools_by_complexity("edge_cases")
        assert edge == EDGE_CASE_TOOLS

    def test_get_tools_invalid_complexity(self):
        """Test that invalid complexity raises error."""
        with pytest.raises(ValueError):
            get_tools_by_complexity("invalid")


class TestToolDefinitions:
    """Tests for specific tool definitions."""

    def test_get_weather_tool(self):
        """Test get_weather tool definition."""
        tool = next(t for t in SIMPLE_TOOLS if t.name == "get_weather")
        assert len(tool.get_required_params()) == 1
        assert tool.get_required_params()[0].name == "city"

    def test_send_email_tool(self):
        """Test send_email tool definition."""
        tool = next(t for t in MEDIUM_TOOLS if t.name == "send_email")
        required = tool.get_required_params()
        optional = tool.get_optional_params()

        required_names = [p.name for p in required]
        assert "to" in required_names
        assert "subject" in required_names
        assert "body" in required_names

        optional_names = [p.name for p in optional]
        assert "cc" in optional_names

    def test_empty_tool(self):
        """Test empty_tool has no parameters."""
        tool = next(t for t in EDGE_CASE_TOOLS if t.name == "empty_tool")
        assert len(tool.parameters) == 0

    def test_weird_params_tool(self):
        """Test weird_params tool has unusual parameter names."""
        tool = next(t for t in EDGE_CASE_TOOLS if t.name == "weird_params")
        param_names = [p.name for p in tool.parameters]

        assert "kebab-case-param" in param_names
        assert "$special" in param_names
        assert "_underscore" in param_names

    def test_unicode_tool(self):
        """Test unicode_tool has unicode parameter names."""
        tool = next(t for t in EDGE_CASE_TOOLS if t.name == "unicode_tool")
        param_names = [p.name for p in tool.parameters]

        assert "日本語" in param_names
        assert "émoji" in param_names
