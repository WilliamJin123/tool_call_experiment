"""Base class for tool call formats."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from src.tools.base import Tool


@dataclass
class ToolCall:
    """Represents a parsed tool call from model output."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""  # Original text that was parsed

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ToolCall):
            return False
        return self.name == other.name and self.arguments == other.arguments


class BaseFormat(ABC):
    """Abstract base class for tool call format handlers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this format (e.g., 'json', 'mcp', 'xml')."""
        pass

    @abstractmethod
    def generate_system_prompt(self, tools: list[Tool]) -> str:
        """
        Generate a system prompt that instructs the model how to call tools.

        Args:
            tools: List of tool definitions to include in the prompt.

        Returns:
            System prompt string with tool definitions and format instructions.
        """
        pass

    @abstractmethod
    def parse_response(self, response: str) -> list[ToolCall]:
        """
        Parse model response to extract tool calls.

        Args:
            response: Raw model output text.

        Returns:
            List of parsed ToolCall objects. Empty list if no tool calls found.
        """
        pass

    def validate_call(self, call: ToolCall, tools: list[Tool]) -> tuple[bool, str]:
        """
        Validate a tool call against available tool definitions.

        Args:
            call: The parsed tool call to validate.
            tools: List of available tool definitions.

        Returns:
            Tuple of (is_valid, error_message). Error message is empty if valid.
        """
        # Find the tool definition
        tool = None
        for t in tools:
            if t.name == call.name:
                tool = t
                break

        if tool is None:
            return False, f"Unknown tool: {call.name}"

        # Check required parameters
        for param in tool.parameters:
            if param.required and param.name not in call.arguments:
                return False, f"Missing required parameter: {param.name}"

        # Check for unknown parameters
        known_params = {p.name for p in tool.parameters}
        for arg_name in call.arguments:
            if arg_name not in known_params:
                return False, f"Unknown parameter: {arg_name}"

        # Type checking (basic)
        for param in tool.parameters:
            if param.name in call.arguments:
                value = call.arguments[param.name]
                if not self._check_type(value, param.type):
                    return False, f"Parameter {param.name}: expected {param.type}, got {type(value).__name__}"

        return True, ""

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Basic type checking for parameter values."""
        type_mapping = {
            "str": str,
            "string": str,
            "int": int,
            "integer": int,
            "float": (int, float),
            "number": (int, float),
            "bool": bool,
            "boolean": bool,
            "list": list,
            "array": list,
            "dict": dict,
            "object": dict,
        }

        # Handle optional types like "list[str]" or "dict[str, Any]"
        base_type = expected_type.split("[")[0].lower()

        if base_type not in type_mapping:
            # Unknown type, accept anything
            return True

        expected = type_mapping[base_type]
        return isinstance(value, expected)

    def format_tool_for_prompt(self, tool: Tool) -> str:
        """
        Format a single tool definition for inclusion in the system prompt.
        Subclasses may override this to customize per-tool formatting.

        Args:
            tool: Tool definition to format.

        Returns:
            Formatted tool description string.
        """
        params = []
        for p in tool.parameters:
            param_str = f"{p.name}: {p.type}"
            if p.default is not None:
                param_str += f" = {repr(p.default)}"
            if p.required:
                param_str += " (required)"
            if p.description:
                param_str += f" - {p.description}"
            params.append(param_str)

        params_text = "\n    ".join(params) if params else "None"

        return f"""Tool: {tool.name}
  Description: {tool.description}
  Parameters:
    {params_text}"""
