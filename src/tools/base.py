"""Base classes for tool definitions."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Parameter:
    """Definition of a tool parameter."""

    name: str
    type: str  # e.g., "str", "int", "list[dict]", "dict[str, Any]"
    description: str = ""
    required: bool = True
    default: Any = None

    def __post_init__(self):
        # If default is provided, parameter is optional unless explicitly required
        if self.default is not None and self.required:
            self.required = False


@dataclass
class Tool:
    """Definition of a tool that can be called by the model."""

    name: str
    description: str
    parameters: list[Parameter] = field(default_factory=list)

    def get_required_params(self) -> list[Parameter]:
        """Get list of required parameters."""
        return [p for p in self.parameters if p.required]

    def get_optional_params(self) -> list[Parameter]:
        """Get list of optional parameters."""
        return [p for p in self.parameters if not p.required]

    def to_json_schema(self) -> dict[str, Any]:
        """Convert tool to JSON Schema format (OpenAI-style)."""
        properties = {}
        required = []

        for param in self.parameters:
            prop = {"description": param.description}

            # Map type strings to JSON Schema types
            type_mapping = {
                "str": "string",
                "string": "string",
                "int": "integer",
                "integer": "integer",
                "float": "number",
                "number": "number",
                "bool": "boolean",
                "boolean": "boolean",
                "list": "array",
                "array": "array",
                "dict": "object",
                "object": "object",
            }

            base_type = param.type.split("[")[0].lower()
            prop["type"] = type_mapping.get(base_type, "string")

            # Handle array item types
            if base_type in ("list", "array") and "[" in param.type:
                inner_type = param.type.split("[")[1].rstrip("]")
                inner_base = inner_type.split("[")[0].lower()
                prop["items"] = {"type": type_mapping.get(inner_base, "string")}

            if param.default is not None:
                prop["default"] = param.default

            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
