"""Tool definitions for evaluation."""

from src.tools.base import Tool, Parameter
from src.tools.simple import SIMPLE_TOOLS
from src.tools.medium import MEDIUM_TOOLS
from src.tools.complex import COMPLEX_TOOLS
from src.tools.edge_cases import EDGE_CASE_TOOLS


def get_all_tools() -> list[Tool]:
    """Get all tool definitions."""
    return SIMPLE_TOOLS + MEDIUM_TOOLS + COMPLEX_TOOLS + EDGE_CASE_TOOLS


def get_tools_by_complexity(complexity: str) -> list[Tool]:
    """Get tools by complexity level."""
    match complexity:
        case "simple":
            return SIMPLE_TOOLS
        case "medium":
            return MEDIUM_TOOLS
        case "complex":
            return COMPLEX_TOOLS
        case "edge_cases":
            return EDGE_CASE_TOOLS
        case _:
            raise ValueError(f"Unknown complexity: {complexity}")


__all__ = [
    "Tool",
    "Parameter",
    "get_all_tools",
    "get_tools_by_complexity",
    "SIMPLE_TOOLS",
    "MEDIUM_TOOLS",
    "COMPLEX_TOOLS",
    "EDGE_CASE_TOOLS",
]
