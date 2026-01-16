"""Evaluation prompts with expected outputs."""

from src.prompts.base import EvalPrompt, ExpectedToolCall
from src.prompts.simple import SIMPLE_PROMPTS
from src.prompts.multi_tool import MULTI_TOOL_PROMPTS
from src.prompts.ambiguous import AMBIGUOUS_PROMPTS
from src.prompts.complex import COMPLEX_PROMPTS


def get_all_prompts() -> list[EvalPrompt]:
    """Get all evaluation prompts."""
    return SIMPLE_PROMPTS + MULTI_TOOL_PROMPTS + AMBIGUOUS_PROMPTS + COMPLEX_PROMPTS


def get_prompts_by_type(prompt_type: str) -> list[EvalPrompt]:
    """Get prompts by type."""
    match prompt_type:
        case "simple":
            return SIMPLE_PROMPTS
        case "multi_tool":
            return MULTI_TOOL_PROMPTS
        case "ambiguous":
            return AMBIGUOUS_PROMPTS
        case "complex":
            return COMPLEX_PROMPTS
        case _:
            raise ValueError(f"Unknown prompt type: {prompt_type}")


__all__ = [
    "EvalPrompt",
    "ExpectedToolCall",
    "get_all_prompts",
    "get_prompts_by_type",
    "SIMPLE_PROMPTS",
    "MULTI_TOOL_PROMPTS",
    "AMBIGUOUS_PROMPTS",
    "COMPLEX_PROMPTS",
]
