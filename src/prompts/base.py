"""Base classes for evaluation prompts."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExpectedToolCall:
    """Expected tool call for evaluation."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    partial_match: bool = False  # If True, only check specified args

    def matches(self, actual_name: str, actual_args: dict[str, Any]) -> bool:
        """Check if actual call matches expected."""
        if self.name != actual_name:
            return False

        if self.partial_match:
            # Only check that expected args are present with correct values
            for key, value in self.arguments.items():
                if key not in actual_args:
                    return False
                if actual_args[key] != value:
                    return False
            return True
        else:
            # Exact match required
            return self.arguments == actual_args


@dataclass
class EvalPrompt:
    """An evaluation prompt with expected outputs."""

    id: str
    prompt: str
    expected_calls: list[ExpectedToolCall]
    category: str  # 'simple', 'multi_tool', 'ambiguous', 'complex'
    description: str = ""
    requires_clarification: bool = False  # True if prompt is intentionally ambiguous

    @property
    def expected_tool_names(self) -> list[str]:
        """Get list of expected tool names."""
        return [call.name for call in self.expected_calls]

    @property
    def num_expected_calls(self) -> int:
        """Get number of expected tool calls."""
        return len(self.expected_calls)
