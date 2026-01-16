"""Response parsing utilities."""

import re
from typing import Any

from src.formats.base import ToolCall


class ResponseParser:
    """Utilities for parsing model responses."""

    @staticmethod
    def extract_code_blocks(text: str, language: str | None = None) -> list[str]:
        """Extract code blocks from markdown-formatted text."""
        if language:
            pattern = rf"```{language}\s*(.*?)\s*```"
        else:
            pattern = r"```(?:\w+)?\s*(.*?)\s*```"

        return re.findall(pattern, text, re.DOTALL)

    @staticmethod
    def extract_json_objects(text: str) -> list[str]:
        """Extract potential JSON objects from text using brace matching."""
        objects = []
        depth = 0
        start = -1

        for i, char in enumerate(text):
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

    @staticmethod
    def extract_xml_tags(text: str, tag_name: str) -> list[str]:
        """Extract content of specific XML tags."""
        pattern = rf"<{tag_name}>(.*?)</{tag_name}>"
        return re.findall(pattern, text, re.DOTALL | re.IGNORECASE)

    @staticmethod
    def normalize_tool_name(name: str) -> str:
        """Normalize a tool name for comparison."""
        # Remove common prefixes/suffixes
        name = name.strip()
        name = re.sub(r"^(call_|invoke_|use_|run_)", "", name, flags=re.IGNORECASE)
        name = re.sub(r"(_tool|_function)$", "", name, flags=re.IGNORECASE)
        return name.lower()

    @staticmethod
    def normalize_arguments(args: dict[str, Any]) -> dict[str, Any]:
        """Normalize argument names and values for comparison."""
        normalized = {}

        for key, value in args.items():
            # Normalize key (lowercase, remove common variations)
            norm_key = key.lower().replace("-", "_").replace(" ", "_")

            # Normalize value
            if isinstance(value, str):
                # Strip whitespace
                value = value.strip()
            elif isinstance(value, dict):
                value = ResponseParser.normalize_arguments(value)
            elif isinstance(value, list):
                value = [
                    ResponseParser.normalize_arguments(v) if isinstance(v, dict) else v
                    for v in value
                ]

            normalized[norm_key] = value

        return normalized

    @staticmethod
    def compare_tool_calls(
        expected: ToolCall,
        actual: ToolCall,
        partial_match: bool = False,
    ) -> tuple[bool, str]:
        """
        Compare two tool calls for equality.

        Returns (is_match, reason) tuple.
        """
        # Compare names (normalized)
        expected_name = ResponseParser.normalize_tool_name(expected.name)
        actual_name = ResponseParser.normalize_tool_name(actual.name)

        if expected_name != actual_name:
            return False, f"Tool name mismatch: expected '{expected.name}', got '{actual.name}'"

        # Normalize arguments
        expected_args = ResponseParser.normalize_arguments(expected.arguments)
        actual_args = ResponseParser.normalize_arguments(actual.arguments)

        if partial_match:
            # Only check that expected args are present
            for key, expected_value in expected_args.items():
                if key not in actual_args:
                    return False, f"Missing expected argument: {key}"
                if actual_args[key] != expected_value:
                    # Check for approximate equality for strings
                    if isinstance(expected_value, str) and isinstance(actual_args[key], str):
                        if expected_value.lower() not in actual_args[key].lower():
                            return False, f"Argument '{key}' mismatch: expected '{expected_value}', got '{actual_args[key]}'"
                    else:
                        return False, f"Argument '{key}' mismatch: expected '{expected_value}', got '{actual_args[key]}'"
        else:
            # Exact match required
            if expected_args != actual_args:
                return False, f"Arguments mismatch: expected {expected_args}, got {actual_args}"

        return True, ""

    @staticmethod
    def find_best_match(
        expected: ToolCall,
        actuals: list[ToolCall],
        partial_match: bool = False,
    ) -> tuple[ToolCall | None, float]:
        """
        Find the best matching tool call from a list.

        Returns (best_match, score) tuple where score is 0-1.
        """
        if not actuals:
            return None, 0.0

        best_match = None
        best_score = 0.0

        for actual in actuals:
            is_match, _ = ResponseParser.compare_tool_calls(expected, actual, partial_match)

            if is_match:
                return actual, 1.0

            # Calculate partial score
            score = 0.0

            # Name match
            expected_name = ResponseParser.normalize_tool_name(expected.name)
            actual_name = ResponseParser.normalize_tool_name(actual.name)
            if expected_name == actual_name:
                score += 0.5

            # Argument overlap
            expected_args = set(ResponseParser.normalize_arguments(expected.arguments).keys())
            actual_args = set(ResponseParser.normalize_arguments(actual.arguments).keys())

            if expected_args:
                overlap = len(expected_args & actual_args) / len(expected_args)
                score += 0.5 * overlap

            if score > best_score:
                best_score = score
                best_match = actual

        return best_match, best_score
