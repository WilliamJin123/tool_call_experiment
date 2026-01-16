#!/usr/bin/env python3
"""Verify critical bug fixes in the tool call format experiment codebase."""

from src.models.config import get_all_models
from src.evaluation.runner import EvaluationResult


def test_no_double_prefix():
    """Verify no double-prefix in model keys (Bug #1)."""
    print("Testing Bug #1: Groq model ID double-prefix...")
    for m in get_all_models():
        key = f"{m.provider}/{m.model_id}"
        assert key.count("/") <= 2, f"Double prefix in: {key}"
        # Ensure no provider appears twice
        assert not key.startswith(f"{m.provider}/{m.provider}/"), f"Double prefix in: {key}"
        print(f"  ✓ {key}")
    print("  PASSED\n")


def test_tool_accuracy_jaccard():
    """Verify tool accuracy uses Jaccard similarity (Bug #2)."""
    print("Testing Bug #2: Tool accuracy Jaccard similarity...")

    # Test case: extra tools should reduce accuracy
    result = EvaluationResult(
        model="test",
        format_type="test",
        prompt_id="test",
        response_text="",
        expected_tools=["tool_a"],
        actual_tools=["tool_a", "tool_b"],  # Extra tool
    )
    # Jaccard: intersection(1) / union(2) = 0.5
    assert result.tool_accuracy == 0.5, f"Expected 0.5, got {result.tool_accuracy}"
    print(f"  ✓ Extra tool case: {result.tool_accuracy}")

    # Test case: missing tools should reduce accuracy
    result2 = EvaluationResult(
        model="test",
        format_type="test",
        prompt_id="test",
        response_text="",
        expected_tools=["tool_a", "tool_b"],
        actual_tools=["tool_a"],  # Missing tool
    )
    # Jaccard: intersection(1) / union(2) = 0.5
    assert result2.tool_accuracy == 0.5, f"Expected 0.5, got {result2.tool_accuracy}"
    print(f"  ✓ Missing tool case: {result2.tool_accuracy}")

    # Test case: wrong tools should be 0
    result3 = EvaluationResult(
        model="test",
        format_type="test",
        prompt_id="test",
        response_text="",
        expected_tools=["tool_a"],
        actual_tools=["tool_b"],  # Wrong tool
    )
    # Jaccard: intersection(0) / union(2) = 0.0
    assert result3.tool_accuracy == 0.0, f"Expected 0.0, got {result3.tool_accuracy}"
    print(f"  ✓ Wrong tool case: {result3.tool_accuracy}")

    # Test case: perfect match
    result4 = EvaluationResult(
        model="test",
        format_type="test",
        prompt_id="test",
        response_text="",
        expected_tools=["tool_a", "tool_b"],
        actual_tools=["tool_a", "tool_b"],
    )
    assert result4.tool_accuracy == 1.0, f"Expected 1.0, got {result4.tool_accuracy}"
    print(f"  ✓ Perfect match case: {result4.tool_accuracy}")

    print("  PASSED\n")


def test_json_brace_matching():
    """Verify JSON brace matching handles strings correctly (Bug #6)."""
    print("Testing Bug #6: JSON brace matching with strings...")

    from src.formats.json_format import JSONFormat

    json_format = JSONFormat()

    # Test with braces inside string values
    text = '{"tool": "test", "message": "Use { and } in string"}'
    objects = json_format._extract_json_objects(text)
    assert len(objects) == 1, f"Expected 1 object, got {len(objects)}"
    assert objects[0] == text, f"Expected full object, got {objects[0]}"
    print(f"  ✓ Handles braces in strings: {len(objects)} object extracted")

    # Test with nested objects
    text2 = '{"tool": "test", "params": {"nested": "value"}}'
    objects2 = json_format._extract_json_objects(text2)
    assert len(objects2) == 1, f"Expected 1 object, got {len(objects2)}"
    print(f"  ✓ Handles nested objects: {len(objects2)} object extracted")

    print("  PASSED\n")


def test_function_sig_nested_parens():
    """Verify function signature parsing handles nested parentheses (Bug #11)."""
    print("Testing Bug #11: Function signature nested parentheses...")

    from src.formats.function_sig import FunctionSigFormat

    func_format = FunctionSigFormat()

    # Test with nested parentheses
    text = 'call_tool(data=[(1, 2), (3, 4)], nested=func(a, b))'
    matches = func_format._extract_function_calls(text)
    assert len(matches) >= 1, f"Expected at least 1 match, got {len(matches)}"
    func_name, args = matches[0]
    assert func_name == "call_tool", f"Expected 'call_tool', got {func_name}"
    assert "data=[(1, 2), (3, 4)]" in args, f"Arguments not captured correctly: {args}"
    print(f"  ✓ Nested parens: {func_name}(...)")

    # Test with deeply nested
    text2 = 'outer(inner1(inner2(x)), y)'
    matches2 = func_format._extract_function_calls(text2)
    func_names = [m[0] for m in matches2]
    assert "outer" in func_names, f"Expected 'outer' in matches: {func_names}"
    print(f"  ✓ Deep nesting: {func_names}")

    print("  PASSED\n")


def test_evaluation_result_category():
    """Verify EvaluationResult has category field (Bug #12)."""
    print("Testing Bug #12: EvaluationResult category field...")

    result = EvaluationResult(
        model="test",
        format_type="test",
        prompt_id="test",
        response_text="",
        category="simple",
    )
    assert result.category == "simple", f"Expected 'simple', got {result.category}"
    print(f"  ✓ Category field exists: {result.category}")

    # Check it's in to_dict
    result_dict = result.to_dict()
    assert "category" in result_dict, "category not in to_dict()"
    print(f"  ✓ Category in to_dict(): {result_dict['category']}")

    print("  PASSED\n")


def main():
    print("=" * 60)
    print("Verifying Critical Bug Fixes")
    print("=" * 60 + "\n")

    test_no_double_prefix()
    test_tool_accuracy_jaccard()
    test_json_brace_matching()
    test_function_sig_nested_parens()
    test_evaluation_result_category()

    print("=" * 60)
    print("All critical fixes verified!")
    print("=" * 60)


if __name__ == "__main__":
    main()
