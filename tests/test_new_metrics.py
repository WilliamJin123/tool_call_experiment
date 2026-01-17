
import pytest
from src.evaluation.runner import EvaluationResult
from src.prompts.base import ExpectedToolCall
from src.formats.base import ToolCall
from src.evaluation.metrics import MetricsCalculator

def test_param_accuracy_exact_match():
    """Test param_accuracy with exact matches."""
    result = EvaluationResult(
        model="test",
        format_type="test",
        prompt_id="test",
        response_text="",
        expected_calls=[
            ExpectedToolCall(name="tool1", arguments={"a": 1}),
            ExpectedToolCall(name="tool2", arguments={"b": 2})
        ],
        valid_calls=[
            ToolCall(name="tool1", arguments={"a": 1}),
            ToolCall(name="tool2", arguments={"b": 2})
        ],
        parse_success=True
    )
    assert result.param_accuracy == 1.0

def test_param_accuracy_partial_match_fail():
    """Test param_accuracy with exact match requirement failing."""
    result = EvaluationResult(
        model="test",
        format_type="test",
        prompt_id="test",
        response_text="",
        expected_calls=[
            ExpectedToolCall(name="tool1", arguments={"a": 1})
        ],
        valid_calls=[
            ToolCall(name="tool1", arguments={"a": 2})  # Wrong value
        ],
        parse_success=True
    )
    assert result.param_accuracy == 0.0

def test_param_accuracy_partial_match_success():
    """Test param_accuracy with partial match allowed."""
    result = EvaluationResult(
        model="test",
        format_type="test",
        prompt_id="test",
        response_text="",
        expected_calls=[
            ExpectedToolCall(name="tool1", arguments={"a": 1}, partial_match=True)
        ],
        valid_calls=[
            ToolCall(name="tool1", arguments={"a": 1, "b": 2})  # Extra arg OK
        ],
        parse_success=True
    )
    assert result.param_accuracy == 1.0

def test_param_accuracy_missing_call():
    """Test param_accuracy with missing call."""
    result = EvaluationResult(
        model="test",
        format_type="test",
        prompt_id="test",
        response_text="",
        expected_calls=[
            ExpectedToolCall(name="tool1", arguments={"a": 1}),
            ExpectedToolCall(name="tool2", arguments={"b": 2})
        ],
        valid_calls=[
            ToolCall(name="tool1", arguments={"a": 1})
        ],
        parse_success=True
    )
    # 1 match out of max(2, 1) = 2
    assert result.param_accuracy == 0.5

def test_param_accuracy_extra_call():
    """Test param_accuracy with extra call."""
    result = EvaluationResult(
        model="test",
        format_type="test",
        prompt_id="test",
        response_text="",
        expected_calls=[
            ExpectedToolCall(name="tool1", arguments={"a": 1})
        ],
        valid_calls=[
            ToolCall(name="tool1", arguments={"a": 1}),
            ToolCall(name="tool2", arguments={"b": 2})
        ],
        parse_success=True
    )
    # 1 match out of max(1, 2) = 2
    assert result.param_accuracy == 0.5

def test_metrics_aggregation():
    """Test that metrics calculator aggregates param_accuracy."""
    results = [
        EvaluationResult(
            model="m1", format_type="f1", prompt_id="p1", response_text="",
            expected_calls=[ExpectedToolCall("t1")],
            valid_calls=[ToolCall("t1")],
            parse_success=True,
            category="c1"
        ),
        EvaluationResult(
            model="m1", format_type="f1", prompt_id="p2", response_text="",
            expected_calls=[ExpectedToolCall("t1")],
            valid_calls=[],
            parse_success=True,
            category="c1"
        )
    ]
    # r1: acc=1.0
    # r2: acc=0.0
    # avg: 0.5
    
    calc = MetricsCalculator()
    metrics = calc.calculate(results)
    
    assert metrics.param_accuracy == 0.5
    assert metrics.by_format["f1"]["param_accuracy"] == 0.5
    assert metrics.by_model["m1"]["param_accuracy"] == 0.5
