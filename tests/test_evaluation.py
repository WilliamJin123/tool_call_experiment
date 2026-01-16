"""Tests for evaluation framework."""

import pytest

from src.formats.base import ToolCall
from src.prompts.base import EvalPrompt, ExpectedToolCall
from src.evaluation.parser import ResponseParser
from src.evaluation.metrics import MetricsCalculator, EvaluationMetrics
from src.tokens.tracker import TokenTracker, TokenUsage
from src.tokens.counter import count_tokens
from src.tokens.overhead import calculate_json_overhead, compare_format_overhead


class TestExpectedToolCall:
    """Tests for ExpectedToolCall."""

    def test_exact_match(self):
        """Test exact matching."""
        expected = ExpectedToolCall(
            name="get_weather",
            arguments={"city": "Tokyo"},
            partial_match=False,
        )

        assert expected.matches("get_weather", {"city": "Tokyo"})
        assert not expected.matches("get_weather", {"city": "Paris"})
        assert not expected.matches("get_weather", {"city": "Tokyo", "extra": "arg"})

    def test_partial_match(self):
        """Test partial matching."""
        expected = ExpectedToolCall(
            name="get_weather",
            arguments={"city": "Tokyo"},
            partial_match=True,
        )

        assert expected.matches("get_weather", {"city": "Tokyo"})
        assert expected.matches("get_weather", {"city": "Tokyo", "extra": "arg"})
        assert not expected.matches("get_weather", {"city": "Paris"})


class TestResponseParser:
    """Tests for ResponseParser utilities."""

    def test_extract_code_blocks(self):
        """Test extracting code blocks."""
        text = '''Here's some code:
```json
{"key": "value"}
```
And more:
```python
print("hello")
```'''

        # Extract all code blocks
        blocks = ResponseParser.extract_code_blocks(text)
        assert len(blocks) == 2

        # Extract only JSON blocks
        json_blocks = ResponseParser.extract_code_blocks(text, "json")
        assert len(json_blocks) == 1
        assert "key" in json_blocks[0]

    def test_extract_json_objects(self):
        """Test extracting JSON objects."""
        text = 'Some text {"tool": "test"} more text {"other": "json"}'
        objects = ResponseParser.extract_json_objects(text)
        assert len(objects) == 2

    def test_extract_xml_tags(self):
        """Test extracting XML tags."""
        text = '<root><name>test</name><name>test2</name></root>'
        names = ResponseParser.extract_xml_tags(text, "name")
        assert len(names) == 2
        assert "test" in names[0]

    def test_normalize_tool_name(self):
        """Test tool name normalization."""
        assert ResponseParser.normalize_tool_name("get_weather") == "get_weather"
        assert ResponseParser.normalize_tool_name("call_get_weather") == "get_weather"
        assert ResponseParser.normalize_tool_name("GET_WEATHER") == "get_weather"
        assert ResponseParser.normalize_tool_name("get_weather_tool") == "get_weather"

    def test_compare_tool_calls_exact(self):
        """Test comparing tool calls for exact match."""
        expected = ToolCall(name="get_weather", arguments={"city": "Tokyo"})
        actual = ToolCall(name="get_weather", arguments={"city": "Tokyo"})

        is_match, reason = ResponseParser.compare_tool_calls(expected, actual)
        assert is_match
        assert reason == ""

    def test_compare_tool_calls_mismatch(self):
        """Test comparing mismatched tool calls."""
        expected = ToolCall(name="get_weather", arguments={"city": "Tokyo"})
        actual = ToolCall(name="get_weather", arguments={"city": "Paris"})

        is_match, reason = ResponseParser.compare_tool_calls(expected, actual)
        assert not is_match
        assert "mismatch" in reason.lower()


class TestTokenTracker:
    """Tests for TokenTracker."""

    def setup_method(self):
        self.tracker = TokenTracker()

    def test_count_tokens(self):
        """Test basic token counting."""
        count = self.tracker.count_tokens("Hello, world!")
        assert count > 0
        assert isinstance(count, int)

    def test_record_usage(self):
        """Test recording usage."""
        usage = TokenUsage(
            model="test-model",
            format_type="json",
            prompt_id="test-1",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )
        self.tracker.record_usage(usage)
        assert len(self.tracker.usage_records) == 1

    def test_get_summary_by_format(self):
        """Test getting summary by format."""
        # Add some test data
        for format_type in ["json", "json", "xml"]:
            usage = TokenUsage(
                model="test-model",
                format_type=format_type,
                prompt_id="test",
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
            )
            self.tracker.record_usage(usage)

        summary = self.tracker.get_summary_by_format()
        assert "json" in summary
        assert "xml" in summary
        assert summary["json"]["count"] == 2
        assert summary["xml"]["count"] == 1


class TestTokenUsage:
    """Tests for TokenUsage dataclass."""

    def test_efficiency_ratio(self):
        """Test efficiency ratio calculation."""
        usage = TokenUsage(
            model="test",
            format_type="json",
            prompt_id="test",
            prompt_tokens=100,
            completion_tokens=100,
            total_tokens=200,
            tool_call_tokens=80,
        )
        assert usage.efficiency_ratio == 0.8

    def test_efficiency_ratio_zero_completion(self):
        """Test efficiency ratio with zero completion tokens."""
        usage = TokenUsage(
            model="test",
            format_type="json",
            prompt_id="test",
            prompt_tokens=100,
            completion_tokens=0,
            total_tokens=100,
        )
        assert usage.efficiency_ratio == 0.0

    def test_overhead_ratio(self):
        """Test format overhead ratio calculation."""
        usage = TokenUsage(
            model="test",
            format_type="json",
            prompt_id="test",
            prompt_tokens=100,
            completion_tokens=100,
            total_tokens=200,
            tool_call_tokens=80,
            overhead_tokens=20,
        )
        assert usage.format_overhead_ratio == 0.25


class TestTokenCounter:
    """Tests for token counting utilities."""

    def test_count_tokens_basic(self):
        """Test basic token counting."""
        count = count_tokens("Hello, world!")
        assert count > 0

    def test_count_tokens_empty(self):
        """Test counting empty string."""
        count = count_tokens("")
        assert count == 0

    def test_count_tokens_consistency(self):
        """Test that same text gives same count."""
        text = "This is a test sentence for token counting."
        count1 = count_tokens(text)
        count2 = count_tokens(text)
        assert count1 == count2


class TestOverheadCalculation:
    """Tests for format overhead calculation."""

    def test_json_overhead(self):
        """Test JSON format overhead calculation."""
        overhead = calculate_json_overhead("get_weather", {"city": "Tokyo"})

        assert overhead.format_type == "json"
        assert overhead.total_tokens > 0
        assert overhead.content_tokens > 0
        assert overhead.overhead_tokens >= 0

    def test_compare_format_overhead(self):
        """Test comparing overhead across formats."""
        comparison = compare_format_overhead("get_weather", {"city": "Tokyo"})

        assert "json" in comparison
        assert "mcp" in comparison
        assert "xml" in comparison
        assert "function_sig" in comparison
        assert "unstructured" in comparison

        # MCP should have more overhead than JSON due to wrapper
        assert comparison["mcp"].overhead_tokens >= comparison["json"].overhead_tokens


class TestMetricsCalculator:
    """Tests for MetricsCalculator."""

    def test_calculate_empty_results(self):
        """Test calculating metrics with no results."""
        calculator = MetricsCalculator()
        metrics = calculator.calculate([])

        assert metrics.total_runs == 0
        assert metrics.parse_success_rate == 0.0

    def test_calculate_basic_metrics(self):
        """Test calculating basic metrics."""
        # Create mock results
        from src.evaluation.runner import EvaluationResult

        results = [
            EvaluationResult(
                model="model-1",
                format_type="json",
                prompt_id="test-1",
                response_text="test",
                parse_success=True,
                expected_tools=["get_weather"],
                actual_tools=["get_weather"],
                latency_ms=100.0,
            ),
            EvaluationResult(
                model="model-1",
                format_type="json",
                prompt_id="test-2",
                response_text="test",
                parse_success=False,
                parse_error="Parse error",
                expected_tools=["get_time"],
                actual_tools=[],
                latency_ms=150.0,
            ),
        ]

        calculator = MetricsCalculator()
        metrics = calculator.calculate(results)

        assert metrics.total_runs == 2
        assert metrics.successful_parses == 1
        assert metrics.failed_parses == 1
        assert metrics.parse_success_rate == 0.5
