"""Evaluation runner for tool call format testing."""

from dataclasses import dataclass, field
from typing import Any
import time
import json
from pathlib import Path

from src.formats.base import BaseFormat, ToolCall
from src.tools.base import Tool
from src.prompts.base import EvalPrompt
from src.models.config import ModelConfig
from src.tokens.tracker import TokenTracker, TokenUsage
from src.tokens.counter import count_tokens
from src.tokens.overhead import calculate_format_overhead
from src.evaluation.metrics import MetricsCalculator, EvaluationMetrics


@dataclass
class EvaluationResult:
    """Result of a single evaluation run."""

    model: str
    format_type: str
    prompt_id: str

    # Raw response
    response_text: str
    raw_response: Any = None

    # Parsed tool calls
    parsed_calls: list[ToolCall] = field(default_factory=list)
    parse_success: bool = False
    parse_error: str = ""

    # Validation
    valid_calls: list[ToolCall] = field(default_factory=list)
    invalid_calls: list[tuple[ToolCall, str]] = field(default_factory=list)

    # Token usage
    token_usage: TokenUsage | None = None

    # Timing
    latency_ms: float = 0.0

    # Expected vs actual
    expected_tools: list[str] = field(default_factory=list)
    actual_tools: list[str] = field(default_factory=list)

    @property
    def tool_accuracy(self) -> float:
        """Accuracy of tool selection."""
        if not self.expected_tools:
            return 1.0 if not self.actual_tools else 0.0

        correct = sum(1 for t in self.actual_tools if t in self.expected_tools)
        return correct / len(self.expected_tools)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model": self.model,
            "format_type": self.format_type,
            "prompt_id": self.prompt_id,
            "response_text": self.response_text,
            "parsed_calls": [{"name": c.name, "arguments": c.arguments} for c in self.parsed_calls],
            "parse_success": self.parse_success,
            "parse_error": self.parse_error,
            "valid_calls": [{"name": c.name, "arguments": c.arguments} for c in self.valid_calls],
            "invalid_calls": [({"name": c.name, "arguments": c.arguments}, err) for c, err in self.invalid_calls],
            "token_usage": self.token_usage.to_dict() if self.token_usage else None,
            "latency_ms": self.latency_ms,
            "expected_tools": self.expected_tools,
            "actual_tools": self.actual_tools,
            "tool_accuracy": self.tool_accuracy,
        }


class EvaluationRunner:
    """Run evaluation across models, formats, and prompts."""

    def __init__(
        self,
        formats: list[BaseFormat],
        tools: list[Tool],
        prompts: list[EvalPrompt],
        models: list[ModelConfig],
        token_tracker: TokenTracker | None = None,
    ):
        self.formats = {f.name: f for f in formats}
        self.tools = tools
        self.prompts = {p.id: p for p in prompts}
        self.models = {f"{m.provider}/{m.model_id}": m for m in models}
        self.token_tracker = token_tracker or TokenTracker()
        self.results: list[EvaluationResult] = []
        self.metrics_calculator = MetricsCalculator()

    def run_single(
        self,
        model_key: str,
        format_name: str,
        prompt_id: str,
    ) -> EvaluationResult:
        """Run a single evaluation."""
        model_config = self.models[model_key]
        format_handler = self.formats[format_name]
        prompt = self.prompts[prompt_id]

        # Generate system prompt
        system_prompt = format_handler.generate_system_prompt(self.tools)
        system_tokens = count_tokens(system_prompt)
        user_tokens = count_tokens(prompt.prompt)

        # Get client
        client = model_config.get_client()

        # Make API call
        start_time = time.time()

        try:
            response = client.chat.completions.create(
                model=model_config.model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt.prompt},
                ],
                max_tokens=model_config.max_tokens,
                temperature=model_config.temperature,
            )
            latency_ms = (time.time() - start_time) * 1000
            response_text = response.choices[0].message.content or ""

        except Exception as e:
            return EvaluationResult(
                model=model_key,
                format_type=format_name,
                prompt_id=prompt_id,
                response_text="",
                parse_success=False,
                parse_error=f"API error: {str(e)}",
                expected_tools=prompt.expected_tool_names,
            )

        # Parse response
        try:
            parsed_calls = format_handler.parse_response(response_text)
            parse_success = True
            parse_error = ""
        except Exception as e:
            parsed_calls = []
            parse_success = False
            parse_error = str(e)

        # Validate calls
        valid_calls = []
        invalid_calls = []
        for call in parsed_calls:
            is_valid, error = format_handler.validate_call(call, self.tools)
            if is_valid:
                valid_calls.append(call)
            else:
                invalid_calls.append((call, error))

        # Calculate tool call tokens and overhead
        tool_call_tokens = 0
        overhead_tokens = 0
        for call in parsed_calls:
            if call.raw_text:
                overhead = calculate_format_overhead(
                    call.raw_text,
                    format_name,
                    call.name,
                    call.arguments,
                )
                tool_call_tokens += overhead.content_tokens
                overhead_tokens += overhead.overhead_tokens

        # Record token usage
        token_usage = self.token_tracker.record_from_response(
            response=response,
            model=model_key,
            format_type=format_name,
            prompt_id=prompt_id,
            system_prompt_tokens=system_tokens,
            user_message_tokens=user_tokens,
            tool_call_tokens=tool_call_tokens,
            overhead_tokens=overhead_tokens,
            success=parse_success and len(valid_calls) > 0,
        )

        result = EvaluationResult(
            model=model_key,
            format_type=format_name,
            prompt_id=prompt_id,
            response_text=response_text,
            raw_response=response,
            parsed_calls=parsed_calls,
            parse_success=parse_success,
            parse_error=parse_error,
            valid_calls=valid_calls,
            invalid_calls=invalid_calls,
            token_usage=token_usage,
            latency_ms=latency_ms,
            expected_tools=prompt.expected_tool_names,
            actual_tools=[c.name for c in valid_calls],
        )

        self.results.append(result)
        return result

    def run_all(
        self,
        models: list[str] | None = None,
        formats: list[str] | None = None,
        prompts: list[str] | None = None,
        progress_callback: Any = None,
    ) -> list[EvaluationResult]:
        """Run evaluation across all combinations."""
        models = models or list(self.models.keys())
        formats = formats or list(self.formats.keys())
        prompts = prompts or list(self.prompts.keys())

        total = len(models) * len(formats) * len(prompts)
        current = 0

        for model_key in models:
            for format_name in formats:
                for prompt_id in prompts:
                    current += 1
                    if progress_callback:
                        progress_callback(current, total, model_key, format_name, prompt_id)

                    try:
                        self.run_single(model_key, format_name, prompt_id)
                    except Exception as e:
                        # Log error but continue
                        print(f"Error in {model_key}/{format_name}/{prompt_id}: {e}")

        return self.results

    def get_metrics(self) -> EvaluationMetrics:
        """Calculate metrics for all results."""
        return self.metrics_calculator.calculate(self.results)

    def export_results(self, path: str | Path) -> None:
        """Export evaluation results to JSON."""
        path = Path(path)

        data = {
            "results": [r.to_dict() for r in self.results],
            "metrics": self.get_metrics().to_dict(),
            "token_summary": {
                "by_format": self.token_tracker.get_summary_by_format(),
                "by_model": self.token_tracker.get_summary_by_model(),
            },
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def print_summary(self) -> None:
        """Print a summary of evaluation results."""
        metrics = self.get_metrics()

        print("\n=== Evaluation Summary ===\n")
        print(f"Total runs: {metrics.total_runs}")
        print(f"Parse success rate: {metrics.parse_success_rate:.1%}")
        print(f"Tool accuracy: {metrics.tool_accuracy:.1%}")
        print(f"Average latency: {metrics.avg_latency_ms:.0f}ms")

        print("\n--- By Format ---")
        for format_name, stats in metrics.by_format.items():
            print(f"\n{format_name}:")
            print(f"  Success rate: {stats['success_rate']:.1%}")
            print(f"  Tool accuracy: {stats['tool_accuracy']:.1%}")

        print("\n--- By Model ---")
        for model, stats in metrics.by_model.items():
            print(f"\n{model}:")
            print(f"  Success rate: {stats['success_rate']:.1%}")
            print(f"  Tool accuracy: {stats['tool_accuracy']:.1%}")

        print("\n--- Token Efficiency ---")
        token_summary = self.token_tracker.get_summary_by_format()
        for format_name, stats in token_summary.items():
            print(f"\n{format_name}:")
            print(f"  Avg tokens: {stats['avg_total_tokens']:.0f}")
            print(f"  Efficiency: {stats['avg_efficiency_ratio']:.1%}")
