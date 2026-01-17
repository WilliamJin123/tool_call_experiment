"""Metrics calculation for evaluation results."""

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.evaluation.runner import EvaluationResult


@dataclass
class EvaluationMetrics:
    """Aggregated metrics from evaluation runs."""

    total_runs: int = 0
    successful_parses: int = 0
    failed_parses: int = 0

    # Success rates
    parse_success_rate: float = 0.0
    tool_accuracy: float = 0.0
    param_accuracy: float = 0.0

    # Latency
    avg_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0

    # By format breakdown
    by_format: dict[str, dict[str, float]] = field(default_factory=dict)

    # By model breakdown
    by_model: dict[str, dict[str, float]] = field(default_factory=dict)

    # By prompt category breakdown
    by_category: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_runs": self.total_runs,
            "successful_parses": self.successful_parses,
            "failed_parses": self.failed_parses,
            "parse_success_rate": self.parse_success_rate,
            "tool_accuracy": self.tool_accuracy,
            "param_accuracy": self.param_accuracy,
            "avg_latency_ms": self.avg_latency_ms,
            "min_latency_ms": self.min_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "by_format": self.by_format,
            "by_model": self.by_model,
            "by_category": self.by_category,
        }


class MetricsCalculator:
    """Calculate metrics from evaluation results."""

    def calculate(self, results: list["EvaluationResult"]) -> EvaluationMetrics:
        """Calculate metrics from a list of evaluation results."""
        if not results:
            return EvaluationMetrics()

        metrics = EvaluationMetrics()
        metrics.total_runs = len(results)

        # Count successes and failures
        metrics.successful_parses = sum(1 for r in results if r.parse_success)
        metrics.failed_parses = metrics.total_runs - metrics.successful_parses

        # Success rates
        metrics.parse_success_rate = metrics.successful_parses / metrics.total_runs

        # Tool accuracy - include ALL results, treating failed parses as 0
        tool_accuracies = [r.tool_accuracy if r.parse_success else 0.0 for r in results]
        metrics.tool_accuracy = sum(tool_accuracies) / len(tool_accuracies)

        # Param accuracy - include ALL results, treating failed parses as 0
        param_accuracies = [r.param_accuracy if r.parse_success else 0.0 for r in results]
        metrics.param_accuracy = sum(param_accuracies) / len(param_accuracies)

        # Latency
        latencies = [r.latency_ms for r in results if r.latency_ms > 0]
        if latencies:
            metrics.avg_latency_ms = sum(latencies) / len(latencies)
            metrics.min_latency_ms = min(latencies)
            metrics.max_latency_ms = max(latencies)

        # By format
        metrics.by_format = self._calculate_by_group(results, lambda r: r.format_type)

        # By model
        metrics.by_model = self._calculate_by_group(results, lambda r: r.model)

        # By category (filter out results without category)
        results_with_category = [r for r in results if r.category]
        if results_with_category:
            metrics.by_category = self._calculate_by_group(results_with_category, lambda r: r.category)

        return metrics

    def _calculate_by_group(
        self,
        results: list["EvaluationResult"],
        key_fn: Any,
    ) -> dict[str, dict[str, float]]:
        """Calculate metrics grouped by a key function."""
        groups: dict[str, list["EvaluationResult"]] = {}

        for result in results:
            key = key_fn(result)
            if key not in groups:
                groups[key] = []
            groups[key].append(result)

        summaries = {}
        for key, group_results in groups.items():
            total = len(group_results)
            successful = sum(1 for r in group_results if r.parse_success)

            # Include ALL results in accuracy, treating failed parses as 0
            tool_accuracies = [r.tool_accuracy if r.parse_success else 0.0 for r in group_results]
            avg_tool_accuracy = sum(tool_accuracies) / len(tool_accuracies) if tool_accuracies else 0.0

            param_accuracies = [r.param_accuracy if r.parse_success else 0.0 for r in group_results]
            avg_param_accuracy = sum(param_accuracies) / len(param_accuracies) if param_accuracies else 0.0

            latencies = [r.latency_ms for r in group_results if r.latency_ms > 0]
            avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

            summaries[key] = {
                "total": total,
                "successful": successful,
                "success_rate": successful / total if total > 0 else 0.0,
                "tool_accuracy": avg_tool_accuracy,
                "param_accuracy": avg_param_accuracy,
                "avg_latency_ms": avg_latency,
            }

        return summaries

    def compare_formats(
        self,
        results: list["EvaluationResult"],
    ) -> dict[str, dict[str, Any]]:
        """Compare performance across formats."""
        by_format = self._calculate_by_group(results, lambda r: r.format_type)

        # Rank formats by success rate and tool accuracy
        ranked = sorted(
            by_format.items(),
            key=lambda x: (x[1]["success_rate"], x[1]["tool_accuracy"]),
            reverse=True,
        )

        comparison = {}
        for rank, (format_name, stats) in enumerate(ranked, 1):
            comparison[format_name] = {
                **stats,
                "rank": rank,
            }

        return comparison

    def compare_models(
        self,
        results: list["EvaluationResult"],
    ) -> dict[str, dict[str, Any]]:
        """Compare performance across models."""
        by_model = self._calculate_by_group(results, lambda r: r.model)

        # Rank models by success rate and tool accuracy
        ranked = sorted(
            by_model.items(),
            key=lambda x: (x[1]["success_rate"], x[1]["tool_accuracy"]),
            reverse=True,
        )

        comparison = {}
        for rank, (model, stats) in enumerate(ranked, 1):
            comparison[model] = {
                **stats,
                "rank": rank,
            }

        return comparison

    def get_failure_analysis(
        self,
        results: list["EvaluationResult"],
    ) -> dict[str, list[dict[str, Any]]]:
        """Analyze failures by type."""
        failures = {
            "parse_errors": [],
            "validation_errors": [],
            "wrong_tool": [],
        }

        for result in results:
            if not result.parse_success:
                failures["parse_errors"].append({
                    "model": result.model,
                    "format": result.format_type,
                    "prompt_id": result.prompt_id,
                    "error": result.parse_error,
                    "response_preview": result.response_text[:200],
                })
            elif result.invalid_calls:
                for call, error in result.invalid_calls:
                    failures["validation_errors"].append({
                        "model": result.model,
                        "format": result.format_type,
                        "prompt_id": result.prompt_id,
                        "tool": call.name,
                        "error": error,
                    })
            elif result.tool_accuracy < 1.0:
                failures["wrong_tool"].append({
                    "model": result.model,
                    "format": result.format_type,
                    "prompt_id": result.prompt_id,
                    "expected": result.expected_tools,
                    "actual": result.actual_tools,
                })

        return failures
