"""Token usage tracking and aggregation."""

import logging
from dataclasses import dataclass, field
from typing import Any
import json
from pathlib import Path

import tiktoken

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    """Track token usage for a single API call."""

    model: str
    format_type: str
    prompt_id: str

    # From API response usage field
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    # Calculated metrics
    system_prompt_tokens: int = 0
    user_message_tokens: int = 0
    tool_call_tokens: int = 0
    overhead_tokens: int = 0

    # Metadata
    success: bool = True
    error_message: str = ""

    @property
    def efficiency_ratio(self) -> float:
        """Ratio of useful content to total output tokens (higher = better)."""
        if self.completion_tokens == 0:
            return 0.0
        return self.tool_call_tokens / self.completion_tokens

    @property
    def format_overhead_ratio(self) -> float:
        """Ratio of overhead to tool call tokens (lower = better)."""
        if self.tool_call_tokens == 0:
            return float("inf")
        return self.overhead_tokens / self.tool_call_tokens

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model": self.model,
            "format_type": self.format_type,
            "prompt_id": self.prompt_id,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "system_prompt_tokens": self.system_prompt_tokens,
            "user_message_tokens": self.user_message_tokens,
            "tool_call_tokens": self.tool_call_tokens,
            "overhead_tokens": self.overhead_tokens,
            "efficiency_ratio": self.efficiency_ratio,
            "format_overhead_ratio": self.format_overhead_ratio if self.tool_call_tokens > 0 else None,
            "success": self.success,
            "error_message": self.error_message,
        }


@dataclass
class TokenTracker:
    """Track and aggregate token usage across experiments."""

    encoding_name: str = "cl100k_base"
    usage_records: list[TokenUsage] = field(default_factory=list)
    _encoder: Any = field(default=None, repr=False)

    def __post_init__(self):
        self._encoder = tiktoken.get_encoding(self.encoding_name)

    def count_tokens(self, text: str) -> int:
        """Count tokens in a string using tiktoken."""
        return len(self._encoder.encode(text))

    def record_usage(self, usage: TokenUsage) -> None:
        """Record a token usage entry."""
        self.usage_records.append(usage)

    def record_from_response(
        self,
        response: Any,
        model: str,
        format_type: str,
        prompt_id: str,
        system_prompt_tokens: int = 0,
        user_message_tokens: int = 0,
        tool_call_tokens: int = 0,
        overhead_tokens: int = 0,
        success: bool = True,
        error_message: str = "",
    ) -> TokenUsage:
        """Record usage from an API response object."""
        # Extract token counts from API response
        usage = response.usage if hasattr(response, "usage") else None

        if usage is None:
            logger.warning(f"No usage data in response for {model}/{format_type}/{prompt_id}")

        token_usage = TokenUsage(
            model=model,
            format_type=format_type,
            prompt_id=prompt_id,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            system_prompt_tokens=system_prompt_tokens,
            user_message_tokens=user_message_tokens,
            tool_call_tokens=tool_call_tokens,
            overhead_tokens=overhead_tokens,
            success=success,
            error_message=error_message,
        )

        self.record_usage(token_usage)
        return token_usage

    def get_summary_by_format(self) -> dict[str, dict[str, float]]:
        """Aggregate statistics by format type."""
        by_format: dict[str, list[TokenUsage]] = {}

        for record in self.usage_records:
            if record.format_type not in by_format:
                by_format[record.format_type] = []
            by_format[record.format_type].append(record)

        summaries = {}
        for format_type, records in by_format.items():
            successful = [r for r in records if r.success]
            if not successful:
                continue

            summaries[format_type] = {
                "count": len(records),
                "success_rate": len(successful) / len(records),
                "avg_prompt_tokens": sum(r.prompt_tokens for r in successful) / len(successful),
                "avg_completion_tokens": sum(r.completion_tokens for r in successful) / len(successful),
                "avg_total_tokens": sum(r.total_tokens for r in successful) / len(successful),
                "avg_efficiency_ratio": sum(r.efficiency_ratio for r in successful) / len(successful),
                "avg_overhead_ratio": sum(
                    r.format_overhead_ratio for r in successful if r.tool_call_tokens > 0
                )
                / max(1, len([r for r in successful if r.tool_call_tokens > 0])),
            }

        return summaries

    def get_summary_by_model(self) -> dict[str, dict[str, float]]:
        """Aggregate statistics by model."""
        by_model: dict[str, list[TokenUsage]] = {}

        for record in self.usage_records:
            if record.model not in by_model:
                by_model[record.model] = []
            by_model[record.model].append(record)

        summaries = {}
        for model, records in by_model.items():
            successful = [r for r in records if r.success]
            if not successful:
                continue

            summaries[model] = {
                "count": len(records),
                "success_rate": len(successful) / len(records),
                "avg_prompt_tokens": sum(r.prompt_tokens for r in successful) / len(successful),
                "avg_completion_tokens": sum(r.completion_tokens for r in successful) / len(successful),
                "avg_total_tokens": sum(r.total_tokens for r in successful) / len(successful),
                "avg_efficiency_ratio": sum(r.efficiency_ratio for r in successful) / len(successful),
            }

        return summaries

    def get_summary_by_format_and_model(self) -> dict[str, dict[str, dict[str, float]]]:
        """Aggregate statistics by format and model combination."""
        by_combo: dict[str, dict[str, list[TokenUsage]]] = {}

        for record in self.usage_records:
            if record.format_type not in by_combo:
                by_combo[record.format_type] = {}
            if record.model not in by_combo[record.format_type]:
                by_combo[record.format_type][record.model] = []
            by_combo[record.format_type][record.model].append(record)

        summaries: dict[str, dict[str, dict[str, float]]] = {}
        for format_type, models in by_combo.items():
            summaries[format_type] = {}
            for model, records in models.items():
                successful = [r for r in records if r.success]
                if not successful:
                    continue

                summaries[format_type][model] = {
                    "count": len(records),
                    "success_rate": len(successful) / len(records),
                    "avg_prompt_tokens": sum(r.prompt_tokens for r in successful) / len(successful),
                    "avg_completion_tokens": sum(r.completion_tokens for r in successful) / len(successful),
                    "avg_total_tokens": sum(r.total_tokens for r in successful) / len(successful),
                }

        return summaries

    def export_results(self, path: str | Path) -> None:
        """Export token usage data to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "records": [r.to_dict() for r in self.usage_records],
            "summary_by_format": self.get_summary_by_format(),
            "summary_by_model": self.get_summary_by_model(),
            "summary_by_format_and_model": self.get_summary_by_format_and_model(),
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def export_csv(self, path: str | Path) -> None:
        """Export token usage data to CSV file."""
        import csv

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "model",
            "format_type",
            "prompt_id",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "system_prompt_tokens",
            "user_message_tokens",
            "tool_call_tokens",
            "overhead_tokens",
            "efficiency_ratio",
            "format_overhead_ratio",
            "success",
            "error_message",
        ]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in self.usage_records:
                row = record.to_dict()
                # Handle infinity values in CSV (same as to_dict handles for JSON)
                if row.get("format_overhead_ratio") == float("inf"):
                    row["format_overhead_ratio"] = None
                writer.writerow(row)
