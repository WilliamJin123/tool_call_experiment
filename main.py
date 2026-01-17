#!/usr/bin/env python3
"""Main test runner for tool call format experiment.

This script runs a comprehensive evaluation of different tool call formats
across multiple LLMs, measuring parsing reliability, accuracy, and token efficiency.
"""

import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving PNGs
import numpy as np
import seaborn as sns

from src.formats import (
    JSONFormat,
    MCPFormat,
    XMLFormat,
    UnstructuredFormat,
    FunctionSigFormat,
)
from src.formats.base import BaseFormat
from src.tools import get_all_tools
from src.tools.base import Tool
from src.prompts import get_all_prompts
from src.prompts.base import EvalPrompt
from src.models.config import get_all_models, get_models_by_provider, ModelConfig
from src.evaluation import EvaluationRunner
from src.evaluation.runner import EvaluationResult
from src.evaluation.metrics import MetricsCalculator
from src.tokens import TokenTracker
from src.tokens.counter import count_tokens
from src.tokens.overhead import calculate_format_overhead

# Set up logger - file handler added after directory creation
logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure logging after results directory exists."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("results/evaluation.log"),
        ],
    )


def setup_directories() -> None:
    """Create output directories if they don't exist."""
    Path("results").mkdir(exist_ok=True)
    Path("results/charts").mkdir(exist_ok=True)


class ThreadSafeProgress:
    """Thread-safe progress tracking and intermediate saving."""

    def __init__(self, total: int):
        self.lock = threading.Lock()
        self.current = 0
        self.total = total
        self.start_time = time.time()
        self.last_save_count = 0

    def update(self, model: str, format_name: str, prompt_id: str) -> int:
        """Update progress counter and display status (thread-safe)."""
        with self.lock:
            self.current += 1
            current = self.current

        elapsed = time.time() - self.start_time
        rate = current / elapsed if elapsed > 0 else 0
        eta = (self.total - current) / rate if rate > 0 else 0

        # Extract short names for display
        model_short = model.split("/")[-1][:20]
        format_short = format_name[:12]
        prompt_short = prompt_id[:15]

        print(
            f"\r[{current:4d}/{self.total}] {model_short:20s} | {format_short:12s} | {prompt_short:15s} "
            f"| {elapsed:.0f}s elapsed | ETA: {eta:.0f}s",
            end="",
            flush=True,
        )

        return current

    def should_save(self, interval: int = 100) -> bool:
        """Check if we should save intermediate results (thread-safe)."""
        with self.lock:
            if self.current - self.last_save_count >= interval:
                self.last_save_count = self.current
                return True
            return False


def run_single_evaluation(
    model_config: ModelConfig,
    format_handler: BaseFormat,
    prompt: EvalPrompt,
    tools: list[Tool],
    token_tracker: TokenTracker,
) -> EvaluationResult:
    """Run a single evaluation and return the result."""
    model_key = f"{model_config.provider}/{model_config.model_id}"

    # Generate system prompt
    system_prompt = format_handler.generate_system_prompt(tools)
    system_tokens = count_tokens(system_prompt)
    user_tokens = count_tokens(prompt.prompt)

    # Get client
    client = model_config.get_client()

    # Make API call - track latency even on errors
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
        latency_ms = (time.time() - start_time) * 1000
        return EvaluationResult(
            model=model_key,
            format_type=format_handler.name,
            prompt_id=prompt.id,
            response_text="",
            parse_success=False,
            parse_error=f"API error: {str(e)}",
            expected_tools=prompt.expected_tool_names,
            latency_ms=latency_ms,
            category=prompt.category,
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
        is_valid, error = format_handler.validate_call(call, tools)
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
                format_handler.name,
                call.name,
                call.arguments,
            )
            tool_call_tokens += overhead.content_tokens
            overhead_tokens += overhead.overhead_tokens

    # Record token usage
    token_usage = token_tracker.record_from_response(
        response=response,
        model=model_key,
        format_type=format_handler.name,
        prompt_id=prompt.id,
        system_prompt_tokens=system_tokens,
        user_message_tokens=user_tokens,
        tool_call_tokens=tool_call_tokens,
        overhead_tokens=overhead_tokens,
        success=parse_success and len(valid_calls) > 0,
    )

    return EvaluationResult(
        model=model_key,
        format_type=format_handler.name,
        prompt_id=prompt.id,
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
        expected_calls=prompt.expected_calls,
        actual_tools=[c.name for c in valid_calls],
        category=prompt.category,
    )


def run_provider_evaluations(
    provider: str,
    models: list[ModelConfig],
    formats: list[BaseFormat],
    prompts: list[EvalPrompt],
    tools: list[Tool],
    progress: ThreadSafeProgress,
    results_lock: threading.Lock,
    all_results: list[EvaluationResult],
    token_tracker: TokenTracker,
    completed: set[tuple[str, str, str]],
) -> None:
    """Run all evaluations for a single provider, skipping already-completed ones."""
    provider_count = 0
    skipped = 0

    for model in models:
        model_key = f"{model.provider}/{model.model_id}"
        for format_handler in formats:
            for prompt in prompts:
                # Skip if already completed
                key = (model_key, format_handler.name, prompt.id)
                if key in completed:
                    skipped += 1
                    continue

                try:
                    result = run_single_evaluation(
                        model,
                        format_handler,
                        prompt,
                        tools,
                        token_tracker,
                    )
                except Exception as e:
                    logger.error(f"Failed: {model_key}/{format_handler.name}/{prompt.id}: {e}")
                    result = EvaluationResult(
                        model=model_key,
                        format_type=format_handler.name,
                        prompt_id=prompt.id,
                        response_text="",
                        parse_success=False,
                        parse_error=f"Exception: {str(e)}",
                        expected_tools=prompt.expected_tool_names,
                        category=prompt.category,
                    )

                # Thread-safe result append
                with results_lock:
                    all_results.append(result)

                # Update progress
                provider_count += 1
                progress.update(model_key, format_handler.name, prompt.id)

                # Save every 10 completions for this provider
                if provider_count % 10 == 0:
                    save_intermediate_results_threadsafe(
                        all_results, token_tracker, results_lock
                    )
                    logger.info(f"[{provider}] Saved at {provider_count} completions")

    logger.info(f"[{provider}] Completed {provider_count} new, skipped {skipped} existing")


def save_intermediate_results_threadsafe(
    results: list[EvaluationResult],
    token_tracker: TokenTracker,
    lock: threading.Lock,
) -> None:
    """Thread-safe intermediate results saving."""
    with lock:
        results_copy = list(results)

    data = {
        "results": [r.to_dict() for r in results_copy],
        "partial": True,
        "count": len(results_copy),
    }
    with open("results/evaluation_results_partial.json", "w") as f:
        json.dump(data, f, indent=2)

    token_tracker.export_csv("results/token_usage_partial.csv")


def load_completed_evaluations(path: str = "results/evaluation_results_partial.json") -> set[tuple[str, str, str]]:
    """Load previously completed evaluations from partial results file.

    Returns:
        Set of (model, format_type, prompt_id) tuples that have been completed.
    """
    completed: set[tuple[str, str, str]] = set()
    path_obj = Path(path)

    if not path_obj.exists():
        logger.info("No partial results found, starting fresh")
        return completed

    try:
        with open(path_obj) as f:
            data = json.load(f)

        for result in data.get("results", []):
            key = (result["model"], result["format_type"], result["prompt_id"])
            completed.add(key)

        logger.info(f"Loaded {len(completed)} completed evaluations from {path}")
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Failed to load partial results: {e}, starting fresh")

    return completed


def load_previous_results(path: str = "results/evaluation_results_partial.json") -> list[EvaluationResult]:
    """Load previous results to continue from where we left off."""
    path_obj = Path(path)
    if not path_obj.exists():
        return []

    try:
        with open(path_obj) as f:
            data = json.load(f)

        results = []
        for r in data.get("results", []):
            # Reconstruct EvaluationResult from dict
            result = EvaluationResult(
                model=r["model"],
                format_type=r["format_type"],
                prompt_id=r["prompt_id"],
                response_text=r.get("response_text", ""),
                parse_success=r.get("parse_success", False),
                parse_error=r.get("parse_error", ""),
                latency_ms=r.get("latency_ms", 0),
                expected_tools=r.get("expected_tools", []),
                actual_tools=r.get("actual_tools", []),
                category=r.get("category", ""),
            )
            results.append(result)

        logger.info(f"Loaded {len(results)} previous results")
        return results
    except Exception as e:
        logger.warning(f"Failed to load previous results: {e}")
        return []


def create_progress_callback(runner: EvaluationRunner, tracker: TokenTracker):
    """Create a progress callback with timing and periodic saving."""
    start_time = time.time()

    def callback(current: int, total: int, model: str, format_name: str, prompt_id: str):
        elapsed = time.time() - start_time
        rate = current / elapsed if elapsed > 0 else 0
        eta = (total - current) / rate if rate > 0 else 0

        # Extract short names for display
        model_short = model.split("/")[-1][:20]
        format_short = format_name[:12]
        prompt_short = prompt_id[:15]

        print(
            f"\r[{current:4d}/{total}] {model_short:20s} | {format_short:12s} | {prompt_short:15s} "
            f"| {elapsed:.0f}s elapsed | ETA: {eta:.0f}s",
            end="",
            flush=True,
        )

        # Log and save every 100 evaluations
        if current % 100 == 0:
            logger.info(f"Progress: {current}/{total} ({100*current/total:.1f}%)")
            save_intermediate_results(runner, tracker)
            logger.info(f"Saved intermediate results at {current}/{total}")

    return callback


def generate_charts(runner: EvaluationRunner, tracker: TokenTracker) -> None:
    """Generate all visualization charts and save as PNG."""
    metrics = runner.get_metrics()
    results = runner.results

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = sns.color_palette("husl", 5)

    # 1. Bar chart: Parse success rate by format
    fig, ax = plt.subplots(figsize=(10, 6))
    formats = list(metrics.by_format.keys())
    success_rates = [metrics.by_format[f]["success_rate"] * 100 for f in formats]
    bars = ax.bar(formats, success_rates, color=colors)
    ax.set_ylabel("Parse Success Rate (%)")
    ax.set_xlabel("Format")
    ax.set_title("Parse Success Rate by Format")
    ax.set_ylim(0, 100)
    for bar, rate in zip(bars, success_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{rate:.1f}%", ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig("results/charts/success_rate_by_format.png", dpi=150)
    plt.close()
    logger.info("Saved: success_rate_by_format.png")

    # 2. Bar chart: Parse success rate by model
    fig, ax = plt.subplots(figsize=(14, 6))
    models = list(metrics.by_model.keys())
    model_success = [metrics.by_model[m]["success_rate"] * 100 for m in models]
    # Shorten model names for display
    model_labels = [m.split("/")[-1][:25] for m in models]
    bars = ax.barh(model_labels, model_success, color=sns.color_palette("viridis", len(models)))
    ax.set_xlabel("Parse Success Rate (%)")
    ax.set_ylabel("Model")
    ax.set_title("Parse Success Rate by Model")
    ax.set_xlim(0, 100)
    for bar, rate in zip(bars, model_success):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f"{rate:.1f}%", ha='left', va='center', fontsize=8)
    plt.tight_layout()
    plt.savefig("results/charts/success_rate_by_model.png", dpi=150)
    plt.close()
    logger.info("Saved: success_rate_by_model.png")

    # 3. Grouped bar chart: Success rate by format × model
    fig, ax = plt.subplots(figsize=(16, 8))
    x = np.arange(len(models))
    width = 0.15

    # Group results by format and model
    format_model_data = {}
    for result in results:
        key = (result.format_type, result.model)
        if key not in format_model_data:
            format_model_data[key] = {"total": 0, "success": 0}
        format_model_data[key]["total"] += 1
        if result.parse_success:
            format_model_data[key]["success"] += 1

    for i, fmt in enumerate(formats):
        rates = []
        for model in models:
            data = format_model_data.get((fmt, model), {"total": 1, "success": 0})
            rates.append(data["success"] / data["total"] * 100 if data["total"] > 0 else 0)
        offset = (i - len(formats)/2 + 0.5) * width
        ax.bar(x + offset, rates, width, label=fmt, color=colors[i % len(colors)])

    ax.set_ylabel("Parse Success Rate (%)")
    ax.set_xlabel("Model")
    ax.set_title("Parse Success Rate by Format and Model")
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=45, ha='right')
    ax.legend(title="Format", loc='upper right')
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig("results/charts/success_rate_format_model.png", dpi=150)
    plt.close()
    logger.info("Saved: success_rate_format_model.png")

    # 4. Line chart: Token efficiency by format
    token_summary = tracker.get_summary_by_format()
    fig, ax = plt.subplots(figsize=(10, 6))

    if token_summary:
        fmt_names = list(token_summary.keys())
        avg_tokens = [token_summary[f].get("avg_total_tokens", 0) for f in fmt_names]
        efficiency = [token_summary[f].get("avg_efficiency_ratio", 0) * 100 for f in fmt_names]

        ax2 = ax.twinx()

        line1 = ax.plot(fmt_names, avg_tokens, 'o-', color='blue', linewidth=2,
                        markersize=8, label='Avg Total Tokens')
        ax.set_ylabel("Average Total Tokens", color='blue')
        ax.tick_params(axis='y', labelcolor='blue')

        line2 = ax2.plot(fmt_names, efficiency, 's--', color='green', linewidth=2,
                         markersize=8, label='Efficiency Ratio')
        ax2.set_ylabel("Efficiency Ratio (%)", color='green')
        ax2.tick_params(axis='y', labelcolor='green')

        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper right')

        ax.set_xlabel("Format")
        ax.set_title("Token Usage and Efficiency by Format")

    plt.tight_layout()
    plt.savefig("results/charts/token_efficiency_by_format.png", dpi=150)
    plt.close()
    logger.info("Saved: token_efficiency_by_format.png")

    # 5. Heatmap: Format × Model success matrix
    fig, ax = plt.subplots(figsize=(14, 8))

    # Build success rate matrix
    success_matrix = []
    for fmt in formats:
        row = []
        for model in models:
            data = format_model_data.get((fmt, model), {"total": 1, "success": 0})
            rate = data["success"] / data["total"] * 100 if data["total"] > 0 else 0
            row.append(rate)
        success_matrix.append(row)

    success_matrix = np.array(success_matrix)

    im = ax.imshow(success_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

    ax.set_xticks(np.arange(len(models)))
    ax.set_yticks(np.arange(len(formats)))
    ax.set_xticklabels(model_labels, rotation=45, ha='right')
    ax.set_yticklabels(formats)

    # Add text annotations
    for i in range(len(formats)):
        for j in range(len(models)):
            text = ax.text(j, i, f"{success_matrix[i, j]:.0f}%",
                          ha="center", va="center", color="black", fontsize=8)

    ax.set_title("Parse Success Rate Heatmap (Format × Model)")
    fig.colorbar(im, ax=ax, label="Success Rate (%)")
    plt.tight_layout()
    plt.savefig("results/charts/success_heatmap.png", dpi=150)
    plt.close()
    logger.info("Saved: success_heatmap.png")

    # 6. Pie chart: Cost distribution by provider (based on token usage)
    fig, ax = plt.subplots(figsize=(10, 8))

    # Aggregate tokens by provider
    provider_tokens = {}
    for record in tracker.usage_records:
        provider = record.model.split("/")[0]
        if provider not in provider_tokens:
            provider_tokens[provider] = 0
        provider_tokens[provider] += record.total_tokens

    if provider_tokens:
        providers = list(provider_tokens.keys())
        tokens = list(provider_tokens.values())

        explode = [0.02] * len(providers)
        wedges, texts, autotexts = ax.pie(
            tokens,
            labels=providers,
            autopct='%1.1f%%',
            explode=explode,
            colors=sns.color_palette("pastel", len(providers)),
            pctdistance=0.75,
        )

        # Add legend with token counts
        legend_labels = [f"{p}: {t:,} tokens" for p, t in zip(providers, tokens)]
        ax.legend(wedges, legend_labels, title="Provider", loc="center left", bbox_to_anchor=(1, 0.5))

        ax.set_title("Token Usage Distribution by Provider")

    plt.tight_layout()
    plt.savefig("results/charts/token_distribution_by_provider.png", dpi=150)
    plt.close()
    logger.info("Saved: token_distribution_by_provider.png")

    # 7. Bonus: Tool accuracy by format
    fig, ax = plt.subplots(figsize=(10, 6))
    tool_acc = [metrics.by_format[f]["tool_accuracy"] * 100 for f in formats]
    param_acc = [metrics.by_format[f]["param_accuracy"] * 100 for f in formats]

    x = np.arange(len(formats))
    width = 0.35

    bars1 = ax.bar(x - width/2, tool_acc, width, label='Tool Accuracy', color='steelblue')
    bars2 = ax.bar(x + width/2, param_acc, width, label='Parameter Accuracy', color='coral')

    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Format")
    ax.set_title("Tool Selection and Parameter Accuracy by Format")
    ax.set_xticks(x)
    ax.set_xticklabels(formats)
    ax.legend()
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig("results/charts/accuracy_by_format.png", dpi=150)
    plt.close()
    logger.info("Saved: accuracy_by_format.png")


def generate_report(runner: EvaluationRunner, tracker: TokenTracker) -> None:
    """Generate a markdown summary report."""
    metrics = runner.get_metrics()
    calculator = MetricsCalculator()
    format_comparison = calculator.compare_formats(runner.results)
    model_comparison = calculator.compare_models(runner.results)
    failures = calculator.get_failure_analysis(runner.results)
    token_summary = tracker.get_summary_by_format()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""# Tool Call Format Experiment - Summary Report

Generated: {timestamp}

## Overview

| Metric | Value |
|--------|-------|
| Total Evaluations | {metrics.total_runs} |
| Successful Parses | {metrics.successful_parses} |
| Failed Parses | {metrics.failed_parses} |
| Overall Parse Success Rate | {metrics.parse_success_rate:.1%} |
| Overall Tool Accuracy | {metrics.tool_accuracy:.1%} |
| Overall Parameter Accuracy | {metrics.param_accuracy:.1%} |
| Average Latency | {metrics.avg_latency_ms:.0f}ms |
| Min Latency | {metrics.min_latency_ms:.0f}ms |
| Max Latency | {metrics.max_latency_ms:.0f}ms |

## Results by Format

| Rank | Format | Success Rate | Tool Accuracy | Param Accuracy | Avg Latency |
|------|--------|-------------|---------------|----------------|-------------|
"""

    # Sort formats by rank
    sorted_formats = sorted(format_comparison.items(), key=lambda x: x[1]["rank"])
    for name, stats in sorted_formats:
        report += f"| {stats['rank']} | {name} | {stats['success_rate']:.1%} | {stats['tool_accuracy']:.1%} | {stats['param_accuracy']:.1%} | {stats['avg_latency_ms']:.0f}ms |\n"

    report += """
## Results by Model

| Rank | Model | Success Rate | Tool Accuracy | Param Accuracy | Avg Latency |
|------|-------|-------------|---------------|----------------|-------------|
"""

    # Sort models by rank
    sorted_models = sorted(model_comparison.items(), key=lambda x: x[1]["rank"])
    for name, stats in sorted_models:
        display_name = name.split("/")[-1][:30]
        report += f"| {stats['rank']} | {display_name} | {stats['success_rate']:.1%} | {stats['tool_accuracy']:.1%} | {stats['param_accuracy']:.1%} | {stats['avg_latency_ms']:.0f}ms |\n"

    report += """
## Results by Prompt Category

| Category | Success Rate | Tool Accuracy | Param Accuracy |
|----------|-------------|---------------|----------------|
"""

    for category, stats in metrics.by_category.items():
        report += f"| {category} | {stats['success_rate']:.1%} | {stats['tool_accuracy']:.1%} | {stats['param_accuracy']:.1%} |\n"

    if token_summary:
        report += """
## Token Usage by Format

| Format | Avg Prompt Tokens | Avg Completion Tokens | Avg Total Tokens | Efficiency Ratio |
|--------|-------------------|----------------------|------------------|------------------|
"""
        for fmt, stats in token_summary.items():
            report += f"| {fmt} | {stats['avg_prompt_tokens']:.0f} | {stats['avg_completion_tokens']:.0f} | {stats['avg_total_tokens']:.0f} | {stats['avg_efficiency_ratio']:.1%} |\n"

    report += f"""
## Failure Analysis

### Parse Errors: {len(failures['parse_errors'])}
"""

    # Show top 5 parse errors
    if failures['parse_errors']:
        report += "\nSample parse errors:\n"
        for err in failures['parse_errors'][:5]:
            report += f"- **{err['model']}** ({err['format']}): {err['error'][:100]}...\n"

    report += f"""
### Validation Errors: {len(failures['validation_errors'])}
"""

    if failures['validation_errors']:
        report += "\nSample validation errors:\n"
        for err in failures['validation_errors'][:5]:
            report += f"- **{err['model']}** ({err['format']}): Tool `{err['tool']}` - {err['error'][:80]}...\n"

    report += f"""
### Wrong Tool Selection: {len(failures['wrong_tool'])}
"""

    if failures['wrong_tool']:
        report += "\nSample wrong tool selections:\n"
        for err in failures['wrong_tool'][:5]:
            report += f"- **{err['model']}** ({err['format']}): Expected {err['expected']}, got {err['actual']}\n"

    report += """
## Visualizations

All charts are saved in `results/charts/`:

1. `success_rate_by_format.png` - Parse success rate comparison across formats
2. `success_rate_by_model.png` - Parse success rate comparison across models
3. `success_rate_format_model.png` - Grouped bar chart of success rates
4. `token_efficiency_by_format.png` - Token usage and efficiency metrics
5. `success_heatmap.png` - Format × Model success rate heatmap
6. `token_distribution_by_provider.png` - Token usage by provider
7. `accuracy_by_format.png` - Tool and parameter accuracy by format

## Key Findings

"""

    # Generate key findings
    best_format = sorted_formats[0][0] if sorted_formats else "N/A"
    worst_format = sorted_formats[-1][0] if sorted_formats else "N/A"
    best_model = sorted_models[0][0].split("/")[-1] if sorted_models else "N/A"
    worst_model = sorted_models[-1][0].split("/")[-1] if sorted_models else "N/A"

    report += f"""1. **Best Performing Format**: {best_format} with {sorted_formats[0][1]['success_rate']:.1%} parse success rate
2. **Worst Performing Format**: {worst_format} with {sorted_formats[-1][1]['success_rate']:.1%} parse success rate
3. **Best Performing Model**: {best_model} with {sorted_models[0][1]['success_rate']:.1%} parse success rate
4. **Worst Performing Model**: {worst_model} with {sorted_models[-1][1]['success_rate']:.1%} parse success rate
"""

    if token_summary:
        most_efficient = min(token_summary.items(), key=lambda x: x[1].get('avg_total_tokens', float('inf')))
        least_efficient = max(token_summary.items(), key=lambda x: x[1].get('avg_total_tokens', 0))
        report += f"""5. **Most Token-Efficient Format**: {most_efficient[0]} with {most_efficient[1]['avg_total_tokens']:.0f} avg tokens
6. **Least Token-Efficient Format**: {least_efficient[0]} with {least_efficient[1]['avg_total_tokens']:.0f} avg tokens
"""

    report += """
---
*Generated by Tool Call Format Experiment Framework*
"""

    # Write report
    with open("results/summary_report.md", "w") as f:
        f.write(report)

    logger.info("Saved: summary_report.md")


def save_intermediate_results(runner: EvaluationRunner, tracker: TokenTracker) -> None:
    """Save intermediate results to allow resuming."""
    runner.export_results("results/evaluation_results_partial.json")
    tracker.export_csv("results/token_usage_partial.csv")


def main():
    """Run the full evaluation with provider-level parallelization."""
    print("=" * 60)
    print("Tool Call Format Experiment")
    print("=" * 60)

    # Setup
    setup_directories()
    setup_logging()

    # Initialize formats
    formats = [
        JSONFormat(),
        MCPFormat(),
        XMLFormat(),
        UnstructuredFormat(),
        FunctionSigFormat(),
    ]
    print(f"\nFormats: {[f.name for f in formats]}")

    # Initialize tools
    tools = get_all_tools()
    print(f"Tools: {len(tools)} definitions")

    # Initialize prompts
    prompts = get_all_prompts()
    print(f"Prompts: {len(prompts)} evaluation prompts")

    # Initialize models
    models = get_all_models()
    print(f"Models: {len(models)} model configurations")

    # Calculate total tests
    total_tests = len(models) * len(formats) * len(prompts)
    print(f"\nTotal test cases: {total_tests}")
    print("=" * 60)

    # Initialize tracker
    token_tracker = TokenTracker()

    # Load previous results for resume capability
    completed = load_completed_evaluations()
    all_results: list[EvaluationResult] = load_previous_results()
    remaining = total_tests - len(completed)

    print(f"\nResume status: {len(completed)} completed, {remaining} remaining")

    # Shared state for parallel execution
    results_lock = threading.Lock()
    progress = ThreadSafeProgress(remaining)

    # Group models by provider
    providers = ["cerebras", "groq", "cohere"]

    print(f"\nRunning {len(providers)} providers in parallel...")
    for provider in providers:
        provider_models = get_models_by_provider(provider)
        print(f"  {provider}: {len(provider_models)} models")

    # Run evaluation with provider-level parallelization
    print("\nStarting evaluation...\n")
    start_time = time.time()

    try:
        with ThreadPoolExecutor(max_workers=len(providers)) as executor:
            futures = {
                executor.submit(
                    run_provider_evaluations,
                    provider,
                    get_models_by_provider(provider),
                    formats,
                    prompts,
                    tools,
                    progress,
                    results_lock,
                    all_results,
                    token_tracker,
                    completed,
                ): provider
                for provider in providers
            }

            # Wait for completion with error handling
            for future in as_completed(futures):
                provider = futures[future]
                try:
                    future.result()
                    logger.info(f"Provider {provider} completed")
                except Exception as e:
                    logger.error(f"Provider {provider} failed: {e}")

    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving partial results...")
        save_intermediate_results_threadsafe(all_results, token_tracker, results_lock)
        print("Partial results saved. Exiting.")
        sys.exit(1)

    elapsed = time.time() - start_time
    print(f"\n\nEvaluation complete in {elapsed:.1f}s")
    print("=" * 60)

    # Create an EvaluationRunner with the collected results for export and reporting
    runner = EvaluationRunner(formats, tools, prompts, models, token_tracker)
    runner.results = all_results

    # Export results
    print("\nExporting results...")
    runner.export_results("results/evaluation_results.json")
    token_tracker.export_csv("results/token_usage.csv")
    token_tracker.export_results("results/token_usage.json")
    logger.info("Exported: evaluation_results.json, token_usage.csv, token_usage.json")

    # Generate visualizations
    print("\nGenerating visualizations...")
    generate_charts(runner, token_tracker)

    # Generate summary report
    print("\nGenerating summary report...")
    generate_report(runner, token_tracker)

    # Print summary to console
    runner.print_summary()

    print("\n" + "=" * 60)
    print("All results saved to results/")
    print("=" * 60)


if __name__ == "__main__":
    main()
