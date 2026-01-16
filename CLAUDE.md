# Tool Call Format Experiment

A comprehensive framework to test and compare different tool call formats across various LLMs, measuring parsing reliability, accuracy, and token efficiency.

## Project Overview

This project evaluates how well different LLMs follow various tool calling format specifications. It tests 5 formats across 12 models from 4 providers, using 26 evaluation prompts and 17 tool definitions of varying complexity.

## Implemented Components

### Tool Call Formats (`src/formats/`)

| Format | File | Description |
|--------|------|-------------|
| **JSON** | `json_format.py` | Lenient JSON parsing with regex fallbacks for common model mistakes |
| **MCP** | `mcp_format.py` | Strict Model Context Protocol (JSON-RPC 2.0) validation |
| **XML** | `xml_format.py` | XML tag-based parsing with nested structure support |
| **Unstructured** | `unstructured.py` | Natural language patterns (`TOOL_CALL:` blocks) |
| **Function Signature** | `function_sig.py` | Python-style function calls with AST parsing |

### Tool Definitions (`src/tools/`)

| Complexity | Tools | Examples |
|------------|-------|----------|
| **Simple** | 4 | `get_weather(city)`, `get_time(timezone)`, `calculate(expression)` |
| **Medium** | 4 | `search_files(query, file_type, max_results)`, `send_email(to, subject, body, cc)` |
| **Complex** | 4 | `create_chart(data, chart_type, options)`, `batch_process(items, operations, config)` |
| **Edge Cases** | 5 | `weird_params(kebab-case-param, $special)`, `unicode_tool(日本語, émoji)`, `empty_tool()` |

### Evaluation Prompts (`src/prompts/`)

| Category | Count | Description |
|----------|-------|-------------|
| **Simple** | 8 | Single tool calls ("What's the weather in Tokyo?") |
| **Multi-tool** | 5 | Multiple tool calls in one request |
| **Ambiguous** | 8 | Prompts requiring clarification |
| **Complex** | 9 | Nested structures, unusual parameter names |

### Model Configurations (`src/models/config.py`)

| Provider | Models |
|----------|--------|
| **Cerebras** | zai-glm-4.7, gpt-oss-120b, qwen-3-235b-a22b-instruct-2507, zai-glm-4.6, llama-3.3-70b |
| **Groq** | groq/compound, groq/compound-mini, meta-llama/llama-4-scout-17b-16e-instruct, moonshotai/kimi-k2-instruct-0905 |
| **Gemini** | gemini-2.5-flash |
| **Cohere** | command-a-reasoning-08-2025, command-a-03-2025 |

### Token Tracking (`src/tokens/`)

- **`tracker.py`**: `TokenTracker` class that extracts `prompt_tokens`, `completion_tokens`, `total_tokens` from API responses
- **`counter.py`**: tiktoken utilities for local token counting
- **`overhead.py`**: Format-specific overhead calculation (JSON keys, XML tags, MCP wrapper)
- **`cost.py`**: Cost estimation per provider

### Evaluation Framework (`src/evaluation/`)

- **`runner.py`**: `EvaluationRunner` - iterates models × formats × prompts, captures results
- **`parser.py`**: Response parsing utilities
- **`metrics.py`**: `MetricsCalculator` - parse success rate, tool accuracy, latency

## Usage

```python
from src.formats import JSONFormat, MCPFormat, XMLFormat, UnstructuredFormat, FunctionSigFormat
from src.tools import get_all_tools
from src.prompts import get_all_prompts
from src.models.config import get_all_models
from src.evaluation import EvaluationRunner
from src.tokens import TokenTracker

# Initialize
formats = [JSONFormat(), MCPFormat(), XMLFormat(), UnstructuredFormat(), FunctionSigFormat()]
tools = get_all_tools()
prompts = get_all_prompts()
models = get_all_models()
token_tracker = TokenTracker()

# Run evaluation
runner = EvaluationRunner(formats, tools, prompts, models, token_tracker)
runner.run_all()

# View results
runner.print_summary()
runner.export_results("results/evaluation.json")
token_tracker.export_csv("results/token_usage.csv")
```

## Tests

63 unit tests covering all components:
```bash
uv run pytest tests/ -v
```

## Dependencies

- `keycycle` - API key rotation for multi-provider access
- `tiktoken` - Token counting
- `openai` - OpenAI-compatible client
- `agno` - Required by keycycle

## Environment Setup

Required `.env` file:
```bash
NUM_CEREBRAS=1
CEREBRAS_API_KEY_1=your_key

NUM_GROQ=1
GROQ_API_KEY_1=your_key

NUM_GEMINI=1
GEMINI_API_KEY_1=your_key

NUM_COHERE=1
COHERE_API_KEY_1=your_key
```

---

## Next Steps

### Main Test Script (`main.py`)

Implement a comprehensive test runner that:

1. **Full Matrix Evaluation**
   - Iterate through all 12 models × 5 formats × 26 prompts = 1,560 test cases
   - Handle rate limits and API errors gracefully
   - Save intermediate results to allow resuming

2. **Token Usage Tracking**
   - Extract `usage` field from every API response
   - Track: `prompt_tokens`, `completion_tokens`, `total_tokens`
   - Calculate format overhead per response
   - Aggregate by format, model, and prompt category

3. **Results Analysis**
   - Parse success rate by format/model
   - Tool selection accuracy by format/model
   - Parameter extraction accuracy
   - Average tokens per format
   - Cost comparison across providers

4. **Visualization (save as PNG)**
   - Bar chart: Parse success rate by format
   - Bar chart: Parse success rate by model
   - Grouped bar chart: Success rate by format × model
   - Line chart: Token efficiency by format
   - Heatmap: Format × Model success matrix
   - Pie chart: Cost distribution by provider

5. **Output Files**
   - `results/evaluation_results.json` - Full results
   - `results/token_usage.csv` - Token data
   - `results/summary_report.md` - Human-readable summary
   - `results/charts/` - PNG visualizations

### Implementation Outline

```python
# main.py
import matplotlib.pyplot as plt
from src.formats import JSONFormat, MCPFormat, XMLFormat, UnstructuredFormat, FunctionSigFormat
from src.tools import get_all_tools
from src.prompts import get_all_prompts
from src.models.config import get_all_models
from src.evaluation import EvaluationRunner
from src.tokens import TokenTracker

def main():
    # Setup
    formats = [JSONFormat(), MCPFormat(), XMLFormat(), UnstructuredFormat(), FunctionSigFormat()]
    tools = get_all_tools()
    prompts = get_all_prompts()
    models = get_all_models()
    tracker = TokenTracker()

    # Run with progress callback
    runner = EvaluationRunner(formats, tools, prompts, models, tracker)
    runner.run_all(progress_callback=print_progress)

    # Export results
    runner.export_results("results/evaluation_results.json")
    tracker.export_csv("results/token_usage.csv")

    # Generate visualizations
    generate_charts(runner, tracker)

    # Generate summary report
    generate_report(runner, tracker)

def generate_charts(runner, tracker):
    # Create charts directory
    # Generate and save each chart as PNG
    pass

def generate_report(runner, tracker):
    # Generate markdown summary
    pass

if __name__ == "__main__":
    main()
```

### Additional Dependencies Needed

```bash
uv pip install matplotlib pandas seaborn
```
