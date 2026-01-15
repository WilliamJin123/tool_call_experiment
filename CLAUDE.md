# Tool Call Format Experiment

## Project Overview

This project tests and compares different tool call formats across various Large Language Models (LLMs) to evaluate their effectiveness, parsing reliability, and model compatibility.

## Tool Call Formats to Test

### 1. JSON Format (Less Strict)
Standard JSON object format for tool calls. More forgiving of minor syntax variations.

```json
{
  "tool": "tool_name",
  "parameters": {
    "param1": "value1",
    "param2": "value2"
  }
}
```

### 2. MCP Format (Super Strict)
Model Context Protocol format - requires exact adherence to schema.

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "tool_name",
    "arguments": {
      "param1": "value1"
    }
  }
}
```

### 3. XML Format
XML-based tool calling with structured tags.

```xml
<tool_call>
  <name>tool_name</name>
  <parameters>
    <param1>value1</param1>
    <param2>value2</param2>
  </parameters>
</tool_call>
```

### 4. Unstructured/Instructions Format
Natural language instructions that describe the tool call without rigid structure.

```
Call the tool "tool_name" with the following parameters:
- param1: value1
- param2: value2
```

### 5. Function Signatures Format
Python/TypeScript-style function call syntax.

```python
tool_name(param1="value1", param2="value2")
```

## Tool Definitions

Design tools with varying complexity levels:

### Simple Tools
- `get_weather(city: str)` - Single required parameter
- `get_time(timezone: str = "UTC")` - Optional parameter with default

### Medium Complexity Tools
- `search_files(query: str, file_type: str, max_results: int = 10)` - Multiple parameters
- `send_email(to: str, subject: str, body: str, cc: list[str] = None)` - Mixed types

### Complex Tools
- `create_chart(data: list[dict], chart_type: str, options: dict)` - Nested structures
- `batch_process(items: list[dict], operations: list[str], config: dict)` - Deep nesting

### Edge Case Tools
- `weird_params(kebab-case-param: str, $special: int, _underscore: bool)` - Unusual parameter names
- `unicode_tool(日本語: str, émoji: str)` - Unicode parameters
- `empty_tool()` - No parameters

## Evaluation Prompts

### Simple Prompts
1. "What's the weather in Tokyo?"
2. "Search for Python files containing 'import'"

### Multi-Tool Prompts
1. "Get the weather in Paris and New York, then send an email summary"
2. "Search for all JSON files and create a bar chart of the results"

### Ambiguous Prompts
1. "Help me with the files" (requires clarification)
2. "Process this data" (vague instructions)

### Complex Prompts
1. "Create a pie chart from the sales data where region is 'North' and year > 2020, with custom colors"
2. "Batch process all items with 'pending' status using operations: validate, transform, archive"

## Models Configuration

### Cerebras (via keycycle)
```python
from keycycle import MultiProviderWrapper

wrapper = MultiProviderWrapper.from_env("cerebras", "llama-3.3-70b")
client = wrapper.get_openai_client()
```

Models to test:
- `zai-glm-4.7`
- `gpt-oss-120b`
- `qwen-3-235b-a22b-instruct-2507`
- `zai-glm-4.6`
- `llama-3.3-70b`

### Groq (via keycycle)
```python
wrapper = MultiProviderWrapper.from_env("groq", "meta-llama/llama-4-scout-17b-16e-instruct")
client = wrapper.get_openai_client()
```

Models to test:
- `groq/compound`
- `groq/compound-mini`
- `meta-llama/llama-4-scout-17b-16e-instruct`
- `moonshotai/kimi-k2-instruct-0905`

### Gemini (via keycycle)
```python
wrapper = MultiProviderWrapper.from_env("gemini", "gemini-2.5-flash")
client = wrapper.get_openai_client()
```

Models to test:
- `gemini-2.5-flash`

### Cohere (via keycycle)
```python
wrapper = MultiProviderWrapper.from_env("cohere", "command-a-03-2025")
client = wrapper.get_openai_client()
```

Models to test:
- `command-a-reasoning-08-2025`
- `command-a-03-2025`

## Project Structure

```
tool_call_experiment/
├── CLAUDE.md                    # This file
├── README.md                    # Project overview
├── .env                         # API keys (NUM_CEREBRAS, CEREBRAS_API_KEY_1, etc.)
├── pyproject.toml              # UV project config
│
├── src/
│   ├── __init__.py
│   ├── formats/                # Tool call format definitions
│   │   ├── __init__.py
│   │   ├── base.py            # Base format class
│   │   ├── json_format.py     # JSON format parser/generator
│   │   ├── mcp_format.py      # MCP format parser/generator
│   │   ├── xml_format.py      # XML format parser/generator
│   │   ├── unstructured.py    # Unstructured format parser
│   │   └── function_sig.py    # Function signature format
│   │
│   ├── tools/                  # Tool definitions
│   │   ├── __init__.py
│   │   ├── simple.py          # Simple tools
│   │   ├── medium.py          # Medium complexity tools
│   │   ├── complex.py         # Complex tools
│   │   └── edge_cases.py      # Edge case tools
│   │
│   ├── prompts/                # Eval prompts
│   │   ├── __init__.py
│   │   ├── simple.py          # Simple prompts with expected outputs
│   │   ├── multi_tool.py      # Multi-tool prompts
│   │   ├── ambiguous.py       # Ambiguous prompts
│   │   └── complex.py         # Complex prompts
│   │
│   ├── models/                 # Model configurations
│   │   ├── __init__.py
│   │   └── config.py          # Model/provider configs
│   │
│   ├── evaluation/             # Evaluation logic
│   │   ├── __init__.py
│   │   ├── runner.py          # Test runner
│   │   ├── parser.py          # Response parsing
│   │   └── metrics.py         # Success/failure metrics
│   │
│   └── tokens/                 # Token tracking
│       ├── __init__.py
│       ├── tracker.py         # TokenTracker class
│       ├── counter.py         # Token counting utilities
│       ├── overhead.py        # Format overhead calculation
│       └── cost.py            # Cost estimation
│
├── tests/                      # Unit tests
│   ├── test_formats.py
│   ├── test_tools.py
│   └── test_evaluation.py
│
└── results/                    # Evaluation results
    └── .gitkeep
```

## Implementation Plan

### Phase 1: Core Infrastructure
1. Set up project structure with UV
2. Implement base format class with abstract methods:
   - `generate_system_prompt()` - Generate tool instructions for system prompt
   - `parse_response()` - Parse model response to extract tool calls
   - `validate_call()` - Validate tool call against schema

### Phase 2: Format Implementations
1. Implement each format class:
   - JSON format with lenient parsing
   - MCP format with strict validation
   - XML format with tag-based parsing
   - Unstructured format with regex/NLP parsing
   - Function signature format with AST parsing

### Phase 3: Tool Definitions
1. Define tool schemas in a format-agnostic way
2. Implement tool converters for each format
3. Create expected output mappings for evaluation

### Phase 4: Token Tracking Infrastructure
1. Implement `TokenTracker` class:
   - Integration with `tiktoken` for accurate token counting
   - Support for different tokenizer models per provider
   - Automatic extraction from API response `usage` field
2. Implement format-specific overhead calculators:
   - JSON overhead parser (keys, braces, quotes)
   - XML overhead parser (tags)
   - MCP overhead parser (wrapper fields)
   - Function signature overhead (minimal)
   - Unstructured overhead (delimiters, keywords)
3. Create token analysis utilities:
   - System prompt token counter per format
   - Tool call token extractor
   - Efficiency ratio calculator

### Phase 5: Evaluation Framework
1. Create test runner that:
   - Iterates over models × formats × prompts
   - Makes API calls via keycycle
   - Captures token usage from API response
   - Parses responses
   - Compares against expected outputs
2. Implement metrics:
   - Parse success rate
   - Tool call accuracy
   - Parameter extraction accuracy
   - Format adherence score
   - Token efficiency metrics
   - Cost per successful tool call

### Phase 6: Results & Analysis
1. Run full evaluation matrix
2. Generate comparison reports:
   - Accuracy comparison by format/model
   - Token efficiency comparison by format
   - Cost analysis per format/model combination
   - Overhead breakdown charts
3. Identify format-model compatibility patterns
4. Generate token efficiency recommendations

## Environment Setup

Required `.env` file structure:
```bash
# Cerebras
NUM_CEREBRAS=1
CEREBRAS_API_KEY_1=your_key_here

# Groq
NUM_GROQ=1
GROQ_API_KEY_1=your_key_here

# Gemini
NUM_GEMINI=1
GEMINI_API_KEY_1=your_key_here

# Cohere
NUM_COHERE=1
COHERE_API_KEY_1=your_key_here
COHERE_TIER=free  # or 'pro' or 'enterprise'
```

## Usage Example

```python
from keycycle import MultiProviderWrapper
from src.formats import JSONFormat, MCPFormat, XMLFormat
from src.tools import get_all_tools
from src.evaluation import EvaluationRunner
from src.tokens import TokenTracker

# Initialize provider
wrapper = MultiProviderWrapper.from_env("cerebras", "llama-3.3-70b")
client = wrapper.get_openai_client()

# Initialize format and token tracker
format_handler = JSONFormat()
token_tracker = TokenTracker()

# Get tools
tools = get_all_tools()

# Generate system prompt with tool definitions
system_prompt = format_handler.generate_system_prompt(tools)
system_prompt_tokens = token_tracker.count_tokens(system_prompt)

user_message = "What's the weather in Tokyo?"
user_tokens = token_tracker.count_tokens(user_message)

# Make API call
response = client.chat.completions.create(
    model="llama-3.3-70b",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
)

# Extract token usage from API response
usage = response.usage
print(f"Input tokens: {usage.prompt_tokens}")
print(f"Output tokens: {usage.completion_tokens}")
print(f"Total tokens: {usage.total_tokens}")

# Calculate format overhead
content = response.choices[0].message.content
tool_call_tokens, overhead_tokens = token_tracker.extract_tool_call_tokens(
    content, format_type="json"
)
print(f"Tool call tokens: {tool_call_tokens}")
print(f"Format overhead tokens: {overhead_tokens}")
print(f"Efficiency ratio: {tool_call_tokens / usage.completion_tokens:.2%}")

# Parse response
tool_calls = format_handler.parse_response(content)

# Validate
for call in tool_calls:
    is_valid = format_handler.validate_call(call, tools)
    print(f"Tool: {call.name}, Valid: {is_valid}")

# Record usage for aggregation
token_tracker.record_usage(
    model="llama-3.3-70b",
    format_type="json",
    prompt_type="simple",
    input_tokens=usage.prompt_tokens,
    output_tokens=usage.completion_tokens,
    system_prompt_tokens=system_prompt_tokens,
    tool_call_tokens=tool_call_tokens,
    overhead_tokens=overhead_tokens
)
```

## Token Tracking & Efficiency

Track token usage across formats to compare efficiency and cost implications.

### Token Metrics to Capture

| Metric | Description |
|--------|-------------|
| System Prompt Tokens | Tokens used by tool definitions in system prompt |
| Input Tokens | Total input tokens (system + user message) |
| Output Tokens | Tokens in model response |
| Total Tokens | Input + Output tokens |
| Tool Call Tokens | Tokens specifically for the tool call portion |
| Overhead Tokens | Format boilerplate tokens (tags, keys, etc.) |

### Format Efficiency Comparison

Different formats have different token overhead:

| Format | Expected Overhead | Notes |
|--------|-------------------|-------|
| JSON | Medium | Keys and punctuation add tokens |
| MCP | High | Extra wrapper fields (`jsonrpc`, `method`, etc.) |
| XML | High | Opening/closing tags are verbose |
| Unstructured | Low | Minimal structure, but may be ambiguous |
| Function Signature | Low | Compact syntax, minimal overhead |

### Token Tracking Implementation

```python
from dataclasses import dataclass
from typing import Optional
import tiktoken

@dataclass
class TokenUsage:
    """Track token usage for a single API call."""
    model: str
    format_type: str
    prompt_type: str

    # From API response
    input_tokens: int
    output_tokens: int
    total_tokens: int

    # Calculated metrics
    system_prompt_tokens: int  # Tokens for tool definitions
    user_message_tokens: int   # Tokens for user prompt
    tool_call_tokens: int      # Tokens for tool call in response
    overhead_tokens: int       # Format-specific overhead

    # Efficiency ratios
    @property
    def efficiency_ratio(self) -> float:
        """Ratio of useful content to total tokens (lower overhead = better)."""
        if self.tool_call_tokens == 0:
            return 0.0
        return self.tool_call_tokens / self.output_tokens

    @property
    def format_overhead_ratio(self) -> float:
        """Ratio of overhead to tool call tokens."""
        if self.tool_call_tokens == 0:
            return float('inf')
        return self.overhead_tokens / self.tool_call_tokens


class TokenTracker:
    """Track and aggregate token usage across experiments."""

    def __init__(self, encoding_name: str = "cl100k_base"):
        self.encoder = tiktoken.get_encoding(encoding_name)
        self.usage_records: list[TokenUsage] = []

    def count_tokens(self, text: str) -> int:
        """Count tokens in a string."""
        return len(self.encoder.encode(text))

    def extract_tool_call_tokens(self, response: str, format_type: str) -> tuple[int, int]:
        """Extract tool call tokens and overhead tokens from response."""
        # Implementation varies by format
        pass

    def record_usage(self, usage: TokenUsage) -> None:
        """Record a token usage entry."""
        self.usage_records.append(usage)

    def get_summary_by_format(self) -> dict:
        """Aggregate statistics by format type."""
        pass

    def get_summary_by_model(self) -> dict:
        """Aggregate statistics by model."""
        pass

    def export_results(self, path: str) -> None:
        """Export token usage data to CSV/JSON."""
        pass
```

### Token Counting Strategy

1. **System Prompt Tokens**: Count tokens in the generated system prompt for each format
2. **Response Parsing**: Extract the tool call portion and count its tokens
3. **Overhead Calculation**:
   - For JSON: Count tokens for `{`, `}`, `"tool":`, `"parameters":`, etc.
   - For XML: Count tokens for `<tool_call>`, `</tool_call>`, etc.
   - For MCP: Count tokens for wrapper fields
   - For Function Sig: Minimal (just parentheses and commas)
   - For Unstructured: Count non-content tokens

### Cost Analysis

Calculate estimated costs per format using provider pricing:

```python
@dataclass
class CostEstimate:
    format_type: str
    avg_input_tokens: float
    avg_output_tokens: float

    def calculate_cost(self, input_price_per_1k: float, output_price_per_1k: float) -> float:
        """Calculate cost per API call."""
        input_cost = (self.avg_input_tokens / 1000) * input_price_per_1k
        output_cost = (self.avg_output_tokens / 1000) * output_price_per_1k
        return input_cost + output_cost
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Parse Success | Could the response be parsed into tool calls? |
| Tool Accuracy | Was the correct tool selected? |
| Param Accuracy | Were parameters extracted correctly? |
| Format Score | How well did output match expected format? |
| Latency | Response time per model/format combination |
| Input Tokens | Total input tokens per call |
| Output Tokens | Total output tokens per call |
| Token Efficiency | Ratio of useful content to total tokens |
| Format Overhead | Extra tokens required by format structure |
| Cost per Call | Estimated cost based on token usage |

## Notes

- Use `keycycle.MultiProviderWrapper` for all LLM calls to handle API key rotation
- Each provider requires environment variables in the format `NUM_{PROVIDER}` and `{PROVIDER}_API_KEY_{N}`
- The wrapper provides an OpenAI-compatible client via `get_openai_client()`
- Rate limits are automatically managed per provider/model

## Dependencies

```toml
[project]
dependencies = [
    "keycycle",           # API key rotation
    "tiktoken",           # Token counting (OpenAI tokenizer)
    "openai",             # OpenAI-compatible client
]
```

### Token Counting Notes

- **tiktoken**: Uses OpenAI's tokenizer (`cl100k_base` encoding) as a baseline
- Different models may use different tokenizers; `cl100k_base` provides a reasonable approximation
- For exact token counts, use the `usage` field from API responses when available
- Local token counting is useful for:
  - Pre-calculating system prompt overhead
  - Estimating costs before making API calls
  - Analyzing format-specific overhead patterns
