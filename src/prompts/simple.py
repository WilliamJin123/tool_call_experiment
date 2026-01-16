"""Simple evaluation prompts - single tool calls."""

from src.prompts.base import EvalPrompt, ExpectedToolCall

SIMPLE_PROMPTS = [
    EvalPrompt(
        id="simple_weather_1",
        prompt="What's the weather in Tokyo?",
        expected_calls=[
            ExpectedToolCall(
                name="get_weather",
                arguments={"city": "Tokyo"},
            )
        ],
        category="simple",
        description="Basic weather query for a single city",
    ),
    EvalPrompt(
        id="simple_weather_2",
        prompt="Can you tell me the current weather conditions in New York City?",
        expected_calls=[
            ExpectedToolCall(
                name="get_weather",
                arguments={"city": "New York City"},
                partial_match=True,  # Accept "New York" or "New York City"
            )
        ],
        category="simple",
        description="Weather query with verbose phrasing",
    ),
    EvalPrompt(
        id="simple_time_1",
        prompt="What time is it in Tokyo?",
        expected_calls=[
            ExpectedToolCall(
                name="get_time",
                arguments={"timezone": "Asia/Tokyo"},
                partial_match=True,
            )
        ],
        category="simple",
        description="Time query with city name (needs timezone conversion)",
    ),
    EvalPrompt(
        id="simple_time_2",
        prompt="What's the current UTC time?",
        expected_calls=[
            ExpectedToolCall(
                name="get_time",
                arguments={"timezone": "UTC"},
            )
        ],
        category="simple",
        description="Time query with explicit timezone",
    ),
    EvalPrompt(
        id="simple_calc_1",
        prompt="Calculate 15 times 23",
        expected_calls=[
            ExpectedToolCall(
                name="calculate",
                arguments={"expression": "15 * 23"},
                partial_match=True,
            )
        ],
        category="simple",
        description="Simple multiplication",
    ),
    EvalPrompt(
        id="simple_search_1",
        prompt="Search for Python files containing 'import'",
        expected_calls=[
            ExpectedToolCall(
                name="search_files",
                arguments={"query": "import", "file_type": "py"},
                partial_match=True,
            )
        ],
        category="simple",
        description="File search with type filter",
    ),
    EvalPrompt(
        id="simple_translate_1",
        prompt="Translate 'Hello, how are you?' to Spanish",
        expected_calls=[
            ExpectedToolCall(
                name="translate",
                arguments={
                    "text": "Hello, how are you?",
                    "target_language": "es",
                },
                partial_match=True,
            )
        ],
        category="simple",
        description="Simple translation request",
    ),
    EvalPrompt(
        id="simple_empty_tool",
        prompt="Run the empty tool",
        expected_calls=[
            ExpectedToolCall(
                name="empty_tool",
                arguments={},
            )
        ],
        category="simple",
        description="Call a tool with no parameters",
    ),
]
