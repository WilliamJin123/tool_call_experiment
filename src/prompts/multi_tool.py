"""Multi-tool evaluation prompts - requiring multiple tool calls."""

from src.prompts.base import EvalPrompt, ExpectedToolCall

MULTI_TOOL_PROMPTS = [
    EvalPrompt(
        id="multi_weather_cities",
        prompt="Get the weather in Paris and New York, then send an email summary to team@example.com",
        expected_calls=[
            ExpectedToolCall(
                name="get_weather",
                arguments={"city": "Paris"},
            ),
            ExpectedToolCall(
                name="get_weather",
                arguments={"city": "New York"},
                partial_match=True,
            ),
            ExpectedToolCall(
                name="send_email",
                arguments={"to": "team@example.com"},
                partial_match=True,
            ),
        ],
        category="multi_tool",
        description="Weather for two cities followed by email",
    ),
    EvalPrompt(
        id="multi_search_chart",
        prompt="Search for all JSON files and create a bar chart showing the results",
        expected_calls=[
            ExpectedToolCall(
                name="search_files",
                arguments={"file_type": "json"},
                partial_match=True,
            ),
            ExpectedToolCall(
                name="create_chart",
                arguments={"chart_type": "bar"},
                partial_match=True,
            ),
        ],
        category="multi_tool",
        description="Search then visualize results",
    ),
    EvalPrompt(
        id="multi_task_email",
        prompt="Create a high priority task titled 'Review PR' due 2024-12-31, and email the details to manager@company.com",
        expected_calls=[
            ExpectedToolCall(
                name="create_task",
                arguments={
                    "title": "Review PR",
                    "priority": "high",
                    "due_date": "2024-12-31",
                },
                partial_match=True,
            ),
            ExpectedToolCall(
                name="send_email",
                arguments={"to": "manager@company.com"},
                partial_match=True,
            ),
        ],
        category="multi_tool",
        description="Create task and send notification",
    ),
    EvalPrompt(
        id="multi_time_zones",
        prompt="What time is it in New York, London, and Tokyo?",
        expected_calls=[
            ExpectedToolCall(
                name="get_time",
                arguments={},
                partial_match=True,
            ),
            ExpectedToolCall(
                name="get_time",
                arguments={},
                partial_match=True,
            ),
            ExpectedToolCall(
                name="get_time",
                arguments={},
                partial_match=True,
            ),
        ],
        category="multi_tool",
        description="Time in multiple locations",
    ),
    EvalPrompt(
        id="multi_translate_compare",
        prompt="Translate 'Good morning' to French and Japanese",
        expected_calls=[
            ExpectedToolCall(
                name="translate",
                arguments={"text": "Good morning", "target_language": "fr"},
                partial_match=True,
            ),
            ExpectedToolCall(
                name="translate",
                arguments={"text": "Good morning", "target_language": "ja"},
                partial_match=True,
            ),
        ],
        category="multi_tool",
        description="Translate to multiple languages",
    ),
]
