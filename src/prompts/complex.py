"""Complex evaluation prompts - nested structures and detailed requirements."""

from src.prompts.base import EvalPrompt, ExpectedToolCall

COMPLEX_PROMPTS = [
    EvalPrompt(
        id="complex_chart_filter",
        prompt="Create a pie chart from the sales data where region is 'North' and year > 2020, with custom colors blue and green, title 'North Region Sales'",
        expected_calls=[
            ExpectedToolCall(
                name="create_chart",
                arguments={
                    "chart_type": "pie",
                    "options": {
                        "title": "North Region Sales",
                        "colors": ["blue", "green"],
                    },
                },
                partial_match=True,
            )
        ],
        category="complex",
        description="Chart with filtering and custom options",
    ),
    EvalPrompt(
        id="complex_batch",
        prompt="Batch process all items with 'pending' status using operations: validate, transform, archive. Use parallel processing with 3 retries and 30 second timeout",
        expected_calls=[
            ExpectedToolCall(
                name="batch_process",
                arguments={
                    "operations": ["validate", "transform", "archive"],
                    "config": {
                        "parallel": True,
                        "retry_count": 3,
                        "timeout": 30,
                    },
                },
                partial_match=True,
            )
        ],
        category="complex",
        description="Batch processing with detailed config",
    ),
    EvalPrompt(
        id="complex_report",
        prompt="Generate a PDF report titled 'Q4 Summary' with sections for Sales (include bar chart), Marketing (include pie chart), and Engineering. Author is 'Data Team', version 1.0",
        expected_calls=[
            ExpectedToolCall(
                name="create_report",
                arguments={
                    "title": "Q4 Summary",
                    "format": "pdf",
                    "metadata": {
                        "author": "Data Team",
                        "version": "1.0",
                    },
                },
                partial_match=True,
            )
        ],
        category="complex",
        description="Report with multiple sections and metadata",
    ),
    EvalPrompt(
        id="complex_pipeline",
        prompt="Configure a pipeline named 'ETL Daily' with stages: extract (type: database), transform (type: python), load (type: warehouse). Add a cron trigger for midnight UTC and email notifications on failure to ops@company.com",
        expected_calls=[
            ExpectedToolCall(
                name="configure_pipeline",
                arguments={
                    "name": "ETL Daily",
                },
                partial_match=True,
            )
        ],
        category="complex",
        description="Pipeline with stages, triggers, and notifications",
    ),
    EvalPrompt(
        id="complex_query_report",
        prompt="Query the 'orders' table for columns order_id, customer, total where status = 'completed', limit 50 results, then create a line chart of totals over time",
        expected_calls=[
            ExpectedToolCall(
                name="query_database",
                arguments={
                    "table": "orders",
                    "columns": ["order_id", "customer", "total"],
                    "where": "status = 'completed'",
                    "limit": 50,
                },
                partial_match=True,
            ),
            ExpectedToolCall(
                name="create_chart",
                arguments={
                    "chart_type": "line",
                },
                partial_match=True,
            ),
        ],
        category="complex",
        description="Database query followed by visualization",
    ),
    EvalPrompt(
        id="complex_weird_params",
        prompt="Call weird_params with kebab-case-param set to 'test', $special set to 42, and _underscore set to true",
        expected_calls=[
            ExpectedToolCall(
                name="weird_params",
                arguments={
                    "kebab-case-param": "test",
                    "$special": 42,
                    "_underscore": True,
                },
            )
        ],
        category="complex",
        description="Tool with unusual parameter names",
    ),
    EvalPrompt(
        id="complex_unicode",
        prompt="Use the unicode_tool with 日本語 set to 'こんにちは' and émoji set to 'test'",
        expected_calls=[
            ExpectedToolCall(
                name="unicode_tool",
                arguments={
                    "日本語": "こんにちは",
                    "émoji": "test",
                },
            )
        ],
        category="complex",
        description="Tool with unicode parameter names",
    ),
    EvalPrompt(
        id="complex_mixed_types",
        prompt="Call mixed_types with string_param='hello', int_param=42, float_param=3.14, bool_param=true, list_param=['a', 'b', 'c'], and dict_param={'key': 'value'}",
        expected_calls=[
            ExpectedToolCall(
                name="mixed_types",
                arguments={
                    "string_param": "hello",
                    "int_param": 42,
                    "float_param": 3.14,
                    "bool_param": True,
                    "list_param": ["a", "b", "c"],
                    "dict_param": {"key": "value"},
                },
            )
        ],
        category="complex",
        description="Tool with many different parameter types",
    ),
    EvalPrompt(
        id="complex_nested",
        prompt="Call deeply_nested with config containing three levels: level1 has level2, level2 has level3, and level3 has value set to 'deep'",
        expected_calls=[
            ExpectedToolCall(
                name="deeply_nested",
                arguments={
                    "config": {
                        "level1": {
                            "level2": {
                                "level3": {
                                    "value": "deep",
                                }
                            }
                        }
                    }
                },
            )
        ],
        category="complex",
        description="Deeply nested data structure",
    ),
]
