"""Simple tool definitions - single or few parameters."""

from src.tools.base import Tool, Parameter

SIMPLE_TOOLS = [
    Tool(
        name="get_weather",
        description="Get the current weather for a city",
        parameters=[
            Parameter(
                name="city",
                type="str",
                description="The city name to get weather for (e.g., 'Tokyo', 'New York')",
                required=True,
            ),
        ],
    ),
    Tool(
        name="get_time",
        description="Get the current time in a specific timezone",
        parameters=[
            Parameter(
                name="timezone",
                type="str",
                description="The timezone (e.g., 'UTC', 'America/New_York', 'Asia/Tokyo')",
                required=False,
                default="UTC",
            ),
        ],
    ),
    Tool(
        name="calculate",
        description="Perform a basic arithmetic calculation",
        parameters=[
            Parameter(
                name="expression",
                type="str",
                description="The mathematical expression to evaluate (e.g., '2 + 2', '10 * 5')",
                required=True,
            ),
        ],
    ),
    Tool(
        name="translate",
        description="Translate text to a target language",
        parameters=[
            Parameter(
                name="text",
                type="str",
                description="The text to translate",
                required=True,
            ),
            Parameter(
                name="target_language",
                type="str",
                description="The target language code (e.g., 'es', 'fr', 'ja')",
                required=True,
            ),
        ],
    ),
]
