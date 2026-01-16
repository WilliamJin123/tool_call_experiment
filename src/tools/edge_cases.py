"""Edge case tool definitions - unusual parameter names and unicode."""

from src.tools.base import Tool, Parameter

EDGE_CASE_TOOLS = [
    Tool(
        name="weird_params",
        description="A tool with unusual parameter naming conventions",
        parameters=[
            Parameter(
                name="kebab-case-param",
                type="str",
                description="A parameter with kebab-case naming",
                required=True,
            ),
            Parameter(
                name="$special",
                type="int",
                description="A parameter starting with a special character",
                required=True,
            ),
            Parameter(
                name="_underscore",
                type="bool",
                description="A parameter starting with underscore",
                required=True,
            ),
        ],
    ),
    Tool(
        name="unicode_tool",
        description="A tool with unicode parameter names",
        parameters=[
            Parameter(
                name="日本語",
                type="str",
                description="Japanese text parameter",
                required=True,
            ),
            Parameter(
                name="émoji",
                type="str",
                description="Parameter with accented characters",
                required=True,
            ),
        ],
    ),
    Tool(
        name="empty_tool",
        description="A tool with no parameters",
        parameters=[],
    ),
    Tool(
        name="all_optional",
        description="A tool where all parameters are optional",
        parameters=[
            Parameter(
                name="opt1",
                type="str",
                description="First optional parameter",
                required=False,
                default="default1",
            ),
            Parameter(
                name="opt2",
                type="int",
                description="Second optional parameter",
                required=False,
                default=42,
            ),
            Parameter(
                name="opt3",
                type="bool",
                description="Third optional parameter",
                required=False,
                default=False,
            ),
        ],
    ),
    Tool(
        name="mixed_types",
        description="A tool with many different parameter types",
        parameters=[
            Parameter(
                name="string_param",
                type="str",
                description="A string parameter",
                required=True,
            ),
            Parameter(
                name="int_param",
                type="int",
                description="An integer parameter",
                required=True,
            ),
            Parameter(
                name="float_param",
                type="float",
                description="A float parameter",
                required=True,
            ),
            Parameter(
                name="bool_param",
                type="bool",
                description="A boolean parameter",
                required=True,
            ),
            Parameter(
                name="list_param",
                type="list[str]",
                description="A list of strings",
                required=True,
            ),
            Parameter(
                name="dict_param",
                type="dict",
                description="A dictionary parameter",
                required=True,
            ),
        ],
    ),
    Tool(
        name="deeply_nested",
        description="A tool requiring deeply nested data structures",
        parameters=[
            Parameter(
                name="config",
                type="dict",
                description="Deeply nested config: {level1: {level2: {level3: {value: ...}}}}",
                required=True,
            ),
        ],
    ),
]
