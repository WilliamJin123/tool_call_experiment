"""Complex tool definitions - nested structures and deep nesting."""

from src.tools.base import Tool, Parameter

COMPLEX_TOOLS = [
    Tool(
        name="create_chart",
        description="Create a chart visualization from data",
        parameters=[
            Parameter(
                name="data",
                type="list[dict]",
                description="List of data points, each as a dict with keys matching the chart axes",
                required=True,
            ),
            Parameter(
                name="chart_type",
                type="str",
                description="Type of chart: 'bar', 'line', 'pie', 'scatter', 'area'",
                required=True,
            ),
            Parameter(
                name="options",
                type="dict",
                description="Chart options including 'title', 'x_axis', 'y_axis', 'colors', 'legend'",
                required=True,
            ),
        ],
    ),
    Tool(
        name="batch_process",
        description="Process multiple items with a series of operations",
        parameters=[
            Parameter(
                name="items",
                type="list[dict]",
                description="List of items to process, each with 'id' and 'data' fields",
                required=True,
            ),
            Parameter(
                name="operations",
                type="list[str]",
                description="List of operations to apply: 'validate', 'transform', 'enrich', 'archive'",
                required=True,
            ),
            Parameter(
                name="config",
                type="dict",
                description="Processing configuration with 'parallel', 'retry_count', 'timeout' options",
                required=True,
            ),
        ],
    ),
    Tool(
        name="create_report",
        description="Generate a complex report with multiple sections",
        parameters=[
            Parameter(
                name="title",
                type="str",
                description="Report title",
                required=True,
            ),
            Parameter(
                name="sections",
                type="list[dict]",
                description="List of sections, each with 'heading', 'content', 'charts' fields",
                required=True,
            ),
            Parameter(
                name="metadata",
                type="dict",
                description="Report metadata: 'author', 'date', 'version', 'tags'",
                required=False,
                default=None,
            ),
            Parameter(
                name="format",
                type="str",
                description="Output format: 'pdf', 'html', 'markdown'",
                required=False,
                default="pdf",
            ),
        ],
    ),
    Tool(
        name="configure_pipeline",
        description="Configure a data processing pipeline with multiple stages",
        parameters=[
            Parameter(
                name="name",
                type="str",
                description="Pipeline name",
                required=True,
            ),
            Parameter(
                name="stages",
                type="list[dict]",
                description="Pipeline stages, each with 'name', 'type', 'config', 'dependencies'",
                required=True,
            ),
            Parameter(
                name="triggers",
                type="list[dict]",
                description="Pipeline triggers with 'type' (cron/webhook/manual) and 'config'",
                required=False,
                default=None,
            ),
            Parameter(
                name="notifications",
                type="dict",
                description="Notification settings: 'on_success', 'on_failure', 'channels'",
                required=False,
                default=None,
            ),
        ],
    ),
]
