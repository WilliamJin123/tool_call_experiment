"""Medium complexity tool definitions - multiple parameters with mixed types."""

from src.tools.base import Tool, Parameter

MEDIUM_TOOLS = [
    Tool(
        name="search_files",
        description="Search for files matching a query pattern",
        parameters=[
            Parameter(
                name="query",
                type="str",
                description="The search query or pattern to match",
                required=True,
            ),
            Parameter(
                name="file_type",
                type="str",
                description="The file extension to filter by (e.g., 'py', 'js', 'json')",
                required=True,
            ),
            Parameter(
                name="max_results",
                type="int",
                description="Maximum number of results to return",
                required=False,
                default=10,
            ),
        ],
    ),
    Tool(
        name="send_email",
        description="Send an email to one or more recipients",
        parameters=[
            Parameter(
                name="to",
                type="str",
                description="The primary recipient email address",
                required=True,
            ),
            Parameter(
                name="subject",
                type="str",
                description="The email subject line",
                required=True,
            ),
            Parameter(
                name="body",
                type="str",
                description="The email body content",
                required=True,
            ),
            Parameter(
                name="cc",
                type="list[str]",
                description="List of CC recipient email addresses",
                required=False,
                default=None,
            ),
        ],
    ),
    Tool(
        name="create_task",
        description="Create a new task or todo item",
        parameters=[
            Parameter(
                name="title",
                type="str",
                description="The task title",
                required=True,
            ),
            Parameter(
                name="description",
                type="str",
                description="Detailed description of the task",
                required=False,
                default="",
            ),
            Parameter(
                name="priority",
                type="str",
                description="Task priority: 'low', 'medium', or 'high'",
                required=False,
                default="medium",
            ),
            Parameter(
                name="due_date",
                type="str",
                description="Due date in YYYY-MM-DD format",
                required=False,
                default=None,
            ),
            Parameter(
                name="tags",
                type="list[str]",
                description="List of tags to categorize the task",
                required=False,
                default=None,
            ),
        ],
    ),
    Tool(
        name="query_database",
        description="Execute a database query",
        parameters=[
            Parameter(
                name="table",
                type="str",
                description="The database table to query",
                required=True,
            ),
            Parameter(
                name="columns",
                type="list[str]",
                description="List of columns to select",
                required=False,
                default=None,
            ),
            Parameter(
                name="where",
                type="str",
                description="WHERE clause conditions (e.g., \"status = 'active'\")",
                required=False,
                default=None,
            ),
            Parameter(
                name="limit",
                type="int",
                description="Maximum number of rows to return",
                required=False,
                default=100,
            ),
        ],
    ),
]
