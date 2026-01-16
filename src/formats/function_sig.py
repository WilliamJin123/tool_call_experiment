"""Function signature format for tool calls."""

import ast
import re
from typing import Any

from src.formats.base import BaseFormat, ToolCall
from src.tools.base import Tool


class FunctionSigFormat(BaseFormat):
    """
    Function signature format handler (Python/TypeScript style).

    Expected format:
    tool_name(param1="value1", param2="value2")
    """

    @property
    def name(self) -> str:
        return "function_sig"

    def generate_system_prompt(self, tools: list[Tool]) -> str:
        tools_text = "\n\n".join(self._format_as_function(t) for t in tools)

        return f"""You are a helpful assistant with access to the following functions:

{tools_text}

When you need to call a function, respond with the function call in Python syntax:
```
function_name(param1="value1", param2=123, param3=True)
```

IMPORTANT:
- Use keyword arguments (param=value) for all parameters
- String values must be quoted
- Numbers and booleans should not be quoted
- For lists, use Python list syntax: [1, 2, 3] or ["a", "b", "c"]
- For dicts, use Python dict syntax: {{"key": "value"}}
- Use exact parameter names from the function definitions

You may call multiple functions, each on its own line.
Only include function calls when you want to use a tool. For regular responses, just reply normally."""

    def _format_as_function(self, tool: Tool) -> str:
        """Format a tool as a function signature."""
        params = []
        for p in tool.parameters:
            param_str = p.name
            if p.type:
                param_str += f": {p.type}"
            if p.default is not None:
                param_str += f" = {repr(p.default)}"
            params.append(param_str)

        params_str = ", ".join(params)
        return f"""def {tool.name}({params_str})
    \"\"\"{tool.description}\"\"\"
"""

    def parse_response(self, response: str) -> list[ToolCall]:
        """Parse function-style tool calls from response."""
        tool_calls = []
        seen_raw_texts = set()

        # Pattern for function calls: name(args)
        # Match function calls, being careful with nested parentheses
        func_pattern = r"(\w+)\s*\(([^)]*(?:\([^)]*\)[^)]*)*)\)"

        # Also check code blocks
        code_blocks = re.findall(r"```(?:python)?\s*(.*?)\s*```", response, re.DOTALL)
        text_to_search = response + "\n" + "\n".join(code_blocks)

        matches = re.findall(func_pattern, text_to_search)

        for func_name, args_text in matches:
            # Skip common Python built-ins and non-tool calls
            if func_name in ("print", "len", "str", "int", "float", "bool", "list", "dict", "set", "tuple"):
                continue

            raw_text = f"{func_name}({args_text})"

            # Skip duplicates
            if raw_text in seen_raw_texts:
                continue
            seen_raw_texts.add(raw_text)

            try:
                arguments = self._parse_arguments(args_text)
                tool_calls.append(
                    ToolCall(
                        name=func_name,
                        arguments=arguments,
                        raw_text=raw_text,
                    )
                )
            except Exception:
                # If parsing fails, try a simpler approach
                arguments = self._parse_arguments_simple(args_text)
                if arguments is not None:
                    tool_calls.append(
                        ToolCall(
                            name=func_name,
                            arguments=arguments,
                            raw_text=raw_text,
                        )
                    )

        return tool_calls

    def _parse_arguments(self, args_text: str) -> dict[str, Any]:
        """Parse function arguments using AST."""
        if not args_text.strip():
            return {}

        # Create a fake function call for AST parsing
        fake_code = f"f({args_text})"

        try:
            tree = ast.parse(fake_code, mode="eval")
            call = tree.body

            if not isinstance(call, ast.Call):
                return {}

            arguments = {}

            # Handle keyword arguments
            for keyword in call.keywords:
                if keyword.arg is not None:
                    arguments[keyword.arg] = self._ast_to_value(keyword.value)

            # Handle positional arguments (less common but possible)
            # We can't map these without parameter names, so skip them

            return arguments
        except SyntaxError:
            raise ValueError(f"Could not parse arguments: {args_text}")

    def _ast_to_value(self, node: ast.expr) -> Any:
        """Convert AST node to Python value."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Str):  # Python < 3.8 compatibility
            return node.s
        elif isinstance(node, ast.Num):  # Python < 3.8 compatibility
            return node.n
        elif isinstance(node, ast.NameConstant):  # Python < 3.8 compatibility
            return node.value
        elif isinstance(node, ast.List):
            return [self._ast_to_value(elt) for elt in node.elts]
        elif isinstance(node, ast.Dict):
            return {
                self._ast_to_value(k): self._ast_to_value(v) for k, v in zip(node.keys, node.values) if k is not None
            }
        elif isinstance(node, ast.Tuple):
            return tuple(self._ast_to_value(elt) for elt in node.elts)
        elif isinstance(node, ast.Set):
            return {self._ast_to_value(elt) for elt in node.elts}
        elif isinstance(node, ast.Name):
            # Handle common names
            name = node.id
            if name == "True":
                return True
            elif name == "False":
                return False
            elif name == "None":
                return None
            else:
                return name  # Return as string
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            # Handle negative numbers
            return -self._ast_to_value(node.operand)
        else:
            # For complex expressions, return as string representation
            return ast.unparse(node) if hasattr(ast, "unparse") else str(node)

    def _parse_arguments_simple(self, args_text: str) -> dict[str, Any] | None:
        """Simple regex-based argument parsing as fallback."""
        if not args_text.strip():
            return {}

        arguments = {}

        # Pattern: param_name=value
        # This is simpler but less accurate than AST
        pattern = r"(\w+)\s*=\s*"
        parts = re.split(pattern, args_text)

        # parts alternates: [before_first_param, param1, value1, param2, value2, ...]
        if len(parts) < 3:
            return None

        i = 1
        while i < len(parts) - 1:
            param_name = parts[i]
            value_text = parts[i + 1].strip()

            # Remove trailing comma if present
            if value_text.endswith(","):
                value_text = value_text[:-1].strip()

            arguments[param_name] = self._parse_simple_value(value_text)
            i += 2

        return arguments

    def _parse_simple_value(self, value_str: str) -> Any:
        """Parse a simple value string."""
        value_str = value_str.strip()

        # Handle quoted strings
        if (value_str.startswith('"') and value_str.endswith('"')) or (
            value_str.startswith("'") and value_str.endswith("'")
        ):
            return value_str[1:-1]

        # Try to evaluate as Python literal
        try:
            return ast.literal_eval(value_str)
        except (ValueError, SyntaxError):
            pass

        # Return as string
        return value_str
