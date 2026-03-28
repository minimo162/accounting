"""Tool registry for managing retrieval tools."""

from typing import Any

from .base import BaseTool
from ..context import AgentContext


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):
        self._tools[tool.name] = tool

    def get_schemas(self) -> list[dict[str, Any]]:
        return [
            {"type": "function", "function": t.get_schema()}
            for t in self._tools.values()
        ]

    def execute(self, tool_name: str, context: AgentContext, **kwargs) -> tuple[str, dict]:
        if tool_name not in self._tools:
            return f"Unknown tool: {tool_name}", {"error": f"Unknown tool: {tool_name}"}
        return self._tools[tool_name].execute(context, **kwargs)
