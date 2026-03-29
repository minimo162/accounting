"""Tool registry for managing retrieval tools."""

from typing import Any

from .base import BaseTool
from ..context import AgentContext
from ..query_rewrite import QueryExpander


class ToolRegistry:
    _FOLLOWUP_FILTERED_TOOL_NAMES = {"keyword_search", "semantic_search"}
    _EXACT_FILTERED_TOOL_NAMES = {"read_document"}
    _EXACT_FOLLOWUP_FILTERED_TOOL_NAMES = {"hybrid_search"}

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):
        self._tools[tool.name] = tool

    def get_schemas(self, context: AgentContext | None = None) -> list[dict[str, Any]]:
        hide_followup_searches = bool(
            context and any(log.tool_name == "hybrid_search" for log in context.retrieval_logs)
        )
        is_exact_query = bool(context and QueryExpander.is_exact_query(context.question))
        hide_exact_tools = is_exact_query
        hide_exact_followup_searches = bool(
            context and is_exact_query and any(log.tool_name == "hybrid_search" for log in context.retrieval_logs)
        )
        return [
            {"type": "function", "function": t.get_schema()}
            for t in self._tools.values()
            if not (hide_followup_searches and t.name in self._FOLLOWUP_FILTERED_TOOL_NAMES)
            and not (hide_exact_tools and t.name in self._EXACT_FILTERED_TOOL_NAMES)
            and not (hide_exact_followup_searches and t.name in self._EXACT_FOLLOWUP_FILTERED_TOOL_NAMES)
        ]

    def execute(self, tool_name: str, context: AgentContext, **kwargs) -> tuple[str, dict]:
        if tool_name not in self._tools:
            return f"Unknown tool: {tool_name}", {"error": f"Unknown tool: {tool_name}"}
        return self._tools[tool_name].execute(context, **kwargs)
