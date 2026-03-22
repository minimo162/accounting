"""Base tool interface for A-RAG retrieval tools."""

from abc import ABC, abstractmethod
from typing import Any

from ..context import AgentContext


class BaseTool(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def get_schema(self) -> dict[str, Any]:
        """Return OpenAI function-calling tool schema."""
        ...

    @abstractmethod
    def execute(self, context: AgentContext, **kwargs) -> tuple[str, dict]:
        """Execute the tool, return (result_text, log_dict)."""
        ...
