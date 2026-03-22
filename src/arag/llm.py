"""OpenAI-compatible LLM client for Cerebras with cost tracking."""

import json
import logging
from typing import Any

import httpx
import tiktoken

from .config import LLMConfig

logger = logging.getLogger(__name__)

# Pricing per 1M tokens (input, output)
PRICING: dict[str, tuple[float, float]] = {
    "gpt-oss-120b": (0.60, 0.60),
    "default": (1.0, 1.0),
}


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = httpx.Client(timeout=300)
        # Use cl100k_base as a reasonable tokenizer for multilingual content
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        return len(self.tokenizer.encode(text))

    def count_message_tokens(self, messages: list[dict]) -> int:
        """Count total tokens across all messages."""
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if content and isinstance(content, str):
                total += len(self.tokenizer.encode(content))
            # Account for tool_calls in assistant messages
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    total += len(self.tokenizer.encode(fn.get("name", "")))
                    total += len(self.tokenizer.encode(fn.get("arguments", "")))
        return total

    def calculate_cost(self, usage: dict) -> float:
        """Calculate cost from usage dict."""
        model = self.config.model
        input_price, output_price = PRICING.get(model, PRICING["default"])
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        return (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        url = f"{self.config.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        response = self.client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

        choice = data["choices"][0]
        message = choice["message"]
        usage = data.get("usage", {})
        cost = self.calculate_cost(usage)

        return {
            "message": message,
            "usage": usage,
            "cost": cost,
            "finish_reason": choice.get("finish_reason"),
        }

    async def achat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        url = f"{self.config.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        choice = data["choices"][0]
        message = choice["message"]
        usage = data.get("usage", {})
        cost = self.calculate_cost(usage)

        return {
            "message": message,
            "usage": usage,
            "cost": cost,
            "finish_reason": choice.get("finish_reason"),
        }

    async def astream_chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        """Async streaming chat - yields content deltas."""
        url = f"{self.config.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        async with httpx.AsyncClient(timeout=300) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content")
                            if content:
                                yield content
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue

    def stream_chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        """Streaming chat for SSE responses."""
        url = f"{self.config.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        with self.client.stream("POST", url, headers=headers, json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        yield json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
