"""Gemini LLM client with tool calling support."""

import json
import logging
from typing import Any

import tiktoken
from google import genai
from google.genai import types

from .config import LLMConfig

logger = logging.getLogger(__name__)


def _openai_tools_to_gemini(tools: list[dict]) -> list[types.Tool]:
    """Convert OpenAI-format tool schemas to Gemini Tool objects."""
    function_declarations = []
    for tool in tools:
        fn = tool["function"]
        params = fn.get("parameters", {})
        properties = {}
        for prop_name, prop_def in params.get("properties", {}).items():
            schema_type = prop_def.get("type", "STRING").upper()
            type_map = {
                "STRING": "STRING",
                "INTEGER": "INTEGER",
                "NUMBER": "NUMBER",
                "BOOLEAN": "BOOLEAN",
                "ARRAY": "ARRAY",
                "OBJECT": "OBJECT",
            }
            gemini_type = type_map.get(schema_type, "STRING")
            prop_schema = {
                "type": gemini_type,
                "description": prop_def.get("description", ""),
            }
            if gemini_type == "ARRAY" and "items" in prop_def:
                items_type = prop_def["items"].get("type", "STRING").upper()
                prop_schema["items"] = {"type": type_map.get(items_type, "STRING")}
            properties[prop_name] = prop_schema

        fd = types.FunctionDeclaration(
            name=fn["name"],
            description=fn.get("description", ""),
            parameters={
                "type": "OBJECT",
                "properties": properties,
                "required": params.get("required", []),
            } if properties else None,
        )
        function_declarations.append(fd)
    return [types.Tool(function_declarations=function_declarations)]


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = genai.Client(api_key=config.api_key)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def count_message_tokens(self, messages: list[dict]) -> int:
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if content and isinstance(content, str):
                total += len(self.tokenizer.encode(content))
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    total += len(self.tokenizer.encode(fn.get("name", "")))
                    total += len(self.tokenizer.encode(fn.get("arguments", "")))
        return total

    def _build_contents_and_config(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> tuple[list, types.GenerateContentConfig]:
        """Build Gemini contents and config from OpenAI-format messages.

        Uses raw Gemini Content objects stored in _gemini_content when available
        to preserve thought signatures for tool calling.
        """
        contents = []
        system_instruction = None

        for msg in messages:
            role = msg.get("role", "user")

            if role == "system":
                system_instruction = msg.get("content", "")
                continue

            # If we have the raw Gemini content (preserves thought signatures), use it
            raw_content = msg.get("_gemini_content")
            if raw_content is not None:
                contents.append(raw_content)
                continue

            if role == "user":
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=msg.get("content", ""))],
                ))
            elif role == "assistant":
                parts = []
                if msg.get("content"):
                    parts.append(types.Part.from_text(text=msg["content"]))
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        fn = tc["function"]
                        args = json.loads(fn["arguments"]) if fn["arguments"] else {}
                        parts.append(types.Part.from_function_call(
                            name=fn["name"], args=args,
                        ))
                if parts:
                    contents.append(types.Content(role="model", parts=parts))
            elif role == "tool":
                func_name = msg.get("_func_name", "function")
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_function_response(
                        name=func_name,
                        response={"result": msg.get("content", "")},
                    )],
                ))

        config = types.GenerateContentConfig(
            temperature=temperature if temperature is not None else self.config.temperature,
            max_output_tokens=max_tokens or self.config.max_tokens,
        )
        if system_instruction:
            config.system_instruction = system_instruction
        if tools:
            config.tools = _openai_tools_to_gemini(tools)

        return contents, config

    def _parse_response(self, response) -> dict[str, Any]:
        """Parse Gemini response into OpenAI-compatible format.

        Stores raw Content object in _gemini_content for thought signature preservation.
        """
        if not response.candidates:
            logger.warning("Gemini returned no candidates")
            return {
                "message": {"role": "assistant", "content": "回答を生成できませんでした。再度お試しください。"},
                "usage": {},
                "cost": 0.0,
                "finish_reason": "error",
            }

        candidate = response.candidates[0]
        content = candidate.content

        if content is None:
            logger.warning(f"Gemini returned empty content, finish_reason={candidate.finish_reason}")
            return {
                "message": {"role": "assistant", "content": "回答を生成できませんでした。再度お試しください。"},
                "usage": {},
                "cost": 0.0,
                "finish_reason": candidate.finish_reason.name if candidate.finish_reason else "error",
            }

        message: dict[str, Any] = {
            "role": "assistant",
            "_gemini_content": content,  # Preserve raw content for next turn
        }

        tool_calls = []
        text_parts = []

        for part in (content.parts or []):
            if part.function_call:
                fc = part.function_call
                tool_calls.append({
                    "id": f"call_{fc.name}_{len(tool_calls)}",
                    "function": {
                        "name": fc.name,
                        "arguments": json.dumps(dict(fc.args) if fc.args else {}),
                    },
                })
            elif part.text:
                text_parts.append(part.text)

        if tool_calls:
            message["tool_calls"] = tool_calls
            message["content"] = None
        else:
            message["content"] = "\n".join(text_parts)

        usage = {}
        if response.usage_metadata:
            usage = {
                "prompt_tokens": response.usage_metadata.prompt_token_count or 0,
                "completion_tokens": response.usage_metadata.candidates_token_count or 0,
                "total_tokens": response.usage_metadata.total_token_count or 0,
            }

        return {
            "message": message,
            "usage": usage,
            "cost": 0.0,
            "finish_reason": candidate.finish_reason.name if candidate.finish_reason else "stop",
        }

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        contents, config = self._build_contents_and_config(messages, tools, temperature, max_tokens)
        import time as _time
        for attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model=self.config.model,
                    contents=contents,
                    config=config,
                )
                return self._parse_response(response)
            except Exception as e:
                logger.warning(f"Gemini API error (attempt {attempt+1}/3): {e}")
                if attempt < 2:
                    _time.sleep(1 * (attempt + 1))
                else:
                    logger.error(f"Gemini API failed after 3 attempts: {e}")
                    return {
                        "message": {"role": "assistant", "content": "APIエラーが発生しました。再度お試しください。"},
                        "usage": {},
                        "cost": 0.0,
                        "finish_reason": "error",
                    }

    async def achat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        contents, config = self._build_contents_and_config(messages, tools, temperature, max_tokens)
        for attempt in range(3):
            try:
                response = await self.client.aio.models.generate_content(
                    model=self.config.model,
                    contents=contents,
                    config=config,
                )
                return self._parse_response(response)
            except Exception as e:
                logger.warning(f"Gemini API error (attempt {attempt+1}/3): {e}")
                if attempt < 2:
                    import asyncio
                    await asyncio.sleep(1 * (attempt + 1))
                else:
                    logger.error(f"Gemini API failed after 3 attempts: {e}")
                    return {
                        "message": {"role": "assistant", "content": "APIエラーが発生しました。再度お試しください。"},
                        "usage": {},
                        "cost": 0.0,
                        "finish_reason": "error",
                    }
