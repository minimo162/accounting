"""LLM client supporting OpenAI-compatible APIs (Cerebras, etc.) and Gemini."""

import json
import logging
from typing import Any

import tiktoken

from .config import LLMConfig

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        if config.provider == "gemini":
            from google import genai
            self.client = genai.Client(api_key=config.api_key)
            self._backend = "gemini"
        else:
            from openai import OpenAI, AsyncOpenAI
            self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)
            self.aclient = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
            self._backend = "openai"

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

    # ── OpenAI-compatible backend ──

    def _openai_tools(self, tools: list[dict] | None) -> list[dict] | None:
        if not tools:
            return None
        # Tools from registry already have {"type": "function", "function": {...}} format
        return tools

    def _clean_messages_for_openai(self, messages: list[dict]) -> list[dict]:
        """Remove Gemini-specific fields and ensure OpenAI compatibility."""
        cleaned = []
        for msg in messages:
            m = {k: v for k, v in msg.items() if not k.startswith("_")}
            if m.get("role") == "tool" and "tool_call_id" not in m:
                m["tool_call_id"] = "call_unknown"
            cleaned.append(m)
        return cleaned

    def _parse_openai_response(self, response) -> dict[str, Any]:
        choice = response.choices[0]
        message: dict[str, Any] = {"role": "assistant"}

        if choice.message.tool_calls:
            message["content"] = None
            message["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in choice.message.tool_calls
            ]
        else:
            message["content"] = choice.message.content or ""

        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens or 0,
                "completion_tokens": response.usage.completion_tokens or 0,
                "total_tokens": response.usage.total_tokens or 0,
            }

        return {
            "message": message,
            "usage": usage,
            "cost": 0.0,
            "finish_reason": choice.finish_reason or "stop",
        }

    def _openai_chat(self, messages, tools, temperature, max_tokens) -> dict[str, Any]:
        import time as _time
        cleaned = self._clean_messages_for_openai(messages)
        kwargs = {
            "model": self.config.model,
            "messages": cleaned,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
        }
        oa_tools = self._openai_tools(tools)
        if oa_tools:
            kwargs["tools"] = oa_tools

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(**kwargs)
                return self._parse_openai_response(response)
            except Exception as e:
                logger.warning(f"OpenAI-compat API error (attempt {attempt+1}/3): {e}")
                if attempt < 2:
                    _time.sleep(1 * (attempt + 1))
                else:
                    logger.error(f"API failed after 3 attempts: {e}")
                    return {
                        "message": {"role": "assistant", "content": "APIエラーが発生しました。再度お試しください。"},
                        "usage": {},
                        "cost": 0.0,
                        "finish_reason": "error",
                    }

    async def _openai_achat(self, messages, tools, temperature, max_tokens) -> dict[str, Any]:
        cleaned = self._clean_messages_for_openai(messages)
        kwargs = {
            "model": self.config.model,
            "messages": cleaned,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
        }
        oa_tools = self._openai_tools(tools)
        if oa_tools:
            kwargs["tools"] = oa_tools

        for attempt in range(3):
            try:
                response = await self.aclient.chat.completions.create(**kwargs)
                return self._parse_openai_response(response)
            except Exception as e:
                logger.warning(f"OpenAI-compat API error (attempt {attempt+1}/3): {e}")
                if attempt < 2:
                    import asyncio
                    await asyncio.sleep(1 * (attempt + 1))
                else:
                    logger.error(f"API failed after 3 attempts: {e}")
                    return {
                        "message": {"role": "assistant", "content": "APIエラーが発生しました。再度お試しください。"},
                        "usage": {},
                        "cost": 0.0,
                        "finish_reason": "error",
                    }

    # ── Gemini backend ──

    def _gemini_chat(self, messages, tools, temperature, max_tokens) -> dict[str, Any]:
        from google.genai import types
        contents, config = self._build_gemini_contents_and_config(messages, tools, temperature, max_tokens)
        import time as _time
        for attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model=self.config.model,
                    contents=contents,
                    config=config,
                )
                return self._parse_gemini_response(response)
            except Exception as e:
                logger.warning(f"Gemini API error (attempt {attempt+1}/3): {e}")
                if attempt < 2:
                    _time.sleep(1 * (attempt + 1))
                else:
                    return {
                        "message": {"role": "assistant", "content": "APIエラーが発生しました。再度お試しください。"},
                        "usage": {},
                        "cost": 0.0,
                        "finish_reason": "error",
                    }

    async def _gemini_achat(self, messages, tools, temperature, max_tokens) -> dict[str, Any]:
        contents, config = self._build_gemini_contents_and_config(messages, tools, temperature, max_tokens)
        for attempt in range(3):
            try:
                response = await self.client.aio.models.generate_content(
                    model=self.config.model,
                    contents=contents,
                    config=config,
                )
                return self._parse_gemini_response(response)
            except Exception as e:
                logger.warning(f"Gemini API error (attempt {attempt+1}/3): {e}")
                if attempt < 2:
                    import asyncio
                    await asyncio.sleep(1 * (attempt + 1))
                else:
                    return {
                        "message": {"role": "assistant", "content": "APIエラーが発生しました。再度お試しください。"},
                        "usage": {},
                        "cost": 0.0,
                        "finish_reason": "error",
                    }

    def _build_gemini_contents_and_config(self, messages, tools, temperature, max_tokens):
        from google.genai import types

        def _openai_tools_to_gemini(tools_list):
            fds = []
            for t in tools_list:
                fn = t if "name" in t else t.get("function", t)
                params = fn.get("parameters", {})
                props = {}
                type_map = {"STRING": "STRING", "INTEGER": "INTEGER", "NUMBER": "NUMBER",
                            "BOOLEAN": "BOOLEAN", "ARRAY": "ARRAY", "OBJECT": "OBJECT"}
                for pn, pd in params.get("properties", {}).items():
                    gt = type_map.get(pd.get("type", "STRING").upper(), "STRING")
                    ps = {"type": gt, "description": pd.get("description", "")}
                    if gt == "ARRAY" and "items" in pd:
                        ps["items"] = {"type": type_map.get(pd["items"].get("type", "STRING").upper(), "STRING")}
                    props[pn] = ps
                fds.append(types.FunctionDeclaration(
                    name=fn["name"], description=fn.get("description", ""),
                    parameters={"type": "OBJECT", "properties": props, "required": params.get("required", [])} if props else None,
                ))
            return [types.Tool(function_declarations=fds)]

        contents = []
        system_instruction = None
        for msg in messages:
            role = msg.get("role", "user")
            if role == "system":
                system_instruction = msg.get("content", "")
                continue
            raw = msg.get("_gemini_content")
            if raw is not None:
                contents.append(raw)
                continue
            if role == "user":
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text=msg.get("content", ""))]))
            elif role == "assistant":
                parts = []
                if msg.get("content"):
                    parts.append(types.Part.from_text(text=msg["content"]))
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        fn = tc["function"]
                        args = json.loads(fn["arguments"]) if fn["arguments"] else {}
                        parts.append(types.Part.from_function_call(name=fn["name"], args=args))
                if parts:
                    contents.append(types.Content(role="model", parts=parts))
            elif role == "tool":
                func_name = msg.get("_func_name", "function")
                contents.append(types.Content(role="user", parts=[
                    types.Part.from_function_response(name=func_name, response={"result": msg.get("content", "")})
                ]))

        config = types.GenerateContentConfig(
            temperature=temperature if temperature is not None else self.config.temperature,
            max_output_tokens=max_tokens or self.config.max_tokens,
        )
        if system_instruction:
            config.system_instruction = system_instruction
        if tools:
            config.tools = _openai_tools_to_gemini(tools)
        return contents, config

    def _parse_gemini_response(self, response) -> dict[str, Any]:
        if not response.candidates:
            return {"message": {"role": "assistant", "content": "回答を生成できませんでした。"}, "usage": {}, "cost": 0.0, "finish_reason": "error"}
        candidate = response.candidates[0]
        content = candidate.content
        if content is None:
            return {"message": {"role": "assistant", "content": "回答を生成できませんでした。"}, "usage": {}, "cost": 0.0, "finish_reason": "error"}

        message: dict[str, Any] = {"role": "assistant", "_gemini_content": content}
        tool_calls, text_parts = [], []
        for part in (content.parts or []):
            if part.function_call:
                fc = part.function_call
                tool_calls.append({"id": f"call_{fc.name}_{len(tool_calls)}", "function": {"name": fc.name, "arguments": json.dumps(dict(fc.args) if fc.args else {})}})
            elif part.text:
                text_parts.append(part.text)
        if tool_calls:
            message["tool_calls"] = tool_calls
            message["content"] = None
        else:
            message["content"] = "\n".join(text_parts)

        usage = {}
        if response.usage_metadata:
            usage = {"prompt_tokens": response.usage_metadata.prompt_token_count or 0, "completion_tokens": response.usage_metadata.candidates_token_count or 0, "total_tokens": response.usage_metadata.total_token_count or 0}
        return {"message": message, "usage": usage, "cost": 0.0, "finish_reason": candidate.finish_reason.name if candidate.finish_reason else "stop"}

    # ── Public interface ──

    def chat(self, messages, tools=None, temperature=None, max_tokens=None) -> dict[str, Any]:
        if self._backend == "gemini":
            return self._gemini_chat(messages, tools, temperature, max_tokens)
        return self._openai_chat(messages, tools, temperature, max_tokens)

    async def achat(self, messages, tools=None, temperature=None, max_tokens=None) -> dict[str, Any]:
        if self._backend == "gemini":
            return await self._gemini_achat(messages, tools, temperature, max_tokens)
        return await self._openai_achat(messages, tools, temperature, max_tokens)
