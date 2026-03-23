"""ReAct-style agent loop for A-RAG with full context tracking."""

import json
import logging
import re
from typing import Any, AsyncGenerator

from .config import Config
from .context import AgentContext
from .llm import LLMClient
from .prompt import SYSTEM_PROMPT
from .tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


WRAP_UP_HINT = (
    "【システム通知】検索ステップが多くなっています。"
    "これまでに収集した情報で十分回答できる場合は、追加検索せずに最終回答を提供してください。"
    "完全な情報が得られなくても、現在の情報に基づいて回答し、不足部分はその旨を明記してください。"
)


class Agent:
    NUDGE_AT_LOOP = 8  # After this many loops, hint the LLM to wrap up

    def __init__(self, config: Config, tools: ToolRegistry, chunk_map: dict[str, dict] | None = None):
        self.config = config
        self.llm = LLMClient(config.llm)
        self.tools = tools
        self.chunk_map = chunk_map or {}
        self.max_loops = config.agent.max_loops
        self.max_token_budget = config.agent.max_token_budget
        self.verbose = config.agent.verbose

    def _maybe_nudge(self, messages: list[dict], loop_idx: int):
        """Inject a wrap-up hint if we've been searching too long."""
        if loop_idx == self.NUDGE_AT_LOOP:
            messages.append({"role": "user", "content": WRAP_UP_HINT})

    def _build_initial_messages(self, question: str, history: list[dict] | None = None) -> list[dict]:
        """Build initial messages with optional conversation history."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if history:
            # Include recent conversation turns as context
            # Keep last 2 turns to avoid token overflow
            recent = history[-2:]
            context_text = "以下は直前の会話です。ユーザーの新しい質問が前の話題の続きかどうかを判断し、続きであれば前の文脈を踏まえて検索・回答してください。全く別の話題であれば、独立した質問として扱ってください。\n\n"
            for turn in recent:
                q = turn["question"] if isinstance(turn, dict) else turn.question
                a = turn["answer"] if isinstance(turn, dict) else turn.answer
                context_text += f"ユーザー: {q}\nアシスタント: {a[:500]}\n\n"
            messages.append({"role": "user", "content": context_text})
            messages.append({"role": "assistant", "content": "理解しました。前の会話の文脈を踏まえて、新しい質問に回答します。"})

        messages.append({"role": "user", "content": question})
        return messages

    def run(self, question: str, history: list[dict] | None = None) -> dict[str, Any]:
        """Synchronous run: returns final answer with metadata."""
        context = AgentContext()
        messages = self._build_initial_messages(question, history)
        tool_schemas = self.tools.get_schemas()
        total_cost = 0.0

        for loop_idx in range(self.max_loops):
            if self.verbose:
                logger.info(f"Loop {loop_idx + 1}/{self.max_loops}")

            self._maybe_nudge(messages, loop_idx)

            # Token budget check
            current_tokens = self.llm.count_message_tokens(messages)
            if current_tokens > self.max_token_budget:
                logger.info(f"Token budget exceeded: {current_tokens} > {self.max_token_budget}")
                answer, cost = self._force_final_answer(messages)
                total_cost += cost
                return self._build_result(
                    answer, context, loop_idx + 1, "budget_exceeded", total_cost
                )

            response = self.llm.chat(messages=messages, tools=tool_schemas)
            message = response["message"]
            total_cost += response.get("cost", 0.0)
            messages.append(message)

            tool_calls = message.get("tool_calls")
            if not tool_calls:
                return self._build_result(
                    message.get("content", ""), context, loop_idx + 1, "natural", total_cost
                )

            # Execute tool calls
            for tc in tool_calls:
                func_name = tc["function"]["name"]
                try:
                    func_args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    func_args = {}
                    logger.warning(f"Failed to parse tool arguments: {tc['function']['arguments']}")

                if self.verbose:
                    logger.info(f"  Tool: {func_name}({func_args})")

                result_text, tool_log = self.tools.execute(func_name, context, **func_args)

                # Trajectory logging
                context.add_trajectory_entry(
                    loop=loop_idx + 1,
                    tool_name=func_name,
                    arguments=func_args,
                    tool_result=result_text,
                    tool_log=tool_log,
                )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result_text,
                    "_func_name": func_name,
                })

        # Max loops exceeded
        answer, cost = self._force_final_answer(messages)
        total_cost += cost
        return self._build_result(answer, context, self.max_loops, "max_loops", total_cost)

    async def arun(self, question: str, history: list[dict] | None = None) -> dict[str, Any]:
        """Async run."""
        context = AgentContext()
        messages = self._build_initial_messages(question, history)
        tool_schemas = self.tools.get_schemas()
        total_cost = 0.0

        for loop_idx in range(self.max_loops):
            self._maybe_nudge(messages, loop_idx)

            # Token budget check
            current_tokens = self.llm.count_message_tokens(messages)
            if current_tokens > self.max_token_budget:
                answer, cost = await self._aforce_final_answer(messages)
                total_cost += cost
                return self._build_result(
                    answer, context, loop_idx + 1, "budget_exceeded", total_cost
                )

            response = await self.llm.achat(messages=messages, tools=tool_schemas)
            message = response["message"]
            total_cost += response.get("cost", 0.0)
            messages.append(message)

            tool_calls = message.get("tool_calls")
            if not tool_calls:
                return self._build_result(
                    message.get("content", ""), context, loop_idx + 1, "natural", total_cost
                )

            for tc in tool_calls:
                func_name = tc["function"]["name"]
                try:
                    func_args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    func_args = {}

                result_text, tool_log = self.tools.execute(func_name, context, **func_args)

                context.add_trajectory_entry(
                    loop=loop_idx + 1,
                    tool_name=func_name,
                    arguments=func_args,
                    tool_result=result_text,
                    tool_log=tool_log,
                )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result_text,
                    "_func_name": func_name,
                })

        answer, cost = await self._aforce_final_answer(messages)
        total_cost += cost
        return self._build_result(answer, context, self.max_loops, "max_loops", total_cost)

    async def arun_stream(self, question: str, history: list[dict] | None = None) -> AsyncGenerator[dict, None]:
        """Async streaming run - yields events for SSE.

        Tool-calling loops use non-streaming (need full response for tool parsing).
        Final answer is streamed token-by-token via answer_delta events.
        """
        context = AgentContext()
        messages = self._build_initial_messages(question, history)
        tool_schemas = self.tools.get_schemas()
        total_cost = 0.0

        for loop_idx in range(self.max_loops):
            yield {"type": "status", "data": f"検索中... (ステップ {loop_idx + 1})"}

            self._maybe_nudge(messages, loop_idx)

            # Token budget check
            current_tokens = self.llm.count_message_tokens(messages)
            if current_tokens > self.max_token_budget:
                yield {"type": "status", "data": "回答を生成中..."}
                async for event in self._astream_final_answer(messages, context, loop_idx + 1, "budget_exceeded", total_cost):
                    yield event
                return

            response = await self.llm.achat(messages=messages, tools=tool_schemas)
            message = response["message"]
            total_cost += response.get("cost", 0.0)
            messages.append(message)

            tool_calls = message.get("tool_calls")
            if not tool_calls:
                # LLM produced final answer - strip chunk refs and simulate streaming
                answer = self._strip_chunk_refs(message.get("content", ""))
                yield {"type": "status", "data": "回答を生成中..."}
                chunk_size = 8
                for i in range(0, len(answer), chunk_size):
                    yield {"type": "answer_delta", "data": answer[i:i + chunk_size]}
                yield {"type": "answer_done", "data": answer}
                # Send references individually to avoid oversized SSE events
                for ref in self._get_referenced_chunks(context):
                    yield {"type": "reference", "data": ref}
                yield {
                    "type": "done",
                    "data": {
                        "loops": loop_idx + 1,
                        "stop_reason": "natural",
                        **context.get_summary(),
                        "total_cost": total_cost,
                    },
                }
                return

            for tc in tool_calls:
                func_name = tc["function"]["name"]
                try:
                    func_args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    func_args = {}

                yield {
                    "type": "tool_call",
                    "data": {"tool": func_name, "args": func_args},
                }
                result_text, tool_log = self.tools.execute(func_name, context, **func_args)

                context.add_trajectory_entry(
                    loop=loop_idx + 1,
                    tool_name=func_name,
                    arguments=func_args,
                    tool_result=result_text,
                    tool_log=tool_log,
                )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result_text,
                    "_func_name": func_name,
                })

        # Max loops - force final answer with streaming
        yield {"type": "status", "data": "回答を生成中..."}
        async for event in self._astream_final_answer(messages, context, self.max_loops, "max_loops", total_cost):
            yield event

    async def _astream_final_answer(
        self,
        messages: list[dict],
        context: AgentContext,
        loops: int,
        stop_reason: str,
        total_cost: float,
    ) -> AsyncGenerator[dict, None]:
        """Force a final answer and simulate streaming output."""
        answer, cost = await self._aforce_final_answer(messages)
        answer = self._strip_chunk_refs(answer)
        total_cost += cost

        chunk_size = 8
        for i in range(0, len(answer), chunk_size):
            yield {"type": "answer_delta", "data": answer[i:i + chunk_size]}
        yield {"type": "answer_done", "data": answer}
        for ref in self._get_referenced_chunks(context):
            yield {"type": "reference", "data": ref}
        yield {
            "type": "done",
            "data": {
                "loops": loops,
                "stop_reason": stop_reason,
                **context.get_summary(),
                "total_cost": total_cost,
            },
        }

    def _force_final_answer(self, messages: list[dict]) -> tuple[str, float]:
        """Force the LLM to produce a final answer without tool calls."""
        force_prompt = (
            "これ以上ツールを呼び出さないでください。"
            "これまでに収集した情報に基づいて、最終的な回答を提供してください。"
            "情報が不十分な場合は、その旨を明記した上で、得られた情報の範囲で回答してください。"
            "推測は避け、文書に基づいた回答のみを行ってください。"
        )
        messages_copy = messages + [{"role": "user", "content": force_prompt}]
        response = self.llm.chat(messages=messages_copy, tools=None, temperature=0.0)
        return response["message"].get("content", ""), response.get("cost", 0.0)

    async def _aforce_final_answer(self, messages: list[dict]) -> tuple[str, float]:
        """Async force final answer."""
        force_prompt = (
            "これ以上ツールを呼び出さないでください。"
            "これまでに収集した情報に基づいて、最終的な回答を提供してください。"
            "情報が不十分な場合は、その旨を明記した上で、得られた情報の範囲で回答してください。"
            "推測は避け、文書に基づいた回答のみを行ってください。"
        )
        messages_copy = messages + [{"role": "user", "content": force_prompt}]
        response = await self.llm.achat(messages=messages_copy, tools=None, temperature=0.0)
        return response["message"].get("content", ""), response.get("cost", 0.0)

    @staticmethod
    def _strip_chunk_refs(text: str) -> str:
        """Remove any remaining Chunk ID references from the answer."""
        # Remove patterns like: 【Chunk 123】, [Chunk 123], (Chunk 123), Chunk123, Chunk 123
        text = re.sub(r'[【\[\(]\s*Chunk\s*\d+\s*[】\]\)]', '', text)
        text = re.sub(r'\bChunk\s*\d+\b', '', text)
        # Clean up any resulting double spaces or orphaned punctuation
        text = re.sub(r'  +', ' ', text)
        text = re.sub(r' ([。、，,.])', r'\1', text)
        return text

    def _get_referenced_chunks(self, context: AgentContext) -> list[dict]:
        """Get full text of all chunks that were referenced (read or found via search)."""
        # Collect all chunk IDs from read_chunks AND search retrieval logs
        all_chunk_ids: set[str] = set(context.read_chunk_ids)
        for log in context.retrieval_logs:
            chunk_ids = log.metadata.get("chunk_ids", [])
            all_chunk_ids.update(chunk_ids)

        refs = []
        seen_sources = set()
        for chunk_id in sorted(all_chunk_ids, key=lambda x: int(x) if x.isdigit() else 0):
            chunk = self.chunk_map.get(chunk_id)
            if chunk:
                source = chunk.get("source", "")
                # Deduplicate by source to avoid showing multiple pages of same doc
                # unless they were explicitly read
                if chunk_id in context.read_chunk_ids or source not in seen_sources:
                    seen_sources.add(source)
                    refs.append({
                        "id": chunk_id,
                        "source": source,
                        "text": chunk["text"],
                    })
        return refs

    def _build_result(
        self,
        answer: str,
        context: AgentContext,
        loops: int,
        stop_reason: str,
        total_cost: float,
    ) -> dict[str, Any]:
        return {
            "answer": self._strip_chunk_refs(answer),
            "loops": loops,
            "stop_reason": stop_reason,
            "total_cost": total_cost,
            **context.get_summary(),
            "trajectory": context.trajectory,
            "references": self._get_referenced_chunks(context),
        }
