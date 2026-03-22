"""ReAct-style agent loop for A-RAG with full context tracking."""

import json
import logging
from typing import Any, AsyncGenerator

from .config import Config
from .context import AgentContext
from .llm import LLMClient
from .prompt import SYSTEM_PROMPT
from .tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class Agent:
    def __init__(self, config: Config, tools: ToolRegistry, chunk_map: dict[str, dict] | None = None):
        self.config = config
        self.llm = LLMClient(config.llm)
        self.tools = tools
        self.chunk_map = chunk_map or {}
        self.max_loops = config.agent.max_loops
        self.max_token_budget = config.agent.max_token_budget
        self.verbose = config.agent.verbose

    def run(self, question: str) -> dict[str, Any]:
        """Synchronous run: returns final answer with metadata."""
        context = AgentContext()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        tool_schemas = self.tools.get_schemas()
        total_cost = 0.0

        for loop_idx in range(self.max_loops):
            if self.verbose:
                logger.info(f"Loop {loop_idx + 1}/{self.max_loops}")

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
                })

        # Max loops exceeded
        answer, cost = self._force_final_answer(messages)
        total_cost += cost
        return self._build_result(answer, context, self.max_loops, "max_loops", total_cost)

    async def arun(self, question: str) -> dict[str, Any]:
        """Async run."""
        context = AgentContext()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        tool_schemas = self.tools.get_schemas()
        total_cost = 0.0

        for loop_idx in range(self.max_loops):
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
                })

        answer, cost = await self._aforce_final_answer(messages)
        total_cost += cost
        return self._build_result(answer, context, self.max_loops, "max_loops", total_cost)

    async def arun_stream(self, question: str) -> AsyncGenerator[dict, None]:
        """Async streaming run - yields events for SSE."""
        context = AgentContext()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        tool_schemas = self.tools.get_schemas()
        total_cost = 0.0

        for loop_idx in range(self.max_loops):
            yield {"type": "status", "data": f"検索中... (ステップ {loop_idx + 1})"}

            # Token budget check
            current_tokens = self.llm.count_message_tokens(messages)
            if current_tokens > self.max_token_budget:
                answer, cost = await self._aforce_final_answer(messages)
                total_cost += cost
                yield {"type": "answer", "data": answer}
                yield {
                    "type": "done",
                    "data": {
                        "loops": loop_idx + 1,
                        "stop_reason": "budget_exceeded",
                        **context.get_summary(),
                        "total_cost": total_cost,
                        "references": self._get_referenced_chunks(context),
                    },
                }
                return

            response = await self.llm.achat(messages=messages, tools=tool_schemas)
            message = response["message"]
            total_cost += response.get("cost", 0.0)
            messages.append(message)

            tool_calls = message.get("tool_calls")
            if not tool_calls:
                # Final answer
                answer = message.get("content", "")
                yield {"type": "answer", "data": answer}
                yield {
                    "type": "done",
                    "data": {
                        "loops": loop_idx + 1,
                        "stop_reason": "natural",
                        **context.get_summary(),
                        "total_cost": total_cost,
                        "references": self._get_referenced_chunks(context),
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
                })

        # Force final answer
        answer, cost = await self._aforce_final_answer(messages)
        total_cost += cost
        yield {"type": "answer", "data": answer}
        yield {
            "type": "done",
            "data": {
                "loops": self.max_loops,
                "stop_reason": "max_loops",
                **context.get_summary(),
                "total_cost": total_cost,
                "references": self._get_referenced_chunks(context),
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

    def _get_referenced_chunks(self, context: AgentContext) -> list[dict]:
        """Get full text of all chunks that were read during the query."""
        refs = []
        for chunk_id in sorted(context.read_chunk_ids, key=lambda x: int(x) if x.isdigit() else 0):
            chunk = self.chunk_map.get(chunk_id)
            if chunk:
                refs.append({
                    "id": chunk_id,
                    "source": chunk.get("source", ""),
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
            "answer": answer,
            "loops": loops,
            "stop_reason": stop_reason,
            "total_cost": total_cost,
            **context.get_summary(),
            "trajectory": context.trajectory,
            "references": self._get_referenced_chunks(context),
        }
