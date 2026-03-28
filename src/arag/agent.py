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

    def __init__(self, config: Config, tools: ToolRegistry, chunk_map: dict[str, dict] | None = None, pdf_sources: dict[str, str] | None = None):
        self.config = config
        self.llm = LLMClient(config.llm)
        self.tools = tools
        self.chunk_map = chunk_map or {}
        self.pdf_sources = pdf_sources or {}
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
            status_msg = "調査中..." if loop_idx == 0 else f"調査中... (ステップ {loop_idx + 1})"
            yield {"type": "status", "data": status_msg}

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
                answer = self._sanitize_answer(message.get("content", ""))
                yield {"type": "status", "data": "回答を生成中..."}
                chunk_size = 8
                for i in range(0, len(answer), chunk_size):
                    yield {"type": "answer_delta", "data": answer[i:i + chunk_size]}
                yield {"type": "answer_done", "data": answer}
                refs, source_url_map = self._get_referenced_chunks(context)
                for ref in refs:
                    yield {"type": "reference", "data": ref}
                summary = context.get_summary()
                summary["read_chunk_count"] = summary.get("chunks_read_count", 0)
                summary["chunks_read_count"] = self._count_cited_references(answer)
                yield {
                    "type": "done",
                    "data": {
                        "loops": loop_idx + 1,
                        "stop_reason": "natural",
                        "source_url_map": source_url_map,
                        **summary,
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
        answer = self._sanitize_answer(answer)
        total_cost += cost

        chunk_size = 8
        for i in range(0, len(answer), chunk_size):
            yield {"type": "answer_delta", "data": answer[i:i + chunk_size]}
        yield {"type": "answer_done", "data": answer}
        refs, source_url_map = self._get_referenced_chunks(context)
        for ref in refs:
            yield {"type": "reference", "data": ref}
        yield {
            "type": "done",
            "data": {
                "loops": loops,
                "stop_reason": stop_reason,
                "source_url_map": source_url_map,
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

    # Chunk ID patterns shared across ref-handling methods
    _CHUNK_SUFFIX = r':(?:p|c)?\d+(?:-\d+)*'
    _FILENAME = r'[\w\-]+\.(?:pdf|xml)'

    @classmethod
    def _number_chunk_refs(cls, text: str) -> tuple[str, list[str]]:
        """Replace inline chunk ID refs with [N] citation markers.

        Scans the answer for bracketed chunk ID references and converts them to
        sequential `[1]`, `[2]` … markers.  Returns the modified text and an
        ordered list of unique chunk IDs (index 0 → citation [1], etc.).
        """
        # Single ID in brackets: （filename.pdf:p5）→ [N]
        single = re.compile(
            rf'[（\(\[【]\s*({cls._FILENAME}{cls._CHUNK_SUFFIX})\s*[）\)\]】]'
        )
        # Multiple IDs in one bracket: （id1、id2）→ [N][M]
        multi = re.compile(
            rf'[（\(\[【]\s*({cls._FILENAME}{cls._CHUNK_SUFFIX}(?:\s*[、，;]\s*{cls._FILENAME}{cls._CHUNK_SUFFIX})+)\s*[）\)\]】]'
        )
        参照_form = re.compile(
            rf'参照:\s*({cls._FILENAME}{cls._CHUNK_SUFFIX})\s*;?'
        )

        ordered: list[str] = []
        num_map: dict[str, int] = {}

        def _num(cid: str) -> str:
            if cid not in num_map:
                ordered.append(cid)
                num_map[cid] = len(ordered)
            return f'[{num_map[cid]}]'

        def replace_multi(m: re.Match) -> str:
            ids = re.findall(rf'{cls._FILENAME}{cls._CHUNK_SUFFIX}', m.group(1))
            return ''.join(_num(cid) for cid in ids)

        def replace_single(m: re.Match) -> str:
            return _num(m.group(1))

        text = multi.sub(replace_multi, text)
        text = single.sub(replace_single, text)
        text = 参照_form.sub(replace_single, text)
        return text, ordered

    @classmethod
    def _strip_chunk_refs(cls, text: str) -> str:
        """Remove any remaining un-numbered chunk ID references from the answer."""
        _s = cls._CHUNK_SUFFIX
        _f = cls._FILENAME

        # Any leftover bracketed forms
        text = re.sub(rf'[（\(\[【]\s*{_f}{_s}\s*[）\)\]】]', '', text)
        # Bare Chunk keyword refs
        text = re.sub(r'[【\[\(]\s*Chunk\s*[\w\-:\.]+\s*[】\]\)]', '', text)
        text = re.sub(r'\bChunk\s*\d+\b', '', text)
        # Bare filename:page refs
        text = re.sub(rf'\b{_f}{_s}\b', '', text)
        # Orphaned 参照:
        text = re.sub(r'参照:\s*;?', '', text)
        # Orphaned bare page-number refs like (-40), （-39）, (p40), （-40-41） left by model
        # Handle ASCII and full-width parens, various dash/minus chars, ASCII and full-width digits
        # All dash/minus variants including U+2011 NON-BREAKING HYPHEN used by some models
        _dash = r'[-－−‐\u2011–—\u2212]?'
        _any_dash = r'[-－−‐\u2011–—\u2212]'
        _digs = r'[0-9０-９]+'
        # Also covers square-bracket forms like [-21] or [p5]
        text = re.sub(
            rf'[（(\[]\s*{_dash}\s*p?{_digs}(?:\s*{_any_dash}\s*p?{_digs})*\s*[）)\]]',
            '', text
        )
        # Strip trailing Japanese/ASCII comma before closing paren: （第25項、） → （第25項）
        text = re.sub(r'([（(][^）)\n]{2,})[、，,]\s*([）)])', r'\1\2', text)
        # Clean up residual empty or punctuation-only parentheses like （、）（;）（ ）
        text = re.sub(r'[（(][\s、;,・]*[）)]', '', text)
        text = re.sub(r'  +', ' ', text)
        text = re.sub(r'\s+([。、，,.])', r'\1', text)
        return text

    @staticmethod
    def _strip_reasoning_preamble(text: str) -> str:
        """Remove model internal reasoning/tool-call leakage before the actual answer.

        Some LLMs emit planning text like "We have many chunks. Need to read them."
        or raw JSON tool-call arguments as part of the final text content.
        Detect these patterns and strip everything up to the start of the real answer.
        """
        # Remove JSON blobs that look like tool call arguments
        text = re.sub(r'\{[^{}]*"chunk_ids"[^{}]*\}', '', text, flags=re.DOTALL)
        text = re.sub(r'\{[^{}]*"query"[^{}]*\}', '', text, flags=re.DOTALL)
        text = re.sub(r'\{[^{}]*"ids"[^{}]*\}', '', text, flags=re.DOTALL)
        # Remove common English reasoning phrases emitted by the LLM
        text = re.sub(r'We(?:\s+have|\s+need\s+to|\s+got)[^\n.]*\.', '', text)
        text = re.sub(r'Need\s+to\s+\w+[^\n.]*\.', '', text)
        text = re.sub(r'Let\s+me\s+\w+[^\n.]*\.', '', text)
        # If there's still a block of ASCII-only preamble before the Japanese answer,
        # find the first line with Japanese text or a ## heading and discard everything before it.
        lines = text.splitlines()
        japanese_start = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith('##') or re.search(r'[\u3040-\u9fff\u30a0-\u30ff\u4e00-\u9fff]', stripped):
                japanese_start = i
                break
        if japanese_start is not None and japanese_start > 0:
            preamble = '\n'.join(lines[:japanese_start])
            # Only strip preamble if it looks like leaked reasoning (contains JSON chars or English words)
            if re.search(r'[{}\[\]]', preamble) or re.search(r'\b[A-Za-z]{4,}\b', preamble):
                text = '\n'.join(lines[japanese_start:])
        # Handle the case where ASCII planning text and Japanese answer are on the same line
        # e.g. "We have the content.## 減損の注記様式" or "We have the content.減損..."
        inline_match = re.match(r'^[A-Za-z\s.,!?:;\[\]{}"\']+?(##\s|\S*[\u3040-\u9fff\u30a0-\u30ff\u4e00-\u9fff])', text)
        if inline_match:
            text = text[inline_match.start(1):]
        return text

    @classmethod
    def _sanitize_answer(cls, text: str, *, number_refs: bool = False) -> tuple[str, list[str]] | str:
        """Normalize answer formatting before returning it to clients.

        When ``number_refs=True`` returns ``(text, ordered_chunk_ids)`` so
        callers can build a numbered reference list matched to inline [N] markers.
        Otherwise returns just the text string (backward-compatible default).
        """
        text = cls._strip_reasoning_preamble(text)
        cited_ids: list[str] = []
        if number_refs:
            text, cited_ids = cls._number_chunk_refs(text)
        text = cls._strip_chunk_refs(text)

        heading_only_bullet = re.compile(
            r"^-\s+\*\*(?P<label>[^*]+)\*\*(?P<suffix>（[^）]+）)?$"
        )
        normalized_lines = []
        previous_blank = False
        for raw_line in text.splitlines():
            line = re.sub(r'^[ \t]+(?=-\s)', '', raw_line)
            line = re.sub(r'[ \t]+$', '', line)
            match = heading_only_bullet.match(line)
            if match:
                label = match.group("label").strip()
                suffix = (match.group("suffix") or "").strip()
                line = f"## {label}{suffix}".rstrip()
            is_blank = not line.strip()
            if is_blank and previous_blank:
                continue
            normalized_lines.append("" if is_blank else line)
            previous_blank = is_blank

        result = "\n".join(normalized_lines).strip()
        if number_refs:
            return result, cited_ids
        return result

    def _get_referenced_chunks(self, context: AgentContext, cited_ids: list[str] | None = None) -> list[dict]:
        """Get chunks for references panel in citation order.

        Priority:
        1. Inline-cited chunk IDs (from [N] markers in answer) — in citation order
        2. Explicitly read via read_chunk tool
        3. Fallback: top search results
        """
        seen: set[str] = set()
        ids_to_show: list[str] = []

        # 1. Inline citations from answer body (determines [N] numbering)
        for cid in (cited_ids or []):
            if cid not in seen:
                ids_to_show.append(cid)
                seen.add(cid)

        # 2. read_chunk calls not already included
        for cid in context.read_chunk_ids:
            if cid not in seen:
                ids_to_show.append(cid)
                seen.add(cid)

        # 3. Search fallback when nothing was explicitly cited/read
        if not ids_to_show:
            for cid in context.searched_chunk_ids[:10]:
                if cid not in seen:
                    ids_to_show.append(cid)
                    seen.add(cid)

        refs = []
        seen_parents: set[str] = set()
        source_url_map: dict[str, str] = {}  # "第13号" -> base PDF URL
        for chunk_id in ids_to_show:
            chunk = self.chunk_map.get(chunk_id)
            if not chunk:
                continue
            parent_id = chunk.get("parent_id", chunk_id)
            if parent_id in seen_parents:
                continue
            seen_parents.add(parent_id)
            ref = {
                "id": chunk_id,
                "source": chunk.get("source", ""),
                "text": chunk["text"],
            }
            filename = chunk.get("file", "")
            if filename and filename in self.pdf_sources:
                base_url = self.pdf_sources[filename]
                pdf_page = chunk.get("pdf_page")
                ref["url"] = f"{base_url}#page={pdf_page}" if pdf_page and pdf_page > 1 else base_url
                # Map "第N号" keys -> page-anchored URL of this chunk
                source = chunk.get("source", "")
                for m in re.finditer(r"第\s*(\d+(?:[\u2010\-]\d+)?)\s*号", source):
                    key = f"第{m.group(1).replace(' ', '')}号"
                    if key not in source_url_map:
                        source_url_map[key] = ref["url"]
            refs.append(ref)
        return refs, source_url_map

    @staticmethod
    def _count_cited_references(answer: str) -> int:
        """Count unique standards / section citations explicitly mentioned in the answer."""
        patterns = [
            r"企業会計基準第\s*\d+\s*号",
            r"企業会計基準適用指針第\s*\d+\s*号",
            r"実務対応報告第\s*\d+\s*号",
            r"IFRS\s*第?\s*\d+\s*号",
            r"第\s*\d+\s*項",
            r"第\s*\d+\s*条",
            r"BC\s*\d+(?:\s*[-‑–]\s*\d+)?",
        ]
        citations: set[str] = set()
        for pattern in patterns:
            for match in re.findall(pattern, answer):
                normalized = re.sub(r"\s+", "", match)
                citations.add(normalized)
        return len(citations)

    def _build_result(
        self,
        answer: str,
        context: AgentContext,
        loops: int,
        stop_reason: str,
        total_cost: float,
    ) -> dict[str, Any]:
        sanitized_answer = self._sanitize_answer(answer)
        references, source_url_map = self._get_referenced_chunks(context)
        return {
            "answer": sanitized_answer,
            "loops": loops,
            "stop_reason": stop_reason,
            "total_cost": total_cost,
            **context.get_summary(),
            "cited_reference_count": self._count_cited_references(sanitized_answer),
            "read_chunk_count": len(context.read_chunk_ids),
            "trajectory": context.trajectory,
            "references": references,
            "source_url_map": source_url_map,
        }
