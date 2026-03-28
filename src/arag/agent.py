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
        self.nudge_at_loop = config.agent.nudge_at_loop or self.NUDGE_AT_LOOP
        self.wrap_up_after_searches = config.agent.wrap_up_after_searches
        self.force_final_after_searches = config.agent.force_final_after_searches
        self.force_final_after_reads = config.agent.force_final_after_reads

    def _count_tool_calls(self, context: AgentContext, tool_name: str) -> int:
        return sum(1 for log in context.retrieval_logs if log.tool_name == tool_name)

    def _should_nudge_wrap_up(self, context: AgentContext) -> bool:
        if context.wrap_up_nudged or self.wrap_up_after_searches <= 0:
            return False
        search_count = self._count_tool_calls(context, "hybrid_search")
        read_count = self._count_tool_calls(context, "read_chunk")
        return search_count >= self.wrap_up_after_searches and read_count >= 2

    def _should_force_wrap_up(self, context: AgentContext) -> bool:
        if self.force_final_after_searches <= 0 or self.force_final_after_reads <= 0:
            return False
        search_count = self._count_tool_calls(context, "hybrid_search")
        read_count = self._count_tool_calls(context, "read_chunk")
        return search_count >= self.force_final_after_searches and read_count >= self.force_final_after_reads

    def _maybe_nudge(self, messages: list[dict], loop_idx: int, context: AgentContext):
        """Inject a wrap-up hint if we've been searching too long."""
        if context.wrap_up_nudged:
            return
        if loop_idx == self.nudge_at_loop or self._should_nudge_wrap_up(context):
            messages.append({"role": "user", "content": WRAP_UP_HINT})
            context.wrap_up_nudged = True

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

            self._maybe_nudge(messages, loop_idx, context)

            if self._should_force_wrap_up(context):
                answer, cost = self._force_final_answer(messages)
                total_cost += cost
                return self._build_result(
                    answer, context, loop_idx + 1, "retrieval_budget", total_cost
                )

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
            self._maybe_nudge(messages, loop_idx, context)

            if self._should_force_wrap_up(context):
                answer, cost = await self._aforce_final_answer(messages)
                total_cost += cost
                return self._build_result(
                    answer, context, loop_idx + 1, "retrieval_budget", total_cost
                )

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

            self._maybe_nudge(messages, loop_idx, context)

            if self._should_force_wrap_up(context):
                yield {"type": "status", "data": "回答を生成中..."}
                async for event in self._astream_final_answer(messages, context, loop_idx + 1, "retrieval_budget", total_cost):
                    yield event
                return

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
                answer, cited_ids = self._sanitize_answer(message.get("content", ""), number_refs=True)
                yield {"type": "status", "data": "回答を生成中..."}
                chunk_size = 8
                for i in range(0, len(answer), chunk_size):
                    yield {"type": "answer_delta", "data": answer[i:i + chunk_size]}
                yield {"type": "answer_done", "data": answer}
                refs, source_url_map = self._get_referenced_chunks(context, cited_ids)
                for ref in refs:
                    yield {"type": "reference", "data": ref}
                summary = self._build_answer_summary(answer, context)
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
        answer, refs, source_url_map = self._finalize_answer(answer, context)
        total_cost += cost

        chunk_size = 8
        for i in range(0, len(answer), chunk_size):
            yield {"type": "answer_delta", "data": answer[i:i + chunk_size]}
        yield {"type": "answer_done", "data": answer}
        for ref in refs:
            yield {"type": "reference", "data": ref}
        summary = self._build_answer_summary(answer, context)
        yield {
            "type": "done",
            "data": {
                "loops": loops,
                "stop_reason": stop_reason,
                "source_url_map": source_url_map,
                **summary,
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
            "見出し以外の本文、箇条書き、まとめ文の末尾には必ず参照したチャンクIDを付け、付けられない文は出力しないでください。"
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
            "見出し以外の本文、箇条書き、まとめ文の末尾には必ず参照したチャンクIDを付け、付けられない文は出力しないでください。"
        )
        messages_copy = messages + [{"role": "user", "content": force_prompt}]
        response = await self.llm.achat(messages=messages_copy, tools=None, temperature=0.0)
        return response["message"].get("content", ""), response.get("cost", 0.0)

    # Chunk ID patterns shared across ref-handling methods
    _CHUNK_SUFFIX = r':(?:p|c)?\d+(?:-\d+)*'
    _FILENAME = r'[\w\-]+\.(?:pdf|xml)'

    def _citation_group_key(self, chunk_id: str) -> str:
        chunk = self.chunk_map.get(chunk_id)
        if not chunk:
            return chunk_id
        return chunk.get("parent_id", chunk_id)

    def _preferred_citation_chunk(self, current_id: str, candidate_id: str) -> str:
        current = self.chunk_map.get(current_id)
        candidate = self.chunk_map.get(candidate_id)
        if not current or not candidate:
            return candidate_id if candidate else current_id

        current_parent = current.get("parent_id", current_id)
        candidate_parent = candidate.get("parent_id", candidate_id)
        current_is_parent = current_parent == current_id
        candidate_is_parent = candidate_parent == candidate_id

        # Prefer the more specific child chunk when the same parent is cited twice.
        if current_is_parent and not candidate_is_parent:
            return candidate_id
        return current_id

    def _number_chunk_refs(self, text: str) -> tuple[str, list[str]]:
        """Replace inline chunk ID refs with [N] citation markers.

        Scans the answer for bracketed chunk ID references and converts them to
        sequential `[1]`, `[2]` … markers.  Returns the modified text and an
        ordered list of displayed chunk IDs (index 0 → citation [1], etc.).
        """
        # Single ID in brackets: （filename.pdf:p5）→ [N]
        single = re.compile(
            rf'[（\(\[【]\s*({self._FILENAME}{self._CHUNK_SUFFIX})\s*[）\)\]】]'
        )
        # Multiple IDs in one bracket: （id1、id2）→ [N][M]
        multi = re.compile(
            rf'[（\(\[【]\s*({self._FILENAME}{self._CHUNK_SUFFIX}(?:\s*[、，;]\s*{self._FILENAME}{self._CHUNK_SUFFIX})+)\s*[）\)\]】]'
        )
        参照_form = re.compile(
            rf'参照:\s*({self._FILENAME}{self._CHUNK_SUFFIX})\s*;?'
        )

        ordered_keys: list[str] = []
        num_map: dict[str, int] = {}
        display_chunk_by_key: dict[str, str] = {}

        def _num(cid: str) -> str:
            key = self._citation_group_key(cid)
            if key not in num_map:
                ordered_keys.append(key)
                num_map[key] = len(ordered_keys)
                display_chunk_by_key[key] = cid
            else:
                display_chunk_by_key[key] = self._preferred_citation_chunk(
                    display_chunk_by_key[key],
                    cid,
                )
            return f'[{num_map[key]}]'

        def replace_multi(m: re.Match) -> str:
            ids = re.findall(rf'{self._FILENAME}{self._CHUNK_SUFFIX}', m.group(1))
            return ''.join(_num(cid) for cid in ids)

        def replace_single(m: re.Match) -> str:
            return _num(m.group(1))

        text = multi.sub(replace_multi, text)
        text = single.sub(replace_single, text)
        text = 参照_form.sub(replace_single, text)
        text = re.sub(r'(\[\d+\])(?:\s*\1)+', r'\1', text)
        ordered = [display_chunk_by_key[key] for key in ordered_keys]
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
        # Keep numbered citation markers like [1], but strip bare page refs in parentheses.
        text = re.sub(
            rf'[（(]\s*{_dash}\s*p?{_digs}(?:\s*{_any_dash}\s*p?{_digs})*\s*[）)]',
            '', text
        )
        # Square-bracket cleanup only targets explicit page-like forms such as [p5] or [-21].
        text = re.sub(
            rf'\[\s*(?:{_any_dash}\s*p?{_digs}(?:\s*{_any_dash}\s*p?{_digs})*|p{_digs}(?:\s*{_any_dash}\s*p?{_digs})*)\s*\]',
            '', text
        )
        # Strip trailing Japanese/ASCII comma before closing paren: （第25項、） → （第25項）
        text = re.sub(r'([（(][^）)\n]{2,})[、，,]\s*([）)])', r'\1\2', text)
        # Clean up residual empty or punctuation-only parentheses like （、）（;）（ ）
        text = re.sub(r'[（(][\s、;,・]*[）)]', '', text)
        text = re.sub(r'  +', ' ', text)
        text = re.sub(r'\s+([。、，,.])', r'\1', text)
        return text

    @classmethod
    def _strip_inline_citation_labels(cls, text: str) -> str:
        """Remove citation-only parentheticals and page labels from answer prose."""
        dash = r'[-－−‐\u2011–—\u2212]'
        standard = (
            r'(?:企業会計基準適用指針|企業会計基準|実務対応報告|移管指針|監査基準|会計基準)'
            r'\s*第\s*\d+(?:[\u2010\-]\d+)?\s*号'
            r'|IFRS\s*第?\s*\d+(?:[\u2010\-]\d+)?\s*号?'
        )
        article = r'第\s*\d+\s*(?:項|条|号)'
        bc = rf'BC\s*\d+(?:\s*{dash}\s*\d+)?'
        page = rf'(?:[Pp]{{1,2}}\.?\s*\d+(?:\s*{dash}\s*\d+)?|(?:ページ|頁)\s*\d+)'
        citation_content = rf'(?:{standard}|{article}|{bc}|{page}|[、，,・／/\s]+)+'

        text = re.sub(rf'[（(]\s*(?:{citation_content})\s*[）)]', '', text)
        text = re.sub(
            rf'\s*[,、，]?\s*(?:[Pp]{{1,2}}\.?\s*\d+(?:\s*{dash}\s*\d+)?|#page=\d+)(?=\s*\[\d+\])',
            '',
            text,
        )
        text = re.sub(r'\s+(\[\d+\])', r'\1', text)
        text = re.sub(r'  +', ' ', text)
        text = re.sub(r'\s+([。、，,.])', r'\1', text)
        return text

    @staticmethod
    def _drop_uncited_lines(text: str) -> str:
        """Keep only headings that lead to cited content and cited body lines."""
        kept_lines: list[str] = []
        pending_headings: list[str] = []
        heading_only_bullet = re.compile(
            r"^-\s+\*\*(?P<label>[^*]+)\*\*(?P<suffix>（[^）]+）)?$"
        )

        def flush_pending_headings() -> None:
            nonlocal pending_headings
            if not pending_headings:
                return
            if kept_lines and kept_lines[-1] != "":
                kept_lines.append("")
            kept_lines.extend(pending_headings)
            pending_headings = []

        for raw_line in text.splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()

            if not stripped:
                if kept_lines and kept_lines[-1] != "":
                    kept_lines.append("")
                continue

            heading_bullet_match = heading_only_bullet.match(stripped)
            if heading_bullet_match:
                label = heading_bullet_match.group("label").strip()
                suffix = (heading_bullet_match.group("suffix") or "").strip()
                pending_headings.append(f"## {label}{suffix}".rstrip())
                continue

            if re.match(r"^#{1,6}\s+", stripped):
                pending_headings.append(stripped)
                continue

            if re.search(r"\[\d+\]", stripped):
                flush_pending_headings()
                kept_lines.append(stripped)

        while kept_lines and kept_lines[-1] == "":
            kept_lines.pop()
        return "\n".join(kept_lines)

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

    def _sanitize_answer(self, text: str, *, number_refs: bool = False) -> tuple[str, list[str]] | str:
        """Normalize answer formatting before returning it to clients.

        When ``number_refs=True`` returns ``(text, ordered_chunk_ids)`` so
        callers can build a numbered reference list matched to inline [N] markers.
        Otherwise returns just the text string (backward-compatible default).
        """
        text = self._strip_reasoning_preamble(text)
        cited_ids: list[str] = []
        if number_refs:
            text, cited_ids = self._number_chunk_refs(text)
        text = self._strip_chunk_refs(text)
        text = self._strip_inline_citation_labels(text)
        if number_refs:
            text = self._drop_uncited_lines(text)

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

    def _finalize_answer(self, answer: str, context: AgentContext) -> tuple[str, list[dict[str, Any]], dict[str, str]]:
        sanitized_answer, cited_ids = self._sanitize_answer(answer, number_refs=True)
        references, source_url_map = self._get_referenced_chunks(context, cited_ids)
        return sanitized_answer, references, source_url_map

    def _get_referenced_chunks(
        self,
        context: AgentContext,
        cited_ids: list[str] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
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

        # 2. read_chunk calls only when the answer has no inline citations
        if not ids_to_show:
            for cid in context.read_chunk_ids:
                if cid not in seen:
                    ids_to_show.append(cid)
                    seen.add(cid)

        # 3. Search fallback when nothing was explicitly cited or read
        if not ids_to_show:
            for cid in context.searched_chunk_ids[:10]:
                if cid not in seen:
                    ids_to_show.append(cid)
                    seen.add(cid)

        refs = []
        seen_parents: set[str] = set()
        source_url_map: dict[str, str] = {}
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
                source = chunk.get("source", "")
                source_label = source.split(">")[0].strip()
                if source_label and source_label not in source_url_map:
                    source_url_map[source_label] = ref["url"]
                # Also keep a generic "第N号" key for short citations.
                for m in re.finditer(r"第\s*(\d+(?:[\u2010\-]\d+)?)\s*号", source):
                    key = f"第{m.group(1).replace(' ', '')}号"
                    if key not in source_url_map:
                        source_url_map[key] = ref["url"]
            refs.append(ref)
        return refs, source_url_map

    @staticmethod
    def _count_cited_references(answer: str) -> int:
        """Count unique visible citations in the answer.

        Prefer numbered markers like [1], [2] that the app renders inline.
        Fall back to explicit standard/article patterns for answers without
        numbered chunk refs.
        """
        numbered = {int(m) for m in re.findall(r"\[(\d+)\]", answer)}
        if numbered:
            return len(numbered)

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

    def _build_answer_summary(self, answer: str, context: AgentContext) -> dict[str, Any]:
        summary = context.get_summary()
        summary["read_chunk_count"] = len(context.read_chunk_ids)
        summary["chunks_read_count"] = self._count_cited_references(answer)
        return summary

    def _build_result(
        self,
        answer: str,
        context: AgentContext,
        loops: int,
        stop_reason: str,
        total_cost: float,
    ) -> dict[str, Any]:
        sanitized_answer, references, source_url_map = self._finalize_answer(answer, context)
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
