<script lang="ts">
  import { browser } from '$app/environment';
  import { onMount } from 'svelte';

  interface Reference {
    id: string;
    source: string;
    text: string;
    url?: string;
    display_number?: number;
    doc_type?: string;
    doc_title?: string;
    section_title?: string;
    section_label?: string;
    page_label?: string;
    standard_no?: string | null;
  }

  interface EvidenceCoverage {
    slots?: string[];
    covered_slots?: string[];
    uncovered_slots?: string[];
    coverage_ratio?: number;
  }

  interface AnswerUncertainty {
    present?: boolean;
    insufficient_points?: string[];
    covered_points?: string[];
    coverage_ratio?: number;
    note?: string;
  }

  interface ToolStep {
    tool: string;
    label: string;
    detail?: string;
  }

  interface Message {
    role: 'user' | 'assistant' | 'status';
    content: string;
    metadata?: {
      request_id?: string;
      query_class?: string;
      loops?: number;
      chunks_read_count?: number;
      read_chunk_count?: number;
      total_cost?: number;
      total_retrieved_tokens?: number;
      stop_reason?: string;
      references?: Reference[];
      source_url_map?: Record<string, string>;
      evidence_coverage?: EvidenceCoverage;
      uncertainty?: AnswerUncertainty;
      steps?: ToolStep[];
    };
  }

  let messages: Message[] = $state([]);
  let input = $state('');
  let loading = $state(false);
  let developerMode = $state(false);
  let chatContainer: HTMLElement;

  const API_BASE = import.meta.env.DEV ? 'http://localhost:8000' : '';

  onMount(() => {
    if (browser) {
      developerMode = window.localStorage.getItem('arag:developer-mode') === '1';
    }
  });

  function setDeveloperMode(enabled: boolean) {
    developerMode = enabled;
    if (browser) {
      window.localStorage.setItem('arag:developer-mode', enabled ? '1' : '0');
    }
  }

  function scrollToBottom() {
    if (chatContainer) {
      requestAnimationFrame(() => {
        chatContainer.scrollTop = chatContainer.scrollHeight;
      });
    }
  }

  function getDisplayedCitationCount(message: Message): number | string {
    return message.metadata?.chunks_read_count ?? '?';
  }

  function getDisplayedReferenceCount(message: Message): number | string {
    return message.metadata?.references?.length ?? message.metadata?.chunks_read_count ?? '?';
  }

  function getDisplayedReadCount(message: Message): number | string {
    return message.metadata?.read_chunk_count ?? '?';
  }

  function getDisplayedRequestId(message: Message): string | null {
    return message.metadata?.request_id ?? null;
  }

  function compactLabel(text: string | null | undefined): string {
    return (text ?? '').replace(/\s+/g, ' ').trim();
  }

  function trimLabel(text: string, limit = 44): string {
    const cleaned = compactLabel(text);
    if (!cleaned) return '';
    if (cleaned.length <= limit) return cleaned;
    return `${cleaned.slice(0, limit - 1).trimEnd()}…`;
  }

  function inferDocType(ref: Reference): string {
    const haystack = compactLabel(ref.doc_title ?? ref.source);
    for (const token of [
      '企業会計基準適用指針',
      '企業会計基準',
      '実務対応報告',
      '会計制度委員会報告',
      '企業会計原則',
      '原価計算基準',
      '注解',
      '法令',
      'IFRS関連情報',
      '中小企業会計',
    ]) {
      if (haystack.includes(token)) return token;
    }
    return '参考資料';
  }

  function getReferenceDocType(ref: Reference): string {
    return compactLabel(ref.doc_type) || inferDocType(ref);
  }

  function getReferenceDocTitle(ref: Reference): string {
    return compactLabel(ref.doc_title) || compactLabel(ref.source.split('>')[0] ?? ref.source);
  }

  function getReferenceSectionLabel(ref: Reference): string {
    return (
      compactLabel(ref.section_label) ||
      trimLabel(compactLabel(ref.section_title) || compactLabel(ref.source.split('>').slice(1).join(' > '))) ||
      compactLabel(ref.page_label) ||
      '該当箇所'
    );
  }

  function getReferencePageLabel(ref: Reference): string {
    return compactLabel(ref.page_label);
  }

  function getReferencePreview(ref: Reference): string {
    const preview = compactLabel(ref.text);
    return preview.length > 200 ? `${preview.slice(0, 200)}…` : preview;
  }

  function extractInsufficientPoints(text: string): string[] {
    const points: string[] = [];
    const seen = new Set<string>();
    const matches = text.matchAll(/^\s*[-・•*]\s*([^:：\n]+?)\s*[:：]\s*今回確認できた根拠では不十分/gm);
    for (const match of matches) {
      const label = compactLabel(match[1]);
      if (label && !seen.has(label)) {
        seen.add(label);
        points.push(label);
      }
    }
    return points;
  }

  function getUncertainty(message: Message): AnswerUncertainty | null {
    const explicit = message.metadata?.uncertainty;
    if (explicit?.present || explicit?.insufficient_points?.length || explicit?.note) {
      return {
        present: Boolean(explicit.present ?? explicit.insufficient_points?.length),
        insufficient_points: explicit.insufficient_points ?? [],
        covered_points: explicit.covered_points ?? [],
        coverage_ratio: explicit.coverage_ratio ?? message.metadata?.evidence_coverage?.coverage_ratio ?? 0,
        note: explicit.note ?? '',
      };
    }

    const uncovered = message.metadata?.evidence_coverage?.uncovered_slots ?? [];
    const extracted = extractInsufficientPoints(message.content);
    const insufficient = [...new Set([...uncovered, ...extracted].map((item) => compactLabel(item)).filter(Boolean))];
    if (insufficient.length === 0 && !message.content.includes('今回確認できた根拠では不十分')) {
      return null;
    }
    return {
      present: true,
      insufficient_points: insufficient,
      covered_points: message.metadata?.evidence_coverage?.covered_slots ?? [],
      coverage_ratio: message.metadata?.evidence_coverage?.coverage_ratio ?? 0,
      note: insufficient.length > 0 ? '未確定の論点があります。参照カードは確認できた範囲の原典です。' : '',
    };
  }

  function buildToolStep(toolName: string, args: Record<string, unknown>): ToolStep {
    if (toolName === 'hybrid_search' || toolName === 'keyword_search' || toolName === 'semantic_search') {
      const labelMap: Record<string, string> = {
        hybrid_search: '検索',
        keyword_search: 'キーワード検索',
        semantic_search: '意味検索',
      };
      const query = typeof args.query === 'string' ? compactLabel(args.query) : '';
      return {
        tool: toolName,
        label: query ? `${labelMap[toolName]}: 「${query}」` : labelMap[toolName],
      };
    }

    if (toolName === 'read_chunk') {
      const ids = Array.isArray(args.chunk_ids) ? args.chunk_ids : [];
      return {
        tool: toolName,
        label: '条文を確認',
        detail: `${ids.length || 1}件`,
      };
    }

    if (toolName === 'read_document') {
      const docName = compactLabel(String(args.document_name ?? args.filename ?? ''));
      return {
        tool: toolName,
        label: '文書全体を確認',
        detail: docName || '文書指定なし',
      };
    }

    return {
      tool: toolName,
      label: `${toolName} を実行`,
    };
  }

  function trackUiEvent(event: string, payload: Record<string, unknown> = {}) {
    if (!browser) return;
    const detail = { event, ...payload };
    const trackedWindow = window as Window & { __aragUiEvents?: Array<Record<string, unknown>> };
    trackedWindow.__aragUiEvents = trackedWindow.__aragUiEvents ?? [];
    trackedWindow.__aragUiEvents.push(detail);
    window.dispatchEvent(new CustomEvent('arag-ui-event', { detail }));
    void fetch(`${API_BASE}/api/ui-event`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(detail),
      keepalive: true,
    }).catch(() => {});
  }

  function getChunkUrl(ref: Reference): string {
    return ref.url ?? '';
  }

  function getDisplayNumber(ref: Reference, idx: number): number {
    return ref.display_number ?? idx + 1;
  }

  let copiedRefId: string | null = $state(null);
  async function copyChunkText(ref: { id?: string; text: string }) {
    try {
      await navigator.clipboard.writeText(ref.text);
      copiedRefId = ref.id ?? null;
      setTimeout(() => { copiedRefId = null; }, 1500);
    } catch {}
  }

  function scrollToMessage(idx: number) {
    if (chatContainer) {
      // Wait for DOM to update, then scroll
      setTimeout(() => {
        const msgs = chatContainer.querySelectorAll('.message');
        const header = chatContainer.querySelector('header');
        const headerHeight = header?.offsetHeight ?? 60;
        if (msgs[idx]) {
          const msgTop = (msgs[idx] as HTMLElement).offsetTop;
          chatContainer.scrollTop = msgTop - headerHeight - 16;
        }
      }, 50);
    }
  }

  function handleReferenceOpen(ref: Reference, idx: number, message: Message) {
    const uncertainty = getUncertainty(message);
    trackUiEvent('reference_opened', {
      request_id: message.metadata?.request_id ?? null,
      reference_number: getDisplayNumber(ref, idx),
      reference_id: ref.id,
      reference_count: message.metadata?.references?.length ?? 0,
      uncertainty_present: Boolean(uncertainty?.present),
      developer_mode: developerMode,
    });
  }

  async function sendMessage() {
    const question = input.trim();
    if (!question || loading) return;

    const priorAnswerCount = messages.filter((message) => message.role === 'assistant').length;
    trackUiEvent('question_submitted', {
      question_length: question.length,
      prior_answer_count: priorAnswerCount,
      developer_mode: developerMode,
    });
    if (priorAnswerCount > 0) {
      trackUiEvent('followup_submitted', {
        question_length: question.length,
        prior_answer_count: priorAnswerCount,
        developer_mode: developerMode,
      });
    }

    input = '';
    loading = true;
    messages = [...messages, { role: 'user', content: question }];
    scrollToBottom();

    let statusIdx = messages.length;
    messages = [...messages, { role: 'status', content: '検索を開始しています...' }];
    scrollToBottom();

    try {
      // Build conversation history from previous messages
      const history: { question: string; answer: string }[] = [];
      for (let h = 0; h < messages.length - 1; h++) {
        const msg = messages[h];
        if (msg.role === 'user' && h + 1 < messages.length) {
          // Find the next assistant message
          for (let j = h + 1; j < messages.length; j++) {
            if (messages[j].role === 'assistant') {
              history.push({ question: msg.content, answer: messages[j].content });
              break;
            }
          }
        }
      }

      const response = await fetch(`${API_BASE}/api/ask/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, history }),
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      if (!response.body) throw new Error('No response body');

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let answerContent = '';
      let streamingStarted = false;
      let answerIdx = -1;
      let collectedRefs: Reference[] = [];
      let collectedSteps: ToolStep[] = [];

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          try {
            const event = JSON.parse(line.slice(6));

            if (event.type === 'status') {
              if (!streamingStarted) {
                messages[statusIdx] = { role: 'status', content: event.data };
                messages = [...messages];
              }
            } else if (event.type === 'tool_call') {
              const toolName = event.data.tool;
              const args = event.data.args || {};
              collectedSteps = [...collectedSteps, buildToolStep(toolName, args)];
              let statusContent = '';
              if (toolName === 'hybrid_search' || toolName === 'keyword_search' || toolName === 'semantic_search') {
                const labelMap: Record<string, string> = {
                  hybrid_search: '検索',
                  keyword_search: 'キーワード検索',
                  semantic_search: '意味検索',
                };
                const label = labelMap[toolName];
                const query = args.query ? `「${args.query}」` : '';
                statusContent = `${label}中${query ? ' ' + query : ''}...`;
              } else if (toolName === 'read_chunk') {
                const ids = args.chunk_ids || [];
                const count = Array.isArray(ids) ? ids.length : 1;
                statusContent = `条文を読取中... (${count}件)`;
              } else if (toolName === 'read_document') {
                const docName = args.document_name || args.filename || '';
                statusContent = `文書を読取中${docName ? ' — ' + docName : ''}...`;
              } else {
                statusContent = `${toolName}を実行中...`;
              }
              messages[statusIdx] = {
                role: 'status',
                content: statusContent,
              };
              messages = [...messages];
            } else if (event.type === 'answer_delta') {
              // Streaming token
              if (!streamingStarted) {
                streamingStarted = true;
                // Replace status with empty assistant message
                messages[statusIdx] = {
                  role: 'assistant',
                  content: '',
                };
                answerIdx = statusIdx;
                answerContent = '';
                // Scroll to show the top of the answer message
                scrollToMessage(answerIdx);
              }
              answerContent += event.data;
              messages[answerIdx] = {
                ...messages[answerIdx],
                content: answerContent,
              };
              messages = [...messages];
              // Don't auto-scroll during streaming - user reads from the top
            } else if (event.type === 'answer_done') {
              // Final complete answer
              answerContent = event.data;
              if (answerIdx >= 0) {
                messages[answerIdx] = {
                  ...messages[answerIdx],
                  content: answerContent,
                };
                messages = [...messages];
              }
            } else if (event.type === 'answer') {
              // Legacy non-streaming answer (fallback)
              answerContent = event.data;
            } else if (event.type === 'reference') {
              // Collect references sent individually
              collectedRefs = [...collectedRefs, event.data as Reference];
            } else if (event.type === 'error') {
              // Server error
              messages = messages.filter((_, i) => i !== statusIdx);
              messages = [
                ...messages,
                { role: 'assistant', content: event.data },
              ];
              loading = false;
              return;
            } else if (event.type === 'done') {
              // Merge collected references into metadata
              const metadata = {
                ...event.data,
                references: collectedRefs.length > 0 ? collectedRefs : Array.isArray(event.data.references) ? event.data.references : [],
                steps: collectedSteps,
              };
              if (!streamingStarted) {
                // Non-streaming fallback
                messages = messages.filter((_, i) => i !== statusIdx);
                messages = [
                  ...messages,
                  {
                    role: 'assistant',
                    content: answerContent,
                    metadata,
                  },
                ];
              } else {
                // Update metadata on the already-displayed message
                if (answerIdx >= 0) {
                  messages[answerIdx] = {
                    ...messages[answerIdx],
                    metadata,
                  };
                  messages = [...messages];
                }
              }
              trackUiEvent('answer_rendered', {
                request_id: metadata.request_id ?? null,
                reference_count: metadata.references?.length ?? 0,
                uncertainty_present: Boolean(metadata.uncertainty?.present),
                developer_mode: developerMode,
              });
            }
            // Only auto-scroll for non-streaming events (status, tool_call)
            if (!streamingStarted) {
              scrollToBottom();
            }
          } catch {
            // ignore parse errors
          }
        }
      }
    } catch (err) {
      messages = messages.filter((_, i) => i !== statusIdx);
      messages = [
        ...messages,
        { role: 'assistant', content: `エラーが発生しました: ${err}` },
      ];
    } finally {
      loading = false;
    }
  }

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  function resetChat() {
    messages = [];
    input = '';
    loading = false;
  }
</script>

<script lang="ts" module>
  interface MarkdownReference {
    url?: string;
    source: string;
    text: string;
    display_number?: number;
  }

  function getReferenceByNumber(refs: MarkdownReference[], displayNumber: number): MarkdownReference | undefined {
    return refs.find((ref) => ref.display_number === displayNumber) ?? refs[displayNumber - 1];
  }

  function formatMarkdown(text: string, refs: MarkdownReference[] = []): string {
    if (!text) return '';

    // Pre-process: convert literal <br> tags to newlines, but preserve them inside table rows
    // First, temporarily replace <br> inside table rows (lines starting with |) with a placeholder
    text = text.replace(/^(\|.*)\n?$/gm, (match) => match.replace(/<br\s*\/?>/gi, '\u200B'));
    // Then convert remaining <br> tags to newlines
    text = text.replace(/<br\s*\/?>/gi, '\n');

    // Pre-process: extract tables first (they may span across double newlines)
    // A table starts with a line containing |, followed by a separator line with dashes
    const allLines = text.split('\n');
    const segments: { type: 'table' | 'text'; content: string }[] = [];
    let i = 0;

    while (i < allLines.length) {
      // Check if current line starts a table
      if (
        i + 1 < allLines.length &&
        allLines[i].includes('|') &&
        isSeparatorLine(allLines[i + 1])
      ) {
        // Collect all table lines
        const tableLines: string[] = [allLines[i], allLines[i + 1]];
        i += 2;
        while (i < allLines.length && allLines[i].includes('|') && allLines[i].trim() !== '') {
          tableLines.push(allLines[i]);
          i++;
        }
        segments.push({ type: 'table', content: tableLines.join('\n') });
      } else {
        // Collect text lines until next table or end
        const textLines: string[] = [];
        while (i < allLines.length) {
          if (
            i + 1 < allLines.length &&
            allLines[i].includes('|') &&
            isSeparatorLine(allLines[i + 1])
          ) {
            break;
          }
          textLines.push(allLines[i]);
          i++;
        }
        const joined = textLines.join('\n').trim();
        if (joined) {
          segments.push({ type: 'text', content: joined });
        }
      }
    }

    const html: string[] = [];

    for (const seg of segments) {
      if (seg.type === 'table') {
        html.push(renderTable(seg.content.split('\n')));
        continue;
      }

      // Split text into lines and process each line contextually
      const allBlockLines = seg.content.split('\n');
      let currentParagraph: string[] = [];

      const flushParagraph = () => {
        if (currentParagraph.length > 0) {
          const escaped = currentParagraph.map(l => inlineFormat(escapeHtml(l), refs)).join('<br>');
          html.push(`<p>${escaped}</p>`);
          currentParagraph = [];
        }
      };

      const isListLine = (l: string) =>
        l.match(/^\s*[-・•*]\s/) || l.match(/^\s*\d+\.\s/);

      for (let li = 0; li < allBlockLines.length; li++) {
        const line = allBlockLines[li];
        const trimmedLine = line.trim();

        // Empty line = paragraph break
        if (!trimmedLine) {
          flushParagraph();
          continue;
        }

        // Heading
        const headingMatch = trimmedLine.match(/^(#{1,4})\s+(.+)$/);
        if (headingMatch) {
          flushParagraph();
          const level = headingMatch[1].length;
          const content = escapeHtml(headingMatch[2]);
          html.push(`<h${level + 2}>${inlineFormat(content, refs)}</h${level + 2}>`);
          continue;
        }

        // List item (-, ・, •, *, or 1.)
        if (isListLine(line)) {
          flushParagraph();
          // Collect consecutive list items
          const listItems: string[] = [];
          const isOrdered = !!line.match(/^\s*\d+\.\s/);
          while (li < allBlockLines.length && (isListLine(allBlockLines[li]) || allBlockLines[li].trim() === '')) {
            const ll = allBlockLines[li].trim();
            if (ll === '') {
              // Check if next line is also a list item (allow blank lines within lists)
              if (li + 1 < allBlockLines.length && isListLine(allBlockLines[li + 1])) {
                li++;
                continue;
              }
              break;
            }
            const content = ll.replace(/^\s*[-・•*]\s*/, '').replace(/^\s*\d+\.\s*/, '');
            listItems.push(`<li>${inlineFormat(escapeHtml(content), refs)}</li>`);
            li++;
          }
          li--; // Back up one since the for loop will increment
          const tag = isOrdered ? 'ol' : 'ul';
          html.push(`<${tag}>${listItems.join('')}</${tag}>`);
          continue;
        }

        // Regular text line
        currentParagraph.push(trimmedLine);
      }
      flushParagraph();
    }

    return html.join('');
  }

  function escapeHtml(text: string): string {
    return text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }

  function escapeAttribute(text: string): string {
    return text
      .replace(/&/g, '&amp;')
      .replace(/"/g, '&quot;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }

  function inlineFormat(text: string, refs: MarkdownReference[] = []): string {
    return text
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/`(.*?)`/g, '<code>$1</code>')
      .replace(/【(.*?)】/g, '<span class="ref-tag">$1</span>')
      .replace(/\[(\d+)\]/g, (match, num) => {
        const ref = getReferenceByNumber(refs, Number(num));
        if (!ref?.url) return match;
        const safeUrl = escapeAttribute(ref.url);
        return `<a class="ref-link" href="${safeUrl}" target="_blank" rel="noopener noreferrer">[${num}]<span class="ref-link-icon">&#x2197;</span></a>`;
      });
  }

  function isSeparatorLine(line: string): boolean {
    if (!line) return false;
    const trimmed = line.trim();
    // Match separator patterns: |---|---|, | --- | --- |, |:--:|--:|, etc.
    // Must contain at least one dash sequence between pipes
    return /^\|?[\s\-:|\u2014]+\|?$/.test(trimmed) && trimmed.includes('-');
  }

  function parseRow(line: string): string[] {
    const trimmed = line.trim();
    // Split by | and handle leading/trailing pipes
    const parts = trimmed.split('|');
    // Remove empty first/last if line starts/ends with |
    if (parts.length > 0 && parts[0].trim() === '') parts.shift();
    if (parts.length > 0 && parts[parts.length - 1].trim() === '') parts.pop();
    return parts.map(c => c.trim());
  }

  function formatCell(text: string): string {
    // Restore placeholders to <br> for in-cell line breaks, then apply formatting
    const withBreaks = escapeHtml(text).replace(/\u200B/g, '<br>');
    return inlineFormat(withBreaks);
  }

  function renderTable(lines: string[]): string {
    const headers = parseRow(lines[0]);
    const colCount = headers.length;
    const rows = lines.slice(2).filter(l => l.includes('|')).map(l => {
      const cells = parseRow(l);
      while (cells.length < colCount) cells.push('');
      return cells.slice(0, colCount);
    });

    // Desktop: normal table
    let table = '<div class="table-wrap"><table><thead><tr>';
    for (const h of headers) {
      table += `<th>${formatCell(h)}</th>`;
    }
    table += '</tr></thead><tbody>';
    for (const row of rows) {
      table += '<tr>';
      for (const cell of row) {
        table += `<td>${formatCell(cell)}</td>`;
      }
      table += '</tr>';
    }
    table += '</tbody></table></div>';

    // Mobile: card layout
    let cards = '<div class="table-cards">';
    for (const row of rows) {
      cards += '<div class="table-card">';
      for (let c = 0; c < colCount; c++) {
        const label = headers[c] || '';
        const value = row[c] || '';
        cards += `<div class="table-card-field">`;
        cards += `<div class="table-card-label">${formatCell(label)}</div>`;
        cards += `<div class="table-card-value">${formatCell(value)}</div>`;
        cards += `</div>`;
      }
      cards += '</div>';
    }
    cards += '</div>';

    return table + cards;
  }
</script>

<div class="app">
  <div class="scroll-outer" bind:this={chatContainer}>
    <header>
      <div class="header-wrap">
        <div class="header-row">
          <button class="header-inner" onclick={resetChat}>
            <h1>会計基準 Q&A</h1>
            <p class="subtitle">日本の会計基準についてAIが回答します</p>
          </button>
          <div class="view-toggle" data-testid="view-toggle">
            <button
              class:active={!developerMode}
              class="view-toggle-btn"
              data-testid="view-toggle-user"
              onclick={() => setDeveloperMode(false)}
              type="button"
            >
              利用者表示
            </button>
            <button
              class:active={developerMode}
              class="view-toggle-btn"
              data-testid="view-toggle-developer"
              onclick={() => setDeveloperMode(true)}
              type="button"
            >
              開発表示
            </button>
          </div>
        </div>
      </div>
    </header>

    <main>
    {#if messages.length === 0}
      <div class="welcome">
        <div class="welcome-icon">&#x1f4d1;</div>
        <h2>会計基準について質問してください</h2>
        <p>企業会計基準、適用指針、実務対応報告の内容をAIが検索・回答します。</p>
        <div class="examples">
          <button onclick={() => { input = 'のれんの償却期間について教えてください'; sendMessage(); }}>
            のれんの償却期間について
          </button>
          <button onclick={() => { input = '収益認識基準における履行義務の充足とは？'; sendMessage(); }}>
            収益認識の履行義務
          </button>
          <button onclick={() => { input = 'リース会計基準の改正点を教えてください'; sendMessage(); }}>
            リース会計基準の改正
          </button>
        </div>
      </div>
    {/if}

    {#each messages as msg}
      {#if msg.role === 'user'}
        <div class="message user">
          <div class="bubble user-bubble">{msg.content}</div>
        </div>
      {:else if msg.role === 'status'}
        <div class="message status">
          <div class="bubble status-bubble">
            <span class="spinner"></span>
            {msg.content}
          </div>
        </div>
      {:else}
        <div class="message assistant">
          <div class="bubble assistant-bubble" data-testid="assistant-message">
            <div data-testid="answer-content">
              {@html formatMarkdown(msg.content, msg.metadata?.references ?? [])}
            </div>

            {#if msg.metadata}
              <div class="answer-summary" data-testid="answer-summary">
                <span class="answer-chip">検索 {msg.metadata.loops ?? '?'} 回</span>
                <span class="answer-chip">引用 {getDisplayedCitationCount(msg)} 件</span>
                <span class="answer-chip">確認候補 {getDisplayedReadCount(msg)} 件</span>
                {#if getUncertainty(msg)?.present}
                  <span class="answer-chip warning">
                    未確定 {getUncertainty(msg)?.insufficient_points?.length || 'あり'}
                  </span>
                {/if}
              </div>
            {/if}

            {#if getUncertainty(msg)?.present}
              <div class="uncertainty-banner" data-testid="uncertainty-banner">
                <div class="uncertainty-eyebrow">未確定の論点</div>
                <div class="uncertainty-title">この回答は一部の論点で根拠が不足しています</div>
                <p class="uncertainty-copy">
                  {getUncertainty(msg)?.note || '参照カードは、確認できた範囲の原典だけを示しています。'}
                </p>
                {#if getUncertainty(msg)?.insufficient_points?.length}
                  <div class="uncertainty-points">
                    {#each getUncertainty(msg)?.insufficient_points ?? [] as point}
                      <span class="uncertainty-chip">{point}</span>
                    {/each}
                  </div>
                {/if}
              </div>
            {/if}

            {#if msg.metadata?.references?.length}
              <div class="sources-section" data-testid="references-section">
                <div class="sources-header">
                  <div>
                    <div class="sources-label">参照 ({getDisplayedReferenceCount(msg)}件)</div>
                    <div class="sources-caption">本文の引用番号と同じ順に、原典へそのまま移動できます。</div>
                  </div>
                  {#if getUncertainty(msg)?.present}
                    <div class="sources-caution">未確定の論点は上の注意表示で明示しています</div>
                  {/if}
                </div>
                <div class="sources-list">
                  {#each msg.metadata.references as ref, idx}
                    <div class="source-card" data-testid="reference-card" data-reference-number={getDisplayNumber(ref, idx)}>
                      <div class="source-card-header">
                        <div class="source-card-meta-row">
                          <span class="source-ref-number">[{getDisplayNumber(ref, idx)}]</span>
                          <span class="source-doc-type" data-testid="reference-doc-type">{getReferenceDocType(ref)}</span>
                          {#if getReferencePageLabel(ref)}
                            <span class="source-page-chip">{getReferencePageLabel(ref)}</span>
                          {/if}
                        </div>
                        <button class="copy-btn" title="テキストをコピー（PDF内Ctrl+F用）" onclick={() => copyChunkText(ref)}>
                          {copiedRefId === ref.id ? '✓' : '⎘'}
                        </button>
                      </div>
                      <div class="source-card-title-row">
                        {#if ref.url}
                          <a
                            class="source-card-title"
                            data-testid="reference-link"
                            data-reference-number={getDisplayNumber(ref, idx)}
                            href={getChunkUrl(ref)}
                            target="_blank"
                            rel="noopener noreferrer"
                            onclick={() => handleReferenceOpen(ref, idx, msg)}
                          >
                            <span data-testid="reference-doc-title">{getReferenceDocTitle(ref)}</span><span class="ref-link-icon">&#x2197;</span>
                          </a>
                        {:else}
                          <span class="source-card-title" data-testid="reference-doc-title">{getReferenceDocTitle(ref)}</span>
                        {/if}
                      </div>
                      <div class="source-card-section" data-testid="reference-section-label">{getReferenceSectionLabel(ref)}</div>
                      <div class="source-card-text">{getReferencePreview(ref)}</div>
                    </div>
                  {/each}
                </div>
              </div>
            {/if}

            {#if developerMode && msg.metadata}
              <details class="developer-panel" data-testid="developer-panel" open>
                <summary>開発者詳細</summary>
                <div class="developer-grid">
                  <div class="developer-field">
                    <span class="developer-field-label">Query Class</span>
                    <span class="developer-field-value">{msg.metadata.query_class ?? '?'}</span>
                  </div>
                  <div class="developer-field">
                    <span class="developer-field-label">Stop Reason</span>
                    <span class="developer-field-value">{msg.metadata.stop_reason ?? '?'}</span>
                  </div>
                  <div class="developer-field">
                    <span class="developer-field-label">Retrieved Tokens</span>
                    <span class="developer-field-value">{msg.metadata.total_retrieved_tokens ?? '?'}</span>
                  </div>
                  <div class="developer-field">
                    <span class="developer-field-label">Request ID</span>
                    <span class="developer-field-value" data-testid="developer-request-id">{getDisplayedRequestId(msg) ?? '?'}</span>
                  </div>
                </div>
                {#if msg.metadata.steps?.length}
                  <ol class="developer-steps">
                    {#each msg.metadata.steps as step}
                      <li class="developer-step">
                        <span class="developer-step-label">{step.label}</span>
                        {#if step.detail}
                          <span class="developer-step-detail">{step.detail}</span>
                        {/if}
                      </li>
                    {/each}
                  </ol>
                {/if}
              </details>
            {/if}
          </div>
        </div>
      {/if}
    {/each}
    </main>
  </div>

  <footer>
    <div class="input-area">
      <textarea
        bind:value={input}
        data-testid="question-input"
        onkeydown={handleKeydown}
        placeholder="会計基準について質問してください..."
        rows="1"
        disabled={loading}
      ></textarea>
      <button class="send-btn" data-testid="send-button" onclick={sendMessage} disabled={loading || !input.trim()}>
        {#if loading}
          <span class="spinner"></span>
        {:else}
          送信
        {/if}
      </button>
    </div>
  </footer>
</div>

<style>
  .app {
    display: flex;
    flex-direction: column;
    height: 100%;
    height: 100dvh;
    overflow: hidden;
  }

  .scroll-outer {
    flex: 1 1 0;
    min-height: 0;
    overflow-y: auto;
    scrollbar-width: thin;
    scrollbar-color: #3f3f46 transparent;
  }

  .scroll-outer::-webkit-scrollbar {
    width: 8px;
  }

  .scroll-outer::-webkit-scrollbar-track {
    background: #0c0c10;
  }

  .scroll-outer::-webkit-scrollbar-thumb {
    background: #3f3f46;
    border-radius: 4px;
  }

  .scroll-outer::-webkit-scrollbar-thumb:hover {
    background: #52525b;
  }

  header {
    position: sticky;
    top: 0;
    z-index: 10;
    padding: 1rem 1.5rem;
    border-bottom: 1px solid #1e1e28;
    backdrop-filter: blur(12px);
    background: rgba(12, 12, 16, 0.92);
  }

  .header-wrap {
    max-width: 900px;
    margin: 0 auto;
  }

  .header-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
  }

  .header-inner {
    background: none;
    border: none;
    padding: 0;
    cursor: pointer;
    text-align: left;
    font-family: inherit;
    transition: opacity 0.15s ease;
  }

  .header-inner:hover {
    opacity: 0.75;
  }

  .header-inner h1 {
    font-size: 1.25rem;
    font-weight: 700;
    color: #f4f4f5;
    letter-spacing: 0.02em;
  }

  .subtitle {
    font-size: 0.8rem;
    color: #8b8b95;
    margin-top: 0.15rem;
  }

  .view-toggle {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.25rem;
    border: 1px solid #2a2a35;
    border-radius: 999px;
    background: rgba(24, 24, 32, 0.9);
  }

  .view-toggle-btn {
    border: none;
    background: transparent;
    color: #8b8b95;
    padding: 0.45rem 0.8rem;
    border-radius: 999px;
    font-size: 0.74rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .view-toggle-btn.active {
    background: linear-gradient(135deg, #1d4ed8, #2563eb);
    color: #f8fbff;
    box-shadow: 0 8px 18px rgba(37, 99, 235, 0.28);
  }

  main {
    max-width: 900px;
    margin: 0 auto;
    width: 100%;
    padding: 1.5rem;
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .welcome {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    gap: 0.75rem;
    color: #71717a;
    min-height: calc(100dvh - 140px);
  }

  .welcome-icon {
    font-size: 3rem;
  }

  .welcome h2 {
    font-size: 1.1rem;
    font-weight: 500;
    color: #a1a1aa;
  }

  .welcome p {
    font-size: 0.85rem;
    max-width: 400px;
  }

  .examples {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    justify-content: center;
    margin-top: 0.5rem;
  }

  .examples button {
    background: #1a1a23;
    border: 1px solid #2a2a35;
    color: #a1a1aa;
    padding: 0.5rem 1rem;
    border-radius: 0.75rem;
    cursor: pointer;
    font-size: 0.8rem;
    font-family: inherit;
    transition: all 0.15s ease;
  }

  .examples button:hover {
    background: #252530;
    color: #e4e4e7;
    border-color: #3f3f4a;
  }

  .message {
    display: flex;
    animation: fadeIn 0.3s ease;
  }

  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .message.user {
    justify-content: flex-end;
  }

  .message.status, .message.assistant {
    justify-content: flex-start;
  }

  .bubble {
    max-width: 75%;
    padding: 1rem 1.25rem;
    border-radius: 1rem;
    font-size: 0.9rem;
    line-height: 1.7;
    word-break: break-word;
  }

  .user-bubble {
    background: #2563eb;
    color: #f4f4f5;
    border-bottom-right-radius: 0.25rem;
  }

  .assistant-bubble {
    background: #1c1c28;
    color: #d8d8de;
    border-bottom-left-radius: 0.25rem;
    border: 1px solid #32323e;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.2);
  }

  .status-bubble {
    background: transparent;
    color: #71717a;
    font-size: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 0;
  }

  :global(.ref-link) {
    color: #60a5fa;
    text-decoration: none;
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    transition: color 0.15s ease;
  }

  :global(.ref-link:hover) {
    color: #93bbfd;
    text-decoration: underline;
  }

  :global(.ref-link-icon) {
    font-size: 0.7rem;
    opacity: 0.7;
  }

  .answer-summary {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    margin-top: 0.85rem;
  }

  .answer-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.3rem 0.65rem;
    border-radius: 999px;
    border: 1px solid #2c3444;
    background: #171d29;
    color: #b4c2d8;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.02em;
  }

  .answer-chip.warning {
    border-color: #6b4f1d;
    background: #2d2211;
    color: #f0cd87;
  }

  .uncertainty-banner {
    margin-top: 0.85rem;
    padding: 0.95rem 1rem;
    border-radius: 0.85rem;
    border: 1px solid #6b4f1d;
    background:
      linear-gradient(135deg, rgba(133, 77, 14, 0.2), rgba(54, 40, 16, 0.92)),
      #21170b;
  }

  .uncertainty-eyebrow {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #f0cd87;
  }

  .uncertainty-title {
    margin-top: 0.25rem;
    color: #fff0cf;
    font-size: 0.95rem;
    font-weight: 700;
  }

  .uncertainty-copy {
    margin-top: 0.35rem;
    color: #e8d6b3;
    font-size: 0.8rem;
    line-height: 1.6;
  }

  .uncertainty-points {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    margin-top: 0.65rem;
  }

  .uncertainty-chip {
    padding: 0.28rem 0.6rem;
    border-radius: 999px;
    background: rgba(255, 243, 214, 0.09);
    border: 1px solid rgba(240, 205, 135, 0.28);
    color: #ffe4aa;
    font-size: 0.72rem;
    font-weight: 600;
  }

  .sources-section {
    margin-top: 0.9rem;
    padding-top: 0.9rem;
    border-top: 1px solid #2a2a35;
  }

  .sources-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 0.75rem;
    margin-bottom: 0.55rem;
  }

  .sources-label {
    font-size: 0.72rem;
    color: #727888;
    font-weight: 700;
    letter-spacing: 0.04em;
  }

  .sources-caption {
    margin-top: 0.2rem;
    color: #787f8f;
    font-size: 0.74rem;
    line-height: 1.5;
  }

  .sources-caution {
    max-width: 230px;
    padding: 0.35rem 0.55rem;
    border-radius: 0.6rem;
    background: rgba(133, 77, 14, 0.14);
    border: 1px solid rgba(240, 205, 135, 0.14);
    color: #d5b26f;
    font-size: 0.7rem;
    line-height: 1.4;
  }

  .sources-list {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .source-card {
    background:
      radial-gradient(circle at top right, rgba(37, 99, 235, 0.12), transparent 36%),
      #131823;
    border: 1px solid #273044;
    border-radius: 0.85rem;
    padding: 0.8rem 0.9rem;
    width: 100%;
  }

  .source-card-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 0.75rem;
  }

  .source-ref-number {
    color: #f4f4f5;
    font-size: 0.72rem;
    font-variant-numeric: tabular-nums;
    flex-shrink: 0;
  }

  .source-card-meta-row {
    display: inline-flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.4rem;
  }

  .source-doc-type,
  .source-page-chip {
    padding: 0.18rem 0.45rem;
    border-radius: 999px;
    border: 1px solid #30415f;
    background: #1a2334;
    color: #9eb4d8;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.03em;
  }

  .copy-btn {
    background: none;
    border: none;
    cursor: pointer;
    color: #52525b;
    font-size: 0.8rem;
    padding: 0 0.1rem;
    line-height: 1;
    margin-left: auto;
    flex-shrink: 0;
  }
  .copy-btn:hover { color: #a1a1aa; }

  .source-card-title-row {
    margin-top: 0.45rem;
  }

  .source-card-title {
    color: #f5f8ff;
    text-decoration: none;
    display: inline-flex;
    align-items: center;
    gap: 0.28rem;
    font-size: 0.92rem;
    font-weight: 700;
    line-height: 1.5;
  }

  .source-card-title:hover {
    color: #b8d0ff;
    text-decoration: underline;
  }

  .source-card-section {
    margin-top: 0.2rem;
    color: #8ea4c9;
    font-size: 0.76rem;
    font-weight: 600;
  }

  .source-card-text {
    margin-top: 0.5rem;
    color: #98a2b6;
    font-size: 0.74rem;
    line-height: 1.6;
    white-space: pre-wrap;
  }

  .developer-panel {
    margin-top: 0.9rem;
    padding-top: 0.85rem;
    border-top: 1px solid #2a2a35;
  }

  .developer-panel summary {
    cursor: pointer;
    color: #8fa8d4;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.04em;
  }

  .developer-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.6rem;
    margin-top: 0.75rem;
  }

  .developer-field {
    padding: 0.65rem 0.75rem;
    border-radius: 0.7rem;
    border: 1px solid #273044;
    background: #111725;
  }

  .developer-field-label {
    display: block;
    color: #69758d;
    font-size: 0.67rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }

  .developer-field-value {
    display: block;
    margin-top: 0.18rem;
    color: #d4def1;
    font-size: 0.8rem;
    font-variant-numeric: tabular-nums;
  }

  .developer-steps {
    margin: 0.8rem 0 0;
    padding-left: 1.15rem;
    color: #9aa7bf;
  }

  .developer-step {
    margin-bottom: 0.45rem;
  }

  .developer-step-label {
    color: #d4def1;
  }

  .developer-step-detail {
    margin-left: 0.35rem;
    color: #8fa8d4;
    font-size: 0.76rem;
  }

  footer {
    flex-shrink: 0;
    padding: 1rem 1.5rem;
    border-top: 1px solid #1e1e28;
    background: rgba(12, 12, 16, 0.95);
    backdrop-filter: blur(12px);
  }

  .input-area {
    max-width: 900px;
    margin: 0 auto;
    display: flex;
    gap: 0.75rem;
    align-items: flex-end;
  }

  textarea {
    flex: 1;
    background: #141419;
    border: 1px solid #2a2a35;
    color: #e4e4e7;
    padding: 0.75rem 1rem;
    border-radius: 0.75rem;
    font-size: 0.9rem;
    font-family: inherit;
    resize: none;
    outline: none;
    line-height: 1.5;
    min-height: 44px;
    max-height: 120px;
    transition: border-color 0.15s ease;
  }

  textarea:focus {
    border-color: #2563eb;
    box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.15);
  }

  textarea::placeholder {
    color: #52525b;
  }

  .send-btn {
    background: #2563eb;
    color: white;
    border: none;
    padding: 0.75rem 1.5rem;
    border-radius: 0.75rem;
    font-size: 0.85rem;
    font-family: inherit;
    font-weight: 500;
    cursor: pointer;
    white-space: nowrap;
    transition: all 0.15s ease;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    min-height: 44px;
  }

  .send-btn:hover:not(:disabled) {
    background: #1d4ed8;
  }

  .send-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  .spinner {
    display: inline-block;
    width: 14px;
    height: 14px;
    border: 2px solid currentColor;
    border-right-color: transparent;
    border-radius: 50%;
    animation: spin 0.6s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  :global(code) {
    background: #27272a;
    padding: 0.1em 0.35em;
    border-radius: 0.25rem;
    font-size: 0.85em;
  }

  :global(.assistant-bubble p) {
    margin-bottom: 0.6rem;
  }

  :global(.assistant-bubble p:last-child) {
    margin-bottom: 0;
  }

  :global(.assistant-bubble h3),
  :global(.assistant-bubble h4),
  :global(.assistant-bubble h5),
  :global(.assistant-bubble h6) {
    color: #f4f4f5;
    margin: 0.8rem 0 0.4rem;
    font-size: 0.95rem;
  }

  :global(.assistant-bubble h3) {
    font-size: 1rem;
  }

  :global(.assistant-bubble ul),
  :global(.assistant-bubble ol) {
    padding-left: 1.25rem;
    margin-bottom: 0.6rem;
  }

  :global(.assistant-bubble li) {
    margin-bottom: 0.3rem;
  }

  :global(.table-wrap) {
    display: none;
  }

  :global(.ref-tag) {
    color: #60a5fa;
    font-size: 0.8em;
  }

  :global(.cite-link) {
    color: #60a5fa;
    text-decoration: none;
    font-size: 0.78em;
    font-weight: 600;
    vertical-align: super;
    line-height: 0;
    transition: color 0.15s ease;
  }

  :global(.cite-link:hover) {
    color: #93bbfd;
    text-decoration: underline;
  }

  :global(.cite-num) {
    color: #a1a1aa;
    font-size: 0.78em;
    font-weight: 600;
  }

  :global(.table-cards) {
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
    margin: 0.5rem 0;
  }

  :global(.table-card) {
    background: #131318;
    border: 1px solid #2a2a35;
    border-radius: 0.6rem;
    padding: 0.85rem 1rem;
    border-left: 3px solid #2563eb;
  }

  :global(.table-card-field) {
    padding: 0.35rem 0;
    border-bottom: 1px solid #1e1e28;
  }

  :global(.table-card-field:last-child) {
    border-bottom: none;
  }

  :global(.table-card-label) {
    font-size: 0.7rem;
    font-weight: 600;
    color: #60a5fa;
    margin-bottom: 0.15rem;
    letter-spacing: 0.03em;
  }

  :global(.table-card-value) {
    font-size: 0.85rem;
    color: #d8d8de;
    line-height: 1.6;
  }

  @media (max-width: 768px) {
    .bubble {
      max-width: 95%;
    }

    .header-row {
      flex-direction: column;
      align-items: stretch;
    }

    .view-toggle {
      width: 100%;
      justify-content: space-between;
    }

    .view-toggle-btn {
      flex: 1 1 0;
    }

    main {
      padding: 1rem;
    }

    header {
      padding: 0.75rem 1rem;
    }

    .sources-header {
      flex-direction: column;
    }

    .developer-grid {
      grid-template-columns: 1fr;
    }

    footer {
      padding: 0.75rem 1rem;
    }

  }
</style>
