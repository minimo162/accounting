<script lang="ts">
  interface Reference {
    id: string;
    source: string;
    text: string;
    url?: string;
  }

  interface Message {
    role: 'user' | 'assistant' | 'status';
    content: string;
    metadata?: {
      loops?: number;
      chunks_read_count?: number;
      read_chunk_count?: number;
      total_cost?: number;
      references?: Reference[];
    };
  }

  let messages: Message[] = $state([]);
  let input = $state('');
  let loading = $state(false);
  let chatContainer: HTMLElement;

  const API_BASE = import.meta.env.DEV ? 'http://localhost:8000' : '';

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

  function getDisplayedReadCount(message: Message): number | string {
    return message.metadata?.references?.length ?? message.metadata?.read_chunk_count ?? '?';
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

  async function sendMessage() {
    const question = input.trim();
    if (!question || loading) return;

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
              const toolLabels: Record<string, string> = {
                keyword_search: 'キーワード検索',
                semantic_search: '意味検索',
                read_chunk: 'チャンク読取',
              };
              messages[statusIdx] = {
                role: 'status',
                content: `${toolLabels[toolName] || toolName}を実行中...`,
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
                references: collectedRefs.length > 0 ? collectedRefs : event.data.references,
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
  function formatMarkdown(text: string): string {
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
          const escaped = currentParagraph.map(l => inlineFormat(escapeHtml(l))).join('<br>');
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
          html.push(`<h${level + 2}>${inlineFormat(content)}</h${level + 2}>`);
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
            listItems.push(`<li>${inlineFormat(escapeHtml(content))}</li>`);
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

  function inlineFormat(text: string): string {
    return text
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/`(.*?)`/g, '<code>$1</code>')
      .replace(/【(.*?)】/g, '<span class="ref-tag">$1</span>');
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
        <button class="header-inner" onclick={resetChat}>
          <h1>会計基準 Q&A</h1>
          <p class="subtitle">日本の会計基準についてAIが回答します</p>
        </button>
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
          <div class="bubble assistant-bubble">
            {@html formatMarkdown(msg.content)}
            {#if msg.metadata?.references?.length}
              <div class="references">
                <details>
                  <summary>読んだ候補条文 ({getDisplayedReadCount(msg)}件)</summary>
                  <div class="ref-list">
                    {#each msg.metadata.references as ref}
                      <div class="ref-item">
                        <div class="ref-header">
                          {#if ref.url}
                            <a class="ref-source ref-link" href={ref.url} target="_blank" rel="noopener noreferrer">
                              {ref.source}
                              <span class="ref-link-icon">&#x2197;</span>
                            </a>
                          {:else}
                            <span class="ref-source">{ref.source}</span>
                          {/if}
                        </div>
                        <details class="ref-details">
                          <summary>原文を表示</summary>
                          <div class="ref-text">{ref.text}</div>
                        </details>
                      </div>
                    {/each}
                  </div>
                </details>
              </div>
            {/if}
            {#if msg.metadata}
              <div class="meta">
                検索ステップ: {msg.metadata.loops ?? '?'} |
                引用: {getDisplayedCitationCount(msg)}件 |
                候補: {getDisplayedReadCount(msg)}件
              </div>
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
        onkeydown={handleKeydown}
        placeholder="会計基準について質問してください..."
        rows="1"
        disabled={loading}
      ></textarea>
      <button class="send-btn" onclick={sendMessage} disabled={loading || !input.trim()}>
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

  .references {
    margin-top: 0.75rem;
    padding-top: 0.75rem;
    border-top: 1px solid #2a2a35;
  }

  .references details {
    cursor: pointer;
  }

  .references summary {
    font-size: 0.8rem;
    font-weight: 500;
    color: #a1a1aa;
    padding: 0.25rem 0;
    user-select: none;
  }

  .references summary:hover {
    color: #d4d4d8;
  }

  .ref-list {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    margin-top: 0.5rem;
    max-height: 400px;
    overflow-y: auto;
    scrollbar-width: thin;
    scrollbar-color: #27272a transparent;
  }

  .ref-list::-webkit-scrollbar {
    width: 4px;
  }

  .ref-list::-webkit-scrollbar-thumb {
    background: #27272a;
    border-radius: 2px;
  }

  .ref-item {
    background: #131318;
    border: 1px solid #2a2a35;
    border-radius: 0.5rem;
    padding: 0.75rem;
    font-size: 0.8rem;
  }

  .ref-header {
    margin-bottom: 0.35rem;
  }

  .ref-source {
    color: #d4d4d8;
    font-size: 0.8rem;
    font-weight: 500;
  }

  .ref-link {
    color: #60a5fa;
    text-decoration: none;
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    transition: color 0.15s ease;
  }

  .ref-link:hover {
    color: #93bbfd;
    text-decoration: underline;
  }

  .ref-link-icon {
    font-size: 0.7rem;
    opacity: 0.7;
  }

  .ref-details {
    cursor: pointer;
  }

  .ref-details summary {
    font-size: 0.7rem;
    color: #60a5fa;
    padding: 0.15rem 0;
    user-select: none;
  }

  .ref-details summary:hover {
    color: #93bbfd;
  }

  .ref-text {
    color: #a1a1aa;
    line-height: 1.5;
    white-space: pre-wrap;
    font-size: 0.75rem;
    max-height: 300px;
    overflow-y: auto;
    margin-top: 0.35rem;
    padding-top: 0.35rem;
    border-top: 1px solid #27272a;
    scrollbar-width: thin;
    scrollbar-color: #27272a transparent;
  }

  .ref-text::-webkit-scrollbar {
    width: 4px;
  }

  .ref-text::-webkit-scrollbar-thumb {
    background: #27272a;
    border-radius: 2px;
  }

  .meta {
    margin-top: 0.5rem;
    padding-top: 0.5rem;
    border-top: 1px solid #2a2a35;
    font-size: 0.7rem;
    color: #5a5a65;
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

    main {
      padding: 1rem;
    }

    header {
      padding: 0.75rem 1rem;
    }

    footer {
      padding: 0.75rem 1rem;
    }

  }
</style>
