<script lang="ts">
  interface Reference {
    id: string;
    source: string;
    text: string;
  }

  interface Message {
    role: 'user' | 'assistant' | 'status';
    content: string;
    metadata?: {
      loops?: number;
      chunks_read_count?: number;
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
      const response = await fetch(`${API_BASE}/api/ask/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      if (!response.body) throw new Error('No response body');

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let answerContent = '';

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
              messages[statusIdx] = { role: 'status', content: event.data };
              messages = [...messages];
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
            } else if (event.type === 'answer') {
              answerContent = event.data;
            } else if (event.type === 'done') {
              messages = messages.filter((_, i) => i !== statusIdx);
              messages = [
                ...messages,
                {
                  role: 'assistant',
                  content: answerContent,
                  metadata: event.data,
                },
              ];
            }
            scrollToBottom();
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
      scrollToBottom();
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

    // Split into blocks by double newlines
    const blocks = text.split(/\n\n+/);
    const html: string[] = [];

    for (const block of blocks) {
      const trimmed = block.trim();
      if (!trimmed) continue;

      // Check if it's a markdown table
      const lines = trimmed.split('\n');
      if (lines.length >= 2 && lines[0].includes('|') && lines[1].match(/^\|[\s\-:|]+\|$/)) {
        html.push(renderTable(lines));
        continue;
      }

      // Check if it's a heading
      const headingMatch = trimmed.match(/^(#{1,4})\s+(.+)$/m);
      if (headingMatch && lines.length === 1) {
        const level = headingMatch[1].length;
        const content = escapeHtml(headingMatch[2]);
        html.push(`<h${level + 2}>${inlineFormat(content)}</h${level + 2}>`);
        continue;
      }

      // Check if it's a list
      if (lines.every(l => l.match(/^\s*[-・•]\s/) || l.match(/^\s*\d+\.\s/) || l.trim() === '')) {
        const items = lines.filter(l => l.trim()).map(l => {
          const content = l.replace(/^\s*[-・•]\s*/, '').replace(/^\s*\d+\.\s*/, '');
          return `<li>${inlineFormat(escapeHtml(content))}</li>`;
        });
        const isOrdered = lines[0]?.match(/^\s*\d+\.\s/);
        const tag = isOrdered ? 'ol' : 'ul';
        html.push(`<${tag}>${items.join('')}</${tag}>`);
        continue;
      }

      // Regular paragraph
      const escaped = lines.map(l => inlineFormat(escapeHtml(l))).join('<br>');
      html.push(`<p>${escaped}</p>`);
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

  function renderTable(lines: string[]): string {
    const parseRow = (line: string) =>
      line.split('|').filter((_, i, a) => i > 0 && i < a.length - 1).map(c => c.trim());

    const headers = parseRow(lines[0]);
    const rows = lines.slice(2).filter(l => l.includes('|')).map(parseRow);

    let table = '<div class="table-wrap"><table><thead><tr>';
    for (const h of headers) {
      table += `<th>${inlineFormat(escapeHtml(h))}</th>`;
    }
    table += '</tr></thead><tbody>';
    for (const row of rows) {
      table += '<tr>';
      for (const cell of row) {
        table += `<td>${inlineFormat(escapeHtml(cell))}</td>`;
      }
      table += '</tr>';
    }
    table += '</tbody></table></div>';
    return table;
  }
</script>

<div class="app">
  <header>
    <button class="header-inner" onclick={resetChat}>
      <h1>会計基準 Q&A</h1>
      <p class="subtitle">日本の会計基準についてAIが回答します</p>
    </button>
  </header>

  <main bind:this={chatContainer}>
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
                  <summary>参照した条文 ({msg.metadata.references.length}件)</summary>
                  <div class="ref-list">
                    {#each msg.metadata.references as ref}
                      <div class="ref-item">
                        <div class="ref-header">
                          <span class="ref-source">{ref.source}</span>
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
                参照チャンク: {msg.metadata.chunks_read_count ?? '?'}
              </div>
            {/if}
          </div>
        </div>
      {/if}
    {/each}
  </main>

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
    max-width: 900px;
    margin: 0 auto;
    overflow: hidden;
  }

  header {
    flex-shrink: 0;
    padding: 1rem 1.5rem;
    border-bottom: 1px solid #1e1e26;
    backdrop-filter: blur(8px);
    background: rgba(15, 15, 18, 0.85);
    z-index: 10;
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
    color: #71717a;
    margin-top: 0.15rem;
  }

  main {
    flex: 1 1 0;
    min-height: 0;
    overflow-y: auto;
    padding: 1.5rem;
    display: flex;
    flex-direction: column;
    gap: 1rem;
    scrollbar-width: thin;
    scrollbar-color: #27272a transparent;
  }

  main::-webkit-scrollbar {
    width: 6px;
  }

  main::-webkit-scrollbar-track {
    background: transparent;
  }

  main::-webkit-scrollbar-thumb {
    background: #27272a;
    border-radius: 3px;
  }

  main::-webkit-scrollbar-thumb:hover {
    background: #3f3f46;
  }

  .welcome {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    flex: 1;
    text-align: center;
    gap: 0.75rem;
    color: #71717a;
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
    background: #1e1e26;
    border: 1px solid #27272a;
    color: #a1a1aa;
    padding: 0.5rem 1rem;
    border-radius: 0.75rem;
    cursor: pointer;
    font-size: 0.8rem;
    font-family: inherit;
    transition: all 0.15s ease;
  }

  .examples button:hover {
    background: #27272a;
    color: #e4e4e7;
    border-color: #3f3f46;
  }

  .message {
    display: flex;
  }

  .message.user {
    justify-content: flex-end;
  }

  .message.status, .message.assistant {
    justify-content: flex-start;
  }

  .bubble {
    max-width: 80%;
    padding: 0.75rem 1rem;
    border-radius: 1rem;
    font-size: 0.9rem;
    line-height: 1.6;
    word-break: break-word;
  }

  .user-bubble {
    background: #2563eb;
    color: #f4f4f5;
    border-bottom-right-radius: 0.25rem;
  }

  .assistant-bubble {
    background: #1e1e26;
    color: #e4e4e7;
    border-bottom-left-radius: 0.25rem;
    border: 1px solid #27272a;
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
    border-top: 1px solid #27272a;
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
    background: #16161b;
    border: 1px solid #27272a;
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
    border-top: 1px solid #27272a;
    font-size: 0.7rem;
    color: #52525b;
  }

  footer {
    flex-shrink: 0;
    padding: 1rem 1.5rem;
    border-top: 1px solid #1e1e26;
    background: rgba(15, 15, 18, 0.95);
    backdrop-filter: blur(8px);
  }

  .input-area {
    display: flex;
    gap: 0.75rem;
    align-items: flex-end;
  }

  textarea {
    flex: 1;
    background: #1e1e26;
    border: 1px solid #27272a;
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
    border-color: #3f3f46;
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
    overflow-x: auto;
    margin: 0.5rem 0;
    border-radius: 0.5rem;
    border: 1px solid #27272a;
  }

  :global(table) {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.8rem;
  }

  :global(th) {
    background: #1a1a22;
    color: #d4d4d8;
    font-weight: 600;
    text-align: left;
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid #3f3f46;
    white-space: nowrap;
  }

  :global(td) {
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid #27272a;
    color: #a1a1aa;
    vertical-align: top;
  }

  :global(tr:last-child td) {
    border-bottom: none;
  }

  :global(tr:hover td) {
    background: #1a1a22;
  }

  :global(.ref-tag) {
    color: #60a5fa;
    font-size: 0.8em;
  }
</style>
