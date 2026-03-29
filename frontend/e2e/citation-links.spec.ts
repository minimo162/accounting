import { expect, test } from '@playwright/test';

type Reference = {
  id: string;
  source: string;
  text: string;
  url: string;
  display_number: number;
};

const CASES: Array<{
  question: string;
  answer: string;
  references: Reference[];
  sourceUrlMap: Record<string, string>;
  ambiguousText?: string;
}> = [
  {
    question: 'リース会計基準の改正点を教えてください',
    answer: '## 結論\n借手は使用権資産を計上します[1]。貸手は開示を見直します[2]。',
    references: [
      {
        id: 'std13.pdf:c12',
        source: '企業会計基準第13号 > 第10項',
        text: '第10項 使用権資産を計上する。',
        url: 'https://example.test/std13.pdf#page=5',
        display_number: 1,
      },
      {
        id: 'std20.pdf:c2',
        source: '実務対応報告第20号 > 第2項',
        text: '第2項 追加の注記が必要である。',
        url: 'https://example.test/std20.pdf',
        display_number: 2,
      },
    ],
    sourceUrlMap: {
      企業会計基準第13号: 'https://example.test/std13.pdf#page=5',
      実務対応報告第20号: 'https://example.test/std20.pdf',
    },
  },
  {
    question: '収益認識基準における本人と代理人の区分はどう判断しますか',
    answer: '## 判断\n本人は支配に基づいて総額表示します[1]。代理人は純額表示します[2]。',
    references: [
      {
        id: 'rev29.pdf:c8',
        source: '企業会計基準第29号 > 第47項',
        text: '支配の有無で本人か代理人かを判断する。',
        url: 'https://example.test/rev29.pdf#page=12',
        display_number: 1,
      },
      {
        id: 'rev29.pdf:c9',
        source: '企業会計基準第29号 > 第48項',
        text: '代理人は純額を収益認識する。',
        url: 'https://example.test/rev29.pdf#page=13',
        display_number: 2,
      },
    ],
    sourceUrlMap: {
      企業会計基準第29号: 'https://example.test/rev29.pdf#page=12',
    },
  },
  {
    question: '借手の会計処理を教えてください',
    answer: '## 範囲\n借手（連結子会社を含む）は使用権資産を計上します[1]。企業会計基準第13号という語が本文に出ても、そこ自体はリンクしません。',
    references: [
      {
        id: 'std13.pdf:c12',
        source: '企業会計基準第13号 > 第10項',
        text: '第10項 使用権資産を計上する。',
        url: 'https://example.test/std13.pdf#page=5',
        display_number: 1,
      },
    ],
    sourceUrlMap: {
      企業会計基準第13号: 'https://example.test/std13.pdf#page=5',
    },
    ambiguousText: '企業会計基準第13号',
  },
];

function buildSseBody(answer: string, references: Reference[], sourceUrlMap: Record<string, string>): string {
  const events = [
    { type: 'status', data: '調査中...' },
    { type: 'answer_delta', data: answer },
    ...references.map((ref) => ({ type: 'reference', data: ref })),
    {
      type: 'done',
      data: {
        loops: 2,
        stop_reason: 'natural',
        chunks_read_count: references.length,
        read_chunk_count: references.length,
        total_cost: 0,
        total_retrieved_tokens: 1000,
        source_url_map: sourceUrlMap,
      },
    },
  ];
  return `${events.map((event) => `data: ${JSON.stringify(event)}\n`).join('\n')}\n`;
}

function expectedTextSnippets(answer: string): string[] {
  return answer
    .replace(/^#{1,6}\s+/gm, '')
    .replace(/\[\d+\]/g, '')
    .split(/[\n。]+/)
    .map((part) => part.trim())
    .filter(Boolean);
}

test.describe('citation and reference rendering', () => {
  let servedUrls: string[] = [];

  test.beforeEach(async ({ page, context }) => {
    servedUrls = [];
    await context.route('https://example.test/**', async (route) => {
      servedUrls.push(route.request().url().split('#')[0]);
      await route.fulfill({
        status: 200,
        contentType: 'text/html',
        body: '<html><body>ok</body></html>',
      });
    });

    await page.route('http://localhost:8000/api/ask/stream', async (route) => {
      const payload = route.request().postDataJSON() as { question: string };
      const match = CASES.find((item) => item.question === payload.question);
      if (!match) {
        await route.fulfill({ status: 500, body: 'unknown question' });
        return;
      }
      await route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        body: buildSseBody(match.answer, match.references, match.sourceUrlMap),
      });
    });
  });

  for (const scenario of CASES) {
    test(`renders aligned references for: ${scenario.question}`, async ({ page, context }) => {
      await page.goto('/');
      await page.getByTestId('question-input').fill(scenario.question);
      await page.getByTestId('send-button').click();

      for (const snippet of expectedTextSnippets(scenario.answer)) {
        await expect(page.getByTestId('answer-content')).toContainText(snippet);
      }
      await expect(page.locator('.assistant-bubble .ref-link')).toHaveCount(scenario.references.length);
      await expect(page.getByTestId('reference-card')).toHaveCount(scenario.references.length);

      for (const ref of scenario.references) {
        const card = page.locator(`[data-testid="reference-card"][data-reference-number="${ref.display_number}"]`);
        const link = page.locator(`[data-testid="reference-link"][data-reference-number="${ref.display_number}"]`);
        await expect(card).toContainText(`[${ref.display_number}]`);
        await expect(link).toHaveAttribute('href', ref.url);
      }

      const firstRef = scenario.references[0];
      const popupPromise = page.waitForEvent('popup');
      await page.locator(`[data-testid="reference-link"][data-reference-number="${firstRef.display_number}"]`).click();
      const popup = await popupPromise;
      await popup.waitForLoadState('domcontentloaded');
      expect(servedUrls).toContain(firstRef.url.split('#')[0]);
      await popup.close();

      if (scenario.ambiguousText) {
        await expect(
          page.locator('[data-testid="answer-content"] a').filter({ hasText: scenario.ambiguousText })
        ).toHaveCount(0);
      }
    });
  }
});
