import { expect, test } from '@playwright/test';

type Reference = {
  id: string;
  source: string;
  text: string;
  url: string;
  display_number: number;
  doc_type?: string;
  doc_title?: string;
  section_title?: string;
  section_label?: string;
  page_label?: string;
};

const CASES: Array<{
  question: string;
  answer: string;
  references: Reference[];
  sourceUrlMap: Record<string, string>;
  requestId: string;
  uncertainty?: {
    present: boolean;
    insufficient_points: string[];
    covered_points: string[];
    coverage_ratio: number;
    note: string;
  };
  evidenceCoverage?: {
    slots: string[];
    covered_slots: string[];
    uncovered_slots: string[];
    coverage_ratio: number;
  };
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
        doc_type: '企業会計基準',
        doc_title: '企業会計基準第13号',
        section_title: '第10項',
        section_label: '第10項',
        page_label: 'p.5',
      },
      {
        id: 'std20.pdf:c2',
        source: '実務対応報告第20号 > 第2項',
        text: '第2項 追加の注記が必要である。',
        url: 'https://example.test/std20.pdf',
        display_number: 2,
        doc_type: '実務対応報告',
        doc_title: '実務対応報告第20号',
        section_title: '第2項',
        section_label: '第2項',
        page_label: 'p.1',
      },
    ],
    sourceUrlMap: {
      企業会計基準第13号: 'https://example.test/std13.pdf#page=5',
      実務対応報告第20号: 'https://example.test/std20.pdf',
    },
    requestId: 'req-lease-revision',
    uncertainty: {
      present: false,
      insufficient_points: [],
      covered_points: ['借手', '貸手'],
      coverage_ratio: 1,
      note: '',
    },
    evidenceCoverage: {
      slots: ['借手', '貸手'],
      covered_slots: ['借手', '貸手'],
      uncovered_slots: [],
      coverage_ratio: 1,
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
        doc_type: '企業会計基準',
        doc_title: '企業会計基準第29号',
        section_title: '第47項',
        section_label: '第47項',
        page_label: 'p.12',
      },
      {
        id: 'rev29.pdf:c9',
        source: '企業会計基準第29号 > 第48項',
        text: '代理人は純額を収益認識する。',
        url: 'https://example.test/rev29.pdf#page=13',
        display_number: 2,
        doc_type: '企業会計基準',
        doc_title: '企業会計基準第29号',
        section_title: '第48項',
        section_label: '第48項',
        page_label: 'p.13',
      },
    ],
    sourceUrlMap: {
      企業会計基準第29号: 'https://example.test/rev29.pdf#page=12',
    },
    requestId: 'req-principal-agent',
    uncertainty: {
      present: false,
      insufficient_points: [],
      covered_points: ['本人', '代理人'],
      coverage_ratio: 1,
      note: '',
    },
    evidenceCoverage: {
      slots: ['本人', '代理人'],
      covered_slots: ['本人', '代理人'],
      uncovered_slots: [],
      coverage_ratio: 1,
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
        doc_type: '企業会計基準',
        doc_title: '企業会計基準第13号',
        section_title: '第10項',
        section_label: '第10項',
        page_label: 'p.5',
      },
    ],
    sourceUrlMap: {
      企業会計基準第13号: 'https://example.test/std13.pdf#page=5',
    },
    requestId: 'req-borrower-basic',
    uncertainty: {
      present: false,
      insufficient_points: [],
      covered_points: ['借手'],
      coverage_ratio: 1,
      note: '',
    },
    evidenceCoverage: {
      slots: ['借手'],
      covered_slots: ['借手'],
      uncovered_slots: [],
      coverage_ratio: 1,
    },
    ambiguousText: '企業会計基準第13号',
  },
  {
    question: '借手と貸手の会計処理を教えてください',
    answer: '## 結論\n借手は使用権資産を計上します[1]。\n- 貸手: 今回確認できた根拠では不十分です。',
    references: [
      {
        id: 'std13.pdf:c12',
        source: '企業会計基準第13号 > 第10項',
        text: '第10項 使用権資産を計上する。',
        url: 'https://example.test/std13.pdf#page=5',
        display_number: 1,
        doc_type: '企業会計基準',
        doc_title: '企業会計基準第13号',
        section_title: '第10項',
        section_label: '第10項',
        page_label: 'p.5',
      },
    ],
    sourceUrlMap: {
      企業会計基準第13号: 'https://example.test/std13.pdf#page=5',
    },
    requestId: 'req-partial-answer',
    uncertainty: {
      present: true,
      insufficient_points: ['貸手'],
      covered_points: ['借手'],
      coverage_ratio: 0.5,
      note: '未確定の論点があります。参照カードは確認できた範囲の原典です。',
    },
    evidenceCoverage: {
      slots: ['借手', '貸手'],
      covered_slots: ['借手'],
      uncovered_slots: ['貸手'],
      coverage_ratio: 0.5,
    },
  },
];

function buildSseBody(
  question: string,
  answer: string,
  references: Reference[],
  sourceUrlMap: Record<string, string>,
  requestId: string,
  evidenceCoverage?: Record<string, unknown>,
  uncertainty?: Record<string, unknown>,
): string {
  const events = [
    { type: 'status', data: '調査中...' },
    { type: 'tool_call', data: { tool: 'hybrid_search', args: { query: question } } },
    { type: 'tool_call', data: { tool: 'read_chunk', args: { chunk_ids: references.map((ref) => ref.id) } } },
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
        request_id: requestId,
        query_class: 'simple',
        evidence_coverage: evidenceCoverage ?? {},
        uncertainty: uncertainty ?? { present: false, insufficient_points: [], covered_points: [], coverage_ratio: 1, note: '' },
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
  let uiEvents: Array<Record<string, unknown>> = [];

  test.beforeEach(async ({ page, context }) => {
    servedUrls = [];
    uiEvents = [];
    await context.route('https://example.test/**', async (route) => {
      servedUrls.push(route.request().url().split('#')[0]);
      await route.fulfill({
        status: 200,
        contentType: 'text/html',
        body: '<html><body>ok</body></html>',
      });
    });

    await page.route('http://localhost:8000/api/ui-event', async (route) => {
      uiEvents.push(route.request().postDataJSON() as Record<string, unknown>);
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ ok: true }),
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
        body: buildSseBody(
          match.question,
          match.answer,
          match.references,
          match.sourceUrlMap,
          match.requestId,
          match.evidenceCoverage,
          match.uncertainty,
        ),
      });
    });
  });

  for (const scenario of CASES.filter((item) => !item.uncertainty?.present)) {
    test(`renders aligned references for: ${scenario.question}`, async ({ page, context }) => {
      await page.goto('/');
      await page.getByTestId('question-input').fill(scenario.question);
      await page.getByTestId('send-button').click();

      for (const snippet of expectedTextSnippets(scenario.answer)) {
      await expect(page.getByTestId('answer-content')).toContainText(snippet);
      }
      await expect(page.locator('.assistant-bubble .ref-link')).toHaveCount(scenario.references.length);
      await expect(page.getByTestId('reference-card')).toHaveCount(scenario.references.length);
      await expect(page.getByTestId('uncertainty-banner')).toHaveCount(0);

      for (const ref of scenario.references) {
        const card = page.locator(`[data-testid="reference-card"][data-reference-number="${ref.display_number}"]`);
        const link = page.locator(`[data-testid="reference-link"][data-reference-number="${ref.display_number}"]`);
        await expect(card).toContainText(`[${ref.display_number}]`);
        await expect(card.getByTestId('reference-doc-type')).toHaveText(ref.doc_type ?? '');
        await expect(card.getByTestId('reference-doc-title')).toHaveText(ref.doc_title ?? '');
        await expect(card.getByTestId('reference-section-label')).toHaveText(ref.section_label ?? '');
        await expect(link).toHaveAttribute('href', ref.url);
      }

      await expect(page.getByTestId('developer-panel')).toHaveCount(0);
      await page.getByTestId('view-toggle-developer').click();
      await expect(page.getByTestId('developer-panel')).toBeVisible();
      await expect(page.getByTestId('developer-request-id')).toHaveText(scenario.requestId);

      const firstRef = scenario.references[0];
      const popupPromise = page.waitForEvent('popup');
      await page.locator(`[data-testid="reference-link"][data-reference-number="${firstRef.display_number}"]`).click();
      const popup = await popupPromise;
      await popup.waitForLoadState('domcontentloaded');
      expect(servedUrls).toContain(firstRef.url.split('#')[0]);
      await popup.close();
      expect(uiEvents.some((event) => event.event === 'reference_opened' && event.request_id === scenario.requestId)).toBeTruthy();

      if (scenario.ambiguousText) {
        await expect(
          page.locator('[data-testid="answer-content"] a').filter({ hasText: scenario.ambiguousText })
        ).toHaveCount(0);
      }
    });
  }

  test('shows uncertainty banner only for partial-support answers', async ({ page }) => {
    const scenario = CASES.find((item) => item.uncertainty?.present);
    if (!scenario) test.fail();

    await page.goto('/');
    await page.getByTestId('question-input').fill(scenario!.question);
    await page.getByTestId('send-button').click();

    await expect(page.getByTestId('uncertainty-banner')).toBeVisible();
    await expect(page.getByTestId('uncertainty-banner')).toContainText('貸手');
    await expect(page.getByTestId('reference-card')).toHaveCount(1);
  });

  test('emits followup_submitted after a second user question', async ({ page }) => {
    await page.goto('/');
    await page.getByTestId('question-input').fill(CASES[0].question);
    await page.getByTestId('send-button').click();
    await expect(page.getByTestId('reference-card')).toHaveCount(CASES[0].references.length);

    await page.getByTestId('question-input').fill(CASES[1].question);
    await page.getByTestId('send-button').click();
    await expect(page.getByTestId('reference-card').nth(1)).toBeVisible();

    expect(uiEvents.some((event) => event.event === 'followup_submitted')).toBeTruthy();
    expect(uiEvents.filter((event) => event.event === 'answer_rendered').length).toBeGreaterThanOrEqual(2);
  });
});
