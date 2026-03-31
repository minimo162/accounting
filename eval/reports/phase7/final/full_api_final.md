# Answer Eval Report

- Cases: 21
- Passed: 21
- Failed: 0
- Pass rate: 100%
- Avg latency: 60.8s
- P95 latency: 89.8s
- Avg loops: 5.7
- Avg read_chunk calls: 7.4
- Avg retrieved tokens: 14460
- Avg cited lines: 14.2
- Avg uncited lines: 0.0
- Reference alignment failures: 0
- Missing reference URL failures: 0
- Insufficient detail failures: 0
- Exact clause failures: 0
- Gate mode: hard
- Gate passed: yes
- Blocking failures: 0
- Hard-fail cases: 0
- Soft-fail cases: 0

## Cases

### PASS lease_revision_detail
- Question: リース会計基準の改正点と経過措置を、借手の会計処理・貸手の扱い・関連基準への影響に分けて詳しく教えてください
- Latency: 66.3s
- Loops: 6
- read_chunk calls: 9
- Retrieved tokens: 17309
- Citations: 6
- Cited lines: 20
- Inline citations: 6
- References: 6
- Max citation number: 6
- Uncited lines: 0
- Reference alignment: ok
- Stop reason: evidence_sufficient
- Notes: 長文・複数論点の代表ケース。Phase 7 の full API baseline では 7 loop まで伸びるため、その範囲で detail を落とさず返せるかを監視する。

### PASS lease_revision_detail_depth
- Question: リース会計基準の改正点と経過措置を、借手の会計処理・貸手の扱い・関連基準への影響に分けて詳しく教えてください
- Latency: 64.0s
- Loops: 7
- read_chunk calls: 8
- Retrieved tokens: 17520
- Citations: 8
- Cited lines: 20
- Inline citations: 8
- References: 8
- Max citation number: 8
- Uncited lines: 0
- Reference alignment: ok
- Stop reason: evidence_sufficient
- Notes: detail insufficiency を検知する補助ケース。Phase 7 baseline では 6-7 loop に揺れるため、7 loop 以内で各論点が cited line として残るかを確認する。

### PASS lease_exact_clause
- Question: 企業会計基準第13号第10項では借手のリースをどのように扱いますか
- Latency: 18.2s
- Loops: 3
- read_chunk calls: 2
- Retrieved tokens: 1342
- Citations: 1
- Cited lines: 2
- Inline citations: 1
- References: 1
- Max citation number: 1
- Uncited lines: 0
- Reference alignment: ok
- Stop reason: evidence_sufficient
- Notes: 正確な条項指定クエリ。現行ベースラインでは 7-8 loop 程度で、token 使用量の揺れが大きいため、主に loop 数の悪化を監視する。

### PASS lease_borrower_initial_recognition
- Question: 借手はリース開始日に何を認識しますか。使用権資産とリース負債の関係が分かるように説明してください
- Latency: 89.6s
- Loops: 5
- read_chunk calls: 6
- Retrieved tokens: 14729
- Citations: 4
- Cited lines: 12
- Inline citations: 4
- References: 4
- Max citation number: 4
- Uncited lines: 0
- Reference alignment: ok
- Stop reason: natural
- Notes: 借手の基本処理を確認するケース。Phase 7 の full API baseline では latency が 75 秒台後半から 80 秒台に乗るため、まずは 90 秒を上限に監視する。

### PASS lease_short_term_low_value
- Question: 借手は短期リースや少額リースをどのように扱いますか
- Latency: 30.6s
- Loops: 3
- read_chunk calls: 6
- Retrieved tokens: 4979
- Citations: 4
- Cited lines: 8
- Inline citations: 4
- References: 4
- Max citation number: 4
- Uncited lines: 0
- Reference alignment: ok
- Stop reason: evidence_sufficient
- Notes: 例外的な取扱いの確認。短期・少額のキーワードが落ちないことを見る。

### PASS revenue_principal_agent
- Question: 収益認識基準における本人と代理人の区分はどう判断しますか
- Latency: 46.8s
- Loops: 5
- read_chunk calls: 4
- Retrieved tokens: 10918
- Citations: 2
- Cited lines: 12
- Inline citations: 2
- References: 2
- Max citation number: 2
- Uncited lines: 0
- Reference alignment: ok
- Stop reason: evidence_sufficient
- Notes: 収益認識の代表ケース。現行ベースラインを超える loop 悪化と、判断軸・表示額の欠落を検知する。

### PASS performance_obligation_identification
- Question: 収益認識基準で履行義務はどのように識別しますか。保守サービスや値引きのある契約を念頭に説明してください
- Latency: 59.2s
- Loops: 6
- read_chunk calls: 8
- Retrieved tokens: 17401
- Citations: 7
- Cited lines: 13
- Inline citations: 7
- References: 7
- Max citation number: 7
- Uncited lines: 0
- Reference alignment: ok
- Stop reason: evidence_sufficient
- Notes: 契約の分解説明が必要なケース。現行ベースライン近辺の token 使用量を超えると失敗にする。

### PASS performance_obligation_identification_detail
- Question: 収益認識基準で履行義務はどのように識別しますか。保守サービスや値引きのある契約を念頭に説明してください
- Latency: 87.3s
- Loops: 5
- read_chunk calls: 6
- Retrieved tokens: 11474
- Citations: 4
- Cited lines: 10
- Inline citations: 4
- References: 4
- Max citation number: 4
- Uncited lines: 0
- Reference alignment: ok
- Stop reason: natural
- Notes: 浅い要約ではなく、契約分解の具体論点が cited line として残るかを見る detail ケース。

### PASS revenue_variable_consideration
- Question: 収益認識基準における変動対価はどのように見積もり、いつ収益に反映しますか
- Latency: 56.6s
- Loops: 6
- read_chunk calls: 10
- Retrieved tokens: 18602
- Citations: 8
- Cited lines: 24
- Inline citations: 8
- References: 8
- Max citation number: 8
- Uncited lines: 0
- Reference alignment: ok
- Stop reason: evidence_sufficient
- Notes: 見積りと制約の説明が必要なケース。Phase 7 の full API baseline では retrieved_tokens が 23k 台まで伸びるため、まずは 24k を上限に監視する。

### PASS revenue_contract_modification
- Question: 収益認識基準で契約変更はどのように処理しますか。既存契約の継続か、新しい契約として扱うかの観点で教えてください
- Latency: 57.7s
- Loops: 5
- read_chunk calls: 8
- Retrieved tokens: 16864
- Citations: 4
- Cited lines: 13
- Inline citations: 4
- References: 4
- Max citation number: 4
- Uncited lines: 0
- Reference alignment: ok
- Stop reason: evidence_sufficient
- Notes: 契約変更の分岐説明を確認するケース。継続処理と別個処理の両方に触れたい。

### PASS revenue_point_in_time_vs_over_time
- Question: 収益認識基準では履行義務を一定の期間にわたり充足するのか、一時点で充足するのかをどう判断しますか
- Latency: 57.1s
- Loops: 5
- read_chunk calls: 9
- Retrieved tokens: 17459
- Citations: 7
- Cited lines: 28
- Inline citations: 7
- References: 7
- Max citation number: 7
- Uncited lines: 0
- Reference alignment: ok
- Stop reason: evidence_sufficient
- Notes: 充足時点の判断基準を確認するケース。期間基準と一時点基準の両方に触れるかを見る。

### PASS hedge_accounting_requirements
- Question: 繰延ヘッジを適用するための主な要件を教えてください
- Latency: 44.4s
- Loops: 5
- read_chunk calls: 11
- Retrieved tokens: 3960
- Citations: 4
- Cited lines: 7
- Inline citations: 4
- References: 4
- Max citation number: 4
- Uncited lines: 0
- Reference alignment: ok
- Stop reason: evidence_sufficient
- Notes: 要件列挙の精度確認。形式要件と有効性評価の両方を含めたい。

### PASS deferred_tax_rate_change
- Question: 税効果会計で繰延税金資産・負債の計算に使う税率は何ですか。税率変更時の扱いも教えてください
- Latency: 44.6s
- Loops: 6
- read_chunk calls: 10
- Retrieved tokens: 12855
- Citations: 4
- Cited lines: 4
- Inline citations: 4
- References: 4
- Max citation number: 4
- Uncited lines: 0
- Reference alignment: ok
- Stop reason: evidence_sufficient
- Notes: 税効果会計の基本論点。税率の基準時点と変更時の再計算に触れるかを見る。

### PASS deferred_tax_recoverability
- Question: 税効果会計で繰延税金資産の回収可能性はどのように判断しますか
- Latency: 56.9s
- Loops: 6
- read_chunk calls: 10
- Retrieved tokens: 18910
- Citations: 8
- Cited lines: 21
- Inline citations: 8
- References: 8
- Max citation number: 8
- Uncited lines: 0
- Reference alignment: ok
- Stop reason: evidence_sufficient
- Notes: 回収可能性判断の確認。将来課税所得やスケジューリングへの言及を見たい。

### PASS impairment_indication
- Question: 固定資産の減損会計では、どのような場合に減損の兆候があると判断しますか
- Latency: 60.2s
- Loops: 5
- read_chunk calls: 10
- Retrieved tokens: 16802
- Citations: 4
- Cited lines: 22
- Inline citations: 4
- References: 4
- Max citation number: 4
- Uncited lines: 0
- Reference alignment: ok
- Stop reason: evidence_sufficient
- Notes: 企業会計審議会系の代表ケース。兆候と回収可能価額の基礎語が出るかを見る。

### PASS r_and_d_cost_treatment
- Question: 研究開発費はどのように会計処理しますか。ソフトウェア開発費との違いにも触れてください
- Latency: 64.7s
- Loops: 5
- read_chunk calls: 12
- Retrieved tokens: 23441
- Citations: 5
- Cited lines: 15
- Inline citations: 5
- References: 5
- Max citation number: 5
- Uncited lines: 0
- Reference alignment: ok
- Stop reason: evidence_sufficient
- Notes: 研究開発費等に係る会計基準の確認。2026-03-29 の local baseline では retrieved_tokens が 26k 台まで伸びたため、まずは 30k を上限とする。

### PASS retirement_benefit_actuarial_difference
- Question: 退職給付会計における数理計算上の差異はどのように処理しますか
- Latency: 49.4s
- Loops: 5
- read_chunk calls: 11
- Retrieved tokens: 13359
- Citations: 5
- Cited lines: 14
- Inline citations: 5
- References: 5
- Max citation number: 5
- Uncited lines: 0
- Reference alignment: ok
- Stop reason: evidence_sufficient
- Notes: 退職給付の代表ケース。差異の費用処理と期間配分の説明を確認する。

### PASS consolidation_unrealized_gain_elimination
- Question: 連結会計では、親会社が子会社へ土地を譲渡した場合の未実現損益をどのように処理しますか
- Latency: 71.8s
- Loops: 5
- read_chunk calls: 6
- Retrieved tokens: 14100
- Citations: 4
- Cited lines: 11
- Inline citations: 4
- References: 4
- Max citation number: 4
- Uncited lines: 0
- Reference alignment: ok
- Stop reason: natural
- Notes: 連結会計の新規ケース。親子会社間の土地譲渡に伴う未実現損益の消去に触れられるかを見る。

### PASS consolidation_scope_determination
- Question: 会社計算規則では、どのような子会社を連結の範囲に含め、どのような場合に除外できますか
- Latency: 89.8s
- Loops: 5
- read_chunk calls: 6
- Retrieved tokens: 15332
- Citations: 3
- Cited lines: 20
- Inline citations: 3
- References: 3
- Max citation number: 3
- Uncited lines: 0
- Reference alignment: ok
- Stop reason: natural
- Notes: 連結範囲判定の新規ケース。現行 local baseline では loop 数と retrieved_tokens が高めのため、まずは 12 loop / 24k token を上限に監視する。

### PASS verification_land_revaluation_judgment
- Question: 私は、再評価済み土地の売却損は連結上の未実現損失として必ずしも消去しないと考えています。また、土地再評価差額金は子会社への土地譲渡後もそのまま引き継がれる理解です。依拠条文として連結財務諸表に関する会計基準第36条と土地再評価差額金の会計処理に関するＱ＆Ａを考えています。この判断は妥当ですか。
- Latency: 92.1s
- Loops: 10
- read_chunk calls: 1
- Retrieved tokens: 17724
- Citations: 3
- Cited lines: 7
- Inline citations: 3
- References: 3
- Max citation number: 3
- Uncited lines: 0
- Reference alignment: ok
- Stop reason: natural
- Notes: 判断検証型の新規ケース。土地譲渡・再評価・連結消去に関する論点を分解して確認できるかを、実回答に現れる中核語で監視する。

### PASS cross_reference_land_revaluation
- Question: 土地再評価差額金は子会社への土地譲渡後の連結上どう扱いますか。連結財務諸表に関する会計基準第36条と土地再評価差額金の会計処理に関するＱ＆Ａの両方を踏まえて説明してください
- Latency: 70.5s
- Loops: 11
- read_chunk calls: 3
- Retrieved tokens: 18589
- Citations: 3
- Cited lines: 16
- Inline citations: 3
- References: 3
- Max citation number: 3
- Uncited lines: 0
- Reference alignment: ok
- Stop reason: retrieval_budget
- Notes: クロスリファレンス検索の新規ケース。連結基準と土地再評価Q&Aをまたぐ説明ができるかを、現行 answer の表現に合わせて確認する。
