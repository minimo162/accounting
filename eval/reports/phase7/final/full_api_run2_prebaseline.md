# Answer Eval Report

- Cases: 21
- Passed: 18
- Failed: 3
- Pass rate: 86%
- Avg latency: 59.2s
- P95 latency: 83.6s
- Avg loops: 5.7
- Avg read_chunk calls: 7.2
- Avg retrieved tokens: 14318
- Avg cited lines: 13.5
- Avg uncited lines: 0.0
- Reference alignment failures: 0
- Missing reference URL failures: 0
- Insufficient detail failures: 0
- Exact clause failures: 0
- Gate mode: hard
- Gate passed: yes
- Blocking failures: 0
- Hard-fail cases: 0
- Soft-fail cases: 3

## Cases

### FAIL lease_revision_detail
- Question: リース会計基準の改正点と経過措置を、借手の会計処理・貸手の扱い・関連基準への影響に分けて詳しく教えてください
- Latency: 60.7s
- Loops: 7
- read_chunk calls: 7
- Retrieved tokens: 15413
- Citations: 6
- Cited lines: 12
- Inline citations: 6
- References: 6
- Max citation number: 6
- Uncited lines: 0
- Reference alignment: ok
- Stop reason: evidence_sufficient
- Failures: loops>6 (7)
- Soft failures: loops>6 (7)
- Answer preview: ## リース会計基準の改正点と経過措置
### 1. 改正の背景と目的
国際会計基準審議会（IASB）が2016年にIFRS第16号を、米国財務会計基準審議会（FASB）が同年にTopic 842を公表したことにより、国際的な会計基準と日本のリース会計基準との間に、特に負債の認識において違いが生じ、国際比較上の議論となる可能性があった[1]。これに対応し、国際的な会計基準との整合性を図り財務諸表間の比較可能性を高めること、および借手のすべてのリースについて資産及び負債を計上する会計基準の開発に対するニーズに応えることを目的として改正が行われた[1]。

### 2. 借手の会計処理に関する主な改正点
- **原則的な会計処理**: 借手は、すべてのリースについて、リース開始日に「使用権資産」と「リース負債」を計上する[1]。
- **使用権資産の償却**: 原資産の所有権が借手に移転すると認められるリースの場合、使用権資産の減価償却費は、原資産を自ら所有していたと仮定した場合に適用する減価償却方法と同一の方法により算定する[2]。それ以外のリースでは、定額法等の中から企業の実態に応じた
- Notes: 長文・複数論点の代表ケース。detail を落とさずに 6 loop 以内で返せるかを見る。

### PASS lease_revision_detail_depth
- Question: リース会計基準の改正点と経過措置を、借手の会計処理・貸手の扱い・関連基準への影響に分けて詳しく教えてください
- Latency: 56.5s
- Loops: 6
- read_chunk calls: 9
- Retrieved tokens: 17147
- Citations: 5
- Cited lines: 16
- Inline citations: 5
- References: 5
- Max citation number: 5
- Uncited lines: 0
- Reference alignment: ok
- Stop reason: evidence_sufficient
- Notes: detail insufficiency を検知する補助ケース。各論点が cited line として残っているかを確認する。

### PASS lease_exact_clause
- Question: 企業会計基準第13号第10項では借手のリースをどのように扱いますか
- Latency: 16.8s
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

### FAIL lease_borrower_initial_recognition
- Question: 借手はリース開始日に何を認識しますか。使用権資産とリース負債の関係が分かるように説明してください
- Latency: 75.7s
- Loops: 5
- read_chunk calls: 6
- Retrieved tokens: 14729
- Citations: 4
- Cited lines: 9
- Inline citations: 4
- References: 4
- Max citation number: 4
- Uncited lines: 0
- Reference alignment: ok
- Stop reason: natural
- Failures: latency>75.0s (75.7s)
- Soft failures: latency>75.0s (75.7s)
- Answer preview: ## リース開始日に借手が認識するものと使用権資産・リース負債の関係
借手はリース開始日に、使用権資産とリース負債の両方を認識します[1]。

### リース開始日に認識する項目
借手は、リース開始日に会計基準第34項に従い算定された額によりリース負債を計上します[1]。リース負債の計上額は、原則としてリース開始日において未払である借手のリース料からこれに含まれている利息相当額の合理的な見積額を控除し、現在価値により算定する方法によります[1]。

借手は、リース開始日に算定されたリース負債に、リース開始日までに支払った借手のリース料、付随費用及び資産除去債務に対応する除去費用を加算し、受け取ったリース・インセンティブを控除した額により使用権資産を計上します[1]。

### 使用権資産とリース負債の関係
使用権資産の計上額は、リース開始日に算定されたリース負債の計上額を基礎として、追加の支払項目を加算し、リース・インセンティブを控除して算定されます[2]。これは、ファイナンス・リースであるかオペレーティング・リースであるかにかかわらず、すべてのリースについて適用される単一の会計処理モ
- Notes: 借手の基本処理を確認するケース。2026-03-29 の local baseline では retrieved_tokens が 23k 台まで伸びたため、まずは 26k を上限とする。

### PASS lease_short_term_low_value
- Question: 借手は短期リースや少額リースをどのように扱いますか
- Latency: 24.4s
- Loops: 3
- read_chunk calls: 6
- Retrieved tokens: 4144
- Citations: 3
- Cited lines: 9
- Inline citations: 3
- References: 3
- Max citation number: 3
- Uncited lines: 0
- Reference alignment: ok
- Stop reason: evidence_sufficient
- Notes: 例外的な取扱いの確認。短期・少額のキーワードが落ちないことを見る。

### PASS revenue_principal_agent
- Question: 収益認識基準における本人と代理人の区分はどう判断しますか
- Latency: 35.6s
- Loops: 5
- read_chunk calls: 4
- Retrieved tokens: 10918
- Citations: 2
- Cited lines: 11
- Inline citations: 2
- References: 2
- Max citation number: 2
- Uncited lines: 0
- Reference alignment: ok
- Stop reason: evidence_sufficient
- Notes: 収益認識の代表ケース。現行ベースラインを超える loop 悪化と、判断軸・表示額の欠落を検知する。

### PASS performance_obligation_identification
- Question: 収益認識基準で履行義務はどのように識別しますか。保守サービスや値引きのある契約を念頭に説明してください
- Latency: 77.4s
- Loops: 5
- read_chunk calls: 6
- Retrieved tokens: 11474
- Citations: 6
- Cited lines: 9
- Inline citations: 6
- References: 6
- Max citation number: 6
- Uncited lines: 0
- Reference alignment: ok
- Stop reason: natural
- Notes: 契約の分解説明が必要なケース。現行ベースライン近辺の token 使用量を超えると失敗にする。

### PASS performance_obligation_identification_detail
- Question: 収益認識基準で履行義務はどのように識別しますか。保守サービスや値引きのある契約を念頭に説明してください
- Latency: 75.9s
- Loops: 5
- read_chunk calls: 6
- Retrieved tokens: 11474
- Citations: 6
- Cited lines: 12
- Inline citations: 6
- References: 6
- Max citation number: 6
- Uncited lines: 0
- Reference alignment: ok
- Stop reason: natural
- Notes: 浅い要約ではなく、契約分解の具体論点が cited line として残るかを見る detail ケース。

### FAIL revenue_variable_consideration
- Question: 収益認識基準における変動対価はどのように見積もり、いつ収益に反映しますか
- Latency: 46.4s
- Loops: 6
- read_chunk calls: 12
- Retrieved tokens: 23194
- Citations: 7
- Cited lines: 19
- Inline citations: 7
- References: 7
- Max citation number: 7
- Uncited lines: 0
- Reference alignment: ok
- Stop reason: evidence_sufficient
- Failures: retrieved_tokens>22000 (23194)
- Soft failures: retrieved_tokens>22000 (23194)
- Answer preview: ## 収益認識基準における変動対価の見積もりと収益反映の時期
### 変動対価の見積もり方法
変動対価とは、顧客と約束した対価のうち変動する可能性のある部分を指します[1]。その額の見積りには、以下の2つの方法のうち、企業が権利を得ることとなる対価の額をより適切に予測できる方法を用います[2]。

- 最頻値による方法: 発生し得ると考えられる対価の額における最も可能性の高い単一の金額を用いる方法です。契約において生じ得る結果が2つしかない場合（例: 割増金の条件を達成するか否か）には、適切な見積りとなる可能性があります[2]。
- 期待値による方法: 発生し得ると考えられる対価の額を確率で加重平均した金額を用いる方法です。特性の類似した多くの契約を有している場合には、適切な見積りとなる可能性があります[2]。

見積りにあたっては、契約全体を通じて単一の方法を首尾一貫して適用し、企業が合理的に入手できるすべての情報を考慮して、発生し得ると考えられる対価の額について合理的な数のシナリオを識別します[1]。

### 変動対価の収益への反映時期と制約
変動対価の額に関する不確実性が事後的
- Notes: 見積りと制約の説明が必要なケース。変動対価の中核語が落ちないことを確認する。

### PASS revenue_contract_modification
- Question: 収益認識基準で契約変更はどのように処理しますか。既存契約の継続か、新しい契約として扱うかの観点で教えてください
- Latency: 39.5s
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
- Latency: 64.5s
- Loops: 5
- read_chunk calls: 9
- Retrieved tokens: 17459
- Citations: 7
- Cited lines: 27
- Inline citations: 7
- References: 7
- Max citation number: 7
- Uncited lines: 0
- Reference alignment: ok
- Stop reason: evidence_sufficient
- Notes: 充足時点の判断基準を確認するケース。期間基準と一時点基準の両方に触れるかを見る。

### PASS hedge_accounting_requirements
- Question: 繰延ヘッジを適用するための主な要件を教えてください
- Latency: 40.4s
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
- Latency: 40.2s
- Loops: 6
- read_chunk calls: 9
- Retrieved tokens: 11155
- Citations: 5
- Cited lines: 6
- Inline citations: 5
- References: 5
- Max citation number: 5
- Uncited lines: 0
- Reference alignment: ok
- Stop reason: evidence_sufficient
- Notes: 税効果会計の基本論点。税率の基準時点と変更時の再計算に触れるかを見る。

### PASS deferred_tax_recoverability
- Question: 税効果会計で繰延税金資産の回収可能性はどのように判断しますか
- Latency: 52.4s
- Loops: 6
- read_chunk calls: 10
- Retrieved tokens: 18910
- Citations: 8
- Cited lines: 20
- Inline citations: 8
- References: 8
- Max citation number: 8
- Uncited lines: 0
- Reference alignment: ok
- Stop reason: evidence_sufficient
- Notes: 回収可能性判断の確認。将来課税所得やスケジューリングへの言及を見たい。

### PASS impairment_indication
- Question: 固定資産の減損会計では、どのような場合に減損の兆候があると判断しますか
- Latency: 61.9s
- Loops: 5
- read_chunk calls: 10
- Retrieved tokens: 16802
- Citations: 6
- Cited lines: 25
- Inline citations: 6
- References: 6
- Max citation number: 6
- Uncited lines: 0
- Reference alignment: ok
- Stop reason: evidence_sufficient
- Notes: 企業会計審議会系の代表ケース。兆候と回収可能価額の基礎語が出るかを見る。

### PASS r_and_d_cost_treatment
- Question: 研究開発費はどのように会計処理しますか。ソフトウェア開発費との違いにも触れてください
- Latency: 63.0s
- Loops: 5
- read_chunk calls: 10
- Retrieved tokens: 20416
- Citations: 4
- Cited lines: 24
- Inline citations: 4
- References: 4
- Max citation number: 4
- Uncited lines: 0
- Reference alignment: ok
- Stop reason: evidence_sufficient
- Notes: 研究開発費等に係る会計基準の確認。2026-03-29 の local baseline では retrieved_tokens が 26k 台まで伸びたため、まずは 30k を上限とする。

### PASS retirement_benefit_actuarial_difference
- Question: 退職給付会計における数理計算上の差異はどのように処理しますか
- Latency: 54.2s
- Loops: 5
- read_chunk calls: 9
- Retrieved tokens: 12391
- Citations: 6
- Cited lines: 12
- Inline citations: 6
- References: 6
- Max citation number: 6
- Uncited lines: 0
- Reference alignment: ok
- Stop reason: evidence_sufficient
- Notes: 退職給付の代表ケース。差異の費用処理と期間配分の説明を確認する。

### PASS consolidation_unrealized_gain_elimination
- Question: 連結会計では、親会社が子会社へ土地を譲渡した場合の未実現損益をどのように処理しますか
- Latency: 65.1s
- Loops: 5
- read_chunk calls: 6
- Retrieved tokens: 14433
- Citations: 2
- Cited lines: 10
- Inline citations: 2
- References: 2
- Max citation number: 2
- Uncited lines: 0
- Reference alignment: ok
- Stop reason: natural
- Notes: 連結会計の新規ケース。親子会社間の土地譲渡に伴う未実現損益の消去に触れられるかを見る。

### PASS consolidation_scope_determination
- Question: 会社計算規則では、どのような子会社を連結の範囲に含め、どのような場合に除外できますか
- Latency: 76.2s
- Loops: 5
- read_chunk calls: 6
- Retrieved tokens: 15725
- Citations: 4
- Cited lines: 19
- Inline citations: 4
- References: 4
- Max citation number: 4
- Uncited lines: 0
- Reference alignment: ok
- Stop reason: natural
- Notes: 連結範囲判定の新規ケース。現行 local baseline では loop 数と retrieved_tokens が高めのため、まずは 12 loop / 24k token を上限に監視する。

### PASS verification_land_revaluation_judgment
- Question: 私は、再評価済み土地の売却損は連結上の未実現損失として必ずしも消去しないと考えています。また、土地再評価差額金は子会社への土地譲渡後もそのまま引き継がれる理解です。依拠条文として連結財務諸表に関する会計基準第36条と土地再評価差額金の会計処理に関するＱ＆Ａを考えています。この判断は妥当ですか。
- Latency: 133.2s
- Loops: 11
- read_chunk calls: 1
- Retrieved tokens: 23768
- Citations: 3
- Cited lines: 5
- Inline citations: 3
- References: 3
- Max citation number: 3
- Uncited lines: 0
- Reference alignment: ok
- Stop reason: natural
- Notes: 判断検証型の新規ケース。土地譲渡・再評価・連結消去に関する論点を分解して確認できるかを、実回答に現れる中核語で監視する。

### PASS cross_reference_land_revaluation
- Question: 土地再評価差額金は子会社への土地譲渡後の連結上どう扱いますか。連結財務諸表に関する会計基準第36条と土地再評価差額金の会計処理に関するＱ＆Ａの両方を踏まえて説明してください
- Latency: 83.6s
- Loops: 11
- read_chunk calls: 5
- Retrieved tokens: 18970
- Citations: 4
- Cited lines: 17
- Inline citations: 4
- References: 4
- Max citation number: 4
- Uncited lines: 0
- Reference alignment: ok
- Stop reason: retrieval_budget
- Notes: クロスリファレンス検索の新規ケース。連結基準と土地再評価Q&Aをまたぐ説明ができるかを、現行 answer の表現に合わせて確認する。
