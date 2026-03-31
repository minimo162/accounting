# Answer Eval Report

- Cases: 21
- Passed: 18
- Failed: 3
- Pass rate: 86%
- Avg latency: 58.1s
- P95 latency: 86.6s
- Avg loops: 5.8
- Avg read_chunk calls: 7.1
- Avg retrieved tokens: 13824
- Avg cited lines: 12.8
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
- Latency: 68.6s
- Loops: 7
- read_chunk calls: 7
- Retrieved tokens: 15721
- Citations: 7
- Cited lines: 14
- Inline citations: 7
- References: 7
- Max citation number: 7
- Uncited lines: 0
- Reference alignment: ok
- Stop reason: evidence_sufficient
- Failures: loops>6 (7)
- Soft failures: loops>6 (7)
- Answer preview: ## リース会計基準の改正点と経過措置
### 1. 改正の背景と目的
国際会計基準審議会（IASB）が2016年にIFRS第16号を、米国財務会計基準審議会（FASB）が同年にTopic 842を公表したことにより、国際的な会計基準と日本のリース会計基準との間に、特に負債の認識において違いが生じることとなりました[1]。この国際的な会計基準との整合性を図り、財務諸表間の比較可能性を高めるというニーズに応えるため、新しい会計基準が開発されました[1]。

### 2. 借手の会計処理に関する主な改正点
- **すべてのリースについて資産と負債を計上**: 改正前の基準ではファイナンス・リースとオペレーティング・リースに分類されていましたが、改正後はすべてのリースについて「使用権資産」と「リース負債」を貸借対照表に計上することになりました[1]。
- **使用権資産の償却方法**: 原資産の所有権が借手に移転すると認められるリースの場合、使用権資産の減価償却費は、原資産を自ら所有していたと仮定した場合に適用する減価償却方法と同一の方法により算定します[2]。それ以外のリースでは、定額法等
- Notes: 長文・複数論点の代表ケース。detail を落とさずに 6 loop 以内で返せるかを見る。

### FAIL lease_revision_detail_depth
- Question: リース会計基準の改正点と経過措置を、借手の会計処理・貸手の扱い・関連基準への影響に分けて詳しく教えてください
- Latency: 66.3s
- Loops: 7
- read_chunk calls: 8
- Retrieved tokens: 17520
- Citations: 8
- Cited lines: 17
- Inline citations: 8
- References: 8
- Max citation number: 8
- Uncited lines: 0
- Reference alignment: ok
- Stop reason: evidence_sufficient
- Failures: loops>6 (7)
- Soft failures: loops>6 (7)
- Answer preview: ## リース会計基準の改正点と経過措置について
### 1. 改正の背景と目的
国際会計基準審議会（IASB）が2016年にIFRS第16号を、米国財務会計基準審議会（FASB）が同年にTopic 842を公表したことにより、国際的な会計基準と日本のリース会計基準との間に、特に負債の認識において違いが生じることとなりました[1]。この国際的な会計基準との整合性を図り、財務諸表間の比較可能性を高めることが改正の主要な目的の一つです[1]。

### 2. 借手の会計処理に関する改正点
- **原則的な処理**: 借手のすべてのリースについて、資産（使用権資産）及び負債（リース負債）を計上するモデルが導入されました[1]。これにより、従来のファイナンス・リースとオペレーティング・リースの分類が借手側では廃止されました[2]。
- **使用権資産の償却**: 原資産の所有権が借手に移転すると認められるリースの場合、使用権資産の減価償却費は、原資産を自ら所有していたと仮定した場合に適用する減価償却方法と同一の方法により算定します[3]。それ以外のリースについては、定額法等の中から企業の実態に応
- Notes: detail insufficiency を検知する補助ケース。各論点が cited line として残っているかを確認する。

### PASS lease_exact_clause
- Question: 企業会計基準第13号第10項では借手のリースをどのように扱いますか
- Latency: 19.9s
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
- Latency: 86.6s
- Loops: 5
- read_chunk calls: 6
- Retrieved tokens: 14729
- Citations: 4
- Cited lines: 14
- Inline citations: 4
- References: 4
- Max citation number: 4
- Uncited lines: 0
- Reference alignment: ok
- Stop reason: natural
- Failures: latency>75.0s (86.6s)
- Soft failures: latency>75.0s (86.6s)
- Answer preview: ## 借手はリース開始日に何を認識しますか
借手はリース開始日に、以下の2つの項目を認識します[1]。

- **リース負債**: リース開始日に会計基準第34項に従い算定された額によりリース負債を計上する[1]。
- **使用権資産**: 当該リース負債にリース開始日までに支払った借手のリース料、付随費用及び資産除去債務に対応する除去費用を加算し、受け取ったリース・インセンティブを控除した額により使用権資産を計上する[1]。

## 使用権資産とリース負債の関係について
### 基本的な関係
使用権資産は、リース開始日に算定されたリース負債の計上額を基礎として算定される[2]。具体的には、リース負債にリース開始日までに支払った借手のリース料、付随費用及び資産除去債務に対応する除去費用を加算し、受け取ったリース・インセンティブを控除して算定する[2]。

### 会計処理モデル
企業会計基準及び適用指針は、借手におけるリースの費用配分の方法については、リースがファイナンス・リースであるかオペレーティング・リースであるかにかかわらず、すべてのリースについて使用権資産に係る減価償却費及びリ
- Notes: 借手の基本処理を確認するケース。2026-03-29 の local baseline では retrieved_tokens が 23k 台まで伸びたため、まずは 26k を上限とする。

### PASS lease_short_term_low_value
- Question: 借手は短期リースや少額リースをどのように扱いますか
- Latency: 25.2s
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
- Latency: 36.3s
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
- Latency: 77.3s
- Loops: 5
- read_chunk calls: 6
- Retrieved tokens: 11474
- Citations: 6
- Cited lines: 11
- Inline citations: 6
- References: 6
- Max citation number: 6
- Uncited lines: 0
- Reference alignment: ok
- Stop reason: natural
- Notes: 契約の分解説明が必要なケース。現行ベースライン近辺の token 使用量を超えると失敗にする。

### PASS performance_obligation_identification_detail
- Question: 収益認識基準で履行義務はどのように識別しますか。保守サービスや値引きのある契約を念頭に説明してください
- Latency: 78.8s
- Loops: 5
- read_chunk calls: 6
- Retrieved tokens: 11474
- Citations: 5
- Cited lines: 15
- Inline citations: 5
- References: 5
- Max citation number: 5
- Uncited lines: 0
- Reference alignment: ok
- Stop reason: natural
- Notes: 浅い要約ではなく、契約分解の具体論点が cited line として残るかを見る detail ケース。

### PASS revenue_variable_consideration
- Question: 収益認識基準における変動対価はどのように見積もり、いつ収益に反映しますか
- Latency: 37.7s
- Loops: 6
- read_chunk calls: 7
- Retrieved tokens: 12748
- Citations: 3
- Cited lines: 7
- Inline citations: 3
- References: 3
- Max citation number: 3
- Uncited lines: 0
- Reference alignment: ok
- Stop reason: evidence_sufficient
- Notes: 見積りと制約の説明が必要なケース。変動対価の中核語が落ちないことを確認する。

### PASS revenue_contract_modification
- Question: 収益認識基準で契約変更はどのように処理しますか。既存契約の継続か、新しい契約として扱うかの観点で教えてください
- Latency: 51.3s
- Loops: 5
- read_chunk calls: 8
- Retrieved tokens: 16864
- Citations: 4
- Cited lines: 16
- Inline citations: 4
- References: 4
- Max citation number: 4
- Uncited lines: 0
- Reference alignment: ok
- Stop reason: evidence_sufficient
- Notes: 契約変更の分岐説明を確認するケース。継続処理と別個処理の両方に触れたい。

### PASS revenue_point_in_time_vs_over_time
- Question: 収益認識基準では履行義務を一定の期間にわたり充足するのか、一時点で充足するのかをどう判断しますか
- Latency: 52.6s
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
- Latency: 38.3s
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
- Latency: 42.0s
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
- Citations: 9
- Cited lines: 22
- Inline citations: 9
- References: 9
- Max citation number: 9
- Uncited lines: 0
- Reference alignment: ok
- Stop reason: evidence_sufficient
- Notes: 回収可能性判断の確認。将来課税所得やスケジューリングへの言及を見たい。

### PASS impairment_indication
- Question: 固定資産の減損会計では、どのような場合に減損の兆候があると判断しますか
- Latency: 56.3s
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
- Latency: 60.0s
- Loops: 5
- read_chunk calls: 12
- Retrieved tokens: 23441
- Citations: 4
- Cited lines: 13
- Inline citations: 4
- References: 4
- Max citation number: 4
- Uncited lines: 0
- Reference alignment: ok
- Stop reason: evidence_sufficient
- Notes: 研究開発費等に係る会計基準の確認。2026-03-29 の local baseline では retrieved_tokens が 26k 台まで伸びたため、まずは 30k を上限とする。

### PASS retirement_benefit_actuarial_difference
- Question: 退職給付会計における数理計算上の差異はどのように処理しますか
- Latency: 45.9s
- Loops: 5
- read_chunk calls: 12
- Retrieved tokens: 14852
- Citations: 5
- Cited lines: 12
- Inline citations: 5
- References: 5
- Max citation number: 5
- Uncited lines: 0
- Reference alignment: ok
- Stop reason: evidence_sufficient
- Notes: 退職給付の代表ケース。差異の費用処理と期間配分の説明を確認する。

### PASS consolidation_unrealized_gain_elimination
- Question: 連結会計では、親会社が子会社へ土地を譲渡した場合の未実現損益をどのように処理しますか
- Latency: 57.1s
- Loops: 5
- read_chunk calls: 6
- Retrieved tokens: 14100
- Citations: 3
- Cited lines: 12
- Inline citations: 3
- References: 3
- Max citation number: 3
- Uncited lines: 0
- Reference alignment: ok
- Stop reason: natural
- Notes: 連結会計の新規ケース。親子会社間の土地譲渡に伴う未実現損益の消去に触れられるかを見る。

### PASS consolidation_scope_determination
- Question: 会社計算規則では、どのような子会社を連結の範囲に含め、どのような場合に除外できますか
- Latency: 70.1s
- Loops: 5
- read_chunk calls: 6
- Retrieved tokens: 15955
- Citations: 3
- Cited lines: 18
- Inline citations: 3
- References: 3
- Max citation number: 3
- Uncited lines: 0
- Reference alignment: ok
- Stop reason: natural
- Notes: 連結範囲判定の新規ケース。現行 local baseline では loop 数と retrieved_tokens が高めのため、まずは 12 loop / 24k token を上限に監視する。

### PASS verification_land_revaluation_judgment
- Question: 私は、再評価済み土地の売却損は連結上の未実現損失として必ずしも消去しないと考えています。また、土地再評価差額金は子会社への土地譲渡後もそのまま引き継がれる理解です。依拠条文として連結財務諸表に関する会計基準第36条と土地再評価差額金の会計処理に関するＱ＆Ａを考えています。この判断は妥当ですか。
- Latency: 135.4s
- Loops: 12
- read_chunk calls: 3
- Retrieved tokens: 23875
- Citations: 2
- Cited lines: 5
- Inline citations: 2
- References: 2
- Max citation number: 2
- Uncited lines: 0
- Reference alignment: ok
- Stop reason: natural
- Notes: 判断検証型の新規ケース。土地譲渡・再評価・連結消去に関する論点を分解して確認できるかを、実回答に現れる中核語で監視する。

### PASS cross_reference_land_revaluation
- Question: 土地再評価差額金は子会社への土地譲渡後の連結上どう扱いますか。連結財務諸表に関する会計基準第36条と土地再評価差額金の会計処理に関するＱ＆Ａの両方を踏まえて説明してください
- Latency: 61.4s
- Loops: 11
- read_chunk calls: 2
- Retrieved tokens: 12865
- Citations: 2
- Cited lines: 7
- Inline citations: 2
- References: 2
- Max citation number: 2
- Uncited lines: 0
- Reference alignment: ok
- Stop reason: retrieval_budget
- Notes: クロスリファレンス検索の新規ケース。連結基準と土地再評価Q&Aをまたぐ説明ができるかを、現行 answer の表現に合わせて確認する。
