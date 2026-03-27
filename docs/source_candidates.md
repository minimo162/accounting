# 追加ソース候補

インデックスに新しく含める候補を、既存収録との重複を避けつつ整理したメモ。

前提:

- 既存でかなりカバー済み: ASBJ の企業会計基準 / 適用指針 / 実務対応報告、主要な e-Gov 法令、FSA の一部ガイドライン、JICPA の一部実務指針
- 優先して追加したいのは、既存データを補完する「実務解釈」「IFRS 接続」「中小企業会計」

## 優先度 A

### 1. ASBJ 移管指針

理由:

- JICPA 実務指針から ASBJ へ移った会計系資料を補完できる
- 金融商品、期中、税効果、リースなどの実務質問に効く

収集方法:

- 一覧ページを起点に PDF を収集
- 公開草案ではなく確定版を優先

候補 URL:

- `https://www.asb-j.jp/jp/wp-content/uploads/sites/4/nenjikaizen_20250311_16.pdf` 移管指針の適用
- `https://www.asb-j.jp/jp/news_release/406141.html` 移管指針の適用の公表ページ

実装メモ:

- `scripts/download_extra_pdfs.py` に単発 PDF を追加するより、ASBJ の移管指針一覧ページをたどる scraper を別で持つ方が保守しやすい
- 公開草案 PDF は混ぜない

### 2. JICPA 会計制度委員会研究報告

理由:

- 基準本文や実務指針だけでは拾いにくい実務論点を補完できる
- 補助金、附属明細書、収益認識、内部統制周辺などの質問に強くなる

収集方法:

- 一覧ページを起点に PDF を収集
- 会計制度委員会を優先し、他委員会は会計 Q&A に効くものだけ追加

候補 URL:

- `https://jicpa.or.jp/specialized_field/publication/research_report/` 研究報告一覧
- `https://jicpa.or.jp/specialized_field/publication/files/2-11-13-2-20091216.pdf` 会計制度委員会研究報告第13号
- `https://jicpa.or.jp/specialized_field/publication/files/2-11-9-2-20140402.pdf` 会計制度委員会研究報告第9号

実装メモ:

- `scripts/download_additional_sources.py` に `RESEARCH_REPORT_PAGES` を追加し、一覧ページから PDF を拾う構成が自然
- 監査専用の研究報告は原則除外してよい

### 3. 金融庁 IFRS 関連情報

理由:

- 現状の法令・国内基準だけだと、IFRS 任意適用や指定国際会計基準の質問に弱い
- 連結財規 / 中間財規と IFRS 接続を補完できる

収集方法:

- 一覧ページを起点に、FAQ、開示例、告示資料を収集
- IFRS の本文そのものではなく、日本での取扱い資料を優先

候補 URL:

- `https://www.fsa.go.jp/status/ifrs.html` IFRS 関連情報一覧
- `https://www.fsa.go.jp/news/27/sonota/20160331-5.html` IFRS に基づく連結財務諸表の開示例
- `https://www.fsa.go.jp/news/24/sonota/20130620-2.html` IFRS への対応のあり方に関する当面の方針
- `https://www.fsa.go.jp/news/r6/sonota/20240718/20240718.html` 指定国際会計基準の改正告示

実装メモ:

- HTML ページ主体なので `scripts/process_html_sources.py` の拡張対象
- 告示や別紙が PDF なら PDF として直接取得してよい

## 優先度 B

### 4. 企業内容等の開示に関する関連ページ

理由:

- e-Gov の開示府令だけでは「どう書くか」の実務解像度が不足する
- 開示例、留意事項、EDINET 周辺の質問に効く

候補 URL:

- `https://www.fsa.go.jp/policy/kaiji/kaiji.html` 企業内容等の開示に関する情報
- `https://www.fsa.go.jp/common/law/kaiji/index.html` 開示制度関連のインデックス

実装メモ:

- 既存の `REGULATIONS_PAGES` に近い扱い
- ガイドライン PDF と一覧ページの両方を残す

### 5. 中小企業の会計に関する指針 / 基本要領

理由:

- 現状のソースは上場・金商法寄りで、中小企業実務の質問を十分に拾えない
- 会社計算規則とのあわせ技で非上場企業向け回答の質が上がる

候補 URL:

- `https://www.chusho.meti.go.jp/zaimu/youryou/about/` 中小会計要領について
- `https://www.chusho.meti.go.jp/zaimu/youryou/about/download/0528KaikeiYouryou-1.pdf` 中小企業の会計に関する基本要領
- `https://www.chusho.meti.go.jp/zaimu/kaikei/2005/050803.kaikei_shishin.html` 中小企業の会計に関する指針の公表ページ

実装メモ:

- まずは中小会計要領の PDF と説明ページだけでも価値が高い
- 指針は配布元が複数あり得るため、正本サイトを固定して管理したい

## 優先度 C

### 6. EDINET タクソノミ / 提出者向けガイド

理由:

- 会計基準そのものではないが、開示実務や XBRL タグ付けの質問に有効
- 「どの表示科目をどこに出すか」系の周辺質問を拾える

候補 URL:

- `https://www.fsa.go.jp/search/20241112.html` 2025 年版 EDINET タクソノミ公表ページ
- `https://www.fsa.go.jp/common/law/kaiji/index.html` EDINET 関連導線を含む開示制度インデックス

実装メモ:

- 本体が zip / xsd 中心なら全文 indexing より、ガイド PDF のみ先に入れる方がよい
- 優先度は IFRS と中小企業会計より下

## 今は見送ってよいもの

- IFRS 基準本文そのもの
  - ライセンスと配布条件が重い
  - まずは金融庁の IFRS 関連資料で日本実務を補完する方が費用対効果が高い
- 監査中心の資料一式
  - このアプリの主用途は会計基準 Q&A なので、監査論点は絞った方がよい
- ニュース記事や解説ブログ
  - 公式一次情報に寄せる

## 実装順の提案

1. JICPA 研究報告一覧ページの scraper 追加
2. ASBJ 移管指針の一覧取得
3. FSA IFRS 関連 HTML ページの取り込み
4. 中小会計要領の PDF / HTML 追加

## 重複注意

- e-Gov で収録済みの法令本文と、FSA のガイドライン PDF は役割が違うので共存可
- ただし同じ法令本文の PDF 版と e-Gov XML 版は重複しやすい
- JICPA 実務指針と ASBJ 移管指針は、置き換え関係があるため source 名で判別できるようにしておく
