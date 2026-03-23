"""Update chunk source fields with human-readable document titles."""

import json
import re
import sys

def extract_title(text: str) -> str | None:
    """Extract the document title from page 1 text."""
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    # Look for lines containing standard identifiers
    patterns = [
        r'企業会計基準第\s*[\d０-９]+\s*号',
        r'企業会計基準適用指針第\s*[\d０-９]+\s*号',
        r'実務対応報告第\s*[\d０-９]+\s*号',
        # 企業会計審議会
        r'固定資産の減損に係る会計基準',
        r'退職給付に係る会計基準',
        r'税効果会計に係る会計基準',
        r'外貨建取引等会計処理基準',
        r'研究開発費等に係る会計基準',
        r'連結財務諸表.*原則',
        # JICPA
        r'金融商品会計に関する実務指針',
        r'金融商品会計に関する.*Ｑ＆Ａ',
        r'退職給付会計に関する実務指針',
        r'税効果会計に関する実務指針',
        # SSBJ
        r'サステナビリティ.*基準',
        r'サステナビリティ.*開示',
        # 財務諸表等規則
        r'財務諸表等.*規則',
        r'財務諸表等規則ガイドライン',
        # 監査基準
        r'監査.*基準',
        r'内部統制',
    ]

    for line in lines[:15]:
        # Skip copyright/header lines
        if '複写・転載' in line or '財務会計基準機構' in line and '企業会計基準' not in line:
            continue
        if line.startswith('- ') and len(line) < 10:
            continue

        for pat in patterns:
            match = re.search(pat, line)
            if match:
                # Found a standard identifier, now get the full title
                # Clean the line
                title = line
                title = re.sub(r'^[\s\-－–—]*\d*[\s\-－–—]*', '', title)
                title = title.strip()

                # If this line has the number but not the name, check next lines
                if len(title) < 20:
                    idx = lines.index(line)
                    for next_line in lines[idx+1:idx+4]:
                        next_clean = next_line.strip()
                        if next_clean and not next_clean.startswith('-') and '複写' not in next_clean:
                            title = title + ' ' + next_clean
                            break

                return title[:100]

    return None


def main():
    with open('data/chunks.json', encoding='utf-8') as f:
        chunks = json.load(f)

    # Extract title from first page of each file
    file_titles: dict[str, str] = {}
    for c in chunks:
        if c['page'] == 1 and c['file'] not in file_titles:
            title = extract_title(c['text'])
            if title:
                file_titles[c['file']] = title

    print(f"Extracted titles for {len(file_titles)}/{len(set(c['file'] for c in chunks))} files")

    # Update source field for all chunks
    updated = 0
    for c in chunks:
        title = file_titles.get(c['file'])
        if title:
            c['source'] = f"{title} (p.{c['page']})"
            updated += 1
        # Keep original source if no title extracted

    with open('data/chunks.json', 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Updated {updated}/{len(chunks)} chunks")

    # Show examples
    print("\nExamples:")
    seen = set()
    for c in chunks:
        title = file_titles.get(c['file'])
        if title and c['file'] not in seen:
            seen.add(c['file'])
            print(f"  {c['source']}")
            if len(seen) >= 20:
                break


if __name__ == "__main__":
    main()
