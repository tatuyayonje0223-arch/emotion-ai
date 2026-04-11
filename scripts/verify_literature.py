"""文献DOI検証スクリプト。

docs/literature/*.md から論文テーブルを抽出し、
CrossRef APIで各DOIの実在・雑誌名・著者・年を検証する。

Usage:
    python scripts/verify_literature.py
    python scripts/verify_literature.py --file docs/literature/fear_rage_literature.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

# 許可雑誌リスト（部分一致で照合）
MAJOR_JOURNALS = [
    "nature",
    "science",
    "neuron",
    "nature neuroscience",
    "pnas",
    "proceedings of the national academy",
    "elife",
    "journal of neuroscience",
    "current biology",
    "cell",
    "trends in cognitive sciences",
    "trends in neurosciences",
    "biological psychiatry",
    "annual review of neuroscience",
    "psychological review",
    "nature reviews neuroscience",
    "neuroimage",
    "cerebral cortex",
    "human brain mapping",
    "hormones and behavior",
    "psychoneuroendocrinology",
    "neuropsychopharmacology",
    "molecular psychiatry",
    "nature communications",
    "science advances",
    "frontiers in neuroscience",
    "frontiers in behavioral neuroscience",
    "philosophical transactions",
    "psychological bulletin",
    "neuroscience and biobehavioral reviews",
    "social cognitive and affective neuroscience",
    "emotion",
]


def extract_dois_from_md(filepath: Path) -> list[dict]:
    """Markdownファイルから論文情報を抽出する。

    テーブル行を解析し、DOIがある行を返す。
    """
    text = filepath.read_text(encoding="utf-8")
    papers = []

    # DOIパターン: 10.XXXX/... 形式
    doi_pattern = re.compile(r"10\.\d{4,9}/[^\s|)}\]]+")

    for line in text.splitlines():
        if "|" not in line:
            continue
        # テーブル行の場合
        cells = [c.strip() for c in line.split("|")]
        cells = [c for c in cells if c]  # 空セルを除去

        # DOIを含む行を探す
        line_text = " ".join(cells)
        doi_match = doi_pattern.search(line_text)

        if doi_match:
            doi = doi_match.group().rstrip(".")
            papers.append({
                "doi": doi,
                "raw_line": line.strip(),
                "cells": cells,
                "file": str(filepath),
            })

    # DOIがテーブルに埋め込まれていない場合、本文からも抽出
    for match in doi_pattern.finditer(text):
        doi = match.group().rstrip(".")
        if not any(p["doi"] == doi for p in papers):
            # 前後のコンテキストを取得
            start = max(0, match.start() - 100)
            end = min(len(text), match.end() + 100)
            context = text[start:end].replace("\n", " ")
            papers.append({
                "doi": doi,
                "raw_line": context,
                "cells": [],
                "file": str(filepath),
            })

    return papers


def verify_doi(doi: str) -> dict:
    """CrossRef APIでDOIを検証する。

    Returns:
        {
            "valid": bool,
            "title": str,
            "journal": str,
            "year": int | None,
            "authors": list[str],
            "is_major_journal": bool,
            "error": str | None,
        }
    """
    url = f"https://api.crossref.org/works/{quote(doi, safe='')}"
    headers = {
        "User-Agent": "EmotionAI-LiteratureVerifier/1.0 (mailto:mapshukaku@gmail.com)",
    }

    try:
        req = Request(url, headers=headers)
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        item = data.get("message", {})
        title_list = item.get("title", [])
        title = title_list[0] if title_list else "Unknown"

        # 雑誌名
        container = item.get("container-title", [])
        journal = container[0] if container else "Unknown"

        # 年
        issued = item.get("issued", {})
        date_parts = issued.get("date-parts", [[None]])
        year = date_parts[0][0] if date_parts and date_parts[0] else None

        # 著者
        authors_raw = item.get("author", [])
        authors = [
            f"{a.get('family', '')}, {a.get('given', '')}"
            for a in authors_raw[:3]
        ]

        # 主要雑誌チェック
        journal_lower = journal.lower()
        is_major = any(j in journal_lower for j in MAJOR_JOURNALS)

        return {
            "valid": True,
            "title": title,
            "journal": journal,
            "year": year,
            "authors": authors,
            "is_major_journal": is_major,
            "error": None,
        }

    except HTTPError as e:
        if e.code == 404:
            return {
                "valid": False,
                "title": "",
                "journal": "",
                "year": None,
                "authors": [],
                "is_major_journal": False,
                "error": f"DOI not found (404)",
            }
        return {
            "valid": False,
            "title": "",
            "journal": "",
            "year": None,
            "authors": [],
            "is_major_journal": False,
            "error": f"HTTP {e.code}",
        }

    except (URLError, TimeoutError) as e:
        return {
            "valid": False,
            "title": "",
            "journal": "",
            "year": None,
            "authors": [],
            "is_major_journal": False,
            "error": f"Network error: {e}",
        }

    except Exception as e:
        return {
            "valid": False,
            "title": "",
            "journal": "",
            "year": None,
            "authors": [],
            "is_major_journal": False,
            "error": f"Error: {e}",
        }


def main():
    parser = argparse.ArgumentParser(description="Verify literature DOIs via CrossRef API")
    parser.add_argument("--file", type=str, help="Specific .md file to verify")
    parser.add_argument("--dry-run", action="store_true", help="Extract DOIs without API calls")
    args = parser.parse_args()

    lit_dir = Path(__file__).parent.parent / "docs" / "literature"

    if args.file:
        files = [Path(args.file)]
    else:
        files = sorted(lit_dir.glob("*.md"))

    if not files:
        print(f"No .md files found in {lit_dir}")
        sys.exit(1)

    all_papers = []
    for f in files:
        papers = extract_dois_from_md(f)
        all_papers.extend(papers)
        print(f"[{f.name}] Extracted {len(papers)} DOIs")

    print(f"\nTotal DOIs to verify: {len(all_papers)}")

    if args.dry_run:
        for p in all_papers:
            print(f"  DOI: {p['doi']}")
        sys.exit(0)

    # DOI検証
    results = {
        "total": len(all_papers),
        "valid": 0,
        "invalid": 0,
        "major_journal": 0,
        "non_major": 0,
        "network_error": 0,
        "details": [],
    }

    for i, paper in enumerate(all_papers):
        doi = paper["doi"]
        print(f"  [{i+1}/{len(all_papers)}] Verifying {doi}...", end=" ", flush=True)

        result = verify_doi(doi)
        result["doi"] = doi
        result["source_file"] = paper["file"]
        results["details"].append(result)

        if result["valid"]:
            results["valid"] += 1
            if result["is_major_journal"]:
                results["major_journal"] += 1
                print(f"OK ({result['journal']}, {result['year']})")
            else:
                results["non_major"] += 1
                print(f"OK but non-major: {result['journal']}")
        elif "Network" in (result["error"] or ""):
            results["network_error"] += 1
            print(f"NETWORK ERROR: {result['error']}")
        else:
            results["invalid"] += 1
            print(f"INVALID: {result['error']}")

        # レート制限対策: 1秒待機
        time.sleep(1.0)

    # サマリー
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"Total DOIs:       {results['total']}")
    print(f"Valid:            {results['valid']}")
    print(f"  Major journal:  {results['major_journal']}")
    print(f"  Non-major:      {results['non_major']}")
    print(f"Invalid:          {results['invalid']}")
    print(f"Network errors:   {results['network_error']}")

    if results["invalid"] > 0:
        print(f"\nINVALID DOIs:")
        for d in results["details"]:
            if not d["valid"] and "Network" not in (d["error"] or ""):
                print(f"  {d['doi']} — {d['error']} (from {d['source_file']})")

    if results["non_major"] > 0:
        print(f"\nNON-MAJOR JOURNAL papers:")
        for d in results["details"]:
            if d["valid"] and not d["is_major_journal"]:
                print(f"  {d['doi']} — {d['journal']} (from {d['source_file']})")

    # JSON出力
    output_path = lit_dir / "verification_results.json"
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDetailed results saved to: {output_path}")

    # 成功率
    if results["total"] > 0:
        valid_rate = results["valid"] / results["total"] * 100
        major_rate = results["major_journal"] / results["total"] * 100
        print(f"\nValid rate:         {valid_rate:.1f}%")
        print(f"Major journal rate: {major_rate:.1f}%")

    # 失敗があれば非ゼロ終了
    if results["invalid"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
