"""全論文の包括的検証スクリプト。

1. DOI補完: タイトルからCrossRef APIでDOI検索
2. DOI実在チェック: CrossRef APIで照合
3. アブストラクト取得: CrossRef/PubMed APIから取得
4. 内容照合: 論文テーブルの「Key Finding」とアブストラクトのキーワード一致率

Usage:
    python scripts/verify_literature_full.py
    python scripts/verify_literature_full.py --file docs/literature/fear_rage_literature.md
    python scripts/verify_literature_full.py --phase doi-search   # DOI検索のみ
    python scripts/verify_literature_full.py --phase abstract     # アブストラクト検証のみ
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

POLITE_EMAIL = "mapshukaku@gmail.com"
HEADERS = {"User-Agent": f"EmotionAI-Verifier/2.0 (mailto:{POLITE_EMAIL})"}
LIT_DIR = Path(__file__).parent.parent / "docs" / "literature"
RESULTS_PATH = LIT_DIR / "full_verification_results.json"


# ─── Paper Extraction ───────────────────────────────────────────

def extract_papers_from_md(filepath: Path) -> list[dict]:
    """Markdownからテーブル行を全て抽出。"""
    text = filepath.read_text(encoding="utf-8")
    papers = []
    doi_pattern = re.compile(r"10\.\d{4,9}/[^\s|)}\]]+")
    current_emotion = "unknown"

    for line in text.splitlines():
        # セクションヘッダーから情動名を取得
        m_section = re.match(r"^## (.+)", line)
        if m_section:
            header = m_section.group(1).strip()
            # 番号付きセクション(e.g. "1. Brodmann Area...")はスキップ
            if not re.match(r"^\d+\.", header):
                current_emotion = header

        if "|" not in line or line.strip().startswith("|---") or line.strip().startswith("| #"):
            continue

        cells = [c.strip() for c in line.split("|")]
        cells = [c for c in cells if c]

        if len(cells) < 4:
            continue

        # ヘッダー行スキップ
        if cells[0] in ("#", "Structure ID", "Pathway", "System", "FROM", "Paper"):
            continue

        # テーブル行のID検出
        id_match = re.match(r"^[A-Z]*\d+$", cells[0])
        if not id_match and not re.match(r"^\d+$", cells[0]):
            continue

        # DOI抽出
        line_text = " ".join(cells)
        doi_match = doi_pattern.search(line_text)
        doi = doi_match.group().rstrip(".") if doi_match else None

        # Author(Year) 抽出
        author_year = cells[1] if len(cells) > 1 else ""

        # Title 抽出
        title = cells[2] if len(cells) > 2 else ""

        # Journal 抽出
        journal = cells[3] if len(cells) > 3 else ""

        # Key Finding 抽出 (可変位置)
        # DOI列がある場合: cells = [id, author, title, journal, doi, key_finding, ...]
        # DOI列がない場合: cells = [id, author, title, journal, key_finding, ...]
        key_finding = ""
        for i, c in enumerate(cells):
            if doi_pattern.search(c) or c.lower().startswith("doi"):
                continue
            if i >= 4 and len(c) > 30:  # Key findingは通常長い
                key_finding = c
                break
        if not key_finding and len(cells) > 4:
            key_finding = cells[4]

        papers.append({
            "id": cells[0],
            "author_year": author_year,
            "title": title,
            "journal": journal,
            "doi": doi,
            "key_finding": key_finding,
            "emotion": current_emotion,
            "file": filepath.name,
        })

    return papers


# ─── CrossRef API ────────────────────────────────────────────────

def search_doi_by_title(title: str, author: str = "") -> dict:
    """タイトルからDOIを検索する。"""
    # タイトルを短縮して検索精度を上げる
    clean_title = re.sub(r"[^\w\s]", " ", title)[:200]
    query = clean_title
    if author:
        # 著者名の姓部分を追加
        author_last = re.split(r"[,(&\s]", author)[0].strip()
        if author_last and len(author_last) > 2:
            query = f"{author_last} {clean_title}"

    params = urlencode({"query.bibliographic": query, "rows": "3"})
    url = f"https://api.crossref.org/works?{params}"

    try:
        req = Request(url, headers=HEADERS)
        with urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        items = data.get("message", {}).get("items", [])
        if not items:
            return {"found": False, "doi": None, "score": 0}

        # 最もスコアが高い結果を返す
        best = items[0]
        found_title = (best.get("title", [""])[0] or "").lower()
        search_title = clean_title.lower()

        # タイトル類似度の簡易チェック
        search_words = set(search_title.split())
        found_words = set(found_title.split())
        if search_words and found_words:
            overlap = len(search_words & found_words) / max(len(search_words), 1)
        else:
            overlap = 0

        found_year = best.get("issued", {}).get("date-parts", [[None]])[0][0]

        # 年の一致チェック（author_yearから年を抽出）
        year_match = True
        author_year_num = re.search(r"\b(19|20)\d{2}\b", author or "")
        if author_year_num and found_year:
            year_match = abs(int(author_year_num.group()) - int(found_year)) <= 1

        # 厳格な判定: タイトル重複50%以上 AND 年一致
        is_found = overlap > 0.5 and year_match

        return {
            "found": is_found,
            "doi": best.get("DOI") if is_found else None,
            "score": best.get("score", 0),
            "title_overlap": overlap,
            "year_match": year_match,
            "found_title": best.get("title", [""])[0],
            "found_journal": (best.get("container-title", [""])[0] if best.get("container-title") else ""),
            "found_year": found_year,
        }

    except Exception as e:
        return {"found": False, "doi": None, "score": 0, "error": str(e)}


def fetch_crossref_metadata(doi: str) -> dict:
    """CrossRef APIからメタデータ+アブストラクトを取得。"""
    url = f"https://api.crossref.org/works/{quote(doi, safe='')}"
    try:
        req = Request(url, headers=HEADERS)
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        item = data.get("message", {})
        title = (item.get("title", [""])[0] or "")
        container = item.get("container-title", [])
        journal = container[0] if container else ""
        issued = item.get("issued", {})
        year = (issued.get("date-parts", [[None]])[0][0])
        authors = [f"{a.get('family', '')}" for a in item.get("author", [])[:5]]

        # アブストラクト（CrossRefにある場合）
        abstract = item.get("abstract", "")
        # HTMLタグを除去
        if abstract:
            abstract = re.sub(r"<[^>]+>", "", abstract)

        return {
            "valid": True,
            "title": title,
            "journal": journal,
            "year": year,
            "authors": authors,
            "abstract": abstract,
        }

    except HTTPError as e:
        return {"valid": False, "error": f"HTTP {e.code}"}
    except Exception as e:
        return {"valid": False, "error": str(e)}


# ─── PubMed API ──────────────────────────────────────────────────

def fetch_pubmed_abstract(doi: str) -> str:
    """PubMed E-utilitiesからアブストラクトを取得。"""
    # Step 1: DOIでPubMed IDを検索
    search_url = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        f"?db=pubmed&term={quote(doi)}[doi]&retmode=json"
    )
    try:
        req = Request(search_url, headers=HEADERS)
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        id_list = data.get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return ""

        pmid = id_list[0]

        # Step 2: PMIDでアブストラクトを取得
        time.sleep(0.4)  # NCBI rate limit
        fetch_url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            f"?db=pubmed&id={pmid}&rettype=abstract&retmode=text"
        )
        req2 = Request(fetch_url, headers=HEADERS)
        with urlopen(req2, timeout=15) as resp2:
            text = resp2.read().decode("utf-8")

        return text.strip()

    except Exception:
        return ""


# ─── Content Verification ────────────────────────────────────────

def verify_content(key_finding: str, abstract: str) -> dict:
    """Key Findingとアブストラクトのキーワードマッチングによる内容照合。"""
    if not abstract or not key_finding:
        return {"verified": False, "reason": "no_abstract", "match_score": 0}

    # キーワード抽出（ストップワード除去）
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to",
        "for", "of", "and", "or", "but", "with", "by", "from", "that", "this",
        "it", "not", "as", "be", "has", "have", "had", "do", "does", "did",
        "will", "would", "can", "could", "may", "might", "shall", "should",
        "their", "its", "they", "them", "we", "our", "these", "those",
        "more", "than", "also", "been", "into", "both", "during", "between",
        "such", "when", "which", "while", "after", "before", "through", "about",
    }

    def extract_keywords(text: str) -> set[str]:
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        return {w for w in words if w not in stopwords}

    finding_kw = extract_keywords(key_finding)
    abstract_kw = extract_keywords(abstract)

    if not finding_kw:
        return {"verified": False, "reason": "no_keywords", "match_score": 0}

    overlap = finding_kw & abstract_kw
    score = len(overlap) / len(finding_kw) if finding_kw else 0

    return {
        "verified": score >= 0.25,
        "reason": "keyword_match" if score >= 0.25 else "low_overlap",
        "match_score": round(score, 3),
        "matched_keywords": sorted(list(overlap))[:20],
        "total_finding_keywords": len(finding_kw),
    }


# ─── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Full literature verification")
    parser.add_argument("--file", type=str, help="Specific .md file")
    parser.add_argument("--phase", choices=["doi-search", "abstract", "all"], default="all")
    parser.add_argument("--limit", type=int, default=0, help="Max papers to process (0=all)")
    args = parser.parse_args()

    # ファイル選択
    if args.file:
        files = [Path(args.file)]
    else:
        files = sorted(LIT_DIR.glob("*.md"))
        files = [f for f in files if f.name != "literature_summary_table.md"]

    # 全論文を抽出
    all_papers = []
    for f in files:
        papers = extract_papers_from_md(f)
        all_papers.extend(papers)
        print(f"[{f.name}] {len(papers)} papers extracted")

    print(f"\nTotal papers: {len(all_papers)}")
    print(f"  With DOI: {sum(1 for p in all_papers if p['doi'])}")
    print(f"  Without DOI: {sum(1 for p in all_papers if not p['doi'])}")

    if args.limit > 0:
        all_papers = all_papers[:args.limit]
        print(f"  Limited to first {args.limit}")

    results = []

    for i, paper in enumerate(all_papers):
        print(f"\n[{i+1}/{len(all_papers)}] {paper['author_year']}: {paper['title'][:60]}...")
        entry = {**paper, "verification": {}}

        # Phase 1: DOI検索（DOIがない場合）
        if args.phase in ("doi-search", "all") and not paper["doi"]:
            print(f"  Searching DOI by title...", end=" ", flush=True)
            search = search_doi_by_title(paper["title"], paper["author_year"])
            time.sleep(1.0)

            if search["found"] and search["doi"]:
                print(f"FOUND: {search['doi']} (overlap={search.get('title_overlap', 0):.2f})")
                entry["doi"] = search["doi"]
                entry["doi_source"] = "crossref_search"
                entry["verification"]["doi_search"] = search
            else:
                print(f"NOT FOUND")
                entry["verification"]["doi_search"] = search

        # Phase 2: メタデータ+アブストラクト取得
        if args.phase in ("abstract", "all") and entry.get("doi"):
            doi = entry["doi"]

            # CrossRefからメタデータ
            print(f"  CrossRef metadata...", end=" ", flush=True)
            cr = fetch_crossref_metadata(doi)
            time.sleep(1.0)

            if cr.get("valid"):
                print(f"OK ({cr['journal']}, {cr['year']})")
                entry["verification"]["crossref"] = {
                    "valid": True,
                    "title": cr["title"],
                    "journal": cr["journal"],
                    "year": cr["year"],
                    "has_abstract": bool(cr.get("abstract")),
                }

                abstract = cr.get("abstract", "")

                # CrossRefにアブストラクトがなければPubMedから取得
                if not abstract:
                    print(f"  PubMed abstract...", end=" ", flush=True)
                    abstract = fetch_pubmed_abstract(doi)
                    time.sleep(0.5)
                    if abstract:
                        print(f"OK ({len(abstract)} chars)")
                    else:
                        print("NOT AVAILABLE")

                # 内容照合
                if abstract and paper.get("key_finding"):
                    content_check = verify_content(paper["key_finding"], abstract)
                    entry["verification"]["content"] = content_check
                    status = "MATCH" if content_check["verified"] else "LOW_MATCH"
                    print(f"  Content check: {status} (score={content_check['match_score']:.2f}, "
                          f"keywords={content_check.get('total_finding_keywords', 0)})")
                else:
                    entry["verification"]["content"] = {
                        "verified": False,
                        "reason": "no_abstract_available",
                        "match_score": 0,
                    }
            else:
                print(f"FAIL: {cr.get('error', 'unknown')}")
                entry["verification"]["crossref"] = {"valid": False, "error": cr.get("error")}

        results.append(entry)

    # ─── Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FULL VERIFICATION SUMMARY")
    print("=" * 70)

    total = len(results)
    has_doi = sum(1 for r in results if r.get("doi"))
    doi_found = sum(1 for r in results if r.get("doi_source") == "crossref_search")
    cr_valid = sum(1 for r in results if r.get("verification", {}).get("crossref", {}).get("valid"))
    cr_invalid = sum(1 for r in results if r.get("doi") and not r.get("verification", {}).get("crossref", {}).get("valid") and "crossref" in r.get("verification", {}))
    content_verified = sum(1 for r in results if r.get("verification", {}).get("content", {}).get("verified"))
    content_failed = sum(1 for r in results if r.get("verification", {}).get("content", {}).get("match_score", -1) >= 0 and not r.get("verification", {}).get("content", {}).get("verified"))
    no_abstract = sum(1 for r in results if r.get("verification", {}).get("content", {}).get("reason") in ("no_abstract", "no_abstract_available"))

    print(f"Total papers:           {total}")
    print(f"With DOI:               {has_doi} ({has_doi/total*100:.0f}%)")
    print(f"  DOIs found by search: {doi_found}")
    print(f"CrossRef valid:         {cr_valid}")
    print(f"CrossRef invalid:       {cr_invalid}")
    print(f"Content verified:       {content_verified}")
    print(f"Content low match:      {content_failed}")
    print(f"No abstract available:  {no_abstract}")

    # Emotion-level summary
    print(f"\n--- By Emotion ---")
    emotions = sorted(set(r["emotion"] for r in results))
    for emo in emotions:
        emo_papers = [r for r in results if r["emotion"] == emo]
        emo_verified = sum(1 for r in emo_papers if r.get("verification", {}).get("content", {}).get("verified"))
        emo_doi = sum(1 for r in emo_papers if r.get("doi"))
        print(f"  {emo:20s}: {len(emo_papers):3d} papers, {emo_doi:3d} DOIs, {emo_verified:3d} content-verified")

    # Low match papers (potential hallucinations)
    low_match = [r for r in results
                 if r.get("verification", {}).get("content", {}).get("match_score", 1) < 0.15
                 and r.get("verification", {}).get("content", {}).get("reason") == "low_overlap"]
    if low_match:
        print(f"\n--- POTENTIAL HALLUCINATIONS (content match < 0.15) ---")
        for r in low_match:
            score = r["verification"]["content"]["match_score"]
            print(f"  [{r['id']}] {r['author_year']}: score={score:.2f} | {r['title'][:60]}")

    # Save results
    RESULTS_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"\nDetailed results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
