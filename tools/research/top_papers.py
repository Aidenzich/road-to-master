#!/usr/bin/env python3
"""Pull the top-N papers per top-tier AI conference for a given year.

Data source: Semantic Scholar Graph API (bulk search, no API key needed at
low request rates).

"Best" is operationalized as one of two metrics:
  - influential (default): Semantic Scholar's influentialCitationCount —
    citations where the citing paper actually builds on this one
    (methodology/results), not courtesy citations. Better quality proxy,
    especially against survey/benchmark citation inflation.
  - citations: raw citation count. Most objective, but favors older papers
    within a year window and reference-magnet paper types.

Trust boundary — what is deterministic and what is curated:
  - The CITATION RANKING half is fully deterministic and LLM-free: raw
    Semantic Scholar API numbers, deterministic sort, reproducible run to
    run. Its error sources are the database itself (venue-tagging lag,
    year-of-first-publication semantics), never a model judgment, and the
    DEGRADED flag states when it cannot be trusted.
  - The AWARDS half is deterministic at RUN time (the script only reads
    best_paper_awards.json) but the dataset is CURATED: entries are
    researched from official award pages, historically with LLM
    assistance. The trust anchor is the mandatory source_url on every
    conference-year block — verify it once by hand (a few minutes per
    year) and the dataset is thereafter a static, human-verified fact
    file with no residual LLM dependency. Do not trust an entry whose
    source_url you have not opened; do not add entries without one.
  - Deliberately NOT a scraper: six official sites with divergent,
    yearly-changing layouts would fail silently. A once-a-year manual
    append with a source_url fails loudly and is auditable.

Known caveats (inherent to the source, not bugs):
  - S2's `year` is the FIRST publication year (often the arXiv preprint),
    not the conference edition year. A paper posted to arXiv in 2023 but
    presented at ICLR 2024 shows up under 2023.
  - Recent years are citation-immature: for the current/previous year the
    ranking is closer to "early attention" than "best".
  - Official Best Paper awards are a different notion of best. There is no
    stable API for them, so this tool ships a CURATED dataset
    (best_paper_awards.json, every block carries the primary source_url it
    was verified against). When the citation ranking is credibility-degraded
    (see below) the awards become the primary answer; otherwise they are an
    optional cross-reference (--awards always).

Credibility check (per conference-year, citation ranking is DEGRADED when
any of these holds):
  - median influentialCitationCount of the top slice < 30 (mature years
    score in the hundreds; single digits = S2 venue tagging not backfilled)
  - fewer than 500 venue-tagged papers for that year (backfill incomplete)
  - the queried year is the current calendar year or later (citations
    have not accumulated)

Usage:
  python3 tools/research/top_papers.py --year 2023
  python3 tools/research/top_papers.py --year 2021 --metric citations --top 5
  python3 tools/research/top_papers.py --year 2024 --conf neurips,icml
"""

import argparse
import datetime
import json
import pathlib
import statistics
import sys
import time
import urllib.parse
import urllib.request

API = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
FIELDS = "title,year,venue,citationCount,influentialCitationCount,externalIds,authors,url"
# How deep to fetch (by raw citations) before re-ranking by the chosen
# metric. Influential-vs-raw rank divergence is real but not unbounded;
# 300 is far past where a top-3 influential paper could hide.
POOL = 300
AWARDS_FILE = pathlib.Path(__file__).parent / "best_paper_awards.json"
MIN_MEDIAN_INFLUENTIAL = 30
MIN_TAGGED_TOTAL = 500

CONFS = {
    "neurips": "Neural Information Processing Systems",
    "icml": "International Conference on Machine Learning",
    "iclr": "International Conference on Learning Representations",
    "cvpr": "Computer Vision and Pattern Recognition",
    "acl": "Annual Meeting of the Association for Computational Linguistics",
    "aaai": "AAAI Conference on Artificial Intelligence",
}


def credibility(top: list[dict], total: int, year: int) -> list[str]:
    """Return the list of reasons the citation ranking is degraded (empty = OK)."""
    reasons = []
    if top:
        med = statistics.median(p.get("influentialCitationCount") or 0 for p in top)
        if med < MIN_MEDIAN_INFLUENTIAL:
            reasons.append(f"median influential of top slice is {med:g} < {MIN_MEDIAN_INFLUENTIAL}")
    if total < MIN_TAGGED_TOTAL:
        reasons.append(f"only {total} venue-tagged papers (< {MIN_TAGGED_TOTAL}; S2 backfill incomplete)")
    if year >= datetime.date.today().year:
        reasons.append("queried year is current/future; citations not accumulated")
    return reasons


def load_awards(year: int) -> dict:
    if not AWARDS_FILE.exists():
        return {}
    with open(AWARDS_FILE) as f:
        data = json.load(f)
    return data.get(str(year), {})


def fetch(venue: str, year: int) -> tuple[list[dict], int]:
    """Fetch up to POOL papers for venue+year sorted by raw citations."""
    papers: list[dict] = []
    total = 0
    token = None
    while len(papers) < POOL:
        params = {
            "query": "",
            "venue": venue,
            "year": str(year),
            "sort": "citationCount:desc",
            "fields": FIELDS,
        }
        if token:
            params["token"] = token
        url = API + "?" + urllib.parse.urlencode(params)
        for attempt in range(5):
            try:
                with urllib.request.urlopen(urllib.request.Request(
                        url, headers={"User-Agent": "road-to-master/top_papers"})) as r:
                    payload = json.load(r)
                break
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < 4:
                    wait = 2 ** (attempt + 1)
                    print(f"  429 rate-limited, retrying in {wait}s", file=sys.stderr)
                    time.sleep(wait)
                    continue
                raise
        papers.extend(payload.get("data") or [])
        total = payload.get("total") or total
        token = payload.get("token")
        if not token or not payload.get("data"):
            break
        time.sleep(1.1)
    return papers[:POOL], total


def first_author(p: dict) -> str:
    authors = p.get("authors") or []
    if not authors:
        return "?"
    name = authors[0].get("name", "?")
    return name + (" et al." if len(authors) > 1 else "")


def arxiv_id(p: dict) -> str:
    return (p.get("externalIds") or {}).get("ArXiv") or "-"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--metric", choices=["influential", "citations"],
                    default="influential")
    ap.add_argument("--conf", default=",".join(CONFS),
                    help="comma-separated subset of: " + ",".join(CONFS))
    ap.add_argument("--json", action="store_true", help="emit JSON instead of markdown")
    ap.add_argument("--awards", choices=["auto", "always", "off"], default="auto",
                    help="auto: show curated official awards when the citation ranking "
                         "is credibility-degraded; always: show alongside; off: never")
    args = ap.parse_args()

    keys = [k.strip().lower() for k in args.conf.split(",") if k.strip()]
    unknown = [k for k in keys if k not in CONFS]
    if unknown:
        ap.error(f"unknown conf key(s): {unknown}; valid: {list(CONFS)}")

    sort_key = ("influentialCitationCount" if args.metric == "influential"
                else "citationCount")
    awards = load_awards(args.year) if args.awards != "off" else {}
    out = {}
    degraded = {}
    for key in keys:
        venue = CONFS[key]
        print(f"[fetch] {key} ({venue}) {args.year} ...", file=sys.stderr)
        papers, total = fetch(venue, args.year)
        papers.sort(key=lambda p: (p.get(sort_key) or 0), reverse=True)
        out[key] = papers[: args.top]
        degraded[key] = credibility(out[key], total, args.year)
        time.sleep(1.1)

    if args.json:
        print(json.dumps({
            "ranking": out,
            "degraded": degraded,
            "awards": {k: awards.get(k) for k in keys if awards.get(k)},
        }, indent=2, ensure_ascii=False))
        return 0

    print(f"# Top {args.top} papers per conference — {args.year} "
          f"(metric: {args.metric})\n")
    print("Caveats: S2 year = first-publication (often arXiv) year, not the "
          "conference edition; recent years are citation-immature; official "
          "Best Paper awards are a separate, manually-checked notion.\n")
    for key in keys:
        print(f"## {key.upper()}\n")
        print("| # | Paper | 1st author | Citations | Influential | arXiv |")
        print("|---|-------|-----------|-----------|-------------|-------|")
        for i, p in enumerate(out[key], 1):
            title = p.get("title", "?").replace("|", "\\|")
            link = p.get("url") or ""
            print(f"| {i} | [{title}]({link}) | {first_author(p)} "
                  f"| {p.get('citationCount', 0)} "
                  f"| {p.get('influentialCitationCount', 0)} | {arxiv_id(p)} |")
        print()
        show_awards = (args.awards == "always"
                       or (args.awards == "auto" and degraded[key]))
        if degraded[key]:
            print(f"> ⚠ citation ranking DEGRADED for {key.upper()} {args.year}: "
                  + "; ".join(degraded[key]) + "\n")
        if show_awards:
            entry = awards.get(key)
            if entry:
                print(f"**Official awards ({key.upper()} {args.year})** "
                      f"— [source]({entry['source_url']}):\n")
                for p in entry["papers"]:
                    ax = f" (arXiv:{p['arxiv']})" if p.get("arxiv") else ""
                    print(f"- {p['award']}: *{p['title']}* — {p['first_author']} et al.{ax}")
                print()
            elif args.awards == "auto":
                print(f"> No curated awards for {key.upper()} {args.year} in "
                      f"best_paper_awards.json — add them (with the official source URL) "
                      f"or check the conference site manually.\n")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
