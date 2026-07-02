#!/usr/bin/env python3
"""Mechanically verify every ledger.json quoted_snippet against the paper cache.

This replaces the per-round LLM-invented verification loops (ad-hoc verify-*.txt
files, one-off Python one-liners) with ONE deterministic tool call. The worker
runs it as a pre-handoff self-check; R1 (ProvenanceCritic) runs it FIRST and then
spends its judgment only on what a tool cannot decide (venue-tier sourcing,
claim-kind labeling, sampled LIVE re-fetches to detect cache poisoning).

Usage:
  python3 tools/research/verify_ledger_snippets.py \
      --ledger road-to-master/domains/<domain>/<Topic>/ledger.json \
      --cache  <WORKSPACE_DIR>/.loop-manager/paper-cache/<slug> [--live]

Matching levels per entry (haystack = cached .tex/.txt/.html + entry-cited local
file paths that exist):
  EXACT       raw substring found
  NORMALIZED  found after whitespace-collapse + HTML-tag-strip (acceptable; the
              source's line wrapping / markup interrupts the raw bytes)
  NOT_FOUND   nowhere in the corpus  -> exit 1 (fix the snippet or the citation)
  LIVE_*      same, but matched only in a live re-fetch of source_url (--live)
  SKIPPED_*   non-cached http source without --live / cited path no longer exists

Exit codes: 0 all snippets EXACT/NORMALIZED (or skipped); 1 any NOT_FOUND;
2 invalid input.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
from pathlib import Path

CURL = [
    "curl", "--proto", "=https", "--location", "--silent",
    "--connect-timeout", "30", "--max-time", "60",
    "--max-filesize", str(50 * 1024 * 1024),
]

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


def normalize(text: str, strip_tags: bool) -> str:
    if strip_tags:
        text = TAG_RE.sub("", text)
        text = html.unescape(text)
    return WS_RE.sub(" ", text).strip()


def load_corpus(cache: Path) -> dict[str, tuple[str, str]]:
    """name -> (raw, normalized). Includes .tex/.txt/.html/.xml under the cache."""
    corpus: dict[str, tuple[str, str]] = {}
    if not cache.is_dir():
        return corpus
    for p in sorted(cache.rglob("*")):
        if p.is_file() and p.suffix in (".tex", ".txt", ".html", ".xml", ".bbl", ".bib", ".md"):
            raw = p.read_text(encoding="utf-8", errors="replace")
            corpus[str(p.relative_to(cache))] = (raw, normalize(raw, strip_tags=p.suffix in (".html", ".xml")))
    return corpus


def match_level(snippet: str, raw: str, norm: str) -> str | None:
    if snippet in raw:
        return "EXACT"
    if normalize(snippet, strip_tags=False) in norm:
        return "NORMALIZED"
    return None


def live_fetch(url: str) -> str | None:
    if not url.startswith("https://"):
        return None
    res = subprocess.run(CURL + ["--", url], capture_output=True, text=True)
    return res.stdout if res.returncode == 0 and res.stdout else None


def iter_entries(ledger):
    """Yield (entry_id, source_url, quoted_snippet) from common ledger shapes."""
    items = ledger.get("entries", ledger) if isinstance(ledger, dict) else ledger
    if isinstance(items, dict):
        for key, val in items.items():
            if isinstance(val, dict):
                yield key, val.get("source_url", ""), val.get("quoted_snippet", "")
    elif isinstance(items, list):
        for i, val in enumerate(items):
            if isinstance(val, dict):
                yield val.get("id", f"entry-{i}"), val.get("source_url", ""), val.get("quoted_snippet", "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--live", action="store_true", help="re-fetch non-cached https sources")
    args = ap.parse_args()

    ledger_path = Path(args.ledger)
    if not ledger_path.is_file():
        print(f"verify_ledger_snippets: ERROR: no ledger at {ledger_path}", file=sys.stderr)
        sys.exit(2)
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"verify_ledger_snippets: ERROR: ledger is not valid JSON: {e}", file=sys.stderr)
        sys.exit(2)

    corpus = load_corpus(Path(args.cache))
    live_cache: dict[str, tuple[str, str] | None] = {}
    results: list[tuple[str, str, str]] = []  # (entry, status, where)

    for entry_id, source_url, snippet in iter_entries(ledger):
        if not snippet:
            continue
        status, where = "NOT_FOUND", "-"
        # 1) cached corpus
        for name, (raw, norm) in corpus.items():
            lvl = match_level(snippet, raw, norm)
            if lvl:
                status, where = lvl, name
                break
        # 2) a cited local file path (e.g. temp code clone)
        if status == "NOT_FOUND" and source_url and not source_url.startswith("http"):
            p = Path(source_url.split(":")[0]) if ":" in source_url else Path(source_url)
            if p.is_file():
                raw = p.read_text(encoding="utf-8", errors="replace")
                lvl = match_level(snippet, raw, normalize(raw, strip_tags=False))
                if lvl:
                    status, where = lvl, str(p)
            else:
                status, where = "SKIPPED_MISSING_PATH", source_url
        # 3) live re-fetch
        if status == "NOT_FOUND" and source_url.startswith("https://"):
            if args.live:
                if source_url not in live_cache:
                    body = live_fetch(source_url)
                    live_cache[source_url] = (body, normalize(body, strip_tags=True)) if body else None
                hit = live_cache[source_url]
                if hit:
                    lvl = match_level(snippet, hit[0], hit[1])
                    if lvl:
                        status, where = f"LIVE_{lvl}", source_url
            elif not corpus:
                status, where = "SKIPPED_LIVE", source_url
        results.append((entry_id, status, where))

    not_found = [r for r in results if r[1] == "NOT_FOUND"]
    width = max((len(r[0]) for r in results), default=10)
    for entry_id, status, where in results:
        print(f"{entry_id:<{width}}  {status:<20}  {where}")
    print(f"\nchecked={len(results)} exact={sum(1 for r in results if r[1]=='EXACT')} "
          f"normalized={sum(1 for r in results if r[1]=='NORMALIZED')} "
          f"live={sum(1 for r in results if r[1].startswith('LIVE'))} "
          f"skipped={sum(1 for r in results if r[1].startswith('SKIPPED'))} "
          f"NOT_FOUND={len(not_found)}")
    sys.exit(1 if not_found else 0)


if __name__ == "__main__":
    main()
