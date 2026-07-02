#!/usr/bin/env python3
"""Fetch a paper ONCE into a shared cache dir for the research-note pipeline.

The cache is consumed by the worker (all renew rounds) and reviewers (R2/R5) so
that only ONE agent ever pays the discovery/fetch/extract cost. R1 keeps doing
its own sampled live re-fetches by design (anti-fabrication) — this tool is not
a substitute for that.

Usage:
  python3 tools/research/fetch_paper.py --url https://arxiv.org/abs/2304.02643 \
      --out <WORKSPACE_DIR>/.loop-manager/paper-cache/<slug>

Behavior:
  - https-only validation (rejects ext::, file://, git://, ssh://, http://,
    leading '-'). No override flag.
  - arXiv URLs: fetches the PDF, the e-print LaTeX source (extracted under
    source/), and the ar5iv HTML rendering. LaTeX is the PREFERRED provenance
    representation for quotes and numeric claims.
  - Non-arXiv URLs: fetches the URL as-is; a 403/challenge answer is reported
    as exit 3 (access blocked) so the caller can run the arXiv-preprint
    fallback / BLOCKED protocol.
  - Extracts plain text from the PDF when a backend exists (pdftotext, then
    pymupdf, then pdfplumber); records "unavailable" otherwise. With LaTeX
    present this is optional.
  - Writes meta.json: every artifact's source_url, sha256, byte size, HTTP
    code, retrieved_at (UTC), extractor used. Reviewers verify cache
    integrity against this.
  - Re-running on a populated cache dir is a NO-OP (exit 0) unless --force.

Exit codes: 0 ok / cache hit; 2 invalid input; 3 access blocked (403/paywall);
4 fetch or extraction error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

CURL = [
    "curl", "--proto", "=https", "--location", "--silent", "--show-error",
    "--connect-timeout", "30", "--max-time", "120",
    "--max-filesize", str(50 * 1024 * 1024),
]

ARXIV_RE = re.compile(
    r"^https://(?:www\.)?arxiv\.org/(?:abs|pdf|e-print)/"
    r"(?P<id>[0-9]{4}\.[0-9]{4,5}|[a-z-]+(?:\.[A-Z]{2})?/[0-9]{7})"
    r"(?P<ver>v[0-9]+)?"
)


def die(code: int, msg: str) -> None:
    print(f"fetch_paper: ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def validate_https(url: str) -> str:
    if url.startswith("-"):
        die(2, "url must not start with '-'")
    if not url.startswith("https://"):
        die(2, "url must be https:// (ext::, file://, git://, ssh://, http:// are rejected; no override)")
    return url


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def curl_fetch(url: str, dest: Path):
    """Fetch url to dest. Returns (http_code, content_type, error)."""
    res = subprocess.run(
        CURL + ["-o", str(dest), "-w", "%{http_code}\t%{content_type}", "--", url],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        return None, None, res.stderr.strip() or f"curl exit {res.returncode}"
    parts = (res.stdout.split("\t") + ["", ""])[:2]
    try:
        return int(parts[0]), parts[1], None
    except ValueError:
        return None, parts[1], f"unparseable http code {parts[0]!r}"


def extract_eprint(archive: Path, out_dir: Path) -> list[str]:
    """Extract .tex/.bbl/.bib members from an arXiv e-print (tar.gz, gz, or raw tex)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[str] = []
    try:
        with tarfile.open(archive, mode="r:*") as tar:
            for member in tar.getmembers():
                name = Path(member.name)
                if member.isfile() and name.suffix in (".tex", ".bbl", ".bib") and ".." not in name.parts:
                    target = out_dir / name.name
                    with tar.extractfile(member) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    extracted.append(target.name)
        return extracted
    except tarfile.TarError:
        pass
    # Not a tar: maybe a gzipped (or raw) single .tex file.
    import gzip
    try:
        data = gzip.decompress(archive.read_bytes())
    except OSError:
        data = archive.read_bytes()
    if b"\\documentclass" in data[:4096] or b"\\begin{document}" in data:
        target = out_dir / "main.tex"
        target.write_bytes(data)
        return [target.name]
    return []


def extract_pdf_text(pdf: Path, dest: Path) -> str:
    """Try pdftotext -> pymupdf -> pdfplumber. Returns extractor name or 'unavailable'."""
    if shutil.which("pdftotext"):
        res = subprocess.run(["pdftotext", "-layout", str(pdf), str(dest)], capture_output=True)
        if res.returncode == 0 and dest.exists():
            return "pdftotext"
    try:
        import fitz  # pymupdf
        with fitz.open(pdf) as doc:
            dest.write_text("\n".join(page.get_text() for page in doc), encoding="utf-8")
        return "pymupdf"
    except Exception:
        pass
    try:
        import pdfplumber
        with pdfplumber.open(pdf) as doc:
            dest.write_text(
                "\n".join((page.extract_text() or "") for page in doc.pages), encoding="utf-8")
        return "pdfplumber"
    except Exception:
        pass
    return "unavailable"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True, help="paper url (https only)")
    ap.add_argument("--out", required=True, help="cache dir, e.g. <WS>/.loop-manager/paper-cache/<slug>")
    ap.add_argument("--force", action="store_true", help="re-fetch even if meta.json exists")
    args = ap.parse_args()

    url = validate_https(args.url.strip())
    out = Path(args.out)
    meta_path = out / "meta.json"
    if meta_path.exists() and not args.force:
        print(f"cache hit: {meta_path} exists — reuse it (pass --force to re-fetch)")
        sys.exit(0)
    out.mkdir(parents=True, exist_ok=True)

    meta = {
        "requested_url": url,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": [],
        "blocked": None,
        "pdf_text_extractor": None,
    }

    def record(name: str, source_url: str, code, ctype):
        p = out / name
        meta["artifacts"].append({
            "name": name, "source_url": source_url, "http_code": code,
            "content_type": ctype, "sha256": sha256_of(p), "bytes": p.stat().st_size,
        })

    m = ARXIV_RE.match(url)
    if m:
        aid = m.group("id") + (m.group("ver") or "")
        fetch_plan = [
            ("paper.pdf", f"https://arxiv.org/pdf/{aid}"),
            ("eprint.bin", f"https://arxiv.org/e-print/{aid}"),
            ("ar5iv.html", f"https://ar5iv.labs.arxiv.org/html/{aid}"),
        ]
        meta["arxiv_id"] = aid
        for name, src in fetch_plan:
            code, ctype, err = curl_fetch(src, out / name)
            if err or code is None:
                if name == "paper.pdf":
                    die(4, f"fetch failed for {src}: {err}")
                print(f"warn: optional fetch failed for {src}: {err or code}", file=sys.stderr)
                (out / name).unlink(missing_ok=True)
                continue
            if code == 403:
                die(3, f"HTTP 403 from {src} — access blocked")
            if code != 200:
                if name == "paper.pdf":
                    die(4, f"HTTP {code} from {src}")
                (out / name).unlink(missing_ok=True)
                continue
            record(name, src, code, ctype)
        eprint = out / "eprint.bin"
        if eprint.exists():
            texs = extract_eprint(eprint, out / "source")
            meta["latex_sources"] = texs
            if texs:
                print(f"LaTeX source extracted: source/{{{', '.join(texs)}}} — PREFERRED provenance representation")
    else:
        code, ctype, err = curl_fetch(url, out / "paper.pdf")
        if err or code is None:
            die(4, f"fetch failed: {err}")
        if code == 403:
            meta["blocked"] = {"http_code": 403, "reason": "403 access block / paywall"}
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            die(3, "HTTP 403 — access blocked (meta.json written); run the arXiv-preprint fallback / BLOCKED protocol")
        if code != 200:
            die(4, f"HTTP {code}")
        if ctype and "pdf" not in ctype:
            (out / "paper.pdf").rename(out / "landing.html")
            record("landing.html", url, code, ctype)
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            die(3, f"non-PDF answer ({ctype}) — likely landing page/paywall; inspect landing.html")
        record("paper.pdf", url, code, ctype)

    pdf = out / "paper.pdf"
    if pdf.exists():
        meta["pdf_text_extractor"] = extract_pdf_text(pdf, out / "paper.txt")
        if meta["pdf_text_extractor"] != "unavailable":
            record("paper.txt", "extracted-from:paper.pdf", None, "text/plain")

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"cache ready: {out} ({len(meta['artifacts'])} artifacts; extractor={meta['pdf_text_extractor']})")


if __name__ == "__main__":
    main()
