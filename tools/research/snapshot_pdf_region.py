#!/usr/bin/env python3
"""Render a PDF page (or a region of it) to PNG — table/figure evidence snapshots.

Numeric claims (benchmark tables, ablation numbers) burn the most review tokens
when every verifier re-hunts them in full text. A cropped snapshot of the exact
table lets a reviewer confirm the numbers by looking at ONE image. Store
snapshots under the paper cache (e.g. paper-cache/<slug>/evidence/) and point
the ledger entry at the snapshot path plus the .tex file:line.

Usage:
  python3 tools/research/snapshot_pdf_region.py --pdf paper.pdf --page 7 \
      --out evidence/table4-lvis.png [--bbox x0,y0,x1,y1] [--zoom 2.0]

--page is 1-based. --bbox is in PDF points (origin top-left, as reported by
pymupdf); omit it to snapshot the whole page, then re-crop with a tighter bbox.

Requires pymupdf. If it is not installed this tool exits 5 with a clear
message — snapshots are best-effort evidence, like the figure backends; fall
back to quoting the .tex table environment lines instead.

Exit codes: 0 ok; 2 invalid input; 5 no renderer backend (pymupdf missing).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--page", required=True, type=int, help="1-based page number")
    ap.add_argument("--out", required=True, help="output .png path")
    ap.add_argument("--bbox", help="x0,y0,x1,y1 in PDF points; omit for full page")
    ap.add_argument("--zoom", type=float, default=2.0, help="render scale (default 2.0)")
    args = ap.parse_args()

    try:
        import fitz  # pymupdf
    except ImportError:
        print("snapshot_pdf_region: ERROR: pymupdf not installed — no PDF renderer on this box. "
              "Fall back to quoting the .tex table/figure environment lines as evidence.",
              file=sys.stderr)
        sys.exit(5)

    pdf = Path(args.pdf)
    if not pdf.is_file():
        print(f"snapshot_pdf_region: ERROR: no such pdf: {pdf}", file=sys.stderr)
        sys.exit(2)

    with fitz.open(pdf) as doc:
        if not 1 <= args.page <= doc.page_count:
            print(f"snapshot_pdf_region: ERROR: page {args.page} out of range 1..{doc.page_count}",
                  file=sys.stderr)
            sys.exit(2)
        page = doc[args.page - 1]
        clip = None
        if args.bbox:
            try:
                x0, y0, x1, y1 = (float(v) for v in args.bbox.split(","))
                clip = fitz.Rect(x0, y0, x1, y1)
            except ValueError:
                print("snapshot_pdf_region: ERROR: --bbox must be x0,y0,x1,y1", file=sys.stderr)
                sys.exit(2)
        mat = fitz.Matrix(args.zoom, args.zoom)
        pix = page.get_pixmap(matrix=mat, clip=clip)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        pix.save(out)
        print(f"snapshot written: {out} ({pix.width}x{pix.height}px, page {args.page}"
              f"{', bbox ' + args.bbox if args.bbox else ', full page'})")


if __name__ == "__main__":
    main()
