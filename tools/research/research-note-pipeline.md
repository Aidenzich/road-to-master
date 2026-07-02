# Research Note Pipeline (loops orchestration)

Research notes are produced by the loops orchestrator using `pipelines/research-note.yaml`
(lives in the loops repo — pipelines are loops execution units). The domain assets live
HERE in road-to-master: this generator, the task template, and the `make research-check`
gate (`tools/research/check_note.py`).

Stamp one task `.md` per paper with the generator, then run the batch through loops:

```bash
# from the road-to-master checkout
node tools/research/new-research-task.mjs \
  --url https://example.org/paper.pdf \
  --concept "Attention Is All You Need" \
  --domain natural_language_processing \
  --out <loops>/prompts/<batch-name>

# from the loops checkout
make run DIR=<loops>/prompts/<batch-name> PIPELINE=pipelines/research-note.yaml CONCURRENCY=1
```

Required fields are `paper_url`, `concept`, and `domain`; `code_repo_url` is optional. The
worker validates `paper_url`, and `code_repo_url` when present, as https-only before any
fetch or clone, writes an evidence ledger, authors the note, and records `<domain>/<Topic>`
in `.loop-manager/research-note-note-path`.

Access-blocked papers (paywall/403/no full text): the worker first tries the arXiv preprint
fallback (verified title+author match only). If no free full text exists, the run ends
BLOCKED — no note of any kind is authored, `.loop-manager/research-note-blocked.json`
records the observed facts, the note gate reports N/A, and the final phase files a tracking
issue on this repo instead of publishing a note PR.

The review phase is `Review: Research-Quality Gate`: first a deterministic command gate runs
`make -C road-to-master research-check NOTE=<domain>/<Topic>`, then five independent
reviewers run in parallel: ProvenanceCritic, TechnicalCorrectness,
Readability&Demystification, Integration&StaticAnalysis, and CriticalEvaluation. R2 and R5
fetch the paper full text themselves before judging. The final phase publishes one
road-to-master PR (or the blocked-run tracking issue).

Run research-note batches with `CONCURRENCY=1`. Domain README tables are shared files, so
parallel notes can race on the same index.

## Paper cache & tool-first verification (token efficiency)

Every consumer used to re-run its own discovery → fetch → extract → verify loop
(worker per renew round, R2/R5 per review round, R1's ad-hoc `verify-*.txt`
scripts) — the same paper was re-acquired 4-5× per round. The tools here
collapse that to ONE fetch and ONE deterministic verifier:

| Tool | Job | Notes |
|-|-|-|
| `fetch_paper.py` | fetch ONCE into `<WS>/.loop-manager/paper-cache/<slug>/` | arXiv: PDF + e-print LaTeX (`source/*.tex`) + ar5iv HTML; `meta.json` records sha256/bytes/urls; idempotent (cache hit = no-op); 403 → exit 3 for the BLOCKED protocol |
| `verify_ledger_snippets.py` | exact-substring check of every `quoted_snippet` vs the cache | EXACT / NORMALIZED (ws-collapse + tag-strip) / NOT_FOUND (exit 1); `--live` re-fetches non-cached https sources; worker pre-handoff self-check AND R1's first step |
| `snapshot_pdf_region.py` | crop a PDF page region to PNG (table/figure evidence) | numeric claims verified by looking at ONE image; needs pymupdf, exits 5 gracefully without it — then quote the `.tex` table environment lines instead |

Division of labor (deliberate):
- **Worker** fetches once, authors from `source/*.tex` when present (LaTeX is the
  preferred provenance representation — line wrapping and HTML markup in
  PDF-extracted text / ar5iv cause false snippet mismatches), runs the verifier
  before handing off, and fixes every NOT_FOUND first.
- **R2/R5** read the cache (same bytes as the worker) — no re-fetch.
- **R1** runs the verifier first, then spends judgment ONLY on what a tool cannot
  decide: venue-tier sourcing, claim-kind labeling, and 2-3 sampled LIVE
  re-fetches compared against `meta.json` sha256s (cache-poisoning check). R1's
  independent live sampling is the anti-fabrication gate — never replace it
  with the cache.

These are agent-invoked tools referenced by the pipeline prompts — deliberately
NOT wired as a pipeline `commands:` gate (a hard gate that needs network is the
same failure class as the pre-#88 paywall gate crash).
