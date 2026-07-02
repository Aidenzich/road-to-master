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
