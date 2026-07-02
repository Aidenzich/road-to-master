# road-to-master research-note contract + gate runner

This directory is the **single source of truth** for "what a valid road-to-master research note
is". It ships a deterministic, **offline**, stdlib-only gate runner that mechanically validates a
note against the contract, plus the machine-readable schema and template the contract is built on.
It supersedes the trivial prompt in `tools/llm/research_tool.md` (a 250-word summary prompt, not a
validator).

A note is a directory `domains/<domain>/<Topic>/` containing:
- `README.md` — the note (universal filename across all 90 existing notes).
- `ledger.json` — the evidence ledger (see `ledger.schema.json`).
- `imgs/` — committed figure files (non-PDF).

## Run it

```bash
# Validate one note (from the repo root). Exits non-zero on ANY violation.
make research-check NOTE=natural_language_processing/AttentionIsAllYouNeed

# Red/green proof: good fixture PASSes, each broken fixture FAILs exactly its gate(s).
make research-check-fixtures

# Direct invocations:
python3 tools/research/check_note.py --note <domain>/<Topic>
python3 tools/research/check_note.py --fixture good
python3 tools/research/check_note.py --path <arbitrary note dir>
python3 tools/research/check_note.py --all-fixtures
```

Requires `python3` (stdlib only — no `pip install`, no lockfile). `make research-check NOTE=<x>`
resolves `domains/<x>` first, falling back to `tools/research/fixtures/<x>`.

## The contract (what the gates enforce)

**All sub-gates are evaluated on every run; the verdict is the UNION of failing gates (no
first-failure short-circuit).** Exit code is non-zero iff ≥1 sub-gate fails, and the printed
verdict lists every failing gate name sorted. `nothing-ran` (missing note/ledger) is a FAIL, never
a silent pass. Each failure prints a concrete `gate · file · expected-vs-actual` reason.

### Note structure (`structure:*`)
Sections must appear **in order**: `## 📇 Academic Context` → first-principles body →
`## 🧪 Critical Assessment` → `## 🔗 Related notes`.

| Gate | Fails when |
|-|-|
| `structure:academic-context` | the `## 📇 Academic Context` heading is missing, or that section does not contain the required `| Field | Value |` academic-value table with at least one data row |
| `structure:section-order` | the present anchor sections are out of canonical order, **or** a first-principles body section precedes `## 📇 Academic Context` (it must be the first `##` section), **or** any section follows `## 🔗 Related notes` (it must be the last) |
| `structure:related-notes` | the `## 🔗 Related notes` heading is missing (mandatory even if empty) |
| `structure:substance` | a technical note's body has no equation (`$…$`), algorithm/code block (```` ``` ````) or table (blog/tech-report relaxed to ≥1 concept sub-heading; tier read from the `Venue Kind` cell) |
| `structure:critical-assessment` | the `## 🧪 Critical Assessment` heading is missing, or the section is trivial (< 400 chars **and** no interrogation **sub-heading** — an actual `###`+ line naming problem-realness / baseline-ablation-dataset-metric / novelty-vs-repackaging / problem-solved / real-world-relevance; a marker word buried in prose does not count) |

The Critical Assessment must interrogate the paper from first principles (problem realness,
baseline/ablation/dataset/metric sufficiency, genuine novelty vs engineering-repackaging / rename /
self-defined-benchmark 射箭畫靶, whether the *claimed* problem is actually solved and real-world
relevant). The deterministic gate only asserts the section **exists and is non-trivial**; judging
the *quality* of the critique is the Task 02 R5 reviewer's job.

### Evidence ledger (`ledger:*`) — 4-way epistemic taxonomy
`kind` is a **4-value enum** (a 2-way split is a FAIL): `paper-claim` (論文聲稱), `evidence-supported`
(證據支持), `reasonable-inference` (合理推論), `unproven-or-doubtful` (未被證明/值得懷疑).

| Gate | Fails when |
|-|-|
| `ledger:schema` | `ledger.json` violates the shape contract in `ledger.schema.json` — the runner hand-rolls a tiny stdlib validator over the schema file itself (enforcing `type` / `required` / `additionalProperties` / `minLength`), so a missing required field (`claim_id`, `claim_text`, `kind`, `target`, `source_url`, `retrieved_at`, `quoted_snippet`; top-level `note_path`, `domain`, `topic`, `no_untrusted_code_executed`, `entries`), a wrong-typed value, or an unknown field FAILs. The value-semantic keywords are deliberately owned by their dedicated gates so each violation is reported once: `enum` (kind) → `ledger:kind-enum`, `const` (attestation) → `attestation`, `pattern` (topic) → `name` |
| `ledger:coverage` | an Academic Context cell with a substantive value has no matching ledger entry (`target: table-cell:<field>`), any matching table-cell row lacks a non-empty `source_url`, or a deterministic first-principles body / valid Critical Assessment claim unit has no exact matching ledger entry (`target: note-section:body-prose-1`, `note-section:body-equation-1`, `note-section:body-table-1`, `note-section:body-code-1`, `note-section:critical-prose-1`, ...). The matching entry's `quoted_snippet` or `claim_text` must match the actual table-cell value or occur in the claim unit, so an unrelated `table-cell:*` or `note-section:*` row is not coverage. For non-code note-section claims, every matching `paper-claim` and `evidence-supported` entry also needs its own non-empty `source_url`; a sourced duplicate does not mask an unsourced duplicate. `reasonable-inference` / `unproven-or-doubtful` entries may be unsourced. Coverage of table cells and non-trivial note claims, including Critical Assessment critique claims, must be 100% |
| `ledger:blank-or-unsourced-cell` | an Academic Context cell is left **blank** — write the literal `unknown`/`unavailable` to make an absence explicit; those two literals (case-insensitive) are the *only* unsourced values that pass |
| `ledger:kind-enum` | any entry's `kind` is outside the 4-value enum |
| `ledger:code-source-not-file` | an `evidence-supported` **code** claim (target contains `code`) is accepted ONLY when its `source_url` is a filesystem path (matches `^(\.?/\|[^:]+/)`, never `^[a-z]+://`) **that resolves to a real file INSIDE the inspected clone** — realpath-confined to the note dir or repo root subtree, so neither an absolute path (`/etc/hosts`) nor a `../` traversal can escape. A URL, `null`, a pathless string, a nonexistent path, or a real file *outside* the clone all FAIL — code can only be "supported" by pointing at the file that was read |
| `ledger:no-endorsement` | the ledger has ONLY `paper-claim`/`evidence-supported` entries **and** no explicit "no material weaknesses found, evidence: …" verdict — an uncritical restatement |

### Safety, provenance & identity

| Gate | Fails when |
|-|-|
| `link` | any relative link (`[t](../x)`, `[t](./y)`) does not resolve in the worktree. `http(s)://` links are not resolved (must appear in the ledger if they back a claim). Image links are owned by `figure` |
| `figure` | a referenced `![](imgs/x)` is missing, is a `.pdf` (gitignored), or has no ledger `figure:imgs/x` source |
| `secret` | a note/ledger snippet contains a secret (AWS `AKIA…`, GitHub `ghp_…`, PEM private key, or an `api_key=…`/`secret_key=…` assignment) |
| `name` | the **actual note folder name** or the ledger `topic` does not match `^[A-Za-z0-9._-]+$` (no `..`, no path separators, no shell metachars). The `--note`/`--fixture` CLI input is additionally pre-validated segment-by-segment **before any filesystem access**, so a hostile `--note d/../evil` is rejected as `gate=name`, never resolved |
| `domain` | `<domain>` is not one of the fixed 8 (enum **derived at runtime** from live `domains/` dirs, so a future domain addition doesn't fork truth) |
| `duplicate` | `domains/<domain>/<Topic>/` already exists with a non-empty `README.md` (and isn't the note being checked) → `DUPLICATE`; the caller decides skip/merge — never silently overwrite |
| `attestation` | the ledger lacks `no_untrusted_code_executed: true`, or an `*.exec-marker` file exists under the note dir |

## Scope boundaries (documented, not silent gaps)
- **Offline & read-only.** Validators never fetch the network and never write/execute. They check an
  already-written note + its ledger.
- **Shape/coverage, not truth.** The gate proves a `quoted_snippet` is *present and covered*, not that
  it is faithful to the source. Snippet fabrication is caught by the pipeline's re-fetch
  ProvenanceCritic reviewer (Task 02 R1), not here.
- **Code source = read file in the clone.** `ledger:code-source-not-file` asserts an
  evidence-supported code claim's `source_url` is a path (not a URL/null/pathless string) whose
  **realpath stays inside** the note dir or repo root subtree — absolute paths and `../`
  traversals that land outside the clone are rejected. Whether the snippet faithfully quotes
  that file is the Task 02 re-fetch reviewer's job.
- **No PDF/figure-extraction backend.** Installing extraction backends is a Task 02/03 runtime
  concern, not the validator's.

## Fixtures (red/green proof)
`fixtures/` holds one GOOD note (passes all gates) and one deliberately-broken note per validator
(each FAILs exactly its intended gate) — including one per boundary *disjunct* (body-before-AC and
body-after-Related for section-order; URL/null/nonexistent-file plus the two clone-escape shapes,
`../`-traversal and absolute-path, for code-source; trivial-section for critical-assessment;
hostile actual-folder-name for name; a schema-shape violation for `ledger:schema`; and an
empty-ledger, null-table-cell-source, null-body-claim-source, unsourced-duplicate body-claim, unrelated-table-cell-ledger, unrelated-body-ledger, missing-Academic-Context-table, and missing/unrelated Critical Assessment ledger fixtures for coverage/structure boundaries) — plus a `multi_fault` note proving the
union/no-short-circuit rule. `fixtures/manifest.json` is the expected-verdict map consumed by `--all-fixtures`;
`fixtures/build_fixtures.py` is the deterministic generator that emits the tree from the GOOD
baseline plus one minimal delta per fixture (re-run it to regenerate). See `manifest.json` for the
full fixture→gate mapping.

## Files
- `check_note.py` — the gate runner (all sub-checks, union-reported).
- `ledger.schema.json` — machine-readable ledger schema, **enforced at runtime** by the
  `ledger:schema` gate (not decorative documentation).
- `templates/note.README.md` — the note scaffold an author fills.
- `fixtures/` — red/green fixtures + manifest + generator.
