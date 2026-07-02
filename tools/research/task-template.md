# Research Note Task: [[concept]]

## Input

paper_url: [[paper_url]]
concept: [[concept]]
domain: [[domain]]
code_repo_url: [[code_repo_url]]
expected_title: [[expected_title]]
expected_venue: [[expected_venue]]
expected_year: [[expected_year]]

## Worker Contract

Validate input before any fetch or clone. `paper_url` and optional `code_repo_url` are
https-only when present; an absent or blank `code_repo_url` means no code repository was
provided and must not block the paper task. Reject ext::, file://, git://, ssh://,
user@host:, http://, and any leading-`-` value. domain must be one of: natural_language_processing, computer_vision,
recommender_system, timeseries, reinforcement_learning, mlops, utils, vision_language.

Invoke the deep-research Skill methodology for evidence gathering: broaden source discovery,
cross-check paper/project/venue metadata, and keep provenance in the ledger instead of relying
on a single search result.

Fetch/cache into an uncommitted cache dir OUTSIDE the note tree. Use `curl --proto '=https'`
for paper fetches; abort PDF fetches over 50 MB; timeout is 30000 ms with 2 retries only for
transient/timeout/5xx failures. HTTP 403 / 404 / 410 are non-retryable terminal outcomes with
no retry storm: 404/410 -> FAIL as dead URL. On 403/paywall/no full text, first try the arXiv
preprint fallback (query `https://export.arxiv.org/api/query` by exact title; accept only on
normalized-title match plus author overlap; on a verified match author the normal note from
the preprint with mandatory arXiv id+version provenance in the ledger). If no verified
preprint exists, the run is BLOCKED: author NO note of any kind (no access-block record, no
domain README row, no note-path file), write the observed facts to
`.loop-manager/research-note-blocked.json`, leave every repo working tree clean, and stop.
Never fabricate evidence from title-only access. See the pipeline's Worker Contract
(`pipelines/research-note.yaml` in the loops repo) for the authoritative BLOCKED protocol.

Run figure-backend preflight for `pdftoppm`, `pdfimages`, python `pymupdf`, and python
`pdfplumber`. Figures are best-effort HTML-first from paper HTML/project page; process at most
12 figures. If no backend exists and no HTML figures are available, record "figures unavailable
(no extraction backend)" and continue text-only.

If a code repository is linked, validate https-only first, then clone only to a temp dir with:
`git -c protocol.ext.allow=never -c protocol.file.allow=never -c protocol.git.allow=never clone --depth=1 -- <url>`.
Use static inspection only - never execute `make`, `pip`, `npm`, `python`, tests, hooks,
binaries, package scripts, or cloned code. Skip clone over 500 MB as "too large". secret-scan
before quoting. Set ledger `no_untrusted_code_executed: true`. There is NO override flag.

Author `domains/<domain>/<Topic>/README.md` from `tools/research/templates/note.README.md` and
create sibling `ledger.json`. Every Academic Context cell and non-trivial claim must have a
ledger entry with kind exactly one of `paper-claim`, `evidence-supported`,
`reasonable-inference`, or `unproven-or-doubtful`. Venue tier is literal `unknown` unless
sourced. Citation count is literal `unavailable` on API 429/no-record, never `0`. A claim with
no source_url cannot be `evidence-supported`.

Write `## 🧪 Critical Assessment`. Interrogate problem realness, baseline/ablation/dataset/
metric sufficiency, novelty-vs-repackaging, self-defined-benchmark / 射箭畫靶, and real-world
relevance. Do not silently endorse or 背書 weak claims.

Append the domain README Papers-table row `| Title | Venue | Year | Code | Review |`. Add only
resolvable `../Related/` cross-links. If there is no related safe/resolvable note, keep the
`## 🔗 Related notes` heading and leave the Related section empty; never invent a link. If the
target folder already exists, SKIP/MERGE after inspecting it; never overwrite.

Write the repo-relative note id `<domain>/<Topic>` to `.loop-manager/research-note-note-path`
from the workspace container so the pipeline command can run:
`make -C road-to-master research-check NOTE=<domain>/<Topic>`.
Exception: a BLOCKED run writes `.loop-manager/research-note-blocked.json` instead and must
NOT write the note-path file.
