# Token-Saving Tools & Papers for Coding Agents — Mechanisms, Costs, User Evidence

> **English** | [繁體中文](./README.zh-TW.md)

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | Projects and papers claiming to save 20\~40% tokens for coding agents like Claude Code / Codex: mechanisms, costs, and user evidence |
| Venue Kind | survey (a market audit across 11 representative projects + 30-odd peripheral tools/papers) |
| Year | 2026 |
| Survey date | 2026-07-16 (all web sources retrieved on the same date unless noted) |
| Method | read-only audit: citations only from pages actually opened and read (official blogs / arXiv full text / GitHub API / HN Algolia / Reddit archive API / static clone); every load-bearing claim ran at least one refutation search; no downloaded code was executed |

> This note is not a single-paper summary but a market-audit survey. The audit target is the *claim itself* that a tool "saves 20\~40% tokens for coding agents." **Every vendor number is treated first as an unverified seed**; whenever it conflicts with a primary source, the primary source wins. Each claim is interrogated with "what is being measured, under what workload, and against which baseline?"

---

## One-Sentence Conclusion

> **The "20\~40% saving" band is itself credible — but it is the residual left after each vendor's 60\~95% marketing number is discounted by independent measurement, not any vendor's original claim.** Risk varies enormously across mechanisms: output-side trimming and lazy loading are nearly lossless; token-level pruning (LLMLingua style) is a disaster for coding; retrieval-instead-of-whole-file is the right direction but implementation quality often eats the theoretical gains.

One comparison says it best: the only **neutral, same-task, same-machine** cross-test (ComputingForGeeks, baseline 284,473 tokens) measured a batch of tools that self-claim 60\~95% savings at **0\~43%**:

| Tool | Self-claimed | Neutral cross-test |
|-|-|-|
| token-savior | −80% | **43%** |
| caveman | 65% | **38% / 37%** |
| ooples token-optimizer-mcp | 95%+ | **23%** |
| RTK | 60\~90% | **0%** (no saving at all when output is clean) |
| code-review-graph | — | **5%** (on small repos it is 0.7x — actually more expensive) |

Users' impression of "20\~40%" lines up with independent measurement — not with the vendor headlines (60\~95%).

---

## The Three Investigation Questions

The three questions this audit answers:

- **Q1 — Mechanism**: through what mechanism does each project reduce tokens? Every mechanism must come with a **concrete example** (an actual file/function in the repo, or an actual algorithm/table in the paper), not just a restatement of marketing language.
- **Q2 — Cost**: beyond token reduction, do these mechanisms affect the LLM's **capability and correctness**? What evaluation evidence does each project provide itself? Is there any independent third-party evaluation? Where is the evidence strong, and where are the gaps?
- **Q3 — User evidence**: do user reports, GitHub issues, and community discussion of problems/bugs point at the Q2 suspicion (dropping key context, editing the wrong file, answering the wrong question, the agent getting lost)?

### Quick-Reference Conclusions

| Question | Conclusion |
|-|-|
| **Q1 Mechanism** | Five mechanism classes: ① token-level prompt compression ② semantic/symbol retrieval replacing whole-file reads ③ context middleware / long-term memory ④ output-side diff-edit ⑤ model routing / caching proxy. See "Mechanism Taxonomy". |
| **Q2 Cost** | Huge variation, and tierable (see risk table below). Key finding: token-level perplexity (a measure of how "surprised" the model is by a span of text) pruning breaks coding identifiers/values/syntax (post-compression AST (Abstract Syntax Tree) validity is only **0.29%**); whereas **line-level task-aware pruning** (SWE-Pruner) achieves −23\~38% tokens on the full SWE-bench Verified while success rate **rises** by 1.2\~1.4 percentage points (pts). |
| **Q3 User evidence** | Heavily corroborates the Q2 suspicion: Serena "hits context limits faster than without it" + silently corrupts code; claude-context index desync "never able to search"; mem0 stores mutually contradictory memories; Claude Code's own /compact is the single largest source of "amnesia after compression" complaints. |

---

## Core: Tiered by Risk (Q2: the cost to capability and correctness)

Sorting every mechanism by "does saving tokens hurt capability" is this survey's most useful output, and it is the overall answer to Q2:

| Risk | Mechanism | Why |
|-|-|-|
| 🟢 **Lowest** | diff / targeted edits (output-side), noisy tool-output compression (RTK style), tool-definition lazy loading, prompt caching | Near-lossless by mechanism; Anthropic's own token-efficient tool use (avg 14%) is the only case that is "deployed at scale with zero regression reports". |
| 🟡 **Medium** | semantic/symbol retrieval replacing whole-file feeding (Serena, claude-context); **line-level task-aware context pruning (SWE-Pruner)** | Independent positive evidence for the direction (Cursor A/B (A/B test): +12.5% accuracy), but the implementation layer (index desync, MCP (Model Context Protocol) fixed overhead, symbol-edit corruption) often eats the gains; SWE-Pruner has the strongest evidence of all, but single-team, zero replication, Python-only. |
| 🔴 **High** | **token-level / perplexity pruning (LLMLingua style) for coding**; classifier-based model routing for agentic coding | Multiple independent sources consistently show corruption of identifiers/values/AST; routing collapses on OOD (out-of-distribution, i.e. test scenarios differ from training data) coding + protocol incompatibility scraps entire tool-use turns. |
| ⚫ **Cannot assess** | zero-methodology behavior-injection frameworks (SuperClaude style), zero-methodology "95%+" MCPs | No benchmark, no methodology; there is even reverse evidence (the framework itself eats 40k+ tokens of context). |

> Note: the 🔴 verdict is **limited to token-level pruning**, and does not extend to line-level task-aware pruning — they are different mechanism families; see "Mechanism Taxonomy ①" and "Deep Dive: SWE-Pruner".

### WRAPUP: Per-Tool Master Table

The master table at a glance. **Mechanism**: A=token-level compression, B=retrieval-instead-of-whole-file, C=context middleware/memory, D=output-side diff, E=routing/caching. **Evidence strength** (same color scheme as capability risk, 🟢=good, 🔴=bad): 🟢 has a public benchmark that is (in principle) reproducible; 🟡 only vendor self-test or too small a sample; 🔴 no methodology or explicitly refuses evaluation (incl. LLM-self-grading only). **Capability risk** follows the tiers above: 🟢 lowest, 🟡 medium, 🔴 high, ⚫ cannot assess (zero methodology — can't even judge whether it hurts capability).

| Project | Mechanism | Claimed saving (measurement basis) | Evidence strength | Capability risk | Negative user signal |
|-|-|-|-|-|-|
| SWE-Pruner | A (line-level) | 23\~54% token (agent total tokens, success rate rises) | 🟢 single-team, zero replication | 🟡 | none found (but small deployment base) |
| Aider (repo-map + diff) | B+D | claims no % (repo-map is an overhead) | 🟢 most honest of all | 🟢 diff (strong models) / 🟡 repo-map | yes: #752 budget overshoot 16x, SEARCH-REPLACE failure cluster |
| claude-context | B | \~40% (30-question localization) | 🟡 n=30, weak model | 🟡 | yes: index desync #145/#226/#232 |
| LLMLingua family | A (token-level) | up to 20x (input tokens, CoT tasks) | 🟡 official narrow / punctured on coding by independent evidence | 🔴 (coding) | yes: #89 error rate +18pts, #136 collapses to 0.02 |
| mem0 | C | >90% (vs replaying full history) | 🟡 highly disputed | 🟡 | yes: #5867 contradictory memories |
| Anthropic official features | C/D/E | 84% / 14\~70% / 85% / 98.7% (internal) | 🟡 internal-only, mechanism transparent | 🟢 | few; only /compact draws many "amnesia" complaints |
| RouteLLM | E | 85% cost saving + retains 95% (no coding) | 🟡 in-domain / collapses OOD | 🔴 (agentic) | strong academic refutation, no notable repo complaints |
| Serena | B | no number (community rumor of 70%, no source found) | 🔴 explicitly refuses testing, LLM self-grades | 🟡 | yes: Reddit "actually uses more", #1529 silent code corruption |
| SuperClaude | C | 70%→30-50% (zero methodology) | 🔴 | ⚫ | yes: #286 framework itself eats 43.8k context |
| claude-code-router | E | no % claim of its own | 🔴 no evaluation | 🔴 protocol incompatibility | yes: #1378 tool use fully broken |
| token-savior | B/C/D | −80% self-claimed → neutral 43% | 🔴 self-made tsbench | 🟡 no rigorous capability test | — |
| caveman | D (forces telegraphic output) | 65% self-claimed → neutral 38% | 🔴 self-test | 🟢 neutral test "same answers"; but terse workloads may net-increase | — |
| ooples token-optimizer-mcp | B/D | 95%+ self-claimed → neutral 23% | 🔴 zero methodology | 🟡 untested | — |
| RTK | D (compresses noisy output) | 60\~90% self-claimed → neutral 0% | 🔴 self-test | 🟢 only compresses noise, low risk | — |
| code-review-graph | B (graph retrieval) | neutral 5% (small repos 0.7x — more expensive) | 🔴 | 🟡 untested | — |
| Others (claude-mem, Headroom, etc.) | C/D | 10x, \~50% (self-claimed) | 🔴 self-test | 🟡 | yes: claude-mem #618 token bloat |

---

## Mechanism Taxonomy (Q1: mechanism + concrete example)

### ① Token-level prompt compression — the LLMLingua family

- **Representative**: LLMLingua / LongLLMLingua / LLMLingua-2 (Microsoft, arXiv 2310.05736 / 2310.06839 / 2403.12968)
- **Mechanism**: a small language model computes each token's perplexity and deletes "low-information" tokens outright.
- **Concrete example**: the official README compresses a 2,365-token GSM8K CoT (chain-of-thought, a prompt asking the model to reason step by step) prompt down to 211 tokens (11.2x). But the official sample output is already corrupted: *"He reanged five of boxes into packages of sixlters each..."* — the marketing page itself displays the mechanism's cost.
- **The official pipeline even bundles a "repair" step**: LongLLMLingua has a subsequence recovery step that restores corrupted entities (e.g. "209" back to "2009") after the fact — effectively an official admission that compression breaks the surface form.
- **Claim basis**: "up to 20x compression with minimal performance loss" measures **input prompt tokens**, with a workload of GSM8K/BBH CoT + GPT-3.5-0613.

### ② Semantic / symbol retrieval replacing whole-file reads

Replace "stuffing the whole file into context" with precise retrieval.

- **Serena MCP** (26.5k★): symbol-level retrieval via LSP (Language Server Protocol, the standard interface an IDE uses to get symbols/definitions/references); tools use a symbol name path (`MyClass/my_method`) instead of reading the whole file.
  - Code: `FindSymbolTool` (`src/serena/tools/symbol_tools.py:132`, whose `include_body` param is annotated "Use judiciously"), `GetSymbolsOverviewTool` (`:36`), `ReplaceSymbolBodyTool` (`:571`).
  - **The real example from its own eval**: `get_symbols_overview` returns \~2.5KB JSON per call vs Grep's \~3KB; the official phrasing for the saving is *"saves \~1 call and some context window tokens per navigation"* — far smaller than the community-rumored "70% saving" (that 70% has no traceable source).
- **claude-context** (Zilliz, 12.1k★): AST-aware chunking → embedding → Milvus vector DB → hybrid retrieval.
  - **The only "in principle reproducible" vendor measurement** (repo `evaluation/README.md`): 30 SWE-bench Verified retrieval subtasks, GPT-4o-mini, tokens 73,373 → 44,449 (**−39.4%**), F1 (harmonic mean of precision and recall; higher = better at finding the right file) 0.40 vs 0.40 unchanged. This is the entire experimental basis for the marketing line "Cut Token Waste by 40%".
  - Weaknesses: n=30, restricted to 2-file changes, uses the weak model that benefits most from retrieval, tests only file localization (F1=0.40 means both groups have a >50% chance of picking the wrong file).
- **Aider repo-map** (the most honest of all): tree-sitter extracts symbols + PageRank ranking, **binary-searched into a fixed token budget** (`aider/repomap.py:47`, default `map_tokens=1024`).
  - **The key difference (an absence-of-claim)**: Aider **never claims repo-map saves X%** — the repo map is an **overhead** that "buys" capability with a fixed budget, not a saving relative to whole-file feeding. This is the exact opposite of the MCP retrieval tools' marketing frame.

### ③ Output-side diff-edit (most directly tied to "output tokens")

- **Mechanism**: diff / SEARCH-REPLACE returns only the changed hunk instead of rewriting the whole file (whole-file = "slow and costly because the LLM has to return the entire file").
- **Representatives**: Aider's `diff`/`udiff` (unified diff) formats, Claude Code's built-in `old_string`/`new_string` Edit tool, OpenAI's `apply_patch` V4A.
- **Concrete positive example**: Aider's unified diff lifted GPT-4 Turbo's lazy-coding benchmark from 20% to 61% (output compression **sometimes actually improves capability**).
- **Cost**: format fragility — same source: "disabling flexible patching yields a 9x increase in editing errors"; weak models actually regress in accuracy when using diff (see "User Evidence", Aider).

### ④ Context middleware / long-term memory

- **mem0** (arXiv 2504.19413): extracts salient facts across sessions into a vector DB, replacing replay of the full dialogue history. Claims "saves more than 90% token cost".
  - **Measurement trap**: the >90% is vs the baseline of "replaying 16k\~26k tokens of full history every time" — token saving is guaranteed by the mechanism; the entire dispute is about accuracy (see "User Evidence").
- **SuperClaude** (23.6k★): a "behavior framework" that injects markdown instruction files into context. The v1 README claimed "70% reduction"; the current version walks it back to "30-50% fewer tokens" — **both generations have zero methodology**.
- **Anthropic official**: context editing + memory tool (claims 84% @ 100-turn web search), Tool Search Tool (85% saving on input-side tool definitions). All numbers are internal and unreplicated, but the mechanism is transparent.

### ⑤ Model routing / caching proxy

- **RouteLLM** (arXiv 2406.18665): trains a strong/weak model router, sending simple tasks to the cheaper model. Claims "85% cost saving + retains 95% of GPT-4 capability", but **only tested on MT Bench/MMLU/GSM8K, no coding**.
- **claude-code-router** (35.8k★): a local proxy that routes Claude Code's various traffic to cheaper models. Makes no % saving claim of its own.
- **prompt caching**: saves **cost, not tokens**, and cache writes carry a 1.25×/2× premium — at low hit rates it is actually more expensive. **It also largely conflicts with most dynamic token-saving methods (see "Should You Use It").**

---

## Deep Dive: SWE-Pruner — the strongest "saves tokens without dropping success rate" evidence in this survey

(arXiv 2601.16746 v4, github.com/Ayanami1314/swe-pruner, 299★)

This one deserves its own section because it is **simultaneously the strongest counterexample to LLMLingua-style pruning and the strongest positive case for line-level pruning**:

- **Two key differences from LLMLingua's mechanism**:
  - **task-aware**: the agent first gives an explicit goal (e.g. "focus on error handling") to guide pruning, instead of a fixed perplexity metric.
  - **line-level, not token-level**: a 0.6B skimmer keeps or deletes whole lines, so it **does not produce LLMLingua-style surface corruption**.
- **Positive numbers (full 500-question SWE-bench Verified)**:
  - Claude Sonnet 4.5: success rate 70.6% → **72.0% (+1.4 pts)**, tokens −23.1%
  - GLM-4.6: 55.4% → 56.6% (+1.2 pts), tokens −38.3%
- **Hard proof of the mechanistic divide (AST validity comparison, paper Table 8)**:

  | Method | Post-compression AST validity |
  |-|-|
  | Full Context (uncompressed) | 98.5% |
  | LLMLingua-2 (token-level) | **0.29%** |
  | LongCodeZip (line-level) | 89.3% |
  | SWE-Pruner (line-level) | 87.3% |

- **Honest boundaries**: at single-turn 8x extreme compression, EM (Exact Match, the fraction of outputs identical word-for-word to the reference) still drops from 40.5 to 31.0 (a substantial regression); Python-only; single-team, zero third-party replication.
- **Rating: very likely a genuinely adoptable method with reference value.** It is the only direction in this survey with full public data behind "saves a lot (−23\~38% tokens) without losing capability (success rate rises)", and its mechanism (line-level task-aware) is sound — worth using as a design reference. The only reason it isn't yet in the 🟢 "adopt with confidence" tier is insufficient external validation (single-team, zero replication, Python-only), not any flaw in the mechanism or the data — so verify it on your own codebase (closure recipe #15) before adopting, rather than either copying it blindly or dismissing it.

---

## User Evidence (Q3: distinguishing engineering bugs from mechanistic regression)

All citations below carry an issue number/link and are traceable. Each is classified as "mechanistic regression" or "engineering bug".

### Corroborating mechanistic regression

- **LLMLingua #89** (open): a user measured on LongBench qasper that the wrong-answer/no-answer rate rose from 45.36% to 63.93% (+18 pts). **#136**: dureader accuracy 0.68 → 0.02. 【mechanistic】
- **Serena**: Reddit r/ClaudeCode "Feels like Serena MCP uses more tokens than without?" — OP "I hit context limits faster with Serena than without", top reply confirms "any MCP will increase token usage". Cursor devs separately note "semantic search over codebases performs worse than letting the model search with bash". 【mechanistic: MCP schema fixed overhead】
- **Aider**: an HN user posted logs showing the repo-map summary "produced multiple factually incorrect descriptions, hallucinating tracking mechanisms that don't exist"; on a 300-file monorepo "the repo map overwhelmed the LLM". 【mechanistic: compressed representation misleads — directly matches the "wrong answer / agent gets lost" suspicion】
- **mem0 #5867** (open): a preference change (Ronaldo→Messi) produces two coexisting contradictory memories, "retrieval becomes ambiguous". An HN user turned off memory because it "keeps stale information and bleeds across projects". 【mechanistic】

### Corroborating engineering bugs (but mostly inherent to that mechanism's risk surface)

- **Serena #1529** (open): `replace_symbol_body` corrupts Go's `type Resampler struct` into `type type Resampler struct`, **and the tool returns `{"result": "OK"}` with no error at all** — silent corruption, directly matching the "corrupts code" suspicion. **#516**: Serena's own `search_for_pattern` returns 32,204 tokens in a single call, blowing past Claude Code's 25k MCP limit (a token-saving tool overshooting).
- **claude-context #145 / #226 / #232**: indexing succeeds but then "codebase not indexed, never able to search"; state-sync bug; any project change requires re-indexing the whole thing.
- **claude-mem #618** (confirmed bug): "Uses too much tokens — claude code consumes all my tokens within 10 messages" (a token-bloat bug in a flagship memory plugin).
- **claude-code-router #1378** (open): DeepSeek + thinking + tool calls hits 400 every turn, "completely unusable as soon as it touches tools". 【mechanistic: protocol incompatibility → entire tool-use turn scrapped, the highest risk for a coding agent】

### Aside: Claude Code /compact is everyone's baseline of comparison, and is itself the largest body of negative evidence

- **#10006**: "every auto-compact, it loses every detail". **#13919**: skills stop working after compaction, task time goes from "\~1 hour" to "5-6+ hours". Engineer's report: "definitely dumber after compaction, doesn't know what files it was just looking at".
- Classification: CLAUDE.md/skills not being re-injected is an **engineering bug** (partly fixed in the current version); the summary dropping mid-task decisions/file lists is **mechanistic lossy compression**.

### Balance: the retrieval direction also has positive independent evidence

- **Cursor's official A/B** (2025-11): same model, semantic search vs grep-only, on average **12.5% higher accuracy** (6.5%\~23.5%); large-codebase code retention +2.6%. — refutes the extreme reading that "retrieval necessarily makes a coding agent dumber".
- But contrast Claude Code's official stance: "Claude Code currently doesn't use RAG (Retrieval-Augmented Generation, retrieve vectors first then feed the snippets to the model); in our testing agentic search outperformed RAG for the way people use Code".

---

## Should You Use It (practical guidance)

- **Want to save with zero risk**: prefer the "near-lossless by mechanism" tools — targeted/diff edits (the format-compliance tax is negligible on Claude 4+ class models), noisy tool-output compression (only helps when output is dirty), tool-definition lazy loading, prompt caching (watch for cache-miss backfire and the write premium). Anthropic's built-in token-efficient tool use requires no adoption action — it's already built in.
- **Large codebase (>20k LoC (lines of code) class) and willing to pay index maintenance cost**: only then are semantic retrieval tools worth it; expect the theoretical gains to be partly eaten by index desync and MCP fixed overhead.
- **The most promising new direction to track**: SWE-Pruner-style **line-level task-aware pruning** is very likely usable and worth referencing — the only approach with full public data behind "saves a lot without losing capability". Before adopting, self-host the skimmer and measure on your own codebase with `ccusage` (closure recipe #15), but don't dismiss it just because it's a preprint from a single team.
- **⚠️ prompt caching largely conflicts with dynamic token-saving methods — don't naively stack them**: prompt caching only earns the 0.1x cache-read discount when the prompt prefix is byte-stable and only appended to at the tail; but most dynamic context-shrinking methods (LLMLingua compression, SWE-Pruner pruning, Anthropic context editing clearing old tool calls, memory/summary rewriting) **mutate the prefix every turn**, and any mutation invalidates the cache from that point on → back to full-price misses + the 1.25×/2× write premium, so stacking the two can cost more than either alone. The two strategies pull in opposite directions: caching wants a stable, append-only context; aggressive pruning wants to mutate-and-shrink. Pick one by cost structure — if cost is dominated by a repeated stable long prefix, choose caching; if by unbounded growing context, choose pruning.
- **Never**: use token-level perplexity pruning (LLMLingua style) for coding; extrapolate 2024-era classifier routing's 85%/95% claims to agentic coding.
- **General rule**: percentages with an unknown denominator are not comparable. Before citing any "saves X%", ask **what is measured (input/output/cost/a single subtask), under what workload, against which baseline**.

---

## 🧪 Critical Assessment

### The boundaries of this audit itself

- **Not a single number was measured on Claude Code**: all vendor numbers were measured on their own harness, model, and workload; extrapolating to Claude Code is inference. Closing the loop properly requires measuring a fixed task set under `claude -p` headless with `ccusage` (see the closure recipes in the original dossier).
- **Reddit retrieval was limited**: this environment gets a blanket 403 from Reddit; most Reddit evidence went through the arctic-shift archive API or is marked secondary.
- **SWE-Pruner's positive evidence is strong, but single-team, zero independent replication, Python-only** — it is "most worth verifying", not "already proven". Its paper Table 3 baseline (62.0%) is inconsistent with Table 1 (70.6%) and does not state the sample size — a discrepancy the paper itself never explains.

### claim-laundering is widespread

Many "token-saving" claims quietly swap their measurement basis when relayed through secondary blogs — e.g. LEANN's "97% less storage" gets mis-relayed as "saves 97% tokens" (storage ≠ tokens); code-context-engine's "94% fewer input tokens" uses a "read the entire file" straw-man baseline. Always go back to the primary page to confirm what dimension is being measured.

### Evidence strength is extremely uneven

It ranges from "public benchmark + AST checking" (SWE-Pruner, Aider) to "officially refuses benchmarks, uses LLM self-grading" (Serena) to "zero-methodology marketing numbers" (SuperClaude, most MCPs). The "Evidence strength" column (🟢/🟡/🔴) in the "WRAPUP Per-Tool Master Table" annotates each one; readers should not mistake "has a star count" for "has evidence".

## 🔗 Related notes

- [Vector Database Comparison](../VectorDatabaseComparison/) — the vector-DB selection behind semantic-retrieval tools (claude-context etc.), also a market-audit survey.
- [Shepherd](../Shepherd/) — programmable control of agentic execution traces, relevant to the design trade-offs of context management / memory middleware.

---
*This note is a read-only market-audit output; all citations were retrieved on 2026-07-16 and, unless noted, come from pages actually opened. Full per-item evidence, 11 deep-dive entries, and 15 closure recipes for open questions are in the original dossier.*
