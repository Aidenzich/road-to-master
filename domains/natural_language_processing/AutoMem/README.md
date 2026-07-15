# AutoMem — Research Note
> **English** | [繁體中文](./README.zh-TW.md)

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | AutoMem: Automated Learning of Memory as a Cognitive Skill |
| Venue | arXiv preprint (2607.01224v1) |
| Year | 2026 |
| Authors | Shengguang Wu, Hao Zhu, Yuhui Zhang, Xiaohan Wang, Serena Yeung-Levy (Stanford University) |
| Official Code | https://github.com/autoLearnMem/AutoMem |
| Venue Kind | paper |

> This note is written from the arXiv preprint `2607.01224v1` (LaTeX source). The paper uses the `neurips_2026` preprint template, but as of the time of writing there is no verifiable formal acceptance record, so the venue tier is recorded as `unknown`; the numbers and narrative in a formally published version may differ from this preprint.

## Introduction

A large language model's context window is its working memory: a fixed-size buffer that can hold only a limited amount of information at once. Long-horizon tasks routinely run tens of thousands of steps, far exceeding this capacity. A common approach in the past has been to bolt external memory on as an "architectural module" — RAG's retrieval database, MemGPT's paging mechanism, Generative Agents' memory stream — all of which fix in advance "how memory operates."

AutoMem takes a different view: **memory management is itself a trainable skill**, not a fixed mechanism. It borrows the cognitive-science notion of metamemory — humans learn "what is worth remembering, when to recall, how to organize what they know" — and hands that capability over to the model to decide for itself. Concretely, it promotes file-system operations (read, write, search, append, create) to "first-class memory actions" in the model's action space, on equal footing with the task actions it uses to operate the environment: in the same forward pass the model can choose either a task action or a memory operation (e.g. `<|APPEND|>` or `<|SEARCH|>`). This way every memory decision is a traceable action in the trajectory.

The paper argues that memory skill improves along two axes: the **structure** that supports it (prompt, file schema, action vocabulary) and the model's **proficiency** in exercising it. Both axes are hard to optimize by hand — a single episode can run $10^4$–$10^5$ steps, a memory mistake may lie dormant for hundreds of steps before its consequence surfaces, and manually reviewing full trajectories line by line is infeasible. AutoMem's core observation is: a sufficiently strong LLM (the meta-LLM) can read an entire episode the way a code reviewer reads a full execution log, pointing out where a memory decision went wrong, and can therefore **automate** the optimization of both axes.

How is success measured? The authors chose three procedurally generated long-horizon games — Crafter, MiniHack, NetHack — all from the BALROG benchmark. The reasons for choosing these games: episodes are long enough that context-window management alone cannot cope; the world is regenerated for every seed, so pretraining memory cannot transfer, forcing the model to genuinely "take notes" (maps, inventory, encounter logs). The main metric is BALROG's **progression rate** (scaled to $[0,100]$), with the base model fixed to `Qwen2.5-32B-Instruct`. Baselines fall into two groups: frontier / open-weight models on the BALROG leaderboard, and the same 32B model with basic sliding-window context management (that sliding-window baseline keeps only the most recent 16 steps of observation, with or without chain-of-thought). The core result: **optimizing memory only, without touching the task-action model weights at all**, raises the 32B base agent's performance by roughly 2–4× — recomputing per environment from v0 to full AutoMem, Crafter, MiniHack, and NetHack are 2.05×, 4.00×, and 4.40× respectively — closing in on the level of frontier systems such as Claude Opus 4.5 and Gemini 3.1 Pro Thinking.

![Overall effectiveness of AutoMem's memory-skill optimization: starting from a base agent equipped with file-system memory (v0), first performing memory scaffold optimization (v0→v5/v4/v2), then stacking memory proficiency training (+train) for additional gains. The blue and orange dots are reference lines for Gemini-3.1-Pro-Thinking and Claude-Opus-4.5 on the BALROG leaderboard.](imgs/teaser.png)

## First Principles

### Inner agent: treating memory as a file system

The inner loop is simply one LLM agent running one episode, paired with a directory on disk that serves as external memory. Each step runs two routines, each corresponding to one facet of memory management:

- **The LOG routine** asks "of what just happened, what is worth recording?" — the agent decides whether to record the environment's response to the last action, and how (append to an existing file, open a new file, or rewrite an entry).
- **The PLAN routine** asks "to act, what do I need to recall?" — the agent searches across files, reads a specific entry or its tail, then commits the next world action.

The key point: memory operations and task actions **share the same action space**, which is what makes memory a "learnable skill" rather than a fixed mechanism. Because every memory decision is an observable action in the trajectory, the outer loop can observe, evaluate, and optimize it. The paper also argues for a side effect: optimizing the memory structure **incidentally improves task behavior** — better-organized memory reduces redundant exploration and aimless actions, even though the optimizer targets the memory scaffold rather than the game strategy itself.

![AutoMem overview: two automated outer loops optimize a shared inner agent (gray-shaded center). Outer-loop #1 (top): the meta-LLM reads the full episode trajectory and iteratively rewrites the agent scaffold. Outer-loop #2 (bottom): the meta-LLM acts as a training engine, jointly orchestrating data filtering and finetuning to train a memory specialist dedicated to memory operations; the task model (frozen) commits task actions. The center panel shows files such as game_rules, dungeon_map, inventory, and strategy in the memory directory.](imgs/method.png)

### Two outer loops: one shared machine-learning structure

The paper unifies the two loops into one familiar machine-learning form: each loop has parameters $\theta$ to optimize and an update signal $\nabla L$ derived from the meta-LLM's trajectory analysis. Using the paper's own analogy, this can be written as (notation follows the paper; the subscripts are added by this note to distinguish the two loops):

$$\theta_{\text{struct}} \leftarrow \theta_{\text{struct}} + \nabla L_{\text{struct}}, \qquad \theta_{\text{prof}} \leftarrow \theta_{\text{prof}} + \nabla L_{\text{prof}}$$

The difference lies in "what the parameters are and what the update signal looks like":

- **Outer-loop #1 (structure)**: $\theta_{\text{struct}}$ is the agent scaffold (code, prompt, memory-file schema, action vocabulary), and $\nabla L_{\text{struct}}$ is a single **code rewrite**. The meta-LLM (Claude Opus 4.6, `--effort max`) receives the full episode trajectory (the step-by-step log, the generated memory directory, and the agent code itself), and like a code reviewer diagnoses failure patterns and rewrites the scaffold. Every rewrite is **gated on measured improvement**: the rewritten agent runs on the same fixed set of eval seeds, and is kept only if its average progression strictly exceeds the previous version's; a failure gets 1 retry, and failing again restarts from a clean session. In practice it converges in about 2–5 iterations (Crafter to `v5`, MiniHack to `v4`, NetHack to `v2`).

- **Outer-loop #2 (proficiency)**: $\theta_{\text{prof}}$ is the weights of a dedicated memory model, and $\nabla L_{\text{prof}}$ is a "properly configured supervised training step." Here the meta-LLM (Claude Opus 4.7, `--effort max`) acts as the **training engine**, refining "data-filtering logic, data composition, and LoRA training configuration" as a **joint decision** across iterations. A key constraint: every piece of training data is verbatim text the inner model itself produced during episodes; the meta-LLM acts only as a **filter** picking which responses are worth reinforcing, rather than as a teacher generating new answers.

Because memory is treated as a separable skill, only a dedicated **memory specialist** (a LoRA adapter) is finetuned, responsible for the parts of the LOG routine and PLAN routine that consult memory; the **gameplay model** (the untouched base) commits world actions. At inference, the two model instances share a single conversation history, and after the specialist performs its last memory operation it briefly hands off to the gameplay model. There are two benefits: the training signal concentrates on memory operations, undiluted by action-format examples; and the base model's ability to produce well-formatted world actions is fully preserved, since its weights are never touched by finetuning.

### Worked example: NetHack's map deduplication (how the scaffold evolves)

The most illustrative concrete example of "delayed consequence" is NetHack's map file. The base schema (v0) uses an append-only `dungeon_map.txt` that grows without bound: every time the agent revisits the same cell $(x,y)$ it logs another entry, quickly accumulating thousands of duplicate coordinates that bury useful information. After reading the full NetHack trajectory, the meta-LLM diagnoses this and introduces a `<|UPSERT_MAP|>` operation, switching to coordinate-key deduplication — any new observation of tile $(x,y)$ **overwrites** the old entry rather than stacking alongside it. The evolved schema further adds `inventory` and `status` files auto-synced from observations (sparing the model manual READ reconciliation), plus a prefilled `strategy` reference (hard-coding the primary goal "find the stairs to descend" so the model needn't waste operations early in the episode rediscovering the goal).

![NetHack memory-file evolution: v0's dungeon_map is unbounded, append-only, and often accumulates duplicate coordinates (left); after scaffold optimization (right) it switches to <|UPSERT_MAP|> coordinate-key deduplication, and adds auto-synced current_status and inventory as new files plus a prefilled strategy reference.](imgs/mem_filesystem_evol.png)

The effect is quantitative: these changes shrink the memory the agent must carry per step from **138 characters to 6 characters, a 95% reduction**. This is exactly the point of external memory — compressing the information the model must attend to. Putting this worked example back into the main table: NetHack's progression rises from v0's 0.42% to the scaffold-converged 1.57% (×3.74), then to 1.85% after training.

### How the two axes stack numerically

The table below gives the main results across the three environments (progression rate %, mean $\pm$ standard error). Evaluation uses a fixed set of 10 seeds `[42..51]`, but the number of eval episodes differs per environment (following BALROG defaults): Crafter 10 episodes, MiniHack 40 episodes (8 tasks × 5 each), NetHack 5 episodes, so the sample sizes across the three environments are not symmetric. Frontier numbers are taken from the BALROG leaderboard; the `Qwen2.5-32B-Instruct` rows all use the same BALROG harness.

| Agent | Crafter (%) | MiniHack (%) | NetHack (%) |
|-|-|-|-|
| *Frontier proprietary (BALROG)* | | | |
| Gemini-3-Pro | 57.3 ± 4.4 | 40.0 ± 7.7 | 6.8 ± 3.2 |
| Gemini-3.1-Pro-Thinking | 55.0 ± 6.4 | 27.5 ± 7.1 | 2.6 ± 0.3 |
| Claude-Opus-4.5 | 49.5 ± 3.1 | 27.5 ± 7.1 | 2.0 ± 0.5 |
| *Open-weight (BALROG)* | | | |
| DeepSeek-R1 (671B) | 36.4 ± 3.8 | 25.0 ± 6.8 | 1.4 ± 0.5 |
| Qwen2.5-72B-Instruct | 27.3 ± 3.6 | 5.0 ± 3.4 | 0.3 ± 0.3 |
| Qwen2.5-7B-Instruct | 16.4 ± 3.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| *32B + basic context management* | | | |
| sliding window | 19.55 ± 3.46 | 2.50 ± 2.47 | 0.00 ± 0.00 |
| + chain-of-thought | 17.27 ± 2.71 | 10.00 ± 4.74 | 0.00 ± 0.00 |
| *32B + AutoMem (ours)* | | | |
| memory-as-file-system, v0 | 25.00 ± 5.50 | 7.50 ± 4.16 | 0.42 ± 0.37 |
| + scaffold opt. (loop #1) | 47.27 ± 2.05 | 27.50 ± 7.06 | 1.57 ± 0.35 |
| + memory training (loop #2) | **51.36 ± 3.81** | **30.00 ± 7.25** | **1.85 ± 0.44** |

Scaffold optimization alone (with weights untouched throughout) roughly doubles to triples progression: Crafter 25.0→47.27 (×1.89), MiniHack 7.5→27.5 (×3.67), NetHack 0.42→1.57 (×3.74). Training the memory specialist then stacks a complementary gain: Crafter +4.09, MiniHack +2.5, NetHack +0.28, on the order of one to two scaffold iterations.

What exactly did scaffold optimization change in behavior? The paper measures four indicators (v0 vs converged version, all lower-is-better):

![The impact of scaffold optimization on game and memory behavior. Left: unproductive game action rate (the fraction of steps stuck or oscillating back and forth) drops 32–65% across the three environments (Crafter 28→18, MiniHack 60→21, NetHack 60→41). Right three panels: redundant WRITE down 68–83%, empty SEARCH (queries returning no results) down 13–50%, and input context tokens per step down 3–30%.](imgs/scaffold_analysis_bars.png)

Now look at the internalization of behavior brought by training: on the evolved scaffold, comparing the base model with the trained specialist on "how many writes correspond to each SEARCH in the LOG phase" (lower means searching before writing) — Crafter 0.84→0.39 (−54%), MiniHack 2.89→0.82 (−72%), NetHack 4.66→1.31 (−72%). That is, training internalized the "consult-before-write" discipline that the scaffold had previously only encouraged via prompt, into the weights of the memory model.

### Qualitative case analysis: behavior evolution across the three environments

The aggregate table compresses the gains of the two axes into a few average numbers, but does not show how "behavior" changes. The paper picks one representative eval episode for each environment and places the three optimization stages side by side (base v0 → evolved scaffold → retrained specialist), making memory's impact on behavior directly visible. Note that the progression labeled in each panel here is the value of a **single representative episode**, which is a different thing from the earlier main table's cross-episode aggregate mean (Crafter 10, MiniHack 40, NetHack 5 episodes) — in particular NetHack's 1.85% inside its panel is the single-episode value of that scaffold's run, which happens to collide with the loop #2 average number in the main table; do not conflate the two.

In Crafter, v0 only repeatedly collects wood and gets stuck in a survival loop (progression 9%); the evolved scaffold organizes the map and inventory well enough to craft stone tools, build a furnace, and mine iron ore (55%, 12/22 achievements); the retrained specialist, while maintaining the same crafting level, also proactively eats to sustain survival (59%, 13/22).

![Representative Crafter episodes at the three optimization stages. Left: the v0 base only repeatedly collects wood, no tools (progression 9%); center: the evolved scaffold organizes map and inventory, crafts stone tools, builds a furnace, mines iron ore (55%, 12/22 achievements); right: the retrained specialist balances eating for survival at the same crafting level (59%, 13/22).](imgs/qual_crafter.png)

MiniHack's Corridor-R3 is a navigation task requiring traversal of branching corridors to reach the goal staircase. v0 circles inside the starting room unable to find the exit; the evolved scaffold explores farther but still exhausts the step limit without reaching the staircase, both with progression 0%; the retrained specialist traverses the branching corridors, reaches the goal staircase, and solves the task (100%). This is the most vivid panel of "scaffold changes alone are not enough — proficiency training brings a qualitative change": under the same memory structure, this run's outcome flips from 0% to 100%. The paper does not decompose this single episode's success into a single cause, but at the aggregate level, the trained specialist's "writes per SEARCH in the LOG phase" on MiniHack drops from 2.89 to 0.82 (−72%, see the previous section), exhibiting a cross-episode "search-before-write" tendency, which can be viewed as one clue behind this kind of behavioral improvement.

![Representative MiniHack (Corridor-R3) episodes at the three optimization stages. Left: v0 circles the starting room, cannot find the exit (progression 0%); center: the evolved scaffold explores farther but still cannot reach the staircase (0%); right: the retrained specialist traverses the branching corridors, reaches the goal staircase, and solves the task (100%).](imgs/qual_minihack.png)

NetHack best embodies "delayed consequence." v0 uses an append-only map that keeps re-logging the same tile, dying within a few hundred steps with the character stuck at experience level 1 (progression 0%); the evolved coordinate-key-deduplicated map (see the earlier memory-file evolution figure) lets the agent survive thousands of steps and rise to level 2 (1.85%); the retrained specialist, in this run, consults the map before writing rather than blindly appending, also surviving longer in the same run and climbing to level 4 (2.42%). It must be emphasized: this is a **single representative episode** the paper picked, and the caption of Figure 6 only records that the "search-before-write" behavior and the "survives longer, rises to level 4" outcome co-occurred in the same run; it performs no ablation or causal decomposition on this run, so one cannot conclude that proactive retrieval alone extended survival. What the character-level ladder 1→2→4 provides is a directly countable observation that "memory deduplication / proactive retrieval" and "survival time" co-occur at the single-run scale, not causal proof of the two.

![Representative NetHack episodes at the three optimization stages (single-run values, not the main-table average). Left: v0 uses an append-only map re-logging the same tile, dying within a few hundred steps at experience level 1 (Xp:1/0, progression 0%); center: the evolved coordinate-key-deduplicated map survives thousands of steps and rises to level 2 (Xp:2/37, 1.85%); right: the retrained specialist consults the map before writing, survives longer, and climbs to level 4 (Xp:4/83, 2.42%).](imgs/qual_nle.png)

## 🧪 Critical Assessment

### Is the problem real and important

The memory bottleneck of long-horizon agents is a real problem: NetHack's $10^4$–$10^5$-step episodes indeed far exceed any context window, and frontier models achieve only single-digit progression on NetHack (Gemini-3-Pro 6.8%, Claude-Opus-4.5 2.0%), showing this is an unsolved direction worth investing in. The framing "treat memory as a trainable skill rather than a fixed module" is also not a re-skin: it turns memory decisions into observable, optimizable actions in the action space, and this observability argument is a substantive design choice rather than marketing language.

### Are baselines, ablations, data, and metrics sufficient

This is where the paper most needs careful interpretation. **The frontier comparison is not a level playing field**: the Gemini/Claude numbers are taken from the BALROG leaderboard's "vanilla" setting, without AutoMem's scaffold or two-model deployment applied, whereas the 32B model enjoys a per-environment tailored scaffold + dedicated specialist. So "32B catches up to frontier" is strictly "a 32B with scaffold vs a frontier without scaffold" — if frontier models were also given the same scaffold, the ceiling is unknown. The paper writing this as "closing the gap" is still cautious, but readers can easily over-read it as model-capability equivalence.

**Small samples, large variance**: every number is evaluated on the fixed 10 eval seeds `[42..51]`, but the actual number of eval episodes varies by environment — Crafter 10, MiniHack 40 (8 tasks × 5 each), NetHack 5 — so the sample sizes across the three environments are not symmetric. NetHack's values fall in 0.42–1.85 while the standard error reaches ±0.35–0.44, and the +0.28 "gain" between 1.57 and 1.85 is nearly indistinguishable within the error band — calling it the "complementary gain" of loop #2 is on very weak evidence for NetHack (Crafter's +4.09 is relatively solid).

**NetHack's absolute level is near the floor**: the so-called "catching up to frontier" on NetHack is 1.85% vs 2.0–2.6%, with both sides in a near-failure regime; the claim of "reaching frontier level" here is thin in both statistical and practical terms, and what is genuinely convincing is Crafter (51.36 vs 49.5) and MiniHack (30.0 vs 27.5).

### The overlap risk of self-defined convergence points and eval seeds

A structural concern: **the acceptance gate for scaffold optimization uses the same set of eval seeds `[42..51]`** — "the rewritten agent runs on the same fixed set of eval seeds, and is kept only if average progression rises." And the finally reported table is also evaluated on these 10 seeds. This means loop #1 **selects directly against the test seeds**, and the converged versions (v5/v4/v2) are effectively the best picked out on these seeds, leaving room for overfitting to the evaluation set; each environment also picks a different convergence iteration (v5/v4/v2), which looks more like post-hoc point selection. By contrast, loop #2's training seeds are explicitly disjoint from the eval seeds (Crafter 100, MiniHack 400, NetHack 50 episodes, with random and disjoint seeds), making this half of the experimental design far cleaner. The authors do not report scaffold performance on a held-out set of seeds, which is the comparison I think most needs to be added.

### Can scaffold optimization leak task knowledge rather than improve memory

The scaffold loop can rewrite code, prompts, file schemas, and the memory action surface after a frontier meta-LLM reads full trajectories; its improvement gate constrains only progression, not *how* the gain was obtained. Therefore it can in principle hard-code task-specific strategy hints, preferred search queries, answer-like strings, or synonym lists into the harness, making an apparent memory gain partly a task-policy gain. The paper's prefilled NetHack `strategy` reference ("find the stairs to descend") is a benign, explicit example of task knowledge entering the scaffold. Procedurally regenerated worlds make literal static-map memorization less plausible, but do not rule out exploiting recurring task structure or the fixed evaluation seeds. A stronger evaluation would audit every scaffold diff for injected task knowledge, forbid answer-bearing prompt/schema additions, optimize only on development seeds, and report a final held-out-seed result.

### The missing mature retrieval baseline

AutoMem compares its file-system-memory v0 against sliding-window context baselines, not against a mature retrieval stack such as chunked episodic memory with BM25, hybrid BM25/vector retrieval, query rewriting/reranking, or a deterministic structured state store. Thus the results show that *this optimized file-system harness* beats its own v0 and sliding windows; they do not establish that it beats a strong RAG/BM25 memory system. The decisive ablation would hold a mature retrieval harness fixed and separately test (a) the AutoMem-style memory specialist and (b) scaffold optimization, with all variants evaluated on held-out seeds. That design would reveal whether the gain comes from learned memory control, from a better retrieval substrate, or from task-specific harness engineering.

### Is it the credit of "memory," or of the frontier meta-LLM

The narrative "a 32B can catch up to frontier" has a cost that is easy to overlook: the meta-LLM driving both loops is Claude Opus 4.6 / 4.7 with `--effort max`. What actually does the diagnosing, code rewriting, data filtering, and LoRA tuning is the frontier model; the 32B is merely the object being optimized. So a more accurate version of the conclusion is "using a frontier meta-LLM for offline optimization, one can push a single 32B's long-horizon memory behavior up to frontier inference level" — this does have accessibility value at deployment time (what runs is the 32B), but reading it as "a small model autonomously catches up to a large model" misunderstands the source of the leverage.

### Novelty vs repackaging of prior work, and real-world relevance

The positioning is largely honest: in related work the paper clearly distinguishes MemAct (memory-as-action, but acting on the context window rather than external files), MemSkill (memory as an evolvable skill), and MeMo (frozen base + dedicated memory model, but encoding static document knowledge rather than the agent's memory-management decisions). AutoMem's differentiating selling point is "optimizing both the structure and proficiency axes simultaneously," and this coverage argument holds up. But note its limited generalizability: the paper admits memory is **episodic** (each episode starts from scratch, with no cross-episode persistence), and each of the three environments trains its own scaffold + specialist that cannot be shared — a clear gap remaining from "real-world memory-intensive tasks," which the authors honestly list in the limitations. Overall this is a well-designed, solid piece of work whose critical space concentrates on "comparison fairness and evaluation-set selection," rather than a flawless result to accept wholesale.

## One-minute version

- **The memory bottleneck**: An LLM's context window is like a fixed-size buffer that cannot hold long-horizon tasks running tens of thousands of steps. Each NetHack run can last $10^4$–$10^5$ steps, and even the strongest frontier models have only single-digit progression on the BALROG leaderboard (Gemini-3-Pro 6.8%, Claude-Opus-4.5 2.0%).
- **Memory as action**: AutoMem treats reading and writing files as "first-class actions" on par with interacting with the environment, leaving it to the model to decide when to remember and what to remember. In the same forward pass, the model can choose either a game action or `<|APPEND|>` / `<|SEARCH|>` to operate the external memory files.
- **The power of optimizing memory only**: Without touching the game-strategy weights at all, merely automatically optimizing the structure and usage proficiency of the memory files doubles a small model's performance — the 32B base thereby improves 4.00× on MiniHack.
- **The fairness and limits of the comparison**: The so-called "small model catches up to large model" comparison is not a level playing field — the Claude and Gemini used as references are taken from BALROG's vanilla setting and were not equipped with this tailored memory system; moreover, the acceptance gate deciding scaffold quality uses the very same set of eval seeds as the final test, leaving room for overfitting to the evaluation set.
- **The value is in offline deployment**: What actually does the diagnosing and rewriting is the most capable frontier meta-LLM (Claude Opus 4.6/4.7); the small model is merely optimized offline and then enjoys the results. In practice one can use a large model as a "coach" to tune offline and run only the 32B at deployment to save cost, but this does not mean a small model can autonomously catch up to a large model.

## 🔗 Related notes

- [Reflexion](../Reflexion/) — likewise lets an agent improve across episodes via self-reflection, but using linguistic reflection rather than a trainable memory skill; AutoMem in its related work categorizes it as a contrast within the embodied-agent paradigm.
- [LoRA](../Lora/) — the memory specialist is precisely finetuned with a LoRA adapter.
- [Agent-as-a-Judge](../Agent-as-a-Judge/) — using an agent to review an agent, sharing the trajectory-level review spirit of AutoMem's "meta-LLM as code reviewer / training engine."
- [Meta-Harness](../MetaHarness/) — also Stanford work on "automatically optimizing agent systems," and AutoMem's scaffold loop is structurally similar to its diagnose-and-revise structure.
