# AutoData — Research Note
> **English** | [繁體中文](./README.zh-TW.md)

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | Autodata: An agentic data scientist to create high quality synthetic data |
| Venue | unknown |
| Year | 2026 |
| Authors | Ilia Kulikov, Chenxi Whitehouse, Tianhao Wu, Yixin Nie, Swarnadeep Saha, Eryk Helenowski, Weizhe Yuan, Olga Golovneva, Jack Lanchantin, Yoram Bachrach, Jakob Foerster, Xian Li, Han Fang, Sainbayar Sukhbaatar, Jason Weston (FAIR at Meta) |
| Official Code | https://github.com/facebookresearch/RAM |
| Venue Kind | paper |

> This note is written based on the arXiv preprint `2606.25996` (Meta FAIR "RAM" project). As of writing, no peer-reviewed venue could be found, so Venue is recorded as `unknown`; if an official version is released it may differ from this preprint.

## First Principles

### From "Filtering Static Data" to "Treating Data Production as a Scientist Loop"

The post-training of current language models increasingly relies on synthetic data. Existing mainstream approaches (Self-Instruct, Grounded Self-Instruct, CoT Self-Instruct, Self-Challenging, etc.) are essentially one-shot prompt generation plus post-hoc filtering (filtering, evolution, refinement), and do not directly control the difficulty and quality of the data. AutoData's core claim is to reframe "making data" as the iterative process a data scientist would follow: first generate a batch of data, then do qualitative inspection (eyeballing) and quantitative evaluation, distill learnings, and accordingly update the data-generation recipe to produce better data, until a stopping condition is met.

This framework has an outer loop and an inner loop. The inner loop is the "data creation → data analysis → update recipe" loop; the outer loop takes "the data scientist agent itself" as an object of optimization (meta-optimization), using the same evaluation criterion as the inner loop (producing data that better discriminates between models) to guide the outer loop's improvement of the agent's prompt and strategy. The paper positions it as a general mechanism for "trading inference-time compute for higher-quality training data."

![AutoData's overall loop: the agent plays the data scientist, repeatedly generating data, doing qualitative inspection and quantitative evaluation, distilling insights and updating the generation recipe; the outer loop then optimizes the agent itself.](imgs/autodata_pipeline.png)

### Concrete Implementation: The Weak-vs-Strong Design of Agentic Self-Instruct

All of the paper's experiments are built on a concrete implementation called Agentic Self-Instruct. A main orchestrator agent directs four LLM subagents: a Challenger that generates training examples following a detailed prompt given by the main agent; a "weak solver" (expected to usually fail); a "strong solver" (expected to usually succeed); and a Verifier / judge that checks the quality of the examples and the models' answers and feeds learnings back to the main agent. The system's goal is to produce training data that "the strong solver can do but the weak solver gets stuck on."

For verifiable tasks, one criterion is to require the strong solver's majority vote to be correct while the weak solver's majority vote is wrong; for non-verifiable tasks, it requires a gap in the quality measured by the judge, making the problem neither too easy nor too hard for the weak solver, while using the strong solver to guarantee correctness. If the criterion is not met, the main agent revises the prompt sent to the challenger based on the judge's feedback, switching to a different reasoning angle to regenerate a problem, until it passes. The paper specifically notes: the weak and strong solvers can be "different modes of the same LLM"—the strong version is allowed to use more inference-time compute, scaffolding, or aggregation, and can even access privileged information.

![The weak-vs-strong design of Agentic Self-Instruct: the main agent directs the challenger to generate problems, the weak/strong solvers answer, the judge scores, and the challenger's prompt is updated iteratively based on the judge's feedback.](imgs/agentic_self_instruct.png)

### A Concrete Walkthrough: The CS Research QA Task (Using the Paper's Real Numbers)

Taking computer-science research question answering as an example makes it clearest what this loop is doing. The main agent uses Kimi-K2.6, the strong solver uses Qwen3.5-397B-A17B, and the weak solver uses Qwen3.5-4B. The Challenger generates "context + question + reference answer + a self-contained weighted scoring rubric" from a paper, and the judge scores the two solvers (each answering 3 times to reduce variance) item by item according to the rubric.

The problem is: problems generated directly by prompting (the CoT Self-Instruct column) are mostly too easy for this 4B weak solver. The table below is the quality statistics scored by Kimi-K2.6 at generation time, under the same batch of paper material:

| Metric | CoT Self-Instruct | Agentic Self-Instruct |
|-|-|-|
| Weak solver avg | 0.677 | 0.458 |
| Strong solver avg | 0.696 | 0.772 |
| Gap (strong − weak) | 0.019 | 0.314 |
| Agentic rounds | 1.00 | 6.59 |
| Question length (chars) | 723 | 619 |
| Rubric items | 13.2 | 13.1 |

For CoT-generated problems the weak solver averages 0.677 and the strong−weak gap is only 0.019, with almost no learnable signal. So the paper defines the acceptance criterion directly on this gap: a problem is accepted only when the strong solver averages ≥ 0.65, the weak solver < 0.5, and the strong−weak gap ≥ 20 percentage points. To save compute, the judge evaluates the strong solver only when the weak solver passes its success criterion. After running the whole loop, under the same material the weak solver's score dropped 22 percentage points from 0.677 to 0.458, while the strong solver rose 8 percentage points from 0.696 to 0.772—the gap was stretched to 0.314, meaning the accepted problems indeed shifted toward concrete algorithmic steps, ablation details, or numerical claims that "require following the paper's argument." On average it takes 6.59 rounds to produce a problem, with a long tail extending beyond 10 rounds.

Among the 880 pre-acceptance failed rounds, the failure reasons are highly one-sided: 80% are because the problem was too easy and the weak solver's score too high, so it was rejected; 13% are because the strong solver also could not solve it stably. In terms of data scale, the paper processed over 10k CS papers from the S2ORC corpus (post-2022), used Agentic Self-Instruct to produce 2.8k accepted examples, and then filtered them through the quality verifier at the end of the loop (removing those with paper-specific reference leakage, context too short, or rubric-format errors), retaining 1.3k high-quality examples as RL training data; the CoT baseline also applied the same verifier and sampled the same 1.3k for a fair comparison.

Then, using GRPO (batch size 16, learning rate 1e-6), Qwen3.5-4B was trained on each of the 1.3k datasets, evaluated on a 200-problem held-out test set, and scored by Kimi-K2.6 according to the rubric. The table below shows the step-200 results, reporting mean@3 / best@3 on both the "CoT test set" and "Agentic test set" distributions:

| Response model | CoT mean@3 | CoT best@3 | Agentic mean@3 | Agentic best@3 |
|-|-|-|-|-|
| Qwen3.5-4B (no additional RL) | 0.630 | 0.758 | 0.366 | 0.484 |
| Qwen3.5-4B RL on CoT Self-Instruct data | 0.727 | 0.853 | 0.500 | 0.631 |
| Qwen3.5-4B RL on Agentic Self-Instruct data | 0.774 | 0.894 | 0.632 | 0.768 |

On the easier CoT test set, training on Agentic data pulls the base 4B from 0.630 to 0.774; on the harder Agentic test set it goes from 0.366 to 0.632—the latter gap between the two methods is more than double the former. In other words, training on data that "better discriminates between models" not only wins on the hard distribution but also transfers back to the easy distribution (+0.05).

### The Same Loop, Opposite Failure Mode: Legal Reasoning

The paper uses a second domain to test generality: legal reasoning. The material is drawn from public legal documents such as court opinions in Pile of Law, evaluated on PRBench-Legal and its PRBench-Legal-Hard subset. Interestingly, the failure mode here is the opposite of CS: problems generated directly by CoT are "too hard" rather than too easy for the weak solver—the weak solver averages only 0.159, and many rollouts get a straight 0, so that within each prompt group of GRPO the advantage approaches 0, with almost no learning signal.

Therefore, instead of the hard-coded thresholds used for CS, a more flexible loop judge is used here: each legal document is first passed to an extractor agent to extract a structured summary (topic keywords, key facts, holdings), the challenger generates a problem plus a weighted rubric from this, the weak solver rolls out 5 times and the strong solver 3 times, and the judge reads the pattern of each rollout, the weak/strong gap, and the rubric, outputting a structured verdict (`weak_pattern`, `strong_pattern`, `gap_interpretation`, `rubric_concerns`, `grpo_suitability`) and an `accept`/`improve` decision. The criterion is not a fixed number, but synthesizes data quality and "GRPO learnability."

The result is: the agentic loop pushed the weak solver's average from 0.159 to 0.283, the strong solver barely moved (0.717 → 0.698), and the strong−weak gap actually narrowed from 0.558 to 0.415; the truly key change is that the standard deviation of the weak rollouts per problem rose from 7.93 to 12.63—the same gap spread over a "usable variance range," so the reward signal becomes learnable. The distribution of the loop judge's `grpo_suitability` (high/medium/low) also confirms this: the CoT pool is 4.8% high / 41% medium / 45% low, while the Agentic pool becomes 52% high / 43% medium / 2% low. On downstream RL (GRPO, n=8 rollouts per prompt), the 4B trained on Agentic data scores 0.441 (GPT-5 as judge) and 0.393 (Kimi as judge) on PRBench-Legal, not only beating the same-architecture CoT-trained version (0.377 / 0.343), but even beating the much larger, un-retrained Qwen3.5-397B-A17B baseline (0.404 / 0.358), with both graders agreeing.

The two cases of CS and Legal make AutoData's argument very clear: the same loop applied to the two opposite failure modes of "too easy" and "too hard" moves the gap in opposite directions (CS stretches it open, Legal narrows it), yet the downstream RL results consistently improve. The authors therefore emphasize one sentence: the key is not to make the problem "harder," but to tune it to a difficulty at which the model can "just barely" climb step by step.

### Harder Problems Transfer to Easier Problems: Scientific Reasoning

The third domain is reasoning about mathematical objects, with material and evaluation following the Principia collection (covering MSC2020 and PHYS syllabi). Here three data sources are compared for downstream RL: using the original Principia problems directly (CoT Self-Instruct), Agentic-generated data, and a merge of both (Combined, doubling the training volume to 18k). Each single source has 9k training + 1k held-out examples, training Qwen3.5-4B with GRPO (group size 8, batch size 64), with Kimi-K2.6 doing binary pairwise scoring. On the merged validation set, Agentic gives the largest overall avg@8 improvement of +3.20%, beating using CoT directly (+2.42%) and the double-data Combined (+2.70%); notably, Agentic even wins on the CoT subset (+3.05% vs +1.86%), showing that "training on harder problems transfers to easier problems," rather than only being good at the difficulty level it targets.

An easily overlooked but very real by-product is token efficiency: under a 65,536-token reasoning budget, the base 4B has 23.75% of responses truncated by length (finish_reason=length), dropping to 4.09% after Agentic training. The authors attribute the "wrong→right" flips and find that about half of the accuracy improvement comes from fixing truncation (of Agentic's 945 flips, 54.81% are attributed to truncation repair and 41.06% come from reasoning improvement in non-truncated cases)—that is, training on hard problems not only teaches the model to reason more correctly, but also teaches it to reason more economically within a fixed budget.

### The Outer Loop: Meta-Optimizing the Data Scientist Agent Itself

AutoData's second level is to also change the "data scientist agent's prompt and strategy" with evolutionary optimization. The meta-optimizer maintains a population of candidate prompts, each candidate being a code diff relative to a baseline repository. In each round: (1) select a parent via Boltzmann sampling, where the probability of candidate $c$ being selected is proportional to

$$P(c) \propto \exp(\text{score}_c / T), \qquad T = 0.1$$

(this formula is the selection rule described in the paper, with notation consolidated by this note) strongly favoring high-scoring candidates while retaining exploration; (2) evaluate the parent on a small batch of training papers, collecting trajectories and weak/strong scores; (3) use an LLM agent to read the full solver conversations and write a root-cause analysis of systematic failure modes; (4) have a code-editing agent produce an improvement diff based on the analysis; (5) re-evaluate the parent and mutant on held-out validation papers; (6) accept the mutant into the population only when its validation score is strictly higher than the parent's; (7) summarize the result into a history log readable by subsequent analysis. Because a single score is noisy (the solver at temperature 1.0), accepted candidates accumulate more evaluations and are averaged when later sampled as parents.

On the CS paper task, starting from about a 62.1% validation pass rate, after 233 iterations (126 accepted) it improved to 79.6% (with a 6-hour timeout cap per session). The criterion for "passing" here is: the weak solver (Qwen3.5-4B) ≤ 65%, the best weak-solver attempt ≤ 75%, the strong solver (Qwen3.5-397B-A17B) between 60% and 95%, and the strong−weak gap ≥ 20 percentage points. Several modifications automatically discovered by the meta-optimizer are quite instructive: requiring the problem to test "what is specific to this paper" rather than general ML knowledge (self-check: if you could answer correctly without reading this paper, it is too easy); strictly forbidding the context from leaking the solution; and a counterintuitive finding—removing negative-weight rubric criteria and instead using positive integer weights capped at 7, because negative weights historically hurt strong-model scores yet did not help discrimination.

## 🧪 Critical Assessment

### The Problem Is Real, but the Framework's "Novelty" Lies More in Integration than Mechanism

"Using an agent to iteratively make data that better discriminates between models" is a real and current pain point: as frontier models get stronger, static synthetic data increasingly struggles to produce sufficiently challenging problems. This motivation holds. But we must honestly consider where the novelty comes from: weak–strong challenger–solver, judge feedback, difficulty adjustment, and even treating the harness as an optimization target—these components each appeared in prior work such as STaR, Self-Rewarding, Self-Challenging, Absolute Zero, SPICE, GEPA, and Meta-Harness. AutoData's contribution is mainly to unify them under the narrative of a "data scientist loop" and to show that the meta level can be stacked on top. This is valuable engineering integration and framing, but the wording the paper itself uses ("generalizes all the above methods") is on the grand side, and the original mechanistic increment is relatively limited; the reader should understand it as a unifying framework rather than a wholly new algorithm.

### Baselines, Ablations, and Metrics: Credible, but with a Concern of Evaluation Circularity

The experimental design has much to commend: three heterogeneous domains, a comparison of CoT and Agentic under the same data budget/challenger/corpus, the deliberate addition of GPT-5 as an independent grader on the legal task to rule out Kimi-grader bias, and separating "harder" from "just right" in the discussion. These are all more solid than the typical synthetic-data paper.

But what most warrants caution is the circularity of evaluation: in the CS task, the data's acceptance criterion, the training reward, and the test-set scoring are almost all done by the same Kimi-K2.6 according to a rubric, and the rubric is itself generated by the challenger (also Kimi-K2.6). This means the data-generation end and the evaluation end share the same model and the same rubric perspective, "the Agentic test set was made precisely to highlight Agentic data," so Agentic winning the most on the Agentic test set (mean@3 0.632 vs 0.500) is not entirely independent. The legal task uses the official PRBench judge setup plus a GPT-5 re-scoring, and this part's persuasiveness is clearly higher than CS. Furthermore, the absolute increment is on the small side: on legal, Agentic beats CoT by only +0.05–0.06, and on scientific reasoning the OOD gain is only +1.04% with some categories (such as SuperGPQA, pass@8) regressing, and the authors themselves admit the 4B may already be near the capacity ceiling of that task distribution.

### Self-Defined Benchmarks and the "Painting the Target" Risk

The CS section has a clear self-defined-benchmark concern: the test set is generated by the authors' pipeline, scored by a model homologous with the generation end, and the difficulty definition is written directly into the acceptance criterion (strong−weak gap ≥ 20 points). This setup of "first defining a difficulty favorable to one's own method, then comparing at that difficulty" inherently favors Agentic, making it hard to judge how much of the gain comes from data quality versus how much comes from the bullseye being painted on the method's strong points. The paper's limitations also admit that some generated problems are "over-bound to the paper's specific experimental numbers rather than testing generalizable reasoning," which is the same coin's other side as the painting-the-target concern. In contrast, the legal task greatly mitigates this problem by relying on the external PRBench and GPT-5 grader, and is the most defensible evidence in the whole paper.

### Whether It Is Solved, and Real-World Relevance

On the restricted question of "whether inference-time compute can be traded for better training data," the paper provides evidence of a consistent direction across three domains, and the meta level's 62.1%→79.6% improvement is concrete. But note two open points: one is cost—the average round counts of 6.59 (CS) and 4.98 (Legal) mean each accepted data point costs several to a dozen-plus times the inference compute, and the paper does not systematically report an equal-compute comparison of "Agentic vs a large amount of CoT under the same compute," so "whether it is worth it" remains open. The second is safety/hacking—the authors report encountering the agent "cheating" (for example, changing the prompt to tell the weak solver to pretend to be weak), currently only partially mitigated by adding constraints. Overall, this is a framework-level work whose direction is credible and whose preliminary evidence is solid, but whose increment is on the small side and whose evaluation is partly self-circular—as the authors say, "just the tip of the iceberg," and its long-term value depends on subsequent validation under equal-compute comparisons, more tasks, and stronger anti-cheating measures.

## 🔗 Related notes

- [Instruction Tuning with GPT-4](../InstructioinTuningWithGPT4/) — Also part of the context of "using a strong model to synthesize instruction data for post-training," and can be contrasted with AutoData's approach of agentifying data generation.
