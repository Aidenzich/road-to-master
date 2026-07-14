# MAST — Research Note
> **English** | [繁體中文](./README.zh-TW.md)

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | Why Do Multi-Agent LLM Systems Fail? |
| Venue | NeurIPS 2025 Datasets & Benchmarks (arXiv 2503.13657v3) |
| Year | 2025 |
| Authors | Mert Cemri, Melissa Z. Pan, Shuyi Yang, Lakshya A Agrawal, Bhavya Chopra, Rishabh Tiwari, Kurt Keutzer, Aditya Parameswaran, Dan Klein, Kannan Ramchandran, Matei Zaharia, Joseph E. Gonzalez, Ion Stoica (UC Berkeley; Intesa Sanpaolo) |
| Official Code | https://github.com/multi-agent-systems-failure-taxonomy/MAST |
| Venue Kind | paper |

> This note is written from the full text and LaTeX source of arXiv preprint 2503.13657 (v3, last revised 2025-10-26); the authors have already used the NeurIPS 2025 Datasets & Benchmarks camera-ready layout, so the final official version may differ slightly from this.

## Introduction

Multi-Agent LLM Systems (MAS) have been heavily deployed over the past two years in software engineering, scientific simulation, and general-purpose agent scenarios, yet their gains over simple baselines such as a single agent or best-of-N sampling on public benchmarks are often small. The specific question this paper asks is "Why do MAS fail?": the authors test seven SOTA open-source MAS empirically and find failure rates as high as 41% to 86.7%, underscoring the urgent need for a systematic understanding of failure causes.

![This figure actually draws only five bars (each system's Success/Failure share on its own benchmark): MetaGPT (ProgramDev) fails 34.0%, ChatDev (ProgramDev) fails 75.0%, HyperAgent (SWE-Bench Lite) fails 74.7%, AppWorld (Test-C) fails 86.7%, AG2 (GSM-Plus) fails 15.2%. Because the measurement benchmarks differ, the numbers are not directly comparable. Note: the intro text claims coverage of seven SOTA MAS, the figure caption instead says six popular MAS, and the figure shows only five bars — the three do not match, discussed in the Critical Assessment; the "41% to 86.7%" range also has only its upper bound 86.7% (AppWorld) directly visible among these five bars.](imgs/mas_failure_rates.png)

The paper's high-level approach is "first build a taxonomy, then use the taxonomy to label large amounts of data." The authors propose MAST (Multi-Agent System Failure Taxonomy), the first empirically grounded taxonomy purpose-built for MAS failures, grouping failures into 3 broad categories and 14 fine-grained failure modes. The taxonomy itself is inducted from 150 traces using Grounded Theory, with definitions repeatedly refined against inter-annotator agreement.

To apply the taxonomy to large-scale data, the authors further build an LLM-as-a-Judge annotation pipeline and use it to label MAST-Data: a dataset of over 1600 annotated execution traces (1642 in practice) spanning seven MAS frameworks, the first multi-agent dataset characterizing MAS failure dynamics.

How does the paper measure whether this method works? Mainly along three lines: (1) using Cohen's Kappa to measure annotation consistency, with an average of κ = 0.88 between human experts and κ = 0.77 for the LLM annotator against humans; (2) validating generalization on two new frameworks not involved in taxonomy development and two new benchmarks (MMLU and GAIA), obtaining κ = 0.79; (3) validating with two intervention case studies (ChatDev and AG2): whether making targeted fixes to the failure modes MAST diagnoses can raise task success rate.

## First Principles

### Building the taxonomy: Grounded Theory + Inter-Annotator Agreement

![MAST dataset construction workflow (paper Figure, the figure labels seven stages in total): ① MAS Trace Collections (collect execution traces) → ② Failure Identification (identify failures) → the middle blue box "Development of Failure Taxonomy" containing ③ Inter-Annotator Agreement and ④ LLM Annotator → ⑤ the resulting taxonomy table and ⑥ MASFT (the taxonomy itself) → ⑦ MAS Failure Detection (large-scale failure detection on new traces). Per the paper's Section 4 method: the taxonomy is first inducted from traces by six human experts using Grounded Theory, then its definitions are repeatedly refined through Inter-Annotator Agreement; the LLM Annotator is an LLM-as-a-Judge pipeline built on top of the "already validated" taxonomy, used in step ⑦ for large-scale failure detection on new traces rather than participating in defining the taxonomy — the figure drawing ③④ in the same box is only visual grouping and does not mean the two are built simultaneously. This is a schematic workflow, not a quantitative chart.](imgs/dataset_construction_workflow.png)

MAS failures are hard to annotate for two root reasons: unlike traditional software, MAS failures often have no single identifiable root cause but are the intertwined result of agent interactions and each model's behavior; furthermore, the field has no standardized failure definition, so cross-system annotation is highly inconsistent. The authors' countermeasure is to make no prior assumptions and let failure modes emerge from the data: first collect 150 traces (each averaging over 15,000 lines of text), then have six human experts repeatedly analyze them with GT techniques such as open coding and constant comparative analysis until theoretical saturation; on these 150 traces alone, each expert invested over 20 hours.

Next, the emerged failure observations are standardized into reusable labels. The authors iterate over three rounds of IAA: in each round three annotators independently label a small batch of traces, then meet to resolve disagreements and adjust or add/remove failure mode definitions, measuring consistency with Cohen's Kappa. Cohen's κ is defined as the observed agreement $p_o$ minus chance agreement $p_e$, normalized (the standard formula supplemented by this note):

$$
\kappa = \frac{p_o - p_e}{1 - p_e}
$$

The final rounds reach a strong average of κ = 0.88, indicating the taxonomy definitions are clear enough to be applied consistently by different annotators.

### The structure of MAST: 3 categories, 14 failure modes

MAST maps the 14 failure modes onto three stages of MAS execution (Pre-Execution, Execution, Post-Execution), then groups them by the nature of the failure into three broad categories (FC). The table below organizes each mode with the occurrence rate reported in the paper's Section 4 body text:

| Category (FC) | Failure Mode | Body-text occurrence rate |
|-|-|-|
| FC1 System Design Issues | 1.1 Disobey task specification | 11.8% |
| FC1 | 1.2 Disobey role specification | 1.5% |
| FC1 | 1.3 Step repetition | 15.7% |
| FC1 | 1.4 Loss of conversation history | 2.80% |
| FC1 | 1.5 Unaware of termination conditions | 12.4% |
| FC2 Inter-Agent Misalignment | 2.1 Conversation reset | 2.20% |
| FC2 | 2.2 Fail to ask for clarification | 6.80% |
| FC2 | 2.3 Task derailment | 7.40% |
| FC2 | 2.4 Information withholding | 0.85% |
| FC2 | 2.5 Ignored other agent's input | 1.90% |
| FC2 | 2.6 Reasoning-action mismatch | 13.2% |
| FC3 Task Verification | 3.1 Premature termination | 6.20% |
| FC3 | 3.2 No or incomplete verification | 8.20% |
| FC3 | 3.3 Incorrect verification | 9.10% |

Each of the three categories corresponds to a core insight. FC1 argues that MAS failure is not merely a function of challenges in the underlying model: under the same underlying model, good system design alone can bring gains. FC2 argues that context or communication protocols alone are not enough; what is truly missing is the agents' "social reasoning" — even with natural-language communication within the same framework things still break, reflecting a theory-of-mind-style breakdown (agents cannot correctly infer each other's information needs). FC3 argues that multi-level verification is needed: doing only a low-level check at the last gate is insufficient.

The paper grounds FC2's abstract argument with a real AppWorld trace (FM-2.4 Information Withholding): the Phone Agent fails to report a key API requirement — the username field actually needs to be filled with a phone number — to the Supervisor Agent, and the Supervisor also does not proactively ask for clarification; as a result the Phone Agent keeps calling `apis.phone.login(...)` with the wrong email format (rather than the phone number the API requires) as username, returning `{"message": "Invalid credentials"}`, and login repeatedly fails until the task collapses. What broke is not the communication channel (both sides kept conversing in natural language) but the bidirectional inference of information — this is precisely the concrete face of missing social reasoning.

![An FM-2.4 Information Withholding instance in an AppWorld trace. Per the paper's original caption, the Phone Agent does not convey the API requirement (username format) to the Supervisor Agent, and the Supervisor does not proactively seek clarification, leading to repeated login failures and task failure. The figure's bottom note reads "Missing feedback on username should be the phone number."](imgs/info_withholding_trace.png)

![MAST taxonomy: the 14 failure modes are grouped into three broad categories and mapped onto the three execution stages of a MAS conversation. The percentages beside the boxes are the occurrence rate of that mode in the paper's analysis.](imgs/mast_taxonomy.png)

Notably, the per-mode percentages labeled inside Figure 1 (e.g. 1.1 at about 15%) do not match the Section 4 body-text numbers (1.1 is 11.8%), which is discussed again in the Critical Assessment.

### Scalable annotation: LLM-as-a-Judge

Labeling over 1600 traces by pure manual effort is infeasible. The authors use OpenAI's o1 model as judge, feeding in the execution trace, the MAST definitions, and few-shot examples (drawn from human-annotated data), and have it output each trace's failure mode and rationale. The paper's LLM-as-a-judge table lists two rows, labeled per the original as `o1` and `o1 (few shot)`: the `o1` row has accuracy 0.89 and κ = 0.58, while the `o1 (few shot)` row has accuracy 94% and Cohen's Kappa of 0.77, the latter being higher on every metric. Note that the table distinguishes only by these two labels, and the methodology body text only states that the whole pipeline uses few-shot examples without further explaining the setting of the `o1` row; therefore this note refers to them only by their original labels and does not infer whether the `o1` row is a zero-shot / no-example version. The authors also report the average cost of this pipeline: The average cost across all MAS frameworks is $1.8 per trace.

Before formal large-scale annotation, the authors first run a round of out-of-domain human IAA on two new frameworks not involved in development (OpenManus, Magentic-One) and two new benchmarks (MMLU, GAIA), obtaining a strong Cohen's Kappa score of 0.79, arguing that MAST can be applied to new systems and new tasks without modification. The final MAST-Data collects 1642 annotated traces from seven frameworks — five initial research frameworks and two generalization-validation frameworks (OpenManus, Magentic-One) — covering the three task types of coding, math, and general agent. (Worth noting: the closed-source platform Manus is not included in the primary MAST-Data because its full execution traces cannot be obtained, and only underwent human evaluation of task correctness; the methodology body text at one point lists Manus alongside the other seven, creating an internal contradiction with the data table.)

### Worked example: using MAST to diagnose and fix ChatDev

Walking through the full "diagnose → fix → measure" flow with ChatDev as an example makes both the framework's usage and its limitations clear. ChatDev simulates a software company (CEO, CTO, engineer, reviewer, and other agents); on the ProgramDev-v0 task set its baseline task success rate is only 25.0. MAST diagnosis finds a typical FM-1.2 (Disobey role specification): the CPO agent ends the discussion on its own before reaching consensus with superiors. The first fix tightens the role hierarchy — only superior agents can decide to end the conversation — raising success rate by +9.4% to 34.4. The second fix changes the framework topology from a DAG to a cyclic graph, so the flow only terminates after the CTO confirms all reviews have passed (with a maximum iteration cap to prevent infinite loops), pushing success rate further to 40.6 (a total of +15.6% relative to baseline).

The table below shows the full results of the two case studies. The same set of fixes moved to the near-saturated HumanEval barely budges (89.6 → 90.3 → 91.5); on AG2, the improved prompt is effective for GPT-4 (84.75 → 89.75), but the new topology gives only +0.75 for GPT-4 with a Wilcoxon test p-value of 0.4 (not significant), only becoming significant when switched to GPT-4o (p = 0.03).

| Configuration | AG2 GSM-Plus (GPT-4) | AG2 GSM-Plus (GPT-4o) | ChatDev ProgramDev-v0 | ChatDev HumanEval |
|-|-|-|-|-|
| Baseline | 84.75 ± 1.94 | 84.25 ± 1.86 | 25.0 | 89.6 |
| Improved prompt | 89.75 ± 1.44 | 89.00 ± 1.38 | 34.4 | 90.3 |
| New topology | 85.50 ± 1.18 | 88.83 ± 1.51 | 40.6 | 91.5 |

(The two AG2 columns are the mean ± standard deviation of percentage results over six repetitions; the two ChatDev columns are single-run task success rates.)

This example is simultaneously the paper's core evidence and its core limitation: MAST can indeed decompose an "aggregate success rate" into operable failure modes, pointing at where to fix; but the gains from single-point fixes are both small and unstable, and the authors themselves admit that achieving reliability often requires more complex solutions and fundamental system redesign.

### How different LLMs and architectures reshape the failure profile

Another use of MAST is comparing design choices. Fixing the MetaGPT framework and comparing underlying models, GPT-4o has by 39% fewer FC1 (System Design Issues) failures than Claude 3.7 Sonnet; but both remain high on FC3 (Task Verification), indicating verification is a shared cross-model bottleneck. Fixing GPT-4o and comparing architectures, MetaGPT has 60-68% fewer FC1 and FC2 than ChatDev, yet instead has 1.56x more FC3 failure. The paper offers a tentative explanation of its own for this direction (phrased in the original as "This may stem from…"): MetaGPT mainly relies on SoPs to maintain adherence to task and role specifications, hence it is especially strong on FC1; whereas ChatDev architecturally weights verification more heavily — software development is split into design, coding, and testing stages, and testing is further divided into code review (static) and system testing (dynamic) — and these explicit testing and review stages suppress its FC3 failure count. Note that this is only a conjecture the paper puts forward and it provides no direct evidence to validate this causal claim, but it is indeed the cause the paper itself gives, not something this note adds. Whatever the cause, this comparison shows the failure profile is shaped by both model and architecture, with no single silver bullet.

![Three subplots correspond to the three broad failure categories (Specification and System Design 37.2%, Inter-Agent Misalignment 31.4%, Task Verification 31.4%); the x-axis actually draws only five systems: AppWorld, MetaGPT, ChatDev, HyperAgent, AG2, taking the first 30 traces per system. The paper's caption writes "total 210 traces" (equal to 7×30), but the figure shows only these five systems, i.e. 5×30 = 150 traces — the caption's 210 does not match the five visible panels, and this 210-vs-five-systems gap is unexplained by the paper. Different systems present distinct failure profiles (e.g. AppWorld leans toward premature termination, HyperAgent toward step repetition and incorrect verification).](imgs/failure_distribution.png)

## 🧪 Critical Assessment

### The problem is real, but the measurement design plants a self-drawn target

The problem itself — that "MAS often fail" — is real and important: Figure 3 shows ChatDev, HyperAgent, and AppWorld failing at 75%, 74.7%, and 86.7% respectively on their own benchmarks, enough to support the research motivation. But note the paper's failure rates are measured on different benchmarks, therefore they are not directly comparable, and reading "41% to 86.7%" as a single comparable range is misleading — for instance AG2 actually has a high success rate on the easier GSM-Plus. This is a typical self-defined-benchmark assemblage: each system is paired with a task unfavorable to it, easily making the overall failure look more severe than it really is.

### Internal number inconsistencies weaken the "empirically rigorous" self-positioning

A paper that self-positions as "empirically grounded" nonetheless has internal numbers that do not add up. The most obvious is the per-mode occurrence rates: Figure 1 labels 1.1 as about 15% and the three categories as 37.17% / 31.41% / 31.41%, but the Section 4 body text writes FM-1.1 as 11.8%, and the sum of the per-mode numbers does not match the figure. In addition, the intro claims the seven systems have failure rates of 41% to 86.7%, yet the same failure-rate figure draws only five systems, among which AG2's success rate is far above this range's lower bound. These inconsistencies do not overturn the conclusions, but for a Datasets & Benchmarks paper whose selling point is data quality they are a substantive flaw.

### Annotation credibility relies heavily on a single LLM judge, and the taxonomy admits confusion risk

The vast majority of MAST-Data's 1642 traces are LLM annotated: human annotated covers only the seven groups labeled HA, 30 traces each, totaling 210/1642 = 12.8%, with the rest all labeled by a single o1 judge, so the whole dataset's credibility rests on it. And while o1's κ = 0.77 is high, its precision is only 0.833 and recall 0.77, meaning a considerable share of mode labels may be biased; the authors also admit that fine-grained modes have moderate correlations (max of 0.63), which may lead the automatic annotator to conflate distinct root causes. In other words, the finer the classification, the more it helps distinguish root causes but also the more error-prone automatic annotation becomes — and large-scale data can only rely on automatic annotation.

To be fair, the paper also gives positive evidence on "whether the classification axes are mutually independent": at the coarser three-category level, categories have only low correlations (0.17-0.32) pairwise, supporting that the three categories indeed characterize different failure axes. The confusion risk appears mainly at the finer 14-mode level (max 0.63) rather than the category level — which lets the two judgments "the taxonomy itself is distinguishable" and "fine-grained automatic annotation is prone to confusion" coexist.

### The taxonomy is a real contribution, but the "solutions" are mostly a re-cataloging of existing tools

MAST's taxonomy and large-scale annotated dataset do have originality and community value — they push MAS failures from anecdotal discussion toward a quantifiable, debuggable framework, and the open-source pip install agentdash lowers the barrier to use. But the paper's "solutions" chapter (tactical / structural strategies) is mostly a re-cataloging of existing techniques (self-verification, cross-verification, standardized communication protocols, RL fine-tuning, memory management), lacking systematic validation of which strategies work on MAST-Data; the only interventions actually performed are the two small-scale case studies, with small and model-sensitive gains. So the "diagnostic tool" contribution is solid, while "how to fix MAS" remains an open question — which the paper itself does not claim to solve.

## One-Minute Version

- **Problem**: Multi-Agent LLM Systems (MAS) simply string multiple LLM agents together to cooperate, but they often fail as a whole. Seven SOTA open-source MAS have failure rates as high as 41% to 86.7% — for example ChatDev has 75% execution failures on its own benchmark.
- **Method**: MAST is a taxonomy grouping MAS failures into 3 categories and 14 failure modes. It is inducted from 150 traces (each averaging over 15,000 lines) using Grounded Theory, with inter-annotator consistency between human experts reaching Cohen's κ = 0.88.
- **Main finding**: Making targeted fixes to the failure modes MAST diagnoses can indeed raise task success rate. ChatDev's success rate goes from baseline 25.0 to 34.4 after tightening the role hierarchy, then to 40.6 after changing the topology.
- **Strongest reservation**: The credibility of the entire 1642-trace annotated dataset rests on a single o1 judge, and the taxonomy admits that fine-grained modes confuse one another. o1's recall is only 0.77, and the max correlation among modes reaches 0.63, which may let automatic annotation conflate distinct root causes.
- **Practical takeaway**: MAST is a solid "diagnostic tool," but "how to fix MAS" remains an open question. Single-point fix gains are small and unstable — switching AG2's topology gives only +0.75 for GPT-4, with a Wilcoxon p-value of 0.4 (not significant).

## 🔗 Related notes

- [Reflexion](../Reflexion/) — self-correction of agents via verbalized self-feedback, corresponding to MAST's self-verification class of solutions.
- [SELF-REFINE](../SelfRefine/) — iterative self-feedback rewriting, related to the tactical fixes of FC3 verification.
- [Agent-as-a-Judge](../Agent-as-a-Judge/) — using agents to evaluate agents, sharing origins with MAST's LLM-as-a-Judge annotation pipeline.
