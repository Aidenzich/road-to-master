# FineTuning-vs-ICL-vs-RAG — Research Note
> **English** | [繁體中文](./README.zh-TW.md)

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | When to fine-tune, when to use in-context learning, and when to use RAG: what the empirical evidence supports about their trade-offs |
| Venue | unknown |
| Year | unknown |
| Authors | unknown |
| Official Code | unknown |
| Venue Kind | survey |

## 📚 Sources

| # | Title | Venue | Year | arXiv / URL | Access |
|-|-|-|-|-|-|
| 1 | Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs (Ovadia et al.) | arXiv (preprint) | 2023 | [2312.05934](https://arxiv.org/abs/2312.05934) | fetched |
| 2 | RAG vs Fine-tuning: Pipelines, Tradeoffs, and a Case Study on Agriculture | arXiv (preprint) | 2024 | [2401.08406](https://arxiv.org/abs/2401.08406) | fetched |
| 3 | Few-shot Fine-tuning vs. In-context Learning: A Fair Comparison and Evaluation (Mosbach et al.) | arXiv (preprint) | 2023 | [2305.16938](https://arxiv.org/abs/2305.16938) | fetched |
| 4 | RAFT: Adapting Language Model to Domain Specific RAG | arXiv (preprint) | 2024 | [2403.10131](https://arxiv.org/abs/2403.10131) | fetched |

All four use the full arXiv text as their evidence source (preprints; the official conference versions may differ slightly in wording); every number and quotation in this note has been checked back against each paper's own `source/*.tex`. No source was blocked.

## First Principles

### First, Split the Question Correctly: Knowledge and Behavior Are Two Different Things

The most common mistake in a practical decision table is to conflate "making the model know new facts" (knowledge injection) with "making the model switch to a different behavior/tone/format" (behavior adaptation), and then answer both with a single table. The four papers in fact each measure something different: Ovadia et al. measure the ability to inject new facts into the model, Mosbach et al. measure few-shot classification-task adaptation (behavior), the Agriculture paper touches both, and RAFT uses fine-tuning to teach the model "how to read retrieved context." Put side by side, the answer is no longer a three-way pick, but "is what you want to inject knowledge or behavior."

### Knowledge Injection: RAG Almost Consistently Beats Unsupervised Fine-Tuning

Ovadia et al. directly test the belief in the seed decision table that "fine-tuning is for injecting knowledge," and the conclusion is the opposite: unsupervised fine-tuning offers some improvement, but RAG consistently outperforms it, both for existing knowledge and entirely new knowledge, and moreover LLMs struggle to learn new factual information through unsupervised fine-tuning. Across the three models Llama2-7B, Mistral-7B, and Orca2-7B, the average accuracy gain brought by RAG is consistently greater than that of FT.

![Average accuracy gain of FT / RAG / FT+RAG across models, with RAG consistently higher than FT](imgs/knowledge_injection_gain.png)

### A Fully Walked-Through Cross-Method Comparison: the Current Events Task

Walking through Ovadia's current events task (entirely new facts the model never saw during pretraining) is the clearest. Taking Mistral-7B as an example, the log-likelihood accuracy of the four settings is respectively: base model 0.481, jumping to 0.875 with RAG added, while regular fine-tuning (FT-reg) reaches only 0.504, and even with multi-paraphrase augmentation (FT-par) it reaches only 0.588. That is, on the task of "injecting new facts," taking the same corpus and doing unsupervised fine-tuning on it barely writes the knowledge in (0.481→0.504), whereas putting the same corpus into context to let the model retrieve on the fly nearly doubles the accuracy. The authors further find that to teach new knowledge via fine-tuning, the knowledge must be repeated in numerous ways: current-events accuracy is a monotonically increasing function of the number of paraphrases. What this worked example illustrates is the mechanism the seed table never mentions—unsupervised fine-tuning struggles to write a "one-time fact" into the weights, whereas RAG converts the memory problem into a retrieval problem.

### But Whether Fine-Tuning Succeeds or Fails Depends Heavily on Which Kind of Fine-Tuning It Is

Ovadia's pessimistic conclusion has a key boundary: the authors explicitly state they focused on unsupervised training as their primary fine-tuning method, as opposed to instruction-tuning or RL-based methods. When the Agriculture paper switches to supervised Q&A fine-tuning, the conclusion changes: they see an accuracy increase of over 6 p.p. when fine-tuning the model and this is cumulative with RAG, which increases accuracy by 5 p.p. further. Taking GPT-4 as an example, base 75% → fine-tuned 81% → fine-tuned+RAG 86%. The same paper also shows fine-tuning can make the model learn to answer specific questions across geographic regions, with answer similarity raised from 47% to 72%. So on the question "can fine-tuning inject knowledge," the two papers appear contradictory on the surface but are actually testing different fine-tuning: unsupervised continued training vs supervised task/style fine-tuning—this is precisely the trap of comparability.

### "Fine-Tuning Is High and Stable, ICL Is Only Good for Few-Shot" Also Gets Discounted

The seed table assumes fine-tuning brings high and consistent performance, while ICL is only good for limited data and rapid prototyping. Mosbach et al. point out that past comparisons claiming fine-tuning generalizes worse OOD and is prone to learning spurious correlations were in fact an illusion obtained by using models of different sizes. After controlling for model size (125M–30B) and number of examples (16 examples), fine-tuned language models can in fact generalize well out-of-domain, and both approaches generalize similarly; they exhibit large variation. In other words, the adjective "consistent" holds for neither—the variance of both FT and ICL is high. The genuinely stable difference lies elsewhere: ICL requires large models to work in contrast to FT, which works well even with small models, which makes ICL unfriendly to low-resource languages; and FT benefits more from additional samples than ICL does.

### The Three-Way Framework Itself Collapses: Hybrids Like RAFT

When you treat "fine-tuning vs RAG" as mutually exclusive options, RAFT directly proves that the best approach is often to combine the two. The Agriculture paper already shows that the gains of FT and RAG can stack; RAFT goes further: rather than treat retrieval as a plug-in, it is better to train the model to ignore those documents that don't help in answering the question, which they call distractor documents. RAFT (LLaMA2-7B) scores 73.30 / 35.28 / 74.00 / 84.95 / 86.86 on PubMed / HotPot / HuggingFace / Torch Hub / TensorFlow, almost across-the-board surpassing domain-specific fine-tuning (DSF) and GPT-3.5+RAG. The key counterintuitive point is: introducing RAG to a domain-specifically fine-tuned (DSF) model doesn't invariably lead to better outcomes—DSF scores 61.06 on HuggingFace, and directly adding RAG (DSF+RAG) drops instead to 42.59, because the model was not trained to read context. RAFT's contribution is to treat "reading retrieved context" as a behavior to be fine-tuned in.

### A Pragmatic Comparison Table

| Aspect | Fine-tuning (supervised) | In-context learning | RAG |
|-|-|-|-|
| Best at injecting | Behavior / style / format, task skills | Temporary task demonstrations | Factual knowledge, updatable sources |
| Injecting one-time new facts | Weak (unsupervised continued training especially weak) | Only what is placed in context | Strong and updatable in real time |
| Requires training / compute | Required | Not required | Not required (needs retrieval infrastructure) |
| Small-model usability | High | Low (needs large models) | Medium |
| Consistency / variance | High variance | High variance | Relatively controllable |
| Stackability | Stacks with RAG (should even be trained together) | Partially overlaps with RAG | Stacks with FT |

## 🧪 Critical Assessment

### The Four Papers' Experimental Setups Are Not Actually Directly Comparable

Setting these four up as a "FT vs ICL vs RAG showdown" is dangerous, because their fine-tuning is not the same thing at all. Ovadia uses unsupervised continued training, Agriculture uses supervised Q&A, Mosbach uses pattern-based fine-tuning, and RAFT uses supervised fine-tuning with chain-of-thought. Ovadia itself lists "only testing unsupervised FT" as a limitation. Therefore the external validity of the statement "RAG beats FT" has a boundary: it holds for the specific operation of "throwing the raw corpus into unsupervised continued training," and cannot be generalized to "any fine-tuning loses to RAG."

### The Realism of the Benchmarks and the Sufficiency of the Metrics

The headline numbers of three of them are built on fairly narrow evaluations. Ovadia uses multiple-choice log-likelihood accuracy throughout, which is a very weak proxy for "whether the model truly understands and can generate"; the Agriculture paper's correctness and conciseness are judged by GPT-4 as the judge, which amounts to using one LLM's preferences to define another LLM's goodness; RAFT's PubMed is a yes/no binary task, and the authors themselves admit they see no significant gain over DSF+RAG on that dataset. None of these are self-defined benchmarks "painted around the arrow after it hit," but the singularity of the metrics still makes the "who is better" conclusion more fragile than it appears.

### External Validity for the Reader's Practical Question

What the reader really wants to ask is "which should I choose right now," yet these numbers mostly come from 2023–2024 7B–13B open-source models or GPT-3.5/4. Mosbach specifically warns that these methods are still poorly understood, and that the effects are extremely sensitive to pattern, seed, and number of examples. For the current generation of models (long context, native tool-use, stronger instruction following), not a single paper provides head-to-head data; directly applying these pass values to today's models is an unverified extrapolation.

### The Parts Not Yet Answered

No paper answers the cost-curve question of "at what knowledge-update frequency does the amortized cost of re-fine-tuning become lower than running RAG retrieval over the long term"; the seed table lists "limited compute" simultaneously in the applicability columns of both ICL and RAG, but none of the four papers gives a comparable total cost of ownership measurement. The direction RAFT points out—"what gets fine-tuned in is the behavior of reading context"—is the most promising integration direction, but it is only validated on a few domain-specific QA tasks, and whether it generalizes to open-ended generation remains an open question. Therefore this note's overall judgment of the "three-way decision table" is: on the two points of conflating knowledge with behavior and presupposing the three are mutually exclusive, the evidence does not support it—this belongs to a framework not yet substantiated, even dubious, rather than a conclusion that can be copied directly.

## 🔗 Related notes

- [Lora](../Lora/)
- [GNN-RAG](../GNN-RAG/)
- [SAG-SQL-RAG](../SAG-SQL-RAG/)
- [PromptLanguageCodingAccuracy](../PromptLanguageCodingAccuracy/)
