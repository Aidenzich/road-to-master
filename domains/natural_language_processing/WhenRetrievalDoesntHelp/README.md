# When Retrieval Doesn't Help: A Large-Scale Study of Biomedical RAG — Research Note
> **English** | [繁體中文](./README.zh-TW.md)

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | When Retrieval Doesn't Help: A Large-Scale Study of Biomedical RAG |
| Venue | BioNLP Workshop at ACL 2026 (arXiv notes accepted; this note is written from the arXiv preprint LaTeX source) |
| Year | 2026 |
| Authors | Erfan Nourbakhsh, Rocky Slavin, Ke Yang, Anthony Rios (The University of Texas at San Antonio) |
| Official Code | https://github.com/erfan-nourbakhsh/BioMedicalRAG |
| Venue Kind | paper |

> This note is written from the LaTeX source of the arXiv preprint (arXiv:2606.04127); if the official camera-ready version adjusts anything, defer to the final published version for numbers and wording. Table figures and quotations are all taken from the preprint `source/acl_latex.tex`.

## First Principles

Biomedical QA is a high-stakes setting: a single factual error can lead to a harmful clinical decision, while large language models (LLMs) are inherently prone to producing "fluent but wrong" hallucinations, and their knowledge goes stale as the training corpus is frozen. Retrieval-augmented generation (RAG) is seen as a remedy because it splices external evidence into the prompt at inference time, in principle improving factuality, traceability, and knowledge recency all at once. This optimistic expectation is not idle: the most representative prior work, the MIRAGE/MedRAG study, reported that retrieval improves biomedical QA accuracy by as much as 18% relative to chain-of-thought prompting, which also pushed the research focus toward "which corpus, which retriever should I pick."

This paper challenges precisely the scope of that premise. The authors point out that prior systematic medical RAG studies mostly focused on large proprietary or 70B-class models (GPT-3.5, GPT-4, Mixtral-8×7B, Llama2-70B), and evaluated almost exclusively in a zero-shot multiple-choice (MCQ) format; whether those gains carry over to the 7B–8B models that are actually deployable and run on a single GPU is an open question. At the same time, past evaluations also leaned toward expert-level exam questions and rarely touched consumer health questions from laypeople (laymen), or community-generated retrieval sources.

So the paper maxes out the evaluation matrix all at once: 5 open-source instruction-tuned models (spanning 7B to 72B), 10 biomedical QA datasets (spanning both layman and expert, open-ended and MCQ), 4 retrieval methods (BM25, TF-IDF, MedCPT, Hybrid RRF), and 4 retrieval corpora (PubMed abstracts, medical textbooks, Yahoo Answers, HealthCareMagic), and always against a "no-retrieval" (w/o RAG) baseline, in order to isolate the contribution of retrieval itself. The key to this design is that every experimental condition is a (retriever, corpus, query dataset) triple, so the effects of the three choices — "swap the model," "swap the corpus," "swap the retriever" — can be separated and compared.

The table below lays out the entire evaluation space so it's easy to compare which dimension is the dominant factor:

| Evaluation dimension | Options | Count |
|-|-|-|
| Backbone model | Qwen2.5-7B, Llama-3.1-8B, Mistral-7B, Llama-3.1-70B, Qwen2.5-72B | 5 (7B–72B) |
| Retrieval method | No-retrieval (baseline), BM25, TF-IDF, MedCPT, Hybrid(RRF) | 4 + baseline |
| Retrieval corpus | PubMed/BioASQ, Medical Textbooks, Yahoo Answers, HealthCareMagic | 4 (expert×2, layman×2) |
| Evaluation dataset | layman: MeQSum, MedRedQA, MedicationQA, MASH-QA, iCliniq; expert: BioASQ-B, MedQuAD, MedQA-USMLE, MedMCQA, MMLU-Medical | 10 |

Let's first walk through the concrete case where "retrieval should help most," so we can feel how large the gap is. The place where retrieval gain concentrates the most is the open-ended BioASQ task: LLaMA-3.1-8B has a ROUGE-L of 21.65 without retrieval, and jumps to 27.43 after adding the BioASQ/PubMed corpus (+5.78, LLaMA-3.1-8B improves from 21.65 to 27.43 ROUGE-L); and the single largest jump within the BioASQ column actually falls on LLaMA-3.1-70B (21.93 → 28.98, +7.05). The reason it can produce such pretty numbers is that the evaluation questions are themselves drawn from PubMed literature, so the retrieved abstracts are practically the answer source. But when we average 8B over 7 open-ended datasets, the maximum retrieval benefit over the baseline is only 1.18 points (the maximum retrieval benefit over no-retrieval is 1.18 points: 13.06 → 14.24); the average gains of the other four models are even smaller. In other words, those eye-catching +5.78/+7.05 are local phenomena propped up by a single, highly corpus-aligned dataset, and once flattened across the whole they are diluted to almost nothing.

The differences among the four retrieval methods themselves are equally tiny. Take the most effortless RRF (Reciprocal Rank Fusion) as an example: it requires no training, and simply fuses the two ranked lists from BM25 and MedCPT by rank — a document $d$ ranked $r$ in a list gets a score of $\frac{1}{k+r}$; taking $k=60$, it sums the two lists' scores and re-ranks:

$$
\mathrm{RRF}(d) = \sum_{i \in \{\mathrm{BM25},\,\mathrm{MedCPT}\}} \frac{1}{k + r_i(d)}, \quad k = 60
$$

All retrieval conditions fix top-$k=5$ documents spliced in front of the prompt. Notably, even though MedCPT is a dense retriever specifically trained with contrastive learning on PubMed search logs, it does not systematically outperform the purely lexical BM25 on closed-question accuracy; Hybrid has a slight edge in some settings, but no single method wins consistently. This result itself hints that the bottleneck is not "whether the retriever is smart enough."

Condensing the main results into an open-ended ROUGE-L average comparison of "no-retrieval vs. best corpus" reveals the magnitude of the gain:

| Model | w/o RAG | Best corpus (BioASQ) | Δ |
|-|-|-|-|
| LLaMA-3.1-8B | 13.06 | 14.24 | +1.18 |
| LLaMA-3.1-70B | 14.22 | 14.66 | +0.44 |
| Mistral-7B | 13.64 | 14.44 | +0.80 |
| Qwen2.5-7B | 12.91 | 13.56 | +0.65 |
| Qwen2.5-72B | 13.56 | 13.91 | +0.35 |

On MCQ (closed-question accuracy), the situation is even more unfavorable to retrieval: for smaller models, retrieval often "actually subtracts points." Mistral-7B drops all the way from 75.7 without retrieval to 68.6–72.3 across the corpora; the best retrieval settings of LLaMA-3.1-8B and Qwen2.5-7B are also generally below their own no-retrieval baseline. What really determines the score is the backbone: Qwen2.5-72B's 85.6 with no retrieval at all is already more than 2 points higher than the best retrieval setting of any 7B model (exceeds the best retrieval configuration of any 7B model by over 2 points). That is, with the same budget, rather than spending it on a better retriever or corpus, you'd be better off directly swapping in a larger generation model.

So, if we guarantee that "what gets retrieved is definitely relevant evidence," can retrieval cash in on its promise? The authors designed a clean oracle control: using the BioASQ corpus as the source, evaluated on PubMedQA (yes/no/maybe research questions), using an LLM-as-a-judge to determine whether the retrieved context is sufficient to answer (we use an LLM-as-a-judge framework to determine whether the retrieved context contains enough information), then picking out 100 questions where "all retrieval methods retrieved context judged relevant" for comparison.

The table below places clean (guaranteed relevant) and noisy (mixing 20 more unrelated documents into the relevant evidence) contexts side by side:

| Model | w/o RAG | BM25 (clean) | BM25 (noisy) |
|-|-|-|-|
| LLaMA-3.1-8B | 0.410 | 0.580 | 0.300 |
| LLaMA-3.1-70B | 0.410 | 0.660 | 0.260 |
| Mistral-7B | 0.460 | 0.510 | 0.310 |
| Qwen2.5-7B | 0.410 | 0.380 | 0.310 |
| Qwen2.5-72B | 0.380 | 0.350 | 0.250 |

This table is the pivot of the paper's whole argument. Even when context is guaranteed relevant, the gain is still limited and inconsistent: LLaMA3.1-70B improves substantially with BM25 retrieval (0.410 → 0.660), but Qwen2.5-72B barely moves and Qwen2.5-7B even drops slightly — "retrieving the right evidence" does not equal "being able to use the evidence." More crucially, the fragility: just adding 20 more unrelated documents into the relevant evidence (When we add 20 unrelated documents to the retrieved evidence) makes the score collapse dramatically, in most cases worse than no retrieval at all (the 70B's 0.660 drops to 0.260). Put together, these two tables point their arrows at the same conclusion: the real bottleneck is the model's ability to "use evidence," not the ability to "find evidence."

![Open-ended ROUGE-L stays almost flat across top-k (1,3,5,10,25,50); the height of each bar group is determined by the model, not by k](imgs/top-k-rougel.png)

The ablation experiments further cement this reading. Sweeping the number of retrieved documents top-$k$ from 1 to 50, open-ended ROUGE-L enters a plateau after $k=5$, changing by less than 0.2 points between $k=5$ and $k=50$ (ROUGE-L changes by less than 0.2 points between $k=5$ and $k=50$) — stuffing in more documents does not bring in more useful signal. Closed questions expose the small models' constitutional problems even better: LLaMA-3.1-8B reaches 72.83% at $k=5$ and then starts to decline, while Mistral-7B declines steadily after $k=3$, with only 51.22% left at $k\geq25$. Long context is a burden rather than an asset for small models.

![Bar chart of closed-question accuracy across top-k (1,3,5,10,25,50). Legend colors: Qwen2.5-7B purple, Llama-3.1-8B teal-green, Mistral-7B yellow, Llama-3.1-70B red, Qwen2.5-72B dark blue. Llama-3.1-70B (red) and Qwen2.5-7B (purple) stay around 77–80 throughout and barely change with k; Llama-3.1-8B (teal-green) reaches 72.83% at k=5 then drops to about 62.5 at k≥25; Mistral-7B (yellow) declines from k=3 onward, only about 51.22 at k≥25. The damage from long context concentrates on the small models.](imgs/topk-close-accuracy.png)

This phenomenon of "the evidence-utilization bottleneck not vanishing with k" appears not only in ROUGE-L. Laying out the other five open-ended metrics (BERTScore, METEOR, BLEU, ROUGE-2, ROUGE-1) by top-$k$ as well, each metric's bars stay almost flat between $k=1$ and $k=50$; the authors report that after $k=5$ no measurable gain is seen on any metric or any model (additional passages beyond $k{=}5$ provide no measurable benefit in any metric for any model). That is, the "plateau" is not an artifact of one metric but a signal saturation consistent across metrics.

![Appendix figure: bar charts of the open-ended BERTScore, METEOR, BLEU, ROUGE-2, ROUGE-1 metrics each across top-k (1,3,5,10,25,50), with the five models in the same color scheme. Within each subplot the bars barely change with k, presenting a consistent plateau across all metrics, confirming that retrieving more documents no longer brings in useful signal](imgs/topk-open-metrics.png)

![Open-ended ROUGE-L across shot count: 7–8B models collapse at 10-shot, 70B models are almost unaffected](imgs/few-shot-rougel.png)

The few-shot ablation states "small models can't hold up long context" even more bluntly. The larger models (LLaMA-3.1-70B, Qwen2.5-72B) barely change between 1, 3, 5, 10 shots, but the 7–8B models show a cliff at 5, 10 shots: LLaMA-3.1-8B's closed-question accuracy collapses from 82.89% (1-shot) to 10.06% (10-shot) (accuracy collapses from 82.89% (1-shot) to 10.06% (10-shot)), and open-ended ROUGE-L also drops from 14.29 to 8.38. Interestingly, 3-shot is the sweet spot for open-ended (3-shot prompting is the sweet spot, LLaMA-3.1-8B to 17.19, Qwen2.5-7B to 17.22), and it degrades beyond that. This curve is of the same origin as the RAG conclusion: once the prompt gets long, the small models' ability to locate the target instruction collapses first, and evidence utilization naturally becomes moot.

## 🧪 Critical Assessment

### Can the metrics actually measure "whether retrieval helped"

The whole paper's conclusion is built on reference-based metrics (ROUGE-L, BLEU, METEOR, BERTScore, accuracy), and the authors themselves admit in the Limitations that these metrics do not directly measure factuality or evidence grounding (though our experiments do not directly measure evidence utilization or grounding). This is a real tension: a model might answer correctly on parametric knowledge alone, without using retrieval at all; or it might copy the surface phrasing of the retrieved text yet not improve medical correctness. For open-ended biomedical QA, n-gram overlap is inherently a weak signal (BLEU is generally below 2 on lay datasets), so the statement "retrieval gain is only 1–2 points" may partly be that the metric is insensitive to "useful retrieval," rather than retrieval truly being ineffective. The direction of the conclusion is credible, but the "small" magnitude should be discounted a bit.

### Is that BioASQ +5.78 a real gain or an artifact of evaluation alignment

The only place retrieval clearly helps is open-ended BioASQ, whose retrieval corpus (PubMed) and evaluation answers (BioASQ is itself grounded in PubMed literature) come from the same source. When the corpus and reference are highly homologous, the retrieved abstracts and the gold answers share a large amount of vocabulary, and a surface-overlap metric like ROUGE-L gets systematically inflated. The authors read it as "the local benefit of domain-matched retrieval," but a more conservative reading is: this looks more like the evaluation design leaking the answer, rather than the model truly understanding medicine better because of retrieval. This also explains why, once you leave BioASQ and flatten across the other six datasets, it nearly zeroes out — the truly generalizable gain may be even smaller than 1.18 points.

### Is the oracle analysis enough to support the causal claim "the model can't use evidence"

The clean/noisy oracle control is the paper's most powerful mechanistic evidence, but its extrapolation should be handled carefully: it is done only in the setting of 100 PubMedQA questions, a single BioASQ corpus, with relevance filtered by an LLM-as-a-judge. Using an LLM to judge "whether the context is sufficient to answer" itself introduces judgment bias, and the sample of 100 questions makes a 0.02–0.03 difference hard to distinguish from noise. The authors' conclusion is actually worded with some reservation (admitting they did not directly measure evidence utilization), but the note body and figures make it easy to directly equate "relevant evidence brought no gain" with "the model can't use evidence" — this is a reasonable but not-yet-directly-measured inference, not an already-proven causal claim.

### Breadth is a strength, but the lack of variance and single greedy decoding are concerns

The 5×10×4×4 matrix is the paper's most solid contribution, and the consistency across ten datasets makes "retrieval gain is tiny" not look like the happenstance of a single benchmark. But the cost is that all numbers come from single, FP16, greedy decoding with at most 300 tokens generated per question (greedy decoding and a maximum of 300 newly generated tokens), and the whole paper reports no variance or significance test from any repeated runs. When the key conclusion itself is "the difference is only 1–2 points," the lack of error bars is a double-edged sword: it does support "the gain is small," yet it also leaves fine-grained rankings like "Hybrid slightly wins" or "the BioASQ corpus is best on average" unable to rule out being sampling noise.

### For practitioners: swap the backbone first, or fix retrieval first

As a deployment guide, the paper's core message is robust and useful — on 7B–72B open-source models, investing resources in a larger backbone is usually more worthwhile than investing in a better retriever or corpus. But this prescription has clear boundaries: the experiments do not include GPT-4-class proprietary frontier models (do not include proprietary frontier systems), and deliberately do not touch adaptive retrieval, re-ranking, iterative retrieval, or task-specialized chunking (does not explore more complex retrieval strategies such as adaptive retrieval, document re-ranking). So the correct conclusion is not "biomedical RAG is useless," but "a standard RAG pipeline with fixed top-k and no re-ranking, paired with small-to-medium open-source models, cannot cash in on the gains promised by prior large-model studies." Reading it as the former would over-extrapolate to settings the paper never actually tested.

## One-minute version

- Where the problem comes from: past high expectations for biomedical RAG were mostly built on large proprietary models; prior work claimed retrieval can boost accuracy by up to 18%, but whether this holds on 7B–8B models that run on a single GPU had never been answered.
- What they did: cross 5 open-source models, 10 datasets, 4 retrieval methods, and 4 corpora into one big matrix, and pair every setting against a "no retrieval at all" baseline, to isolate retrieval's own contribution.
- Main finding: the improvement from retrieval is both small and fragile — just mixing 20 unrelated documents into the correct evidence makes the 70B model's score collapse from 0.660 to 0.260, in most cases worse than no retrieval at all.
- Don't over-interpret: because it didn't test GPT-4-class frontier models, and deliberately didn't touch advanced techniques like re-ranking or adaptive retrieval, this paper's conclusion cannot be read as "biomedical RAG is useless," only that "a standard pipeline with fixed top-k and no re-ranking, paired with small-to-medium open-source models, cannot cash in on the gains promised by large-model studies."
- For practitioners: on small-to-medium open-source models, spending budget on swapping in a larger backbone is usually more worthwhile than upgrading the retriever or corpus — Qwen2.5-72B has 85.6 with no retrieval at all, more than 2 points higher than the best retrieval setting of any 7B model.

## 🔗 Related notes

- [GNN-RAG](../GNN-RAG/)
- [SAG: Query-Time Dynamic Hyperedge SQL-RAG](../SAG-SQL-RAG/)
- [Fine-tuning vs In-context Learning vs RAG](../FineTuning-vs-ICL-vs-RAG/)
