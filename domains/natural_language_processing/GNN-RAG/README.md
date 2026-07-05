# GNN-RAG — Research Note
> **English** | [繁體中文](./README.zh-TW.md)

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | GNN-RAG: Graph Neural Retrieval for Large Language Model Reasoning |
| Venue | arXiv (preprint) |
| Year | 2024 |
| Authors | Costas Mavromatis, George Karypis |
| Official Code | https://github.com/cmavro/GNN-RAG |
| Venue Kind | paper |

> This is a reading note written from the arXiv preprint `2405.20139`; the manuscript uses the NeurIPS 2024 preprint style, and the official published version (camera-ready) may differ from this. All numbers and quotations are based on the preprint LaTeX source files.

## First Principles

### Problem Background: KGQA's Bottleneck Is "Retrieval," Not "Generation"

The task this paper addresses is Question Answering over Knowledge Graphs (KGQA): given a knowledge graph $\mathcal{G}$ made of `(head, relation, tail)` triples and a natural-language question $q$, the model is required to extract the set of entities $\{a\}$ in the graph that correctly answer $q$. During training there are only question–answer pairs, with no annotated "path to the answer," so it is a weakly-supervised setting. The authors split KGQA into two stages: first, from the complete KG with millions of facts, use entity linking and PageRank to extract a question-relevant dense subgraph $\mathcal{G}_q$; then hand it to a reasoning model to output the answer. The whole pipeline is strung together in the form of retrieval-augmented generation (RAG), verbalizing the KG facts into text and stuffing them into the LLM's prompt.

RAG's performance on KGQA depends heavily on "which facts are retrieved." The authors point out that the real difficulty lies on the retrieval side: a KG easily has millions of facts, and fetching the correct information requires effective graph-processing ability, while fetching irrelevant information instead interferes with the LLM's reasoning. Existing approaches either rely on the LLM to retrieve hop-by-hop, which cannot handle complex graph structures and fails on multi-hop questions; or they must rely on the internal knowledge of an ultra-large model like GPT-4 to fill the gaps in retrieval. In other words, the paper points the finger at the "the retriever is not strong enough" link that RAG narratives often overlook.

### GNN-RAG's Core Design: Using the GNN as a Dense-Subgraph Reasoner

![The GNN-RAG framework: the GNN reasons over the dense subgraph to produce candidate answers and their corresponding shortest paths; the paths are verbalized (optionally merged with RA) and handed to the LLM for RAG](imgs/fig2.png)

GNN-RAG's central claim is: although a GNN does not understand natural language the way an LLM does, it is inherently good at handling complex graph structures, and can be exactly "repurposed" as a retriever. The whole flow has three steps: first, a GNN does message-passing reasoning over the dense subgraph, classifying each node as "answer / non-answer," and takes out candidate answers whose probability exceeds a threshold; second, extract the shortest paths from the question entities to these candidate answers as KG reasoning paths; third, verbalize these paths into text of the form `{question entity} → {relation} → {entity} → … → {answer entity}` and, together with the question, feed them to the downstream LLM for RAG. In this division of labor, the GNN is responsible for extracting useful information from the graph, and the LLM is responsible for the final question answering using its language-understanding ability.

### The GNN's Message Passing and "Question–Relation Matching"

The GNN treats KGQA as a node-classification problem. At layer $l$, the representation $\mathbf{h}_v^{(l)}$ of node $v$ is updated by aggregating neighbor messages, and the message passing is conditioned on the question $q$:

$$
\mathbf{h}_v^{(l)} = \psi\!\Big(\mathbf{h}_v^{(l-1)},\; \sum_{v' \in \mathcal{N}_v} \omega(q, r)\cdot \mathbf{m}_{vv'}^{(l)}\Big)
$$

The key here is $\omega(q, r)$: it measures how relevant the relation $r$ of the fact $(v, r, v')$ is to the question $q$, which amounts to doing "question–relation matching." In a theorem in the appendix, the authors prove that in the ideal case, if $\omega$ can assign 1 to relevant facts and 0 to irrelevant facts, the GNN's sum-operator can "filter out" irrelevant information and keep only the question-relevant subgraph, achieving optimal reasoning. This also points out the GNN's Achilles' heel—its success or failure rides entirely on $\omega$, this semantic-matching function.

Since the implementation of $\omega(q, r)$ depends on a shared pretrained LM to encode the question and relation representations (a common form being $\phi(\mathbf{q}^{(k)} \odot \mathbf{r})$), the authors' clever move is: rather than swap out different GNN architectures, swap the LM inside $\omega$. They train two GNNs, one using SBERT and one using $\text{LM}_{\text{SR}}$ (an LM pretrained for question–relation matching over the KG). Experiments show that although these two GNNs fetch different KG information, both can improve RAG, so their paths can subsequently be unioned to form an ensemble.

### From Reasoning Paths to the LLM's RAG

After obtaining the reasoning paths, the authors verbalize them into text and feed them to the downstream LLM. Since the LLM is very sensitive to the prompt template and the way graph information is phrased, the authors do RAG prompt tuning on an open-source, trainable model (a LLaMA2-Chat-7B): fine-tuning it with the training set's question–answer pairs, with the prompt "Based on the reasoning paths, please answer the given question". During training it is fed the shortest paths from the question entities to the answer, while at inference these are replaced with the paths retrieved by GNN-RAG. It is worth noting that this 7B model and the RAG fine-tuning approach are themselves carried over from the comparison method RoG; what GNN-RAG truly replaces is the "retriever" part.

### Why the GNN Is Suited to Multi-Hop Retrieval, and Its Limitations

![The landscape of existing KGQA methods: GNN-based methods reason over the dense subgraph, while LLM-based methods (ToG hop-by-hop, RoG generating relation paths) use the same LLM to do both retrieval and reasoning](imgs/fig1.png)

The authors support "why the GNN is a good multi-hop retriever" with a set of retrieval analyses. They train a deep ($L=3$) and a shallow ($L=1$) GNN and measure "Answer Coverage" (whether the retrieved paths contain at least one correct answer—note this evaluates only retrieval, not the final QA) and the number of input tokens (efficiency). The results on WebQSP are as follows:

| Retriever | 1-hop #Input Tok. | 1-hop %Ans. Cov. | 2-hop #Input Tok. | 2-hop %Ans. Cov. |
|-|-|-|-|-|
| RoG (LLM-based) | 150 | 87.1 | 435 | 82.1 |
| GNN ($L=1$) | 112 | 83.6 | 2,582 | 79.8 |
| GNN ($L=3$) | 105 | 82.4 | 357 | 88.5 |

This table tells the story very clearly: on 2-hop questions, the deep GNN uses the fewest tokens (357, far fewer than the shallow GNN's 2,582 and RoG's 435) to achieve the highest answer coverage (88.5%)—it is both more effective and more economical. But on 1-hop questions the situation reverses: here "precise question–relation matching" matters more than "deep graph search," and the LLM retriever (RoG, 87.1%) slightly beats the GNN (82.4%). This complementarity of "deep GNN wins multi-hop, LLM wins single-hop" directly gives rise to the retrieval augmentation of the next section.

### Retrieval Augmentation (RA): Unioning the Strengths of Both Retrievers

The idea of Retrieval Augmentation (RA) is very direct: since the GNN is good at multi-hop and the LLM is good at single-hop, take the union of the reasoning paths retrieved by both at inference time, balancing diversity and answer recall. The paper's default **GNN-RAG+RA** merges the paths of the GNN retriever with those of the RoG LLM retriever. The authors also propose a cheaper alternative, **GNN-RAG+Ensemble**: without calling an LLM, it only unions the paths of the two earlier GNNs paired with different LMs (GNN+SBERT and GNN+$\text{LM}_{\text{SR}}$), avoiding the extra overhead of the LLM retriever's multiple beam-search generations.

### One Concrete Forward Pass (Using the Paper's Real Numbers)

Take the WebQSP question in the figure, "Which language do Jamaican people speak?", and walk through it: subgraph retrieval first uses entity linking and PageRank to fetch from Freebase a dense subgraph with an average of about 1,429.8 entities; the GNN reasons over it, outputs candidate answers (English, Jamaican English, French, Caribbean…) and extracts the shortest paths, for example `Jamaica → language_spoken → Jamaican English`; these paths, once verbalized, are fed to the fine-tuned LLaMA2-Chat-7B, which outputs the final answer. The key is the ledger of efficiency versus effectiveness: the whole GNN-RAG (default) requires **no extra LLM calls** on the retrieval side (#LLM Calls = 0), the median input tokens for WebQSP / CWQ are only 144 / 207, yet it achieves 71.3 / 59.4 F1; by comparison, RoG requires 3 beam-search generations per question, inputs 202 / 325 tokens, and achieves only 70.8 / 56.2 F1. The main results table lays out the overall comparison:

| Method | WebQSP Hit | WebQSP H@1 | WebQSP F1 | CWQ Hit | CWQ H@1 | CWQ F1 |
|-|-|-|-|-|-|-|
| LLaMA2-Chat-7B (No RAG) | 64.4 | — | — | 34.6 | — | — |
| RoG (KG+LLM) | 85.7 | 80.0 | 70.8 | 62.6 | 57.8 | 56.2 |
| ToG+GPT-4 | 82.6 | — | — | 69.5 | — | — |
| GNN-RAG (Ours) | 85.7 | 80.6 | 71.3 | 66.8 | 61.7 | 59.4 |
| GNN-RAG+RA (Ours) | 90.7 | 82.8 | 73.5 | 68.7 | 62.8 | 60.4 |

Reading these rows together: simply switching the retriever from "none" to GNN-RAG raises the 7B LLaMA2's WebQSP Hit from 64.4 to 85.7; adding RA further pushes it to 90.7, surpassing ToG+GPT-4's 82.6 across the board on WebQSP with a 7B model, and approaching it on CWQ as well (68.7 vs 69.5). The authors estimate ToG+GPT-4 costs over $800 overall, while GNN-RAG can be deployed on a single 24GB GPU—this is the core selling point of "a small model + good retrieval" beating "a large model + weak retrieval." The gap is most pronounced on multi-hop and multi-entity questions: relative to RoG, GNN-RAG is higher on F1 by 6.5–17.2 (WebQSP) and 8.5–8.9 (CWQ) percentage points.

## 🧪 Critical Assessment

### Is the Problem Real and Important

The problem that "RAG's quality is limited by the retriever" is real and independently corroborated—the paper cites multiple works showing that fetching noise drags down LLM reasoning. Focusing the diagnosis on the retrieval side, rather than blindly scaling up the generation model, is a defensible direction. KGQA itself is also a knowledge-intensive task with practical significance. What must be noted is that this problem setting is bound to the premise of "already having a clean, structured knowledge graph with high enough answer coverage": the answer coverage of the WebQSP subgraph is 94.9%, but CWQ's is only 79.3%, meaning about twenty percent of CWQ questions have their correct answer simply not in the retrieved subgraph, and the ceiling of any downstream method is capped by the subgraph-extraction step. The paper's contribution actually only operates on the link after "the subgraph is given," and this boundary is glossed over lightly in the narrative.

### Are the Baselines, Ablations, Datasets, and Metrics Sufficient

This paper's empirical work is quite solid: the main results cover five categories—embedding / GNN / LLM / KG+LLM / GNN+LLM—totaling more than twenty baselines, and it comes with multiple ablations such as swapping the GNN (GraftNet/NSM/ReaRev), swapping the underlying LLM (Alpaca, Flan-T5, ChatGPT), dense vs sparse subgraphs, and training-data combinations, and it also honestly reports #LLM Calls and input tokens on the efficiency side. The metrics are chosen reasonably: Hit, Hits@1, and F1 each measure different aspects, and the authors even admit that Hit is more lenient toward the LLM because it only looks at "whether any answer is hit." One gap worth pressing on is statistical significance—for cases like GNN-RAG and RoG both being 85.7 on WebQSP Hit, or F1 differing by only 0.5, the paper gives no error bars or variation over multiple random seeds, making it hard for readers to judge whether some "ties or slight wins" are noise. In addition, the ablations also show that "a weak GNN is not a good retriever" (when GNN-RAG is swapped to GraftNet/NSM, CWQ performance is worse than RoG), meaning the benefit of the whole method depends heavily on a GNN that is already SOTA in itself (ReaRev), rather than on the GNN-RAG framework itself.

### Is This a New Method or a Recombination of Existing Components

In fairness, GNN-RAG's constituent components are almost all off-the-shelf: the GNN retrieval uses ReaRev, the RAG fine-tuning and the 7B LLaMA2 are carried over from RoG, $\text{LM}_{\text{SR}}$ comes from SR, and the subgraph extraction follows NSM's PageRank approach. What is truly original is the joining approach of "using the GNN node-classification output to back out the shortest paths, then feeding the paths as retrieval results to RAG," and the RA union strategy designed on top of it. This is indeed a valuable insight—it connects "the GNN is good at multi-hop, the LLM is good at semantics" via a clean interface (verbalized paths), rather than forcing a latent fusion. But by a strict novelty standard, this is closer to a well-designed system integration and empirical finding than a methodological breakthrough; the paper's own contribution statement (Framework / Effectiveness / Efficiency) also leans toward the engineering and empirical side.

### Is the Evaluation "Tailor-Made" by the Authors, and Its Real-World Relevance

The huge gains on the multi-hop, multi-entity subsets (F1 higher than competitors by 8.9–15.5 percentage points) need to be seen in context: these are precisely the home turf of the GNN's deep graph search, which amounts to picking the slice most favorable to its own mechanism to highlight the advantage, and readers should understand it as a conditional conclusion of "when deep graph search matters," not an across-the-board crushing—in fact, on 1-hop questions the GNN is instead slightly inferior to the LLM retriever, which the authors honestly present. In terms of overall F1 (for example WebQSP's 71.3 vs RoG's 70.8), the gap is actually very small. The biggest reservation about real-world relevance lies in the dependence on a "high-quality KG": the method assumes the existence of a structured, updatable knowledge graph aligned with the question, which holds on academic benchmarks like Freebase/WikiMovies, but in most practical scenarios the cost of building and maintaining a KG is often the very bottleneck—and that is precisely the part this method does not address. Therefore this note's conclusion is credible and persuasive on "KGQA academic benchmarks," but there remains an obvious gap when extrapolating to general open-domain RAG, and its universality should not be overstated.

## 🔗 Related notes

- [BM25](../information_retrieval/BM25/) — Classic sparse-retrieval baseline, useful as a contrast to the graph retrieval here
- [TF-IDF: IDF](../information_retrieval/TFIDF/) — Foundational concept of retrieval weighting
