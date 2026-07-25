# SubmergedKnowledge — Research Note
> **English** | [繁體中文](./README.zh-TW.md)

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | Are LLMs Really Not Knowledgeable? Mining the Submerged Knowledge in LLMs' Memory |
| Venue | ICLR 2026 |
| Year | 2026 |
| Authors | Xingjian Tao, Yiwei Wang, Yujun Cai, Zhicheng Yang, Jing Tang |
| Official Code | unknown |
| Venue Kind | paper |

> This note is based on the arXiv preprint `2412.20846v2` (2026-01-28); that version's title matches the ICLR 2026 OpenReview submission (forum `gvUufgeJvV`), and the official camera-ready content may differ slightly from this version.

> **Correction note (2026-07-26)**: An earlier version of this note claimed that "Figure 4 (LLaMA3-8b, DBPedia) actually plots LLaMA3-70b's data and does not reconcile with the paper's table." Checked against the paper's **actual Figure 4**, that accusation is **factually wrong**: Figure 4's Head is 18.9 / 48.3 / 61.3 / 83.4 / 90.5 (k=1,5,10,50,100), which **matches exactly** the table values this note cites (48.3 / 83.4 / 90.5) — the figure and table were consistent all along, with no 70b mix-up. That erroneous claim has been removed. This correction pass also re-checked **every image committed in this folder (Figures 2/3/5/6/7) against the prose, and the values agree** (e.g. Figure 3a's 9.8% accuracy, Figure 3b's Hits@100 ranking, Figure 7's 56/61/65% uninformative shares all match); **the only things not verifiable here are the paper's prose and any table not included as an image** (abstract/introduction wording, Section 4 phrasing, the "70b table rows" the old note referenced), so cross-check those against the original PDF when citing.

## Introduction

Large language models (LLMs) are often used as "parametric knowledge bases": facts are compressed into the weights, then retrieved back out through generation. But on knowledge-intensive question answering (QA) tasks, models frequently give wrong answers or hallucinate, and the mainstream explanation is "the knowledge was simply never learned into the parameters," so the usual remedy is to make the model bigger and feed it more data. This paper challenges exactly that assumption: it argues that many failures are not "not knowing," but "knowing yet failing to express it."

The paper's core observation is: even when the model ultimately outputs a wrong answer, the correct answer often still appears with high probability in the probability distribution over vocabulary tokens — it just wasn't selected as top-1. The paper uses "the capital of Washington State" as its flagship example — the model outputs "Seattle," yet assigns a high probability score to the correct answer "Olympia." The authors call this kind of knowledge that is "hidden in the distribution but not expressed" submerged knowledge.

![Although the model hides Olympia at the second-highest probability position, it still ultimately outputs the wrong answer Seattle (paper Figure 1, schematic)](imgs/case_example.png)

To quantify this phenomenon, the authors propose the Hits@k metric: a hit is counted as long as the correct answer falls within the top k tokens ranked by logit, decoupled from whether the final output is correct. The evaluation covers one open-domain dataset DBPedia and two specific-domain datasets IMDB (movies) and GoodReads (books), and splits them into head/torso/tail by entity popularity; the models under test are 9 open-source models from 1.5B to 72B (LLaMA2/3, Qwen2, Mistral; the upper bound is Qwen2-72b, though the paper body describes the range as "1.5B to 70B" and does not count this 72B exception), all using greedy decoding at temperature 0. The conclusion is: the "stored knowledge" measured by Hits@k far exceeds the amount reflected by standard accuracy.

The paper's second main thread examines the widely adopted few-shot QA paradigm: letting the model answer "unsure" when it lacks confidence in order to reduce hallucination. The authors argue that this kind of prompting that permits "unsure" actually suppresses low-confidence but correct answers, causing a kind of memory-masking effect, and they design a set of decoding experiments that filter out "unsure"-related tokens to quantify it.

## First Principles

### Why "answering wrong" does not equal "not knowing"

The authors treat logits as the model's "internal knowledge state" before it makes its final token choice. Their analysis points to a stable pattern: even when the top-1 choice is wrong, the token representing the correct information is still often assigned a fairly high probability, especially in specialized domains, where the model verbally says "unsure" yet places the correct term at a high rank. This means that traditional evaluation, which only looks at the final output, systematically underestimates the knowledge actually encoded in the parameters.

### Hits@k: peeling "expression" apart from "storage"

The definition of Hits@k is straightforward: among $N$ questions, the proportion where the correct answer appears within the top-k logits.

$$\text{Hits}@k = \frac{N^{k}_{correct}}{N}$$

where $N^{k}_{correct}$ is the number of questions for which the "correct answer falls within the top-k logits." The authors argue that for a model like LLaMA3 with a vocabulary of about 128,000 tokens, a relatively small k can effectively capture the stored knowledge while maintaining computational efficiency.

The evaluation protocol has two key details worth remembering. First, because the model uses subword tokenization, the authors judge a hit by "string matching": a hit is counted as long as any token in the top-k shares at least three consecutive characters with the ground-truth answer. Second, the value of k is tied to the vocabulary size, and a larger k is a looser scoring criterion. These two points determine what Hits@k is actually measuring, and we will return here later in the critical section.

### A concrete per-k forward-pass example

Put LLaMA3-8b on DBPedia, run one forward pass on a given question, take the output distribution ranked by logit, then progressively relax the hit threshold by k. Using the actual numbers from the paper's appendix tables ($k=5,50,100$), the Hits@k for the three popularity subsets of DBPedia is as follows:

| LLaMA3-8b @ DBPedia | Head | Torso | Tail |
|-|-|-|-|
| Hits@5 | 48.3 | 42.4 | 36.9 |
| Hits@50 | 83.4 | 79.6 | 76.6 |
| Hits@100 | 90.5 | 88.1 | 87.1 |

This table is a microcosm of the whole paper's argument: raising the threshold from top-5 to top-100, the "hit rate" of the same model on the same batch of questions jumps from 48.3% to 90.5%. The authors' reading is that standard accuracy is nearly at the bottom, but the correct answers actually lie densely within the top hundred tokens. This gap of "top-1 can't select it, but the candidate set contains it" is exactly the submerged knowledge they define. As a contrast, LLaMA3-70b reaches 92.1% Hits@100 on DBPedia-Head, whereas the older LLaMA2-70b reaches only 70.5%.

The paper's Figure 4 is the flagship illustration for the same configuration (LLaMA3-8b, DBPedia), directly laying out the Hits@k for $k=1,5,10,50,100$ side by side; Head rises all the way from 18.9% at Hits@1 to 48.3% at Hits@5 and 90.5% at Hits@100, with the steepest segment occurring between k=1→5. These figure values match the table values for the same model above (Hits@5 48.3, Hits@50 83.4, Hits@100 90.5) exactly — the figure and the table are consistent.

![Figure 4: the Hits@k bar chart the paper draws for LLaMA3-8b on DBPedia, with Head/Torso/Tail increasing as k=1,5,10,50,100; Head is 18.9% at k=1, jumps to 48.3% at k=5, and reaches 90.5% at k=100](imgs/fig4_k_selection.png)

### "unsure" suppression and the two-stage decoding probe

Back to the second main thread. The authors observe that in many "model outputs unsure" cases, the correct answer still falls at top-2 or top-3 by logit ranking. To quantify this, they design a two-stage decoding procedure: first filter out the "uninformative tokens" within the top-k (those starting with "uns", empty strings, fewer than three characters, or pure stop words), take the highest-probability one remaining as the candidate answer $a^*$:

$$a^* = \arg\max_{t \in T_k \setminus U} P(t \mid q)$$

Then append $a^*$ back to the original prompt and feed the model again to trigger a new round of decoding. The algorithm body is as follows:

```text
Input: token ranking list L (by logit from high to low), original prompt Prompt_old
i <- 0
while L[i] is an uninformative token:
    remove L[i] from L
    i <- i + 1
a* <- L[i]
Prompt_new <- Prompt_old + a*
Output_new <- LLM(Prompt_new)
```

The paper's Figure 6 gives three concrete cases supporting this mechanism: facing "the common treatment for tuberculosis," LLaMA3-8b outputs "unsure" at top-1, and the top-2 in the figure is the subword "Antib" related to the correct answer Antibiotic (the original figure caption limits this slot to "the correct answer, or a subword related to it," and this case falls into the latter); asked "which country the Thor-Agena comes from," it likewise first produces "unsure," with the complete correct answer USA falling at top-2; asked "Gian Sangheera-Warren's occupation in Game of Thrones," the top-1 is an empty character, and the correct answer Actor ranks at top-3. What the figure marks is the logit rank (Rank 1/2/3), not the logit value, and the message is clear: the model verbally says "not sure," yet the correct token is actually right behind it by one or two ranks.

![Figure 6: LLaMA3-8b's "unsure" case study. The top-1 of all three questions is an uninformative response (unsure or empty character); immediately following are, respectively, Q1's top-2 related subword "Antib" (correct answer Antibiotic), Q2's top-2 complete correct answer USA, and Q3's top-3 complete correct answer Actor](imgs/fig6_unsure_case_study.png)

Using this filtered decoding, a portion of responses originally judged as "unsure" can be restored to the correct answer. Taking DBPedia as an example, LLaMA3-70b's recovery rate rises from greedy's 11.2% (Head) to 23.0% (+11.8), with Torso and Tail at +9.4 and +6.7 respectively. The authors explicitly state that this "unsure filtered decoding" is only an analytical probe for quantifying the memory-masking effect, not a directly deployable method.

### Signals from scale, domain, and popularity

The experiments also bring out several counterintuitive patterns. First, a bigger model does not necessarily mean higher Hits@k: under DBPedia-Head, $k=100$, LLaMA2-13b (70.9%) and LLaMA2-70b (70.5%) are nearly tied, and LLaMA3-8b (90.5%) and LLaMA3-70b (92.1%) differ by only 1.6 percentage points; in other words, quintupling the parameter count barely moves the amount of submerged knowledge.

![Figure 2: the Hits@100 of models of different scales on DBPedia-Head (top row Open Domain) and specific domains (bottom row Specific Domain), with the horizontal axis being parameter scale. This is clearest in the top-left LLaMA open-domain panel: the cyan (light blue) LLaMA2 stays almost flat from 13b to 70b (the red line is LLaMA3, positioned higher up), showing that scale is not the sole driver of submerged knowledge](imgs/fig2_model_size.png)

Even more dramatic, ranking models by accuracy and ranking them by Hits@k flips the order entirely. Figure 3 ranks the 8 models by Accuracy (panel a) and Hits@100 (panel b) respectively: Qwen2-72b has the highest accuracy (17.3%) yet ranks only in the middle of Hits@100 (90.1%); LLaMA2-70b's accuracy ranks second (16.0%), yet its Hits@100 is at the bottom (70.5%); conversely, LLaMA3-70b, with accuracy of only 11.2%, takes first place in Hits@100 (92.1%). Looking only at the final output would read the ranking of "knowledge-retrieval potential" completely backwards.

![Figure 3: the ranking of 8 LLMs under DBPedia-Head, k=100. (a) ranked by Accuracy, Qwen2-72b (17.3%) is highest and LLaMA3-70b (11.2%) is on the low side; (b) ranked by Hits@100, LLaMA3-70b (92.1%) is highest and LLaMA2-70b (70.5%) is at the bottom — the two metrics' rankings are almost opposite](imgs/fig3a_models_acc.png)
![Figure 3(b): the ranking of the same batch of models after re-sorting by Hits@100](imgs/fig3b_models_hits.png)

Second, newer models have higher Hits@k (LLaMA3 clearly beats LLaMA2, regardless of size). Third, the open domain (DBPedia) has higher Hits@k than the specific domains (IMDB, GoodReads), and is less sensitive to popularity; popularity does affect memory, but its impact is smaller than its impact on accuracy. Figure 5 draws this out with cumulative hit curves: on DBPedia (panel a), the three curves Head/Torso/Tail still have a gap of about 20% vs 13% at $k=1$, and as $k$ increases the gap gradually narrows, but around $k\approx100$ the three lines are still visibly separated, and only when $k$ approaches $10^{3}$ (all three lines approaching 99%) do they nearly merge; on the specific domain IMDB (panel b), Tail (green line) clearly lags behind Head/Torso across the entire range of $k$, and even at $k\approx10^{3}$ it still stalls around 90%, below Head (about 97%) and Torso (about 93%), never catching up. By contrast, popularity's effect on the gap among DBPedia's three curves is clearly smaller than its effect on IMDB — open-domain knowledge is more tolerant of popularity, while specific-domain cold entities have an inherent memory shortfall.

![Figure 5(a): LLaMA3-8b's cumulative hit rate vs top-k curve on DBPedia, with the Head/Torso/Tail gap smaller and gradually narrowing as k increases, but only at k≈10^3 do the three lines nearly merge](imgs/fig5a_dbpedia_ranks.png)
![Figure 5(b): the corresponding curves for the same model on IMDB, where Tail (green line) is still clearly below Head/Torso even at k≈10^3, showing that the specific domain is more sensitive to popularity](imgs/fig5b_imdb_ranks.png)

![Distribution of different response types: over half of DBPedia's head/torso/tail are uninformative (Figure 7)](imgs/response_distribution.png)

A key mediating factor is the "uninformative response" (repeated strings, empty strings, and "unsure"). On DBPedia, Head/Torso/Tail have 56%, 61%, 65% of responses respectively belonging to the uninformative category; as popularity drops, the uninformative proportion rises, becoming the main source of accuracy decline. The authors argue that these uninformative responses still hide relevant knowledge, and that "identifying and filtering uninformative responses" is easier than "identifying wrong answers," so filtering them out and dredging up the submerged knowledge has a chance of improving QA performance.

## 🧪 Critical Assessment

### The headline phenomenon holds; what deserves scrutiny is the measurement, not a "figure-vs-table" clash

"Knowing yet failing to express it" is a real and underestimated phenomenon, and measuring it independently of accuracy has value — establish that first. (Correction: an earlier version of this note claimed here that "Figure 4 actually plots 70b data and does not match the table." Checked against the paper, that claim is factually wrong and has been removed — Figure 4's Head is 18.9 / 48.3 / 61.3 / 83.4 / 90.5, matching the table's 48.3 / 83.4 / 90.5 exactly, so figure and table were consistent all along.) What should be challenged is therefore not a non-existent "figure-table inconsistency," but how much to trust the phenomenon's **absolute magnitude** — and that hinges on the two methodological issues below: whether the Hits@k hit criterion is too loose, and whether it is merely an oracle diagnostic that requires knowing the ground truth. Those are where the real scrutiny belongs.

### Hits@k's hit criterion is too loose and may be measuring string coincidence rather than usable knowledge

The metric design is the place that most deserves questioning. The hit definition is "any token in the top-k shares at least three consecutive characters with the answer," and k can be enlarged to 100, while the vocabulary has about 128,000 tokens. Under this setup, "Olympia" could be hit by any token containing an "Oly", "lym", or "mpi" fragment, which amounts to counting the literal overlap of subwords as "possessing knowledge." Opening the candidate set to 100 tokens and then using three-character substring matching makes it hard to rule out the competing explanation that "the high hit rate is merely because a large vocabulary plus loose matching raises the collision probability." The paper's defense of Hits@k's validity (systematic across domains, model rankings that differ from accuracy) are all correlational arguments, and do not directly refute this string-coincidence confounder; a clean control (for example, running the same three-character matching with randomly shuffled answer labels to see how high the false-hit rate is) is absent.

### It lacks a head-to-head comparison with existing submerged-knowledge probing methods, and the recovery gains are small and unstable

On the method side, the observation that "the correct answer is hidden in lower-ranked logits" overlaps heavily with existing work (such as contrastive decoding, confidence-calibration-type methods), and this paper's novelty lies mainly in the framing/naming and the "unsure suppression" angle, rather than the mechanism itself; but the paper does not put Hits@k or unsure filtering head-to-head against any existing baseline. More critically, the authors themselves position the two-stage decoding as "an analytical probe, not a deployable method," and its recovery gains are inherently scattered: LLaMA3-8b on DBPedia-Head is only +3.8 (9.8→13.6), Mistral-7b barely moves (16.5→16.7, +0.2), and on IMDB several cells are even 0.0 or +0.1. This makes the evidence for "LLMs actually know much more" quite weak on the "can be dredged back" side.

### Flaws in internal consistency weaken trust in the numbers

There are several inconsistencies worth being wary of, and most can be confirmed from the committed figures. First, if the body claims "when $k=50$, the Hits@k of head, torso, tail all exceed 80%," that does not match Figure 4's LLaMA3-8b on DBPedia, whose Torso (79.6) and Tail (76.6) are both below 80%. Second, if the paper equates standard accuracy with Hits@1, that clashes with its own figures: Figure 3(a) gives LLaMA3-8b／DBPedia-Head accuracy as 9.8%, while Figure 4 gives the same model's Hits@1 (Head) as 18.9% — the two are not the same quantity. Third, if the experiment section says the models span "1.5B to 70B," Figure 3's model list nonetheless includes the 72B Qwen2-72b (accuracy 17.3%), so even the upper bound doesn't align. Two further flaws are text-only (not verifiable from the images here, kept from the original note): Section 4 writes the dataset as "DBLP" rather than DBPedia; and the LaTeX still uses the `iclr2025_conference` template despite publication at ICLR 2026. None of these are fatal, but they are worth keeping in mind when citing absolute values.

### Hits@k is an oracle metric, and is still a distance away from "really solving the problem"

Even if submerged knowledge really exists, dredging it back with Hits@k presupposes that you already know the correct answer in order to judge whether it is in the top-k — this makes Hits@k essentially an oracle (ground-truth-requiring) diagnostic quantity, rather than a decoding method that can improve accuracy at deployment time. The two-stage decoding, although it does not need an oracle, relies on "filtering out unsure and then taking the next-highest token" to guess, and its gains are, as mentioned above, small and unstable. Therefore, what this paper establishes is a "diagnostic-level" claim (knowledge is often masked), while the more practically valuable problem of "turning the diagnosis into usable knowledge recovery" has not truly been solved by the paper, and the authors' probe positioning honestly acknowledges this point too.

## One-minute version

- When an LLM answers wrong on QA, it is not necessarily because the parameters lack this knowledge; the correct answer often lies with high probability among the top few of the token distribution, it just wasn't selected as the output. The authors call this submerged knowledge.
- They use Hits@k (whether the correct answer falls within the top-k logits) to measure this: LLaMA3-8b on DBPedia-Head rises all the way from 48.3% at Hits@5 to 90.5% at Hits@100, far higher than the character-by-character-correct accuracy.
- Prompting that lets the model answer "unsure" suppresses low-confidence but correct answers; filtering out the "unsure" token and decoding once more can recover a portion of the correct answers (but the gains are scattered, e.g. LLaMA3-8b is only +3.8, Mistral is nearly 0).
- The reservation most worth keeping: the hit criterion is "three-consecutive-character substring matching + top-100" (vocab ≈ 128k tokens), loose enough to possibly count string coincidence as knowledge; Hits@k needs to know the answer first to be computed, an oracle diagnostic metric, not equivalent to a deployable improvement; and the recovery gains are small and unstable (LLaMA3-8b only +3.8, Mistral ≈ 0).

## 🔗 Related notes

<!-- No safely resolvable related notes yet -->
