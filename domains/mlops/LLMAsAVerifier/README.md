# LLM-as-a-Verifier — Research Note
> **English** | [繁體中文](./README.zh-TW.md)

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | LLM-as-a-Verifier: A General-Purpose Verification Framework |
| Venue | arXiv preprint (2607.05391v2, cs.AI) |
| Year | 2026 |
| Authors | Jacky Kwok, Shulu Li, Pranav Atreya, Yuejiang Liu, Yixing Jiang, Chelsea Finn, Marco Pavone, Ion Stoica, Azalia Mirhoseini (Stanford / UC Berkeley / NVIDIA Research) |
| Official Code | https://github.com/llm-as-a-verifier/llm-as-a-verifier |
| Venue Kind | paper |
| Venue Tier | unknown |
| Peer-review status | Not peer-reviewed (arXiv preprint; camera-ready may differ) |
| Citation count | unavailable |

> This note is written from a **static** reading of the arXiv full text `2607.05391v2` (retrieved 2026-07-24) and the official code repository; no paper code was executed.

## Introduction

When a coding agent runs the same problem five times, the probability that at least one of those runs produces a correct patch is often high — but after the fact you **don't know which one**. This paper formally names that gap as a new scaling axis: **verification**. The authors measure how large the gap is on Terminal-Bench V2: by pooling the trajectories across the entire leaderboard and assuming an oracle verifier that always picks correctly, Pass@N reaches 98.9%, nearly solving the whole benchmark; the bottleneck is not generation, but "picking the right one out of a pile of candidates."

![Oracle Pass@K upper-bound curve on Terminal-Bench V2: after pooling leaderboard trajectories, a perfect verifier's coverage rises with the number of sampled trajectories per problem, approaching 98.9% at large N, far above the single-model horizontal reference lines (Claude Mythos 82.0%, Claude Opus 4.7 69.4%)](imgs/oracle_bon_plot.png)

The concrete problem the paper sets out to solve is: **without any additional training, use an off-the-shelf LLM as a verifier to make best-of-N selection accurate enough to approach the oracle upper bound.** The core observation is the failure mode of traditional LLM-as-a-Judge — it makes the model emit a single discrete score token (e.g. the "5" out of 1–5), takes the highest-probability token as the final score, and thereby collapses the entire scoring probability distribution into one integer. On Terminal-Bench, this coarse-grained scoring causes about 27% of trajectory pairs to be "tied" (same score), and the verifier simply cannot tell which is right and which is wrong.

The authors' high-level solution is a single sentence: **don't take the argmax, take the expectation.** Taking the expectation over the same score-token probability distribution yields a continuous score; this continuous signal is then scaled along three axes — (1) score granularity (number of score tokens $G$), (2) repeated evaluation (number of repetitions $K$), and (3) criteria decomposition (splitting a single scoring criterion into $C$ sub-criteria). To make "pairwise comparison of all candidates" feasible on a budget, the paper further proposes the **Probabilistic Pivot Tournament (PPT)**, which lowers the selection cost from $\mathcal{O}(N^2)$ to $\mathcal{O}(Nk)$.

**How success is measured**: the verifier is used as a trajectory reward model to perform best-of-N selection across four benchmarks — Terminal-Bench V2 and SWE-Bench Verified for coding, RoboRewardBench for robotics, and MedAgentBench for medical. The main metric is post-selection task success rate (for RoboRewardBench, pairwise preference accuracy), with comparison groups being Pass@1, the oracle Pass@N upper bound, discrete LLM-as-a-Judge, and trained reward models. It also measures Value-Order Correlation (VOC, progress tracking) and RL sample efficiency. The paper claims state-of-the-art on all four benchmarks: Terminal-Bench V2 86.5%, SWE-Bench Verified 78.2%, RoboRewardBench 87.4%, MedAgentBench 73.3%.

![Headline results on the four benchmarks: red is LLM-as-a-Verifier, gray is the comparison baseline](imgs/sota_bars.png)

## First Principles

### From argmax to expectation: the mathematical difference between a discrete judge and a probabilistic verifier

First, let's define the objects clearly. A verifier faces a task prompt $x$ and a trajectory $\tau$ (the complete sequence of agent–environment interactions $s_1,a_1,\dots,s_H,a_H$). A traditional discrete judge makes the language model output a single score token and takes the highest-probability one as the score; formally, $R_{\mathrm{LM}}(x,\tau)\in\{1,\dots,G\}$, with a resolution of only $1/G$.

LLM-as-a-Verifier keeps the same scoring prompt but instead reads the model's logprob distribution at the scoring position and takes its expectation:

$$
R(x, \tau) = \frac{1}{CK} \sum_{c=1}^{C} \sum_{k=1}^{K} \sum_{g=1}^{G} p_{\theta}(v_g \mid x, c, \tau)\,\phi(v_g)
$$

where $V_{\mathrm{score}}=\{v_1,\dots,v_G\}$ is a set of ordered score tokens, $\phi(v_g)$ maps each token to a scalar score, $p_\theta(v_g\mid x,c,\tau)$ is the probability the model assigns to that token, $C$ is the number of criteria, $K$ the number of repetitions, and $G$ the granularity. This equation corresponds its three summations to the three scaling axes respectively.

A key implementation detail (in the official code `llm_verifier/fine_grained_reward.py`) is worth pointing out: scoring does not use the numbers 1–20, but rather **the 20 letter tokens A–T**, mapped as A=20 (best) to T=1 (worst). The paper explains the reason in a prompt note — "using a letter scale rather than numbers is to let logprob extraction support granularity scaling." This is because the API only returns top-20 logprobs per position: mapping each of the 20 score levels to a separate letter token lets you grab the whole distribution at once; if you used multi-digit numbers, tokenization would split "17" into "1" and "7", and the distribution would fail to line up.

A concrete expectation calculation (numbers are illustrative, not from the paper): suppose the verifier gives $p(\text{A})=0.6,\ p(\text{B})=0.3,\ p(\text{C})=0.1$ at the scoring position, corresponding to scalars 20, 19, 18. The discrete judge only outputs A, which normalizes to 1.0; whereas the expectation is $0.6\times20+0.3\times19+0.1\times18=19.5$, normalizing to $(19.5-1)/(20-1)=0.974$. Two trajectories that both "round to A" must tie under the discrete judge, but the continuous score can distinguish the difference between 0.974 and 0.951 — this is precisely the source of tie elimination.

After obtaining the continuous score there is one step that is easy to skip over but numerically crucial: the paper **first** normalizes $R(x,\tau)$ to $[0,1]$ via the linear map $R\mapsto(R-\phi_{\min})/(\phi_{\max}-\phi_{\min})$, and **then** feeds the difference between the two trajectories' normalized scores into the Bradley–Terry sigmoid:

$$
P(\tau_i \succ \tau_j \mid x) = \frac{1}{1+\exp\!\big(-(R(x,\tau_i)-R(x,\tau_j))\big)}
$$

This $[0,1]$ normalization is not rhetorical — the 0.974 and 0.951 numbers in the worked example above are the normalized values (the raw expectations 19.5 and 19.1 lie on the 1–20 scale); the official `pivot_tournament.py`'s `bradley_terry` docstring also explicitly states "p(a beats b) under the Bradley-Terry model on rewards in [0, 1]", and `extract_score` maps the expectation back to $[0,1]$ before returning. If you fed the un-normalized 1–20 expectations directly into the sigmoid, the score differences would routinely span several units, and the sigmoid would saturate into a nearly hard 0/1 preference, throwing away the fine granularity that the continuous score provides.

### Which kind of error each of the three scaling axes tackles

![All three scaling axes make verification accuracy rise monotonically (Terminal-Bench V2)](imgs/scaling_axes.png)

The three axes are not three ways of saying the same thing, but target three different sources of error in reward estimation:

- **Score granularity $G$**: tackles "resolution" error. Increasing $G$ from 1 to 20 raises pairwise verification accuracy from 73.1% to 77.5%. The paper explains this with a signal-to-noise ratio $\mathrm{SNR}(G)=\mathbb{E}[s_c-s_i]/\sqrt{\mathrm{Var}(s_c-s_i)}$ (correct trajectory score $s_c$ minus incorrect trajectory score $s_i$): as $G$ goes from 1 to 20, SNR rises from 0.775 to 0.799. Note — this is a **very small** SNR change (about +3%), yet it corresponds to +4.4 points of accuracy; the causal strength between the two is questionable (see Critical Assessment).
- **Repeated evaluation $K$**: tackles "variance of a single evaluation." $\frac{1}{K}\sum_k R^{(k)}$ is a Monte Carlo estimate of the expected reward, whose variance shrinks as $\mathcal{O}(1/K)$ while the bias stays unchanged. As $K$ goes from 1 to 16, accuracy rises from 74.7% to 77.5%, with diminishing returns — the paper admits "additional evaluations yield diminishing returns due to correlated biases on hard samples." This axis also transfers across modalities to robotics: on RoboRewardBench, the same repeated evaluation pushes trajectory preference accuracy along $K$ all the way to 87.4% at $K=8$ before saturating. But here is a text-vs-figure inconsistency worth verifying yourself: the paper's main text and figure caption both write "$K=1$ is 81.5% and beats the trained baseline under **every budget**," yet on the same figure the blue line's starting point looks to the eye to be about **79.8%**, falling **below** RoboReward-8B's 81.4% reference line — that is, at the cheapest budget point $K=1$, the verifier actually **loses** to RoboReward-8B, only crossing over at around $K=2$. So the correct statement is "overtakes and pulls ahead after repeated evaluation," not "dominates the whole way."

![RoboRewardBench trajectory preference accuracy rising with the number of repeated evaluations $k$ (1→8): LLM-as-a-Verifier (blue solid line, dots) climbs from about 79.8% ($k{=}1$, below RoboReward-8B's 81.4% green dotted line) to 87.4% ($k{=}8$) and saturates, crossing above RoboReward-8B at around $k{=}2$; throughout it is above LLM-as-a-Judge (70.8%), TOPReward (74.7%), and Robometer-4B (78.8%). The paper's main text records $k{=}1$ as 81.5% and claims "leads at every budget," which does not match the starting point on the figure (see Critical Assessment). The right panel shows the same trend under different max_frames visual contexts](imgs/roborewardbench.png)
- **Criteria decomposition $C$**: tackles the "a single criterion is a bad proxy" problem. For a code agent, "is this trajectory correct?" is split into three sub-criteria — Specification (are all requirements met), Output (does the final output format conform), Errors (are there unresolved error signals in the log/tool output). Any single criterion scores 75.2%–76.4%, while the three-way ensemble reaches 78.3%. These three criteria correspond verbatim in the official repository `criteria/terminal_bench.md`.

![Discrete judge (yellow) vs continuous verifier (green): the right panel shows the judge's tie rate dropping from 26.7% to 5.5%, while the verifier is always zero ties](imgs/judge_vs_verifier.png)

The right side of the figure above is the most compelling chart in the whole paper: the discrete judge has 26.7% of comparisons tied at $K=1$, and only slowly drops to 5.5% at $K=16$ through the averaging of repeated evaluation; the continuous verifier, by contrast, is **always zero ties**. This shows that what the judge buys with a large amount of repeated evaluation is mainly "breaking ties," while the verifier gets a stronger signal in one shot.

### A real worked example: `query-optimize`

The paper uses the Terminal-Bench V2 `query-optimize` task to concretely demonstrate the role of granularity (trajectories produced by Claude Opus 4.5 under the OpenHands harness, scored by Gemini 2.5 Flash). The task is to rewrite a slow SQL query into an equivalent fast query. Both candidates produce a faster query, but their verification procedures differ: the correct one honestly waits for the original query to finish its full 5-minute run on the standard database and then does a `diff`; the failed one never verifies equivalence on the database, but instead builds a new database.

Gemini 2.5 Flash can actually see through this failure mode, but it uses vague, graded phrasing like "slightly cleaner" or "somewhat more direct." When 100 repeated evaluations are run:

| Method | $\#(s_c > s_i)$ correct | tie | $\#(s_c < s_i)$ wrong |
|-|-|-|-|
| Judge (discrete, $G{=}5$, take argmax) | 12/100 | **88/100** | 0/100 |
| Verifier (continuous, $G{=}5$, take expectation) | 69/100 | 0/100 | 31/100 |
| Verifier (continuous, $G{=}20$) | **77/100** | 0/100 | 23/100 |

The discrete judge ties the two 88/100 times; taking the expectation over the **same** 1–5 distribution eliminates all ties and picks correctly 69 times; raising the granularity to 20 further increases it to 77 correct picks. This table makes the value of "argmax → expectation" clearest.

### Probabilistic Pivot Tournament: squeezing the selection cost down to O(Nk)

![The five stages of PPT: candidates → Ring pass → Pivot selection → Pivot tournament → Selection](imgs/pivot_tournament.png)

With pairwise preferences in hand, the most intuitive selection method is round-robin: compare all $\binom{N}{2}$ pairs and accumulate soft wins $w_i \mathrel{+}= P(\tau_i\succ\tau_j)$. But this is $\mathcal{O}(N^2)$, and once $N$ is large it consumes the entire verifier budget. PPT uses three steps to lower it to $\mathcal{O}(Nk)$ ($k\ll N$ pivots):

1. **Ring pass**: draw a random Hamiltonian cycle and only score the $N$ pairs of adjacent candidates on the ring. Because it is a single ring, each candidate is **exactly once in the A position and once in the B position**, so the model's positional bias toward "which of A/B is presented first" cancels out in expectation. This is PPT's mechanism for handling positional bias — not via a debias prompt, but via the symmetric tournament structure.
2. **Pivot selection**: sort by the average preference $w_i/c_i$ from the ring pass and take the top $k$ as the pivot set $\mathcal{P}$. Placing pivots on the empirical leaders means spending the remaining budget on distinguishing among the "most likely correct" candidates, rather than wasting it on obviously weak ones.
3. **Pivot rounds**: only compare (i) each non-pivot against pivots, and (ii) pivots against each other. All comparisons accumulate into the same $w_i,c_i$, and finally $\arg\max_i w_i/c_i$ is taken. Dividing by $c_i$ offsets the bias that "pivots participate in more comparisons."

The total number of comparisons is

$$
N + k(N-k) + \binom{k}{2}
$$

which grows linearly in $N$. For example, $N=20,\ k=5$: $20 + 5\times15 + 10 = 105$ query pairs (the paper calls these pairwise comparisons, one query per pair that simultaneously takes `<score_A>` and `<score_B>`), versus round-robin's $\binom{20}{2}=190$. The official code `llm_verifier/pivot_tournament.py`'s `select_best` implements these three steps line by line, with a default pivot count of $k=2$.

Under the setting of 89 problems, 20 candidates each on Terminal-Bench V2, PPT's budget–accuracy tradeoff is as follows (excerpted from an appendix table):

| Method | Query pairs | Selection accuracy (%) |
|-|-|-|
| pass@1 | — | 52.64 |
| V1 ($3N$ budget) | 4,200 | 65.62 |
| PPT $k{=}3$ | 4,723 | 66.17 |
| PPT $k{=}5$ | 6,609 | 66.27 |
| PPT $k{=}9$ | 9,630 | 67.13 |
| Full Round-Robin | 13,111 | 67.42 |

The key is to see clearly what PPT buys: $k=5$ uses about **half** the budget (6,609/13,111 ≈ 50%) to get 66.27%, still **1.15 points lower** than full round-robin (67.42); raising pivots to $k=9$ uses about 73% of the budget (9,630/13,111) to get 67.13%, which only then narrows the gap to a mere **0.29 points**. And the gain relative to V1 ($3N$ budget, 65.62%) is also thin: $k=5$ is only 0.65 points higher (yet uses 2,409 more query pairs), and $k=9$ is 1.51 points higher. In other words PPT is a method for "approaching round-robin with a linear budget," not for being "more accurate" (see Critical Assessment).

### Verifier score as a progress proxy and RL dense reward

The same continuous score has two more extended uses. The first is **task progress**: score each prefix of a successful trajectory and use Spearman rank correlation (Value-Order Correlation, VOC) to measure "whether the score rises monotonically with the time step." On Terminal-Bench V2, successful trajectories have VOC 0.848 and failed trajectories 0.769, a difference of only 0.079; on robotics' RoboRewardBench, VOC reaches 0.966, beating the trained RoboReward-8B (0.877). The second is **dense reward for RL**: use $\rho_t=R(x,\tau_{1:t})$ as a shaped reward fed to DSRL-SAC (LIBERO `ketchup` task, about 1.8× sample efficiency, final success rate 0.76 vs 0.69) and GRPO (Qwen3-8B on MATH, about 1.1×).

![Verifier score vs time step for the Terminal-Bench V2 `pytorch-model-cli` task: the successful trajectory (green) shows an overall upward trend along READ model.py → INSTALL g++ → INSTALL CPU-only torch → UPDATE hidden_dim → DONE, but is not strictly monotonic in the middle — near UPDATE hidden_dim it first rises to about 0.61, falls back to about 0.585, then climbs again, finally jumping to 1.0 at DONE; the failed trajectory (red), after mis-installing torchvision, exhausting the disk, and triggering a compile error, still climbs slowly in score but clearly lags behind (ending at about 0.36 vs 1.0)](imgs/voc_pytorch_model_cli.png)

Two points are worth mentioning. First, the successful trajectory (green line) is not strictly monotonic: it has several visible dips in the middle (about 0.61→0.585→0.605→0.59) before jumping to 1.0 at the end — this is precisely why VOC uses Spearman **rank** correlation rather than requiring stepwise increase, tolerating a few small dips while still achieving the high correlation of 0.848. Second, the failed trajectory (red line) score does not fully stagnate but still climbs slowly to about 0.36 — this corresponds to the VOC concern raised in the Critical Assessment below: the verifier also gives failed trajectories scores with an upward trend.

![RL sample efficiency: left, LIBERO `ketchup` fine-tuning $\pi_0$ with DSRL-SAC, verifier dense reward (red) reaches the same success rate with about 1.8× fewer environment steps than the sparse baseline (gray) and attains a higher final success rate (0.76 vs 0.69); right, MATH fine-tuning Qwen3-8B with GRPO, verifier reasoning reward (red) about 1.1× sample efficiency over the sparse baseline (gray)](imgs/combined_libero_math.png)

It is worth noting that the official repository `llm_verifier/progress.py`'s docstring itself admits: when the offline `track` scores each checkpoint, the verifier can see the **whole trajectory** (including the ending), so "early checkpoints could in principle be influenced by the visible ending"; only the online `ProgressTracker` fed step by step is structurally blind to the future. This is an important caveat for interpreting VOC (see below).

## 🧪 Critical Assessment

### The problem is real, but the headroom is inflated by pooled sampling

"Verification is the bottleneck" is a real and important problem — the oracle upper bound of best-of-N is indeed far above Pass@1. But the paper's most eye-catching opening number, 98.9% oracle, is the Pass@N after "pooling the trajectories of all the models on the Terminal-Bench V2 **leaderboard**," not the capability of a single generation strategy. Meanwhile, in the experiment table, the oracle Pass@5 for the same benchmark under a single strategy (GPT-5.5) is only 92.1%. Using the cross-model pooled 98.9% as motivation and then the single-model 92.1% as the experimental upper bound amounts to using an unrealizable headroom to exaggerate the "recoverable space."

### The SOTA gains are small, and the four bar charts mix incomparable metrics

Breaking apart the headline, the verifier's actual gain relative to Pass@1 is in fact not large: Terminal-Bench 83.1→86.5 (+3.4), SWE-Bench 76.1→78.2 (+2.1), MedAgentBench 70.2→73.3 (+3.1). Measured as "fraction of oracle headroom recovered," SWE-Bench only recovers $(78.2-76.1)/(84.4-76.1)\approx25\%$. More critically, the "SOTA" bar chart in Figure 1 places the four benchmarks side by side, but RoboRewardBench's 87.4% is **pairwise preference accuracy** (judging which of two videos has more progress), which is not the same quantity as the **task success rate** of the other three; putting preference accuracy and task success rate into the same SOTA chart is an apples-to-oranges presentation. Furthermore, Terminal-Bench's "SOTA" compares "GPT-5.5+Capy+verifier" against leaderboard entries with **different harnesses and different base models** (e.g. GPT-5.5+NexAU-AHE 84.7%), where harness, base model, and verifier all vary simultaneously, so the credit cannot be cleanly attributed to the verifier. RoboRewardBench's "leads throughout" claim also has a text-vs-figure discrepancy: the main text writes $K=1$ as 81.5% and claims it beats the trained baseline at every budget, but the same figure's blue line starts at about 79.8%, actually falling below RoboReward-8B's 81.4% at $K=1$ and only crossing over at around $K=2$ — the most striking narrative, "zero-shot, training-free, and totally crushing the trained model," does not hold at the cheapest single-shot budget point.

### Correlated verifier bias and self-preference risk are not addressed head-on

The credibility of the verification signal depends on whether the verifier and the generator are independent. SWE-Bench's candidate pool contains Gemini 3 Flash, while the verifier is precisely the same-family Gemini 2.5 Flash — self-preference bias between same-family models is a known problem in the LLM-as-a-Judge literature, and the paper provides no controlled comparison for it. Repeated evaluation can only cancel independent noise; it is powerless against **correlated** bias such as "the verifier systematically prefers a certain style of writing" — the paper itself admits in the $K$-scaling section that the gain on hard samples diminishes due to "correlated biases," which is effectively conceding the ceiling of this axis.

### The "training-free, general-purpose" framework actually has two hidden dependencies

The first is **token-level logprobs**: the entire method is built on being able to read the distribution at the scoring position, yet the public APIs of frontier models such as GPT-5.5 and Claude Opus 4.7 do not return logprobs. The paper's two-stage workaround (let the closed model produce reasoning, then have an **API verifier that can return logprobs** — both the paper and the code use Gemini 2.5 Flash — read the logprobs to compute the continuous score) recovers most of the gain. Here a common misunderstanding must be corrected: the paper calls this verifier an "open verifier," meaning its **logprobs are openly readable**, not that it is an open-source model — Gemini 2.5 Flash is Google's closed-source model, and the official implementation `fine_grained_reward.py` also accesses it through `VERTEX_API_KEY` via Vertex AI (`genai.Client(vertexai=True, ...)`, with a comment explicitly stating "Only Vertex AI is supported"). So "general-purpose" is conditional: you first need a backend willing to return token-level logprobs (self-hosted vLLM/SGLang, or a commercial API like Vertex), rather than any arbitrary frontier model being pluggable. There is a finer gap at the code level too: `extract_score` takes the expectation over **the valid score tokens among the top-20 logprobs, re-normalized** (`expected = sum(v*p)/total_p`, then mapped to $[0,1]$), not the expectation over "the whole distribution" as the paper's text literally suggests; and the $G=20$ ceiling is fundamentally dictated by the API's `top_logprobs=20` hard limit (the code comment states outright "the OpenAI API caps this at 20"), not a freely chosen scaling axis. When the backend cannot get a logprob, the code by default records that comparison as a 0.5/0.5 tie (`on_error="tie"`, `fine_grained_reward.py:541-613`), and that tie is only written into the current result, never cached to disk (to avoid a transient API error becoming a permanent false tie). This is not entirely "silent": with the default `progress=True`, the first three errors are `print`ed and the progress bar keeps showing a running error count; nor does it "degenerate into randomness" — PPT breaks equal average preferences and final scores deterministically by ascending candidate index (`select_pivots` in `pivot_tournament.py:67-74` uses `i` as the secondary key, and `select_best` in `87-105` uses `-i` as the secondary key). So when the whole batch of comparisons fails, the selection does not become random but deterministically falls back to the lowest-index candidate — an uninformative default value, not a coin flip.

The second is **hand-designed criteria**: criteria decomposition hand-writes sub-criteria for each domain (the paper also lists "whether criteria can be auto-generated" as future work). So "plug-and-play, no per-domain fine-tuning" does not fully hold at the criteria layer.

### VOC establishes a temporal correlation, but not semantic progress

VOC measures the Spearman correlation between "the ordering of the verifier's scores" and "the ordering of the time steps." Here there is a chronological-correlation ≠ semantic-progress trap: a verifier that merely "gives a higher score because a later prefix has more content and looks more complete" would also get a high VOC, without necessarily truly understanding task progress. Two numbers deepen this concern: (1) the VOC of successful and failed trajectories differs by only 0.079 (0.848 vs 0.769), meaning the verifier also gives **failed** trajectories fairly monotonically rising scores — this weakens the paper's claimed use of "early-warning / rollback before the disk is trashed"; (2) the code admits that during offline scoring the verifier can see the ending, so there is information leakage, and the VOC computed with it may overestimate the "prefix-only" progress judgment ability. The paper calls VOC a "calibrated estimator of task progress," a label that is set too strong.

### The RL and cost evidence is thin

The evidence surface for the RL part is very narrow: off-policy only gets 1.8× on the **single** LIBERO `ketchup` task (n=5 seeds), and on-policy only 1.1× on MATH (saving about 10% of optimizer steps). For a claim as large as "dense reward improves RL," single-task, small-gain evidence is insufficient to support a general conclusion. On cost, one MedAgentBench selection ($N=5,\ C=3,\ K=8$, PPT default $k=2$) needs about $12$ query pairs $\times\,3\times8=288$ verifier forward passes, each also reading 20 logprobs — using nearly 300 scorings to buy +3.1 points, whose latency and token cost the paper's main text does not quantify head-on.

### Wrap-up

This is a contribution that is **right in direction, solid in engineering, but packaged by the SOTA narrative as stronger than it actually is**. The "argmax → expectation" insight for eliminating ties is clean and reusable; PPT's ring-based symmetric debiasing is also elegant. But the generality is constrained by logprob access, the SOTA chart mixes metrics, the gains are broadly small, and the progress/RL evidence is thin. Treating it as "a handy, low-effort best-of-N selector" is appropriate; treating it as "conclusive proof of verification as a new scaling law" is overstated.

## One-minute version

- **The verification bottleneck**: the model's difficulty often lies not in "generation" but in "picking the right one out of a pile of candidates." Pooling the trajectories of the entire Terminal-Bench V2 leaderboard and assuming a perfect verifier that always picks correctly, the success rate can reach 98.9% — plenty of potential, stuck at selection.
- **Continuous scoring replaces argmax**: instead of taking the single highest-probability score token, take the expectation over the whole score distribution. In the `query-optimize` task, discrete argmax ties the correct-vs-wrong pair 88/100 times; taking the expectation over the same distribution eliminates all ties and picks correctly 69/100 times.
- **The training-free gain**: using only an off-the-shelf LLM to compute continuous scores, with no additional training, slightly raises the task success rate — Terminal-Bench 83.1%→86.5%, SWE-Bench 76.1%→78.2%.
- **The hidden API dependency (caveat)**: the whole method is built on being able to read the token logprobs at the scoring position, yet the public APIs of frontier models like GPT-5.5 do not return logprobs; when the code cannot get a logprob it records that comparison as a 0.5/0.5 tie (the first few errors are still printed, not cached to disk), and PPT breaks a full tie deterministically by ascending candidate index — it does not randomize, but falls back to an uninformative lowest-index default.
- **The over-packaged SOTA narrative (caveat)**: the gain is in fact not large — SWE-Bench only recovers about 25% of the oracle headroom; and the front-page SOTA bar chart puts RoboRewardBench's **preference accuracy** and the other benchmarks' **task success rate** into the same chart, which is not directly comparable.

## 🔗 Related notes

- [CodingAgentTokenEfficiency](../CodingAgentTokenEfficiency/) — also in the coding-agent test-time tooling space; this note's TurboAgent agentic best-of-N and its token-cost tradeoffs can be read in comparison.
</content>
</invoke>
