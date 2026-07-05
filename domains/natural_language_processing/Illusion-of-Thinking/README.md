# The Illusion of Thinking — Research Note
> **English** | [繁體中文](./README.zh-TW.md)

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity |
| Venue | unknown |
| Year | 2025 |
| Authors | Parshin Shojaee, Iman Mirzadeh, Keivan Alizadeh, Maxwell Horton, Samy Bengio, Mehrdad Farajtabar (Apple) |
| Official Code | unknown |
| Venue Kind | paper |

> Note: This note is based on the arXiv preprint `2506.06941` (v1). The authors are affiliated with Apple; there is currently no citable source for a formal publication venue (a peer-review venue), so Venue is recorded as `unknown`; the citation count is not speculated on either, in the absence of a reliable API record. Because there is no PDF figure-extraction backend and ar5iv did not render successfully (returning a shell page), this note is text-focused, figures unavailable (no extraction backend).

## First Principles

### What exactly is this paper questioning

Over the past year a batch of so-called "Large Reasoning Models" (LRMs) has appeared — OpenAI o1/o3, DeepSeek-R1, Claude 3.7 Sonnet (Thinking), Gemini Thinking — whose common feature is that, before giving an answer, they first produce a long stretch of "thinking" (a long Chain-of-Thought, including self-reflection). These models score higher on math/programming benchmarks such as MATH and AIME, so the industry commonly treats "able to think" as evidence of progress toward more general intelligence.

This paper's core question is: **the existing evaluation paradigm cannot tell whether these models are actually reasoning.** There are two reasons. First, math and programming benchmarks have a data contamination problem — the problems are very likely already in the training data. The authors themselves observe an anomalous phenomenon: humans perform better on AIME25 than on AIME24 (indicating AIME25 is easier), yet models perform worse on AIME25; this "easier but worse" gap points to the newer benchmark being less contaminated while the older benchmark may have been memorized. Second, these benchmarks only look at whether the final answer is right and completely ignore the structure and quality of the intermediate stretch of "thinking" itself.

The authors' solution is not to propose a new benchmark, but to use four **controllable puzzle environments** as experimental tools: they can finely and monotonically adjust "compositional complexity" while keeping the core logic unchanged, and every step can be precisely judged right or wrong by a simulator — not just the final answer, but every intermediate solution within the "thinking."

### The four puzzles and the mathematical control of complexity

Complexity is controlled by a single integer $N$, and the shortest-solution length of each puzzle is a closed-form function of $N$; this is the yardstick of the whole set of experiments:

- **Tower of Hanoi**: three pegs, $N$ disks of different sizes, with a shortest move count of

$$
M_{\text{Hanoi}}(N) = 2^{N} - 1
$$

  which grows exponentially. Scoring only looks at whether each step is legal and whether the goal state is reached; it does not require optimality.
- **Checker Jumping**: a one-dimensional arrangement of $2N$ red and blue checkers plus one empty slot, with a shortest move count of $(N+1)^2 - 1$ (quadratic growth, e.g. $N=4$ requires 24 steps).
- **River Crossing**: $N$ "actor/agent" pairs must cross the river, boat capacity $k$, with the constraint that no actor can be with another agent while their own agent is absent. $N=2,3$ use $k=2$, and larger use $k=3$.
- **Blocks World**: rearrange blocks from an initial stacking into a specified goal stacking, and only the topmost block of each stack can be moved.

The authors deliberately point out in the text that these four growth rates (exponential, quadratic, near-linear) **cannot be used to compare difficulty across puzzles**, because for the model the true difficulty depends on the training-data distribution rather than asymptotic computational complexity; therefore the analysis mainly tracks behavior "within the same puzzle" as $N$ varies.

### Experimental setup

The main comparison is "thinking/non-thinking pairs of the same backbone": Claude 3.7 Sonnet (extended thinking on/off) and DeepSeek (R1 vs. V3). These are chosen because they emit thinking traces (the o series does not). Claude is given a maximum generation budget of 64k tokens, the DeepSeek series is likewise set to 64k on a local server, and temperature defaults to 1.0; 25 samples are run per puzzle / per complexity level, and responses that do not conform to the specified format are filtered out. When looking purely at final accuracy, o3-mini (medium/high, two settings) is additionally included.

### Three complexity regimes

Comparing thinking models with their non-thinking counterparts under **the same reasoning token budget** (pass@k) yields a three-stage behavior different from math benchmarks:

| Complexity regime | Who wins | Phenomenon |
|-|-|-|
| Low | Non-thinking models | Standard LLM accuracy is comparable or even better, and more token-efficient; thinking models "overthink" |
| Medium | Thinking models | The advantage of long CoT begins to show, and the gap between pairs widens |
| High | Both lose | Both accuracies collapse to near zero; thinking models merely push the collapse point back, and ultimately hit the same wall |

The key is not "thinking models are better," but "good or bad depends on complexity" — in the low-complexity regime, thinking is actually a burden.

### Collapse, and that counter-intuitive token curve

On five thinking models (o3-mini medium/high, DeepSeek-R1, R1-Distill-Qwen-32B, Claude 3.7 Sonnet Thinking), accuracy decreases with $N$ until, beyond a certain model-specific threshold, it **collapses to near zero** (accuracy collapse).

What is truly counter-intuitive is the usage of thinking tokens: the model initially **increases** thinking tokens as complexity rises, but once it approaches that collapse threshold it conversely **reduces** reasoning effort — even though it is still far from the 64k generation ceiling and has budget to spare. This "giving up right when it should try harder" reduction is most pronounced on o3-mini and milder on Claude 3.7 Sonnet Thinking, and is regarded by the authors as an intrinsic inference-time scaling limit.

### What happens inside the "thinking"

Using the simulator, the authors extract every **intermediate solution** from Claude 3.7 Sonnet Thinking's thinking traces, recording its relative position in the trace (normalized to the thinking length using the `cl100k_base` tokenizer) and whether it is correct, obtaining three complexity-related patterns:

- **Easy problems**: the model often "finds the correct solution early on" but keeps exploring wrong solutions afterward — the correct solutions are distributed toward the front and the wrong ones toward the back, which is exactly what the literature calls overthinking, pure waste of compute.
- **Medium problems**: the trend reverses; the model first explores a bunch of wrong solutions and only later hits the correct one.
- **High complexity**: it enters collapse mode, where no correct solution can be found in the thinking at all, and it fixates on some early wrong solution, wasting the entire remaining token budget.

### A sharper blow: even giving it the algorithm does not help

Here is the experiment that best illustrates "thinking is an illusion": the authors directly write **the recursive solution algorithm for Tower of Hanoi into the prompt**, so the model only needs to "execute it" rather than "figure it out itself." In principle, the amount of computation needed to execute a known algorithm is far less than to search for and verify a solution. The result — performance barely improves, and the collapse point is still at almost the same $N$. The authors then worry that this is only because Hanoi's recursive solution is "too famous, already memorized," so they redo it on the less well-known and earlier-collapsing Checker Jumping, reaching a consistent conclusion: **it cannot even copy the steps it was given.** This pushes the problem from "cannot plan" further toward "cannot even reliably execute logical steps step by step."

At the same time, the failure behavior is still **non-monotonic**: on Tower of Hanoi, the model sometimes errs before step 50 at $N=12$, yet can survive over 100 steps at $N=10$; across puzzles it is even more extreme — it can do about 100 steps without error on Tower of Hanoi, yet errs at step 4 on River Crossing $N=3$ (a solution of only 11 steps). This inconsistency points to the model applying a "learned solution distribution" rather than executing a stable algorithm.

### A walkthrough with real numbers (Tower of Hanoi)

Walking through the actual format given to the model in the paper quantifies the phenomena above. The output format is a move list `moves = [[disk id, from peg, to peg], ...]`, with pegs 0-indexed. Take $N=3$ as an example; the initial state is `[[3, 2, 1], [], []]`, and one legal solution is exactly $2^3-1=7$ steps:

```
moves = [[1, 0, 2], [2, 0, 1], [1, 2, 1], [3, 0, 2],
         [1, 1, 0], [2, 1, 2], [1, 0, 2]]
```

Now push $N$ upward and observe the same Claude 3.7 Sonnet Thinking:

- $N=5$: shortest solution $2^5-1 = 31$ steps, the model reaches near-perfect accuracy.
- $N=8$: shortest solution $2^8-1 = 255$ steps. The model collapses. And it is not "erring only at step 255" — **the first error falls at around step 40 (about 15% of the full solution)**, at which point it is still far from the 64k token ceiling. That is, the model is not stuck on output length; it has already gone astray and fixated when the solution has only unfolded 15%.
- $N=10$: shortest solution about $10^3$ steps, the first error at around step 100 (about 10%).

Comparing against DeepSeek-R1 makes it even clearer that "collapse is not sampling noise": on Tower of Hanoi, at temperature 0 (greedy decoding) it collapses at $N=8$; turning on sampling instead keeps 18.2% accuracy at $N=8$ and pushes the collapse to $N=9$. Blocks World is even more dramatic: at temperature 0 it collapses at $N=4$, while turning on sampling keeps 44.1% accuracy at $N=4$ and pushes the collapse to $N=30$. If collapse were caused by "long sequences being error-prone to sample," sampling should make things worse, but in fact it is the opposite — so sampling is not the main cause.

Stringing these numbers together gives the paper's title concrete meaning: a 255-step Tower of Hanoi with completely determined rules and a universally known algorithm, on which the model falls apart after solving 15%, even when the algorithm is fed directly to it. The large stretch of "thinking" it produces looks like reasoning, but cannot be converted into reliable execution of a fixed procedure — this is the "illusion of thinking."

## 🧪 Critical Assessment

### The problem is real, but the "illusion" framing is set stronger than the evidence

"Existing reasoning benchmarks are contaminated, and only look at the final answer" is a solid and important problem, and the AIME24/25 reversal observation is persuasive supporting evidence. The controllable puzzles do indeed fill in the "step-by-step verifiable" piece, and I think this part holds up. But the paper jumps from "the model collapses on these puzzles" to "LRM reasoning is an illusion / lacks generalizable reasoning ability," and this inference is bigger than the evidence in hand. Collapse only proves "it fails on this class of long-horizon, state-tracking-intensive combinatorial tasks," which is not the same as "so-called thinking is generally fake." The title itself is very strong rhetoric.

### The trade-offs in baselines, ablations, and metrics

On the positive side, the thinking/non-thinking same-backbone pairing (Claude on/off, R1/V3) is an appropriate control; the temperature-0 sampling ablation, and moving $N>12$ out of Tower of Hanoi to rule out context-limit concerns, are responsible treatments, and the appendix also directly responds to criticisms such as "is it stuck on context length," "is it caused by sampling," and "is it only a Tower of Hanoi special case." The dubious part is the scoring metric: Tower of Hanoi requires **every step to be correct** to count as success, and a 255-step solution counts as entirely wrong if even one step is wrong. This is a rather harsh all-or-nothing standard for "exponential-length output," and part of the accuracy "collapse" is amplified by this 0/1 metric — if one instead looks at "the number of correct steps before the first error," the picture would be much smoother (the authors do report the first-error position, but the main narrative still uses collapse).

### Is this a new finding, or a repackaging of known phenomena

The overthinking phenomenon and the convergence of thinking and non-thinking on pass@k are both already-cited existing observations; this paper's increment lies in "using controllable complexity to organize these phenomena into three regimes, and quantifying the position of intermediate solutions inside the thinking." This is a valuable reorganization, but one should watch out for a "drawing one's own target" risk: the puzzle environments and complexity yardstick are all defined by the authors, and the evaluation conclusion (that it collapses) happens to fall in the extreme region of this self-defined yardstick. The authors themselves also acknowledge that $N$'s true difficulty to the model depends on the training distribution rather than asymptotic complexity — which is tantamount to admitting that the semantics of the "complexity" x-axis are not clean, and that difficulty across puzzles, or even at different $N$ within the same puzzle, is not necessarily comparable.

### Was the claimed problem really "solved," and its real-world relevance

This is a diagnostic paper that offers no solution, so "solving" is out of the question; its value lies in pointing out limitations. But one must be careful about extrapolating the conclusions to real tasks: in Limitations the authors honestly acknowledge that puzzles are only a narrow slice of reasoning tasks, that most experiments rely on black-box APIs to closed-source models and cannot do mechanism-level analysis, and that the deterministic-simulator assumption that "reasoning can be perfectly verified step by step" does not necessarily hold in open-ended domains. Moreover, the "even giving the algorithm does not improve it" experiment was only done on the two puzzles Tower of Hanoi and Checker Jumping, a rather narrow sample. My reading, therefore, is: this paper successfully **falsifies the stronger claim that "LRMs can reliably execute long-horizon deterministic procedures,"** but is not enough to support the bigger slogan that "reasoning as a whole is an illusion"; as a wake-up call against the industry's overly optimistic narrative, its contribution is real, while reading it as "reasoning models are useless" is an over-extension.

## 🔗 Related notes

- [ChatGPT / InstructGPT (RLHF)](../ChatGPT/)
