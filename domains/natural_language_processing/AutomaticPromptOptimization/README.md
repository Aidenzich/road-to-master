# AutomaticPromptOptimization — Research Note
> **English** | [繁體中文](./README.zh-TW.md)

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | Automatic Prompt Optimization with "Gradient Descent" and Beam Search |
| Venue | EMNLP 2023 |
| Year | 2023 |
| Authors | Reid Pryzant, Dan Iter, Jerry Li, Yin Tat Lee, Chenguang Zhu, Michael Zeng (Microsoft Azure AI) |
| Official Code | https://github.com/microsoft/LMOps/tree/main/prompt_optimization |
| Venue Kind | paper |

> This note is written from the full text and LaTeX source of arXiv preprint `2305.03495v2` (the 2023-10-19 revision); the official venue is EMNLP 2023, and the camera-ready version may differ slightly from the preprint.

## Introduction

The capabilities of large language models (LLMs) depend heavily on the prompt, yet prompts are still written by hand through iterative trial and error. The problem this paper tackles is very concrete: given that you can **only access a black-box LLM through an API** (no access to internal gradients, logits, or weights), how do you automatically rewrite a casually written, vague task description into a precise prompt that improves task performance? Existing approaches either require the model's internal state (soft prompts, AutoPrompt-style token-level tuning) or do "directionless" Monte-Carlo or reinforcement-learning search in the prompt's semantic space; the former does not apply to API users, while the latter is often inefficient or produces human-unreadable strings.

The paper's proposed solution is ProTeGi (Prompt Optimization with Textual Gradients). The core idea is to "translate" numerical gradient descent into a Socratic dialogue in text: run the current prompt on a small batch of training data, collect the examples it gets wrong, then ask an LLM to describe in natural language "why this prompt got these wrong" — this critique is the "textual gradient," which points in the same direction as the numerical gradient's "make performance worse" direction; then ask the LLM to rewrite the prompt in the **opposite semantic direction** of the gradient, equivalent to taking one step along the negative gradient. These rewrite steps are wrapped inside an outer beam-search loop, and best-arm identification (a bandit method) is used to decide which candidate prompts are worth keeping into the next round, thereby cutting down expensive API evaluations.

How does the paper measure success? It runs a preliminary case study on 4 binary-classification tasks: Jailbreak (a self-built set of 452 multilingual jailbreak-detection instances), Ethos (997 English hate-speech instances), Liar (4000 English fake-news instances), and Sarcasm (10000 Arabic sarcasm-detection instances). For each task it randomly draws 50 instances for development and 150 for testing, reporting binary F1 averaged over 3 trials, using the January-2023 version of `gpt-3.5-turbo` by default. Comparison targets include Monte-Carlo (i.e. APE's directionless rewriting), RL (GrIPS/TEMPERA-style phrase-level operations), AutoGPT, and a uniform budget-splitting baseline as the reference for bandit selection. The paper's claim is that ProTeGi beats these state-of-the-art baselines on all four tasks, exceeding MC and RL by 3.9% and 8.2% on average respectively, and in the best case improving over the initial prompt by up to 31%, all while using fewer API calls.

![Overview of one ProTeGi optimization step: an initial jailbreak prompt gets an example wrong on a minibatch (Label True but Prediction False); the LLM-generated "gradient" points out that the prompt assumes attacks will be stated explicitly and ignores indirect or covert techniques, and accordingly rewrites a new prompt that "detects however covert it is," which then passes through bandit selection.](imgs/overview.png)

## First Principles

### The obstacle of discrete optimization, and how "textual gradients" get around it

Once prompt optimization is written as an optimization problem, the goal is to find

$$p^{*} = \arg\max_{p \in \mathcal{L}} \; m(p, \mathcal{D}_{te})$$

where $\mathcal{L}$ is the space of coherent natural language, $m(\cdot)$ is an arbitrary metric function (F1 in this paper), and $\mathcal{D}_{te}$ is the test/development data. The difficulty is that $\mathcal{L}$ is discrete and combinatorially explosive, so you cannot differentiate it directly to do gradient descent. ProTeGi's key move is to replace "differentiation" and "back-propagation" each with an LLM call, using two fixed LLM prompts: one prompt called $\nabla$ generates the gradient, and one prompt called $\delta$ applies the gradient. $\nabla$ always receives the current prompt $p_0$ and its behavior on the minibatch (in particular the errors), and outputs a piece of natural language describing $p_0$'s flaws — this text is the gradient $g$; $\delta$ then takes the gradient $g$ and $p_0$, and edits $p_0$ one step in the opposite semantic direction of $g$, fixing the problem $g$ pointed out.

![A textual dialogue tree mimicking gradient descent: top-left $\nabla$ ("What is wrong with p0?") generates gradients $g_1..g_m$ from $p_0$ and the prediction $\hat{y}$, true value $y$; top-right $\delta$ ("Use g to fix p0") applies each gradient back onto $p_0$ to produce rewritten candidates $p'$, then the mc prompt produces semantically similar paraphrase candidates $p''$, and finally all of them are fed into bandit selection to pick the next round's beam.](imgs/gradient-dialogue-tree.png)

The actual content of $\nabla$ and $\delta$ is a task-agnostic fixed string (all tasks share the same set). Here is the gradient-generating prompt $\nabla$ given in the paper's appendix:

```text
I'm trying to write a zero-shot classifier prompt.

My current prompt is:
"{prompt}"

But this prompt gets the following examples wrong:
{error_string}

give {num_feedbacks} reasons why the prompt could
have gotten these examples wrong.
Wrap each reason with <START> and <END>
```

And the prompt $\delta$ that applies the gradient and produces the rewritten prompt:

```text
I'm trying to write a zero-shot classifier.

My current prompt is:
"{prompt}"

But it gets the following examples wrong:
{error_str}

Based on these examples the problem with this
prompt is that {gradient}

Based on the above information, I wrote
{steps_per_gradient} different improved prompts.
Each prompt is wrapped with <START> and <END>.
```

Notably, there is no explicit learning rate or step size here: the paper chooses to let the LLM decide the rewrite magnitude itself, amounting to a kind of "adaptive step size," and leaves step-size control to future work.

### The beam-search outer loop and the expansion step

A single gradient is only one direction; ProTeGi does not take just one step, but treats these gradient steps as an expansion source for a beam search, repeatedly doing "expand—select." The outer algorithm is as follows:

```text
Algorithm 1: ProTeGi
Require: p0 initial prompt, b beam width, r search depth, m metric function
  B0 <- {p0}
  for i = 1 to r-1:
      C <- {}
      for all p in B_i:
          C <- C ∪ Expand(p)
      B_{i+1} <- Select_b(C, m)
  return argmax_{p in B_r} m(p)
```

`Expand(p)` is where "textual gradient descent" actually lands: first draw a minibatch from the training data, run the current prompt $p$ over it and collect the wrongly answered examples $e$; feed $e$ to $\nabla$ to get gradients $\{g_1,...,g_m\}$; feed each $g_i$ to $\delta$ to rewrite candidates $\{p'_{i1},...,p'_{iq}\}$; finally feed each rewritten candidate to a paraphrase prompt $LLM_{mc}$ to produce semantically similar but differently worded Monte-Carlo successor candidates $p''$, used to explore the local space near the new candidate. This step combines "directional gradient rewriting" with "directionless local paraphrase exploration" within the same expansion.

### Treating beam selection as best-arm identification

After expansion the candidate prompts explode in number, and evaluating each candidate over the full training set is extremely expensive, so the goal of the selection step is: with as few data evaluations as possible, pick out the $b$ best candidates. The paper maps this onto the best-arm identification problem in bandit theory — $n$ candidates are $n$ arms, a candidate's true performance on the data is the arm's hidden value, and "pulling an arm once" is evaluating a prompt once on a random data point. The paper tries four selectors. UCB / UCB-E use an acquisition score to trade off between "exploitation" and "exploration," picking the highest-scoring candidate to evaluate at time step $t$:

$$Q_t(p) + c \sqrt{\frac{\log t}{N_t(p)}}$$

where $Q_t(p)$ is candidate $p$'s current estimated performance, $N_t(p)$ is the total number of times it has been evaluated so far, and $c$ is the exploration coefficient (set to 2.0 in all experiments). The other family is Successive Rejects and the more aggressive Successive Halving, which require no hyperparameter tuning: Successive Rejects splits into $n-1$ stages, each stage evaluating the surviving candidates and eliminating the lowest-scoring one, with the number of evaluation points $n_t$ allocated to each candidate at stage $t$ increasing across stages:

$$n_t = \left\lceil \frac{1}{0.5 + \sum_{i=2}^{T} 1/i} \cdot \frac{B - T}{T + 1 - t} \right\rceil$$

where $B$ is the total query budget; the original text only explicitly defines $B$ near the formula, while $T$ in the formula is not defined at that spot (the algorithm itself separately uses $n$ to denote the number of candidate prompts, so the exact meaning of $T$ has to be inferred from the surrounding context). In theory Successive Rejects is provably optimal for best-arm identification, but the paper's empirical results turn out the opposite (see below).

### A concrete walkthrough of one optimization step

Let's walk through the mechanism above with a real Ethos example. The initial prompt $p_0$ is a casually written engineer's sentence, "Is the following text hate speech?" On the minibatch it got wrong a comment that was sarcastic and indirectly mentioned Muslims: the ground truth is "No (not hate speech)," but $p_0$ predicted "Yes." Feeding this error to $\nabla$, the gradient $g$ it generates is: "This prompt assumes that hate speech must contain explicit, direct wording; but this text is a sarcastic, indirect comment about Muslims, which the model finds harder to recognize as actually not being hate speech." Feeding $g$ to $\delta$, the rewritten ProTeGi candidate $p'$ is: "Does the following text contain language that targets a group of people based on their religion, gender, or other personal characteristics?" — you can see the prompt has gone from a vague question to a more precise annotation instruction, rewritten in a data-driven way.

The arithmetic of the candidate count is also worth laying out, to see where the API calls come from. The paper's settings: minibatch size $|\mathcal{D}_{mini}|=64$, beam width $b=4$, running 6 optimization steps; each time drawing 4 errors into a group, each group generating $m=4$ gradients, each gradient rewritten once (yielding 4 new candidates), and each new candidate producing $p=2$ Monte-Carlo paraphrases, so one group of errors yields roughly $4 + 4\times2 = 12$ candidates. To avoid computational explosion, before bandit selection the paper randomly samples down to 8 successor candidates per parent prompt before selection. The entire process does no hyperparameter search at all; all tasks share the same set of defaults.

### Main evidence

![Curves of F1 versus "evaluation budget per candidate" on the four tasks: ProTeGi (blue) is at the top in all four panels — Jailbreak, Ethos, Liar, Sarcasm — beating MC (orange), RL (green), AutoGPT (red dashed), and the initial prompt p0 (purple dotted); as the evaluation budget increases from about 12 to 50, all methods generally rise.](imgs/main-result.png)

The main results are in the figure above. On average ProTeGi exceeds MC and RL by 3.9% and 8.2% respectively, exceeds the original prompt $p_0$ by 15.3%, and exceeds AutoGPT by 15.2%. There are also notable gaps between the baselines: RL's phrase-level operations barely move the prompt off its starting point on Ethos and Sarcasm, while AutoGPT running 6 rounds of feedback actually makes the starting prompt worse on Jailbreak and Sarcasm. An ablation on beam search itself shows that replacing "flattened enumeration" (No iteration) and "greedy DFS" (Greedy) with beam is better on three tasks: Jailbreak 0.85 vs 0.80/0.82, Liar 0.67 vs 0.63/0.63, Sarcasm 0.88 vs 0.87/0.85. As for the selectors, all approximate best-arm methods beat the uniform budget-splitting baseline, but contrary to theoretical expectation, the UCB-style selectors consistently beat the Successive Rejects-style ones (e.g. on 50-per-prompt Jailbreak, UCB 0.85 vs SR 0.82, SH 0.80). When swapping the underlying model, RLHF-tuned models substantially surpass GPT-3: GPT-4 is highest on Jailbreak (0.88); but on Sarcasm GPT-4 and ChatGPT tie for highest (both 0.86), so it is not a GPT-4 sole win, in contrast to GPT-3's mere 0.73 / 0.55.

## 🧪 Critical Assessment

### The problem is real, but read the "31%" headline number with care

The pain point that "prompts are written by hand through trial and error, and often you only have API access" is real and widespread, and treating black-box, non-parametric, arbitrary-metric as design constraints hits the practical mark. But the "up to 31% improvement" headlined in the abstract is a **single best-case** number, whereas the **average** improvement over the original $p_0$ across tasks reported in the main body is only 15.3%; the average lead over the strongest baseline MC is only 3.9%. Putting the best individual number in the abstract headline while hiding the average in the body is a common way of presenting results; readers should understand the actual benefit in terms of averages like 15.3% / 3.9%, not 31%. Also note that the paper collectively calls these numbers "margin" and does not state at that spot whether they are absolute percentage points or relative percent, so they should be read as the paper's self-reported improvement magnitude, not as precise F1 point differences.

### The evaluation scale is small, and all four tasks are binary classification

The paper itself positions this work as a "limited and preliminary case study." The evidentiary weaknesses are quite concrete: each task uses only 150 test instances, averaged over 3 trials, a sample size so small that a difference in the second decimal place of F1 (e.g. Sarcasm 0.88 vs 0.87) can hardly be said to be statistically meaningful, and the paper reports no significance test or confidence interval for the main results. More fundamentally, all four tasks are binary classification, while the method claims to be applicable to arbitrary tasks such as parsing, chatbot, and summarization — this generality claim is entirely unsupported by experiments and is an unverified extrapolation.

### The self-built Jailbreak benchmark and the inflation risk of self-defined targets

Among the four tasks, the most striking improvements (beam ablation, the largest gains from model swaps) mostly land on Jailbreak, and Jailbreak is precisely the task the authors themselves defined and built — the paper explicitly says "We define jailbreak attack as..." — and this 452-instance multilingual dataset carries human-annotated labels (the paper does not disclose the annotators' identities). Achieving the largest improvement on a self-built, third-party-unverified dataset carries the risk of aligning the method with a self-defined target: we cannot rule out inflation caused by "the initial prompt being especially poor, hence especially large room for improvement," and the paper provides no information on this dataset's annotation consistency or difficulty calibration. In contrast, Ethos, Liar, and Sarcasm are existing public datasets, and their improvement magnitudes (e.g. Ethos moving only within 0.93–0.965 throughout) are much milder.

### The novelty lies in being "directional," but the components are mostly a recombination of existing parts

The method's real novelty is treating "the LLM-generated critique" as a directional gradient to guide discrete search, which is indeed a meaningful step relative to APE's directionless Monte-Carlo or RLPrompt's unreadable outputs. But to be honest, beam search, bandit selectors like UCB / Successive Rejects, and Monte-Carlo paraphrasing are all off-the-shelf parts; the analogy of gradient descent to a textual dialogue is narratively elegant, but in implementation it is "generate a critique with a fixed prompt, then rewrite with a fixed prompt" — two LLM calls, essentially close to iterative rewriting driven by self-critique. It shares the same big family as Self-Refine and Reflexion — self-feedback iterative methods — differing in that the feedback here is explicitly anchored to minibatch errors and wrapped into beam+bandit search-budget control.

### Overfitting and variance: the method "improves then declines"

![Learning curves of F1 versus optimization step (0–7) on the four tasks: Ethos and Sarcasm clearly decline after peaking at around step 3, while Jailbreak and Liar stay roughly flat after about step 3–4. Approximate read-offs from the figure: Ethos falls from about 0.965 at step 3 all the way down to about 0.942 at step 7, and Sarcasm falls from about 0.877 at step 3 to about 0.857 at step 6 (all eyeball read-offs from the figure, not numbers explicitly listed in the paper).](imgs/learning-curves.png)

An observation that undercuts the "the problem is solved" narrative is the learning dynamics themselves: as in the figure above, all tasks reach a peak at around step 3, after which Ethos and Sarcasm even reverse and decline, showing that this process overfits on the training data or gets stuck in a local minimum. The appendix's variance experiment also points out that although ProTeGi is better on average, its variance may be higher (e.g. Ethos's SE 0.003 vs MC's 0.001), which the paper speculates is caused precisely by the semantic directionality of the gradient updates. Note that this variance table uses a different evaluation protocol from the main results — each candidate runs only 6 queries, and each variant is repeated over 12 trials (deliberately lowering the query count to amplify variance), reporting Accuracy and SE rather than the main results' F1, so its numbers should not be directly compared against the main table above. On top of that, the authors state explicitly in the Limitations that even with a small query budget, one optimization run — due to the large number of API calls (including full evaluation of each round's beam candidates) — often runs over 1 hour, which is a real barrier for "real-time or large-scale" applications. Taken together, this is a valuable prototype that introduces directionality into prompt search and whose results are interpretable, but its benefit magnitude, statistical robustness, and cross-task generality all remain at a preliminary stage and should not be over-extrapolated.

## 🔗 Related notes

- [SelfRefine](../SelfRefine/) — belongs to the same family of LLM self-feedback iterative rewriting, repeatedly refining outputs with its own feedback.
- [Reflexion](../Reflexion/) — uses verbalized self-reflection as a feedback signal to improve subsequent attempts, conceptually close to the "textual gradient" feedback mechanism.
