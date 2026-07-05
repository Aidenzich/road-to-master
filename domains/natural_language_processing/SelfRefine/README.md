# Self-Refine — Research Note
> **English** | [繁體中文](./README.zh-TW.md)

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | SELF-REFINE: Iterative Refinement with Self-Feedback |
| Venue | NeurIPS 2023 |
| Year | 2023 |
| Authors | Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, Peter Clark |
| Official Code | https://selfrefine.info/ |
| Venue Kind | paper |

> This note is based on the arXiv preprint `2303.17651v2`, with the formal version included in NeurIPS 2023; details of the camera-ready version may differ slightly from the preprint. All numbers and citations follow the preprint LaTeX source.

## First Principles

The problem Self-Refine seeks to solve is: even for strong language models (LLMs) like GPT-4, the first generated output is often not the best, especially on tasks with multiple objectives (such as dialogue responses) or objectives that are hard to formalize (such as improving code readability). Past iterative correction mostly required training an additional correction model or relying on an external reward model, requiring large amounts of annotation or supervised data. Self-Refine's core idea is: use "the same" LLM to simultaneously play the three roles of generator, feedback provider, and refiner, without any additional training, without supervised data, and without reinforcement learning, improving the output relying solely on iterative self-feedback and refinement at the test stage.

![Self-Refine high-level flow: the same model M generates an output, produces feedback on its own output, and then refines according to the feedback, with feedback and refinement alternating until the stopping condition holds](imgs/self-refine-overview.png)

The figure above is a high-level illustration of the method: given an input, the model first produces a version of output and sends it back to the same model to obtain feedback, the feedback is then sent back to the model to refine the previous draft, and the two steps of feedback and refinement are executed repeatedly until a stopping condition is triggered, with the entire process involving no human assistance.

The method relies on only three few-shot prompts: the initial generation prompt $p_{gen}$, the feedback prompt $p_{fb}$, and the refinement prompt $p_{refine}$. These three prompts each use a small number of in-context examples to give the same base model the abilities to generate, provide feedback, and refine respectively, so the whole thing is "supervision-free" — the only supervision signal is hidden in the few-shot examples. The first step is initial generation: given input $x$, prompt $p_{gen}$, and model $\mathcal{M}$, the model produces the initial output $y_0$.

$$
y_0 = \mathcal{M}(p_{gen} \| x)
$$

The second step is feedback. The same model $\mathcal{M}$ produces feedback $fb_t$ on its own just-produced output $y_t$ according to the feedback prompt $p_{fb}$. The key is that the feedback must be "actionable and specific": actionable means the feedback should contain a concrete action that is likely to improve the output, and specific means the feedback should point out the exact segment of the output to modify. For example, in code optimization, the feedback would simultaneously point out multiple aspects such as efficiency and readability, and directly name the specific place to change, such as a for loop.

$$
fb_t = \mathcal{M}(p_{fb} \| x \| y_t)
$$

The third step is refinement: the model rewrites the most recent version of the output into a new version according to the feedback. To make the model remember the attempts of the past few rounds and avoid repeating the same mistakes, in the implementation it is not fed only the latest $(y_t, fb_t)$; instead, all past outputs and feedback are concatenated onto the prompt, so that the model learns from past mistakes. Therefore the refinement step is in fact a form that preserves the full history.

$$
y_{t+1} = \mathcal{M}(p_{refine} \| x \| y_0 \| fb_0 \| \cdots \| y_t \| fb_t)
$$

The two steps of feedback and refinement alternate until the stopping condition $\mathrm{stop}(fb_t, t)$ holds: the condition can be reaching a specified number of iteration steps, or extracting a stopping indicator (such as a scalar score) from the feedback. In the paper's experimental setup, each task is iterated at most 4 times, and greedy decoding with temperature 0.7 is used for all settings; to keep the comparison across different models consistent, even for instruction-adept models like ChatGPT and GPT-4, feedback and refinement are uniformly implemented with few-shot prompts.

The paper evaluates Self-Refine on 7 diverse tasks spanning natural language and source-code generation, with GPT-3.5 (`text-davinci-003`), ChatGPT (`gpt-3.5-turbo`), and GPT-4 as base models, and Codex (`code-davinci-002`) additionally tested for code tasks. All tasks use "the same base model but without feedback-refinement iteration" as the control, and the abstract claims that across all evaluated tasks, Self-Refine's outputs are preferred over conventional single-step generation under both human and automatic metrics, with an average absolute improvement in task performance of about 20%. The table below shows the main results of the three main models on the 7 tasks.

| Task | GPT-3.5 Base | GPT-3.5 +SR | ChatGPT Base | ChatGPT +SR | GPT-4 Base | GPT-4 +SR |
|-|-|-|-|-|-|-|
| Sentiment Reversal | 8.8 | 30.4 (↑21.6) | 11.4 | 43.2 (↑31.8) | 3.8 | 36.2 (↑32.4) |
| Dialogue Response | 36.4 | 63.6 (↑27.2) | 40.1 | 59.9 (↑19.8) | 25.4 | 74.6 (↑49.2) |
| Code Optimization | 14.8 | 23.0 (↑8.2) | 23.9 | 27.5 (↑3.6) | 27.3 | 36.0 (↑8.7) |
| Code Readability | 37.4 | 51.3 (↑13.9) | 27.7 | 63.1 (↑35.4) | 27.4 | 56.2 (↑28.8) |
| Math Reasoning (GSM8K) | 64.1 | 64.1 (0) | 74.8 | 75.0 (↑0.2) | 92.9 | 93.1 (↑0.2) |
| Acronym Generation | 41.6 | 56.4 (↑14.8) | 27.2 | 37.2 (↑10.0) | 30.4 | 56.0 (↑25.6) |
| Constrained Generation | 28.0 | 37.0 (↑9.0) | 44.0 | 67.0 (↑23.0) | 15.0 | 45.0 (↑30.0) |

The main results read with two obvious patterns. Preference-type tasks (dialogue response, sentiment reversal, acronym generation) have the largest gains; for example, in the dialogue response task GPT-4's preference score is pulled all the way from 25.4 to 74.6, an absolute improvement of 49.2. In contrast, math reasoning GSM8K has almost no improvement (GPT-4 goes from 92.9 to 93.1), and the paper's own explanation for the cause is that the model struggles to judge whether a reasoning chain has errors — a seemingly fluent reasoning chain fools the model into thinking "everything looks good"; empirically ChatGPT gives "everything looks fine" feedback for 94% of instances, so there is simply nothing to refine.

### One concrete forward refinement: turning a brute-force solution into dynamic programming

Let us walk through a real example from the code optimization task. The input is a program with a brute-force solution that needs to make up a certain amount; the `pie` baseline copies the slow-version logic almost verbatim, using six nested loops to enumerate all coin combinations, only changing the input-reading part, without improving efficiency at all; the initial version roughly looks like this.

```python
# Slow version: six nested loops enumerating
def solve(amount):
  best_price = (amount + 199) // 200 * 380
  for a in range(amount // 200 + 1):
    for c1 in range(amount // 1500 + 1):
      if a*200 + b*300 == amount:
        price = a*380 + b*550
        if price < best_price:
          best_price = price
  return best_price
```

Self-Refine first produces feedback, diagnosing that "this program is very slow because it uses six nested loops to enumerate all coin combinations for payment," and suggests switching to a more efficient approach. The model then rewrites the program into a dynamic programming solution according to this feedback, reducing the time complexity to $\mathcal{O}(amount*coins)$. This corresponds precisely to the 8.7 percentage points by which GPT-4 raises the optimization rate on Code Optimization from 27.3% to 36.0% in the main results. The rewritten program is as follows.

```python
# Fast version: dynamic programming
def solve(amount):
  coins = [200, 300]
  prices = [380, 550]
  dp = [float('inf')] * (amount + 1)
  dp[0] = 0
  for i in range(len(coins)):
    for j in range(coins[i], amount+1):
      dp[j] = min(dp[j], dp[j - coins[i]] + prices[i])
  return dp[amount]
```

Do multiple rounds of iteration actually help? The paper lays out the scores after each round of iteration, and on average output quality improves as the number of iterations increases. The table below shows the round-by-round scores for three tasks ($y_0$ to $y_3$, averaged across three models).

| Task | $y_0$ | $y_1$ | $y_2$ | $y_3$ |
|-|-|-|-|-|
| Code Optimization | 22.0 | 27.0 | 27.9 | 28.8 |
| Sentiment Reversal | 33.9 | 34.9 | 36.1 | 36.8 |
| Constrained Generation | 29.0 | 40.3 | 46.7 | 49.7 |

From the table one can see that the gains are mainly concentrated in the first one or two rounds: Code Optimization climbs from 22.0 to 28.8, Constrained Generation climbs from 29.0 to 49.7, but the marginal improvement of each additional round gradually shrinks, exhibiting clear diminishing returns. The paper also cautions that on tasks with multi-aspect feedback (such as acronym generation) quality does not necessarily increase monotonically, so it switches to assigning numerical scores to each quality aspect in order to select the more balanced output during iteration.

## 🧪 Critical Assessment

### Gains concentrate in low-baseline open-ended tasks, and are almost zero on verifiable tasks

"An LLM's first generation is not optimal and can be improved with self-feedback" is a real and valuable problem, and Self-Refine requires no training at all, applicable to any strong model relying solely on test-stage prompts, and this plug-and-play property makes it very attractive from an engineering standpoint. Note, however, that the paper's most striking gains almost all fall on "preference-type, open-ended" tasks, and the initial baseline scores of such tasks are inherently low (for example, GPT-4's Base on dialogue response is only 25.4), so the room for improvement is naturally large; once a task has clear, verifiable correctness (such as GSM8K), the gain approaches zero. This suggests that what Self-Refine is truly good at is "polishing style and coverage" rather than "correcting factual or logical errors."

### The ablations are solid, but there is a loop risk of GPT-4 being both judge and judged

The ablation design is the strength of this paper: the authors use "Self-Refine feedback vs. generic feedback vs. no feedback" to prove that concrete, actionable feedback is the key (sentiment reversal drops straight to zero without feedback), and use a 1-vs-$k$ sampling comparison to rule out the explanation of "just generating a few more candidates"; these are all more solid than only reporting the main results.

But the metric itself has questionable aspects. Several tasks have no automatic metric and instead use GPT-4 as a proxy for human preference to score, while the outputs being evaluated are themselves often generated and refined by GPT-4 — the judge and the judged are highly homologous, and there is a loop risk of self-preference; although the paper reports a 68–82% correlation with humans, that also means about a quarter to a third of the disagreements are masked by the GPT-4 scoring. Moreover, the human A/B for preference-type tasks is done only on a subset of the outputs, and the representativeness of the sample is not fully explained.

### The novelty lies in "running the loop with pure prompting on a frozen model," not in iterative refinement itself

Iterative refinement of "generate → feedback → refine" already had quite a few prior works (the paper itself cites PEER, Self-Correction, etc.), and Self-Refine's true novelty lies not in the concept but in proving that "without any additional training, a single frozen model, relying purely on few-shot prompts" can run this loop and see effects on strong models. This is a valuable empirical contribution, but it is more like re-verifying and simplifying an existing idea in the era of large models, rather than a brand-new mechanism; understanding it as a strong baseline or a prompting technique is more apt than treating it as a new model.

### Two self-created tasks contributed the largest gains, and website generation is only a qualitative demonstration

What warrants caution is that among the 7 tasks, two (acronym generation and the 20–30-keyword Constrained Generation) are newly created by the authors themselves, and Self-Refine's gains on these two new tasks are especially large — when the evaluation benchmark is defined around the method's own strengths, the persuasiveness of the numbers must be discounted. The authors attribute the high gain of Constrained Generation to "it is easy to miss concepts the first time and one can fill them in afterward," which in fact also shows that the task is inherently especially friendly to iterative remediation. In terms of real-world relevance, the paper uses a website-generation case to demonstrate generalization potential, an alluring direction, but that is only a qualitative demonstration with no quantitative evaluation, belonging to a prospect claim rather than a verified result. Taken together, this paper is not without material weaknesses; its conclusions hold most robustly on "open-ended, low-baseline" tasks, and are quite limited on tasks with objective correctness.

## 🔗 Related notes

<!-- There are currently no relatable notes that can be safely parsed; the heading is kept, without linking for now. -->
