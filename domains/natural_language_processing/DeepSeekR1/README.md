# DeepSeekR1 — Research Note
> **English** | [繁體中文](./README.zh-TW.md)

## 📇 Academic Context

| Field | Value |
|-|-|
| Title | DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning |
| Venue | Nature (vol. 645, pp. 633–638) |
| Year | 2025 |
| Authors | DeepSeek-AI (Daya Guo, Dejian Yang, Zhihong Shao, Peiyi Wang, Junxiao Song, et al.) |
| Official Code | https://github.com/deepseek-ai/DeepSeek-R1 |
| Venue Kind | paper |

## Core Question and One-Sentence Summary

The question this paper asks is blunt: **does a large language model's reasoning ability necessarily have to be taught through human-annotated reasoning traces (chain-of-thought demonstrations)?** DeepSeek-R1's answer is "no." Starting from DeepSeek-V3-Base, the authors **skip the conventional supervised fine-tuning (SFT) warm-up stage** and train directly with pure reinforcement learning (pure RL), using only an automatically verifiable reward signal of "is the final answer correct or not." With just this, the model spontaneously develops long-chain reasoning behaviors such as self-checking, verification, and backtracking, and on verifiable tasks like AIME, Codeforces, and GPQA it approaches or even matches OpenAI-o1-1217. This "don't teach first, just provide incentives" line of thinking is the biggest divergence between this paper and past RLHF pipelines.

### GRPO: Replacing the Value Network with Group Relative Advantage

The backbone of RL training is Group Relative Policy Optimization (GRPO), which was originally proposed by DeepSeekMath to simplify PPO by removing a value network as large as the policy model. For each question $q$, GRPO samples a whole group of $G$ outputs $\{o_1,\dots,o_G\}$ from the old policy $\pi_{\theta_{old}}$, then uses "the relative magnitude of rewards within the group" as the advantage of each output, and maximizes the following objective:

$$
\mathcal{J}_{GRPO}(\theta)=\mathbb{E}\Big[\frac{1}{G}\sum_{i=1}^{G}\min\Big(\rho_i A_i,\ \mathrm{clip}(\rho_i,1-\epsilon,1+\epsilon)A_i\Big)-\beta\,\mathbb{D}_{KL}(\pi_\theta\|\pi_{ref})\Big],\quad \rho_i=\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}
$$

The key is that the advantage $A_i$ no longer requires a learned critic; instead it is obtained directly by **standardizing the rewards of the same group of $G$ rollouts**, with the formula:

$$
A_i=\frac{r_i-\mathrm{mean}(\{r_1,\dots,r_G\})}{\mathrm{std}(\{r_1,\dots,r_G\})}
$$

This step turns the "absolute reward" into a "ranking relative to peers." The benefit is saving the memory and training cost of a value model; the cost is that the variance of the advantage estimate depends on the within-group samples, so a sufficiently large $G$ (16 in this paper) and a sufficiently large batch are needed.

### Reward Design: Rules Only, No Neural Reward Model

For R1-Zero's reasoning tasks, the reward is entirely rule-based, formed by the equally weighted sum of two parts: an **accuracy reward** (math problems require the answer to be placed in a specified format such as a box, compared by rules; programming problems are thrown into a compiler and run against test cases) and a **format reward** (forcing the model to wrap its thinking process inside `<think>...</think>` tags):

$$
Reward_\text{rule}=Reward_\text{acc}+Reward_\text{format}
$$

The authors deliberately **do not use a neural reward model** (neither outcome-based nor process-based), the reason being that they observed neural reward models are easily exploited by reward hacking under large-scale RL, and are costly to retrain. This "use rules whenever possible, avoid a model" trade-off is the precondition for R1-Zero's stable training.

### R1-Zero: Long-Chain Reasoning Emerging Spontaneously Under Pure RL

R1-Zero's training recipe is almost minimalist in its cleanliness: learning rate 3e-6, KL coefficient 0.001, rollout temperature 1, sampling 16 outputs per problem, with a maximum length of 32,768 tokens before step 8.2k and relaxed to 65,536 afterwards, trained for 10,400 steps in total (about 1.6 epochs), batch size 512. No SFT, no human reasoning examples, no value network—only rule-based rewards.

Under this setup, the most dramatic observation is: the model's average pass@1 on AIME 2024 climbs from an initial 15.6 all the way up to 77.9, and with self-consistency applied (majority vote over 16 answers, cons@16) it further reaches 86.7, surpassing the average level of human contestants. The left half of the figure below is this learning curve.

![R1-Zero's accuracy during training on AIME 2024 (pass@1 and cons@16)](imgs/r1zero_aime.png)

Accompanying the rising accuracy is a **spontaneous growth in response length**: the model was never told to "think longer," yet during training it gradually stretched each response from a few hundred tokens to nearly ten thousand tokens, using more "thinking time" to explore, verify, and switch strategies. The authors call the moment when a certain intermediate version appeared and said in an anthropomorphic tone "Wait, wait. Wait. That's an aha moment" the "aha moment," and they counted that reflective words like "wait" surged 5 to 7 times in frequency after step 8000.

![Growth of R1-Zero's average response length during training](imgs/r1zero_length.png)

### A Concrete GRPO Update Step: 16 Rollouts on an AIME Problem

Let us ground the formulas above in a real update. Take an AIME math problem as $q$; GRPO draws $G=16$ complete solution rollouts. Suppose 6 of them compute the correct answer and 10 are wrong (all are format-compliant, so the format reward is identical for each and cancels out during standardization, leaving only the accuracy reward $r_i\in\{0,1\}$). The statistics of this group are $\mathrm{mean}=6/16=0.375$ and $\mathrm{std}=\sqrt{0.375\times0.625}\approx0.484$. Plugging into the $A_i$ formula yields the table below (this 6/16 split is a concrete number we assume for illustration, not the paper's original data):

| Rollout type | Count | $r_i$ | $A_i=(r_i-0.375)/0.484$ | Effect on policy |
|-|-|-|-|-|
| Correct | 6 | 1 | $+1.29$ | Raise the generation probability of this CoT |
| Wrong | 10 | 0 | $-0.77$ | Lower the generation probability of this CoT |

The intuition is: **the rarer the correct answers within the group, the larger the positive advantage the few correct ones receive** (if all 16 are correct, the advantages are all 0 and no longer drive updates). The model is thereby pushed to produce more long reasoning trajectories that "end up correct," and behaviors like long CoT, reflection, and backtracking are indirectly induced under this pressure of "only correct answers get a relative advantage"—no step ever explicitly teaches it to reflect.

### From R1-Zero to R1: A Four-Stage Pipeline

R1-Zero is strong, but has two practical flaws: poor readability, and mixing of Chinese and English within the same CoT (language mixing). DeepSeek-R1 therefore switches to a multi-stage pipeline to "tame" it: (1) first do SFT with a few thousand conversational **cold-start** data points that are close to human thinking; (2) run a round of reasoning-oriented RL, adding a language consistency reward; (3) use rejection sampling to generate data, mix in non-reasoning data, and do another round of SFT (about 800K samples in total); (4) a final round of RL covering both reasoning and general data, aligning helpfulness and harmlessness. The figure below is the overall comparison of R1, R1-Zero, and various baseline models.

![Comparison of DeepSeek-R1, R1-Zero, and representative models across benchmarks](imgs/benchmark_overview.png)

The effect of each stage (intermediate checkpoints Dev1/Dev2/Dev3) can be seen clearly in the table below: after cold-start, instruction-following metrics rise sharply, but reasoning metrics temporarily regress (AIME drops from R1-Zero's 77.9 to Dev1's 59.0), then recover via reasoning RL; what truly gets lifted by the final stage are user-preference metrics—AlpacaEval 2.0's LC-winrate soars from R1-Zero's 24.7 to the final R1's 87.6, and ArenaHard from 53.6 to 92.3:

| Benchmark (Metric) | R1-Zero | R1-Dev1 | R1-Dev2 | R1-Dev3 | R1 |
|-|-|-|-|-|-|
| AIME 2024 (Pass@1) | 77.9 | 59.0 | 74.0 | 78.1 | 79.8 |
| IF-Eval (Prompt Strict) | 46.6 | 71.7 | 72.0 | 78.1 | 83.3 |
| AlpacaEval2.0 (LC-winrate) | 24.7 | 50.1 | 55.8 | 62.1 | 87.6 |
| ArenaHard (GPT-4-1106) | 53.6 | 77.0 | 73.2 | 75.6 | 92.3 |

In the cross-model horizontal comparison, the final R1 runs almost neck-and-neck with OpenAI-o1-1217 on math (AIME 2024: R1 79.8 vs o1 79.2; MATH-500: 97.3 vs 96.4), its Codeforces competitive-programming rating of 2029 is slightly below o1's 2061, and on AlpacaEval2.0 (87.6) and ArenaHard (92.3), which require following formats and writing, it leads clearly. Where it genuinely loses out is engineering-type tasks: on Aider-Polyglot, R1's 53.3 trails o1's 61.7, and the authors admit the amount of RL on engineering data is still too small.

### Distillation: Moving a Large Model's Reasoning Ability into Small Models

The paper's second contribution is to use the 800K data points produced by R1 as a teacher and directly do SFT (no RL) on small models from the Qwen2.5 / Llama series, yielding a series of R1-Distill models. The effect is striking: the mere 1.5B-parameter DeepSeek-R1-Distill-Qwen-1.5B scores 28.9 on AIME 2024 pass@1, already surpassing GPT-4o-0513's 9.3 and Claude-3.5-Sonnet's 16.0; the 32B version reaches 72.6:

| Model | AIME24 pass@1 | MATH-500 | GPQA Diamond | LiveCodeBench |
|-|-|-|-|-|
| GPT-4o-0513 | 9.3 | 74.6 | 49.9 | 32.9 |
| DeepSeek-R1-Distill-Qwen-1.5B | 28.9 | 83.9 | 33.8 | 16.9 |
| DeepSeek-R1-Distill-Qwen-32B | 72.6 | 94.3 | 62.1 | 57.2 |
| DeepSeek-R1-Distill-Llama-70B | 70.0 | 94.5 | 65.2 | 57.5 |

The authors also ran a key control: doing the same large-scale RL directly on Qwen2.5-32B-Base (yielding Qwen2.5-32B-Zero) achieves only AIME 47.0, far below the distilled version's 72.6. The conclusion is: **when compute is limited, distilling a strong model into a small model is more cost-effective than letting the small model run large-scale RL on its own**; but to break through the ceiling, a stronger base and larger-scale RL are still needed.

### Training Cost

The paper unusually lays out its costs: estimating at $2 per H800 GPU-hour, R1-Zero cost 101K GPU-hours (about $202K), SFT data generation 5K hours, and R1 itself 41K hours (about $82K), for a total of 147K GPU-hours, about $294K. Relative to pretraining that routinely runs into tens of millions of dollars, this "post-training" cost magnitude is an important reason why this paper can be reproduced en masse.

## 🧪 Critical Assessment

### The Premise of This Path: The Task Must Have a Reliable Verifier

"Can reasoning ability grow without relying on human reasoning annotations, using only verifiable rewards" is a genuine question, and the stakes are high: if it holds, the ceiling of reasoning ability is no longer bound by human demonstrations. R1-Zero's "no SFT warm-up" line provides a fairly convincing existence proof—the AIME pass@1 curve from 15.6 to 77.9 is hard to explain away as data contamination or prompt engineering, and the authors specifically performed 10-gram decontamination (filtering out about six million pretraining texts in the math domain alone). But note: the premise of this path is that **the task must have a reliable, automatically decidable verifier**, and the paper itself admits in its limitations that tasks like writing and open-ended QA which "lack a reliable reward" still have to fall back on SFT + a little RL, so the narrative that "pure RL solves reasoning" actually has a clear boundary of applicability.

### The Fairness of Self-Trained "Zero" Control Groups and Cross-Team Comparisons

On the positive side, the experimental honesty of this paper is fairly high: there are cross-model horizontal comparisons (including o1-1217, GPT-4o, Claude-3.5, DeepSeek-V3), a longitudinal breakdown by stages Dev1/2/3, a direct control of distillation vs RL, an ablation of the language consistency reward, and even an honest record of failed attempts at reward hacking and MCTS/PRM. None of these look like cherry-picking. But a few points deserve a question mark: first, the comparison with OpenAI-o1 uses the other party's publicly reported scores, while the sampling setup (pass@1 estimated with 64 samples, temperature 0.6) is decided by DeepSeek's side, so the fairness of cross-team comparison is inherently limited. Second, controls like R1-Zero-Qwen and Qwen2-Math-7B-Zero are all "Zero" versions self-trained by the authors as targets, belonging to a comparison framework of the authors' own definition, which easily makes the advantage of distillation look amplified. Third, the safety evaluation is self-rated by the authors as "moderate level (roughly on par with GPT-4o)," and they frankly admit R1 can be jailbroken to produce dangerous content, a weakness readers easily overlook in a capability-emphasizing narrative.

### The Novelty Lies at the Combination Level, Not in Any Single Component

An honest distinction is needed. GRPO was not invented by this paper (it comes from DeepSeekMath), and rule-based reward, self-consistency, and CoT are all pre-existing techniques. The true novelty of this paper lies not in any single component, but in **the combination of "removing the SFT warm-up, doing outcome-only RL directly on a sufficiently large base, and empirically observing the emergence of long CoT and reflective behavior," together with the evidence that it scales**. This is a substantial empirical contribution, but one should also avoid being carried away by anthropomorphic narratives like the "aha moment"—the so-called "aha moment" is essentially a statistical rise in the frequency of reflective words, and the paper writes it up quite dramatically; readers should treat it as a phenomenon description rather than a mechanistic explanation. In addition, the term "pure RL" is semantically a bit loose: the finally shipped R1 actually uses four stages including two SFTs, and what is truly pure is R1-Zero, not R1.

### Restricted Verifiable Reasoning Is Nearly Solved; Open Tasks and Tool Use Remain Untouched

Within the restricted scope of "verifiable reasoning," the claim roughly holds and has been widely reproduced by the community (open-source weights + MIT license make it one of the most influential open reasoning models of 2025). But zooming out, several limitations directly weaken the claim that "general reasoning is solved": R1 cannot use tools (search, calculator), its structured output is weak, it exhibits language mixing on non-Chinese-and-English queries, few-shot actually degrades its performance, and on real software engineering tasks (SWE, Aider) it shows almost no improvement over V3. In other words, it approaches o1 on "competition problems with a clean verifier," but on "open tasks where the reward is hard to define," this paper neither claims nor demonstrates that the problem is solved. This is both its boundary and precisely the next direction the authors point to. I think the most durable value of this paper is not necessarily the R1 model itself, but that it clearly identifies "verifier + a sufficiently large base + enough RL compute" as the three levers of reasoning ability—this judgment is an inference that was only gradually validated by the community after this paper, not a conclusion the paper has already proven.

## 🔗 Related notes

<!-- No safely parseable related notes yet; heading retained. -->
