# DeepScalper: A Risk-Aware Reinforcement Learning Framework to Capture Fleeting Intraday Trading Opportunities

## Abstract
- `Reinforcement learning (RL)` techniques have shown great success in many challenging quantitative trading tasks, such as portfolio management and algorithmic trading. 
- Especially, `intraday trading` is one of the most profitable and risky tasks because of the `intraday behaviors` of the financial market that reflect billions of rapidly `fluctuating capitals`. 
- However, a vast majority of existing RL methods focus on the relatively low frequency trading scenarios(e.g., day-level) and fail to capture the `fleeting intraday investment opportunities` due to two major challenges: 
    1. How to effectively train profitable RL agents for intraday investment decision-making, which involves `high-dimensional fine-grained action space` 
    2. How to learn meaningful `multi-modality` market representation to understand the intraday behaviors of the financial market at tick-level.

- Motivated by the efficient workflow of professional human intraday traders, we propose **DeepScalper**, a deep reinforcement learning framework for intraday trading to tackle the above challenges.
- Specifically, **DeepScalper** includes four components: 
    1. A `dueling Qnetwork` with `action branching` to deal with the `large action space` of intraday trading for efficient RL optimization 
    2. A novel `reward function` with a `hindsight bonus` to encourage RL agents making trading decisions with a `long-term horizon of the entire trading day` 
    3. An encoder-decoder architecture to learn multi-modality temporal market embedding, which incorporates both `macro-level` and `micro-level` market information; 
    4. A `risk-aware auxiliary task` to maintain a `striking balance` between maximizing profit and minimizing risk. 
Through extensive experiments on real-world market data spanning over three years on six financial futures (2 stock index and 4 treasury bond), 
- we demonstrate that DeepScalper significantly outperforms many state-of-the-art baselines in terms of four financial criteria. 
- Furthermore, we conduct a series of exploratory and ablative studies to analyze the contributions of each component in DeepScalper.

## Overview of the Framework
![](./assets/overview.png)


## Volcabulary
- scalper [s \cal \per] 倒賣（戲票等）的人，黃牛
- multi-modality 多模態
- risk-aware auxiliary [aug \xi \li \ary] task 風險覺察輔助任務
- extensive 廣泛的