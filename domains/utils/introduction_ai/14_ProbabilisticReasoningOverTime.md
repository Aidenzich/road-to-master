# Chapter 14. Probabilistic Reasoning Over Time
- https://moodle.ncku.edu.tw/pluginfile.php/847129/mod_resource/content/1/Chapter%2014%20Probabilistic%20Reasoning%20over%20Time.pdf

## Inference in Temporal Models
We can formulate the basic inference tasks that must be solved:
- **Filtering**
        $$
            P(X_t|e_{1:t})
        $$
    - Compute the most recent state distribution of **last latent variable** at the **end** of the sequence.
    - Input: **All evidence to date.**
    - Output: **Posterior distribution over the date(Belief state)**.
        
- **Prediction**
    $$
        P(X_{t+k}|e_{1:t}), \, k>0
    $$
    - Compute the posterior distribution over the future state from now.
    - Input: **All evidence to date.**
    - Ouput: **Posterior distrubution over the future state**.
- **Smoothing**
    $$
        P(X_k|e_{1:t}), \, 0 \leq k < t
    $$
    - Compute the posterior distribution over the past state.
    - Input: **All evidence to date.**
    - Ouput: **The posterior distribution over the past state.**
- **Most likely explanation**
    $$
        argmax_{x1:t} P(X_{1:t}|e_{1:t})
    $$
    - Find the sequence of states that is most likely to generate the given observations.
    - Input: **Given observations**.
    - Output: **The most likely evidence**.
    - Most likely hood **Viterbi algorithm**.
- **Learning**
    - Models can learn from observations.
    - Dynamic Bayes net Learning can be done as a by-product of inference.
        - inference provides estimates which can be used to update the models.
        - The overall process is Expectation Maximization algorithm (EM)
### Filtering & Prediction
### Smoothing
#### Complexity
- Both the forward & backward recursions is $O(1)$
- Smoothing with respect to evidence $e_{1:t}$ is $O(t)$
- Smoothing with whole process is $O(t^2)$
- **Forward-Backward Algorithm** 
    - Record the results of forward filtering( from 1 to t), then run the backward( from t to 1).
    - reduce Complexity to $O(t)$ by dynamic programming 

## Hidden Markov Models, HMM
- It's a temporal probabilistic model
    - The state of the process is described by a single **discrete random variable**.
        - Assumed to be a Markov Chain with unobservable(hidden) states $X$.
        - Requires that there be an observable process $Y$(outcomes are influenced by the outcomes of $X$).
    - The possible values of the variable are the possible states of the world.
    - $X$ cannot be observed directly, the goal is to learn about X by observing Y.




### Canonical Problems
[inference tasks](##Inference-in-Temporal-Models)
#### Filtering

- This problem can be handled efficiently using the **forward algorithm**.
#### Smoothing
- Find the distribution of last latent somewhere in the **middle** of a sequence.
- The **forward-backward algorithm** is a good method for computing the smoothed values for all hidden state variables.
#### Most likely explanation

## Kalman Filters
- If the variables were discrete, we could model the system with a [Hidden Markov Models](#Hidden-Markov-Models)
- For handling continuous variables, using **Kalman Filtering algorithm**.
- Assumption data distribution is **Gaussian distribution**, and obtain a **linear Gaussian transition model**:
$$
    P(X_{t+\Delta}= x_{t+\Delta}|X_t = x_t, \dot{X_t} = \dot{x_t})
$$
    - Filtering with a linear Gaussian model produces a Gaussian state distribution **for all time**.
### Extended Kalman filter (EKF)
- attempts to overcome nonlinearities in the system being modeled.
- Nonlinear means the transition model **cannot** be described as a matrix multiplication of the state vector.
### Switching Kalman filter
- **Multiple** Kalman filters run in parallel.
- Each filter using a different model of the system.
- A weighted sum of predictions is used, where the weight depends on how well each filter fits the current data.

## Keeping Track of Many Objects
- When methods commit to an incorrect assignment, the prediction at the next time step may be significantly wrong, leading to more incorrect assignments.
- Particle filtering algorithm maintain a large collection of possible current assignments.
### Nearest-neighbor filter
- To choose a single **best** assignment at each time step.
    - Repeatedly chooses the closest pairing of predicted position and observation
        - Given the predicted positions of the objects at the current time step.
        - Adds that pairing to the assignment.
- **Benefit**
    - Works well when: 
        - The objects are well separated in state space.
        - The prediction uncertainty and observation error are small.
#### Hungarian algorithm
- When there is **more uncertainty** as to the correct assignment.
- Choose **the assignment that maximizes** the joint probability of the current observations given the predicted positions.



