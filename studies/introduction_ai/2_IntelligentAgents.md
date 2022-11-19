# Chapter 2. Intelligent Agents
## Agents
- **Sensors:** Perceiving Environment through sensors
- **Actuators:** Acting upon through Environment by actuators
## Action (agent’s behavior)
$$
Action = f_{agent}(GivenPerceptSequence)
$$
- $f_{agent}$: the mapping from percepts to actions.
## Perfomance Measure
- Percept Sequence is desirable, the agent has performed well.
- Desirability is captured by a **performance measure** that evaluates any given sequence of environment states.
## Rational Agents
- do action that **maximize** it's performance.
- **Rationality** 
    - depends on Performance measure.
    - doesn't require **Omniscience**.
    - not only gathering information but also **Learning**.
    - rational agent should be **Autonomous** - relied on its own percepts rather than the prior knowledge.
## Task Environments
### PEAS
1. ***P**erformance Measure*
    - Measuring the action you make.
2. ***E**nvironment*
    - The environment where agent worked.
3. ***A**ctuator*
    - The tool for agent to interact with environment.
4. ***S**ensors*
    - The tool for agent to perceive the environment.
### Properties of Task Environments
- *Fully/Partially Observable*
    - sensors detect all aspects that are relevant to the choice of action
- *Single/Multi Agent*
    - the performance measure will be influenced by other agent or not.
- *Deterministic/Stochastic*
    - the next state of environment is *completely determined* by the current stateand the agent's action.
- *Episodic/Sequential*
    - next episode depend on the action in previous or not
- *Static/Dynamic*
    - environment can change while an agent is deliberating or not
- *Discrete/Continuous*
    - Has **discrete/continuous** set of percepts and actions
- *Known/Unknown*
    - The agents/designers state of knowledge about the environment

## The Structure of Agents
$$
    agent = architecture + program
$$
- $architectrue$: The computer with physical sensors & actuators for program to run.

### Agent Programs
#### (X) trival agent program
- The table-driven approach's tables could be too huge.
#### Simple Reflex Agents
![Simple Reflex Agents](https://i.imgur.com/maAycNC.png)
- select actions on the basis of current percept, ignoring the rest of the percept history.
- Condition-action rules
- Work only if the correct decision can be made on the basis of only the current percept
    - Another word, **work only if the environment is fully observable**.
    

#### Model-based Reflex Agents
![Model-based-ReflexAgents](https://i.imgur.com/caHoIzQ.png)
- Handle **partial observability**
    - keep track of the part of the world it can’t see now
    - Should maintain some **internal state** that depends on the percept history.
        - Internal State -> How the world evolves -> What my actions do
#### Goal-based Reflex Agents
![Goal-based-ReflexAgents](https://i.imgur.com/9CplljY.png)
- The agent needs goal information that describes situations that are desirable.
- Decision making **different** from condition-action rules
    - Consideration of the future.
        - "What will happen if I do such-and such?"
        - "Will that make me reach the goal?"
#### Utility-based Reflex Agents
![Utility-based-ReflexAgents](https://i.imgur.com/n9G2UUG.png)
- Handle *the uncertainty inherent in stochastic* or *partially observable environments*
- utility function that measures its preferences among states then chooses the action lead to the best expected utility


### Learning Agents
![](https://i.imgur.com/Ii9kZ7C.png)
- **Learning element**
    - responsible for making improvements
    - Using feedback from the **critic** 
        - how the agent is doing.
        - determines how the performance component should be modified to do better in the future. 

- **Performance element**
    - responsible for selecting external actions
- **Problem generator**
    - responsible for suggesting actions that will lead to new and informative experiences. 
    - To suggest the exploratory actions that is suboptimal in the short-run but **might much better actions for the long-run**.
    - A process of modification of the agent's component that bring the closer agreement with the available feedback information.