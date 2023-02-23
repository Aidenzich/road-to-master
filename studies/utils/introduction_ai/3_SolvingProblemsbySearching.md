# Chapter 3. Solving Problems by Searching
## Problem Solving Agent
- One kind of [Goal-based agents](https://hackmd.io/PiAKU9vjQUKrimFi50szNg?view####Goal-based-Reflex-Agents)
- limit ourselves to the simplest kind of task environment: the solution is always a **fixed sequence of actions.**

## Well-Defined Problems and Solution
- States
- Initial state
- Actions
- Transition model
- Goal test 
- *Path cost*
- Solution
    - The action sequence that leads state from initial state to goal state

## Searching for Solutions
### Search Tree
- Initial state as root
- The **branches** are actions and the nodes correspond to states in the state space of the problem
### Measuring Problem-Solving Performance
- Completeness
    - Is the algorithm guaranteed to find a solution?
- Optimality
    - Does the strategy find the optimal solution?
- Time Complexity
    - How long does it take to find a solution?
- Space Complexity
    - How much memory is needed to perform the search?

### Uninformed Search Strategies (Blind search)
- No additional information like goals,etc.
- All algorithm can do is 
    - generate successors.
    - distinguish a goal state from a non-goal state.
- The opposite algorithm is **informed search(heuristic search)**
---
#### Breadth-First Search (BFS)
![BFS](https://i.imgur.com/svKWqy5.gif)
- The root node is expanded first, then all the successors of the root node are expanded next, then their successors, and so on..
- Space Complexity
    $$
        b + b^2 + b^3 +...+ b^d = O(b^d)
    $$
    - $d$: depth 
    - $b$: nodes number
- Time Complexity
    $$
        O(|b|+|d|)
    $$
- Only blind search, resulting in inefficiency.
---
#### Uniform-Cost Search (UCS)
![UCS](https://i.imgur.com/KFPbgfF.gif)

- Expands the node n with the lowest path cost g(n).
- 2 main differences to BFS
    - Applied goal test to a node when selected for expansion. 
    - A better path is found to a node currently on the frontier.
- Optimal in general, expands nodes in order of their optimal path cost.
- guided by path costs rather than depths.
- Complexity
    - Worst case can be greater than $b^d$.
        $$
            O(b^{1+[C^*/\epsilon]})
        $$
        - $C^*$: The cost of the optimal solution.
        - $\epsilon$: Assume that every action costs at least.
    - If all step cost the same, Uniform-Cost Search is similar to Breadth-First Search.
---
#### Depth-First Search (DFS)
![](https://i.imgur.com/lIARwKs.gif)
- Expands the deepest node in the current
- Time Complexity
    $$ O(b^m) $$ 
    - $m$ the maximum depth of any node(may be larger than $d$).
    - May cause more time complexity than BFS
- Space Complexity
    $$
        O(bm)
    $$
    - smaller than BFS
#### Depth-limited Search (Improve of DFS)
- with a **predetermined depth limit**.
- Incompleteness
- Nonoptimal
- Depth limits can be based on knowledge of the problem.
    - leads to a more efficient depth-limited search
    - Case by case 
---
#### Iterative deepening DFS (IDDFS)
![](https://i.imgur.com/r9pSOSo.png)
- Doing the Depth-limited search recursively by  gradually increasing the limit(0,1\~...) until a goal is found.
- Time Complexity
    $$
        O(b^d)
    $$
- Space Complexity
    $$
        O(bd)
    $$
- Benefits of DFS and BFS
    - IDDFS is optimal like Breadth-First Search, but uses much less memory.
    - Like DFS, its memory requirements are modest: $O(bd)$.
    - Like BFS, 
        - Completeness when the branching factor is finite.
        - Optimality when the path cost is a nondecreasing function of the depth.
- Seem wasteful but is not too costly
    - In DFS, most of cost is in the deepest layer(Most of the nodes are in the bottom level)
---
#### Bidirectional Search
![](https://i.imgur.com/a7SwsKo.png)
- Replace the *goal test* with a $check$ that the frontiers of the 2 searches intersect or not.
    - if they do, a solution is found.
- Run forward and backward
    $$ b^{d/2} + b^{d/2} <= b^d $$
- It's difficult to use when facing an abstract goal.
- It's a graph search algorithm
---
### Informed (Heuristic) Search Strategies
- The main difference to [*uninformed search*](###Uninformed-Search-Strategies-(Blind-search)) is the **heuristic function** $h(n)$.
    - heuristic: use problem-specific knowledge beyond the definition of the problem itself.
#### Best-First Search
- Evaluation function $f(n)$
- $h(n)$ = estimated cost of the cheapest path from the state at node n to a goal state. 
---
#### Greedy best-first search
- Use a table  $h_SLD$ to **lead to a solution quickly**.
    $$
        f(n) = h(n)
    $$
    - Expand the node that is closest to the goal
- lack of [*completeness*](###Measuring-Problem-Solving-Performance)
- with a good $h_SLD$, may have lower complexity.


---
#### $A*$ Search
- Most widely known form of *best-first search*
- The estimated cost of the cheapest solution
    $$
        f(n) = g(n) + h(n)
    $$
    - $g(n)$ 
        - the [*path cost*](##Well-Defined-Problems-and-Solution) to reach the node n (i to n).
        - gives the path cost from the start node to node n.
    - $h(n)$ 
        - the cost to get from the node to the goal (n to goals).
        - the estimated cost of the cheapest path from n to the goal.
- Conditions to [*optimality*](###Measuring-Problem-Solving-Performance):
    - *Admissibility*
        - admissible $h(n)$ never overstimates the cost to reach the goal
    - *Consistency(monotonicity)*
        -  required only for applications of A* to graph search.
        - ***Triangle Inequality***
            $$
                h(n) <= c(n, a, n') + h(n')
            $$
            - $n$: every node
            - $a$: any actions
            - $n'$: every successor generated by $a$
            
## Conclusion
| Algorithm\Measure | Completeness | Optimality | Time Complexity | Space Complexity |
|---|---|---|---|---|


