---
title: 'Learning Combinatorial Optimization over Graphs'
date: 2020-04-24
permalink: /posts/2020/04/graph_optimization/
tags:
  - combinatorial_optimization
  - graph_theory
---


## Foundational Research

The seminal paper in this field is Dai et al.'s [Learning Combinatorial Optimization over Graphs](https://arxiv.org/pdf/1704.01665.pdf). The work aims to make a general algorithm that can learn to solve all types of different graph optimization problems, since many of these problems have a similar underlying structure.

Since most graph optimization problems use a greedy algorithm, it would be great if ours did too. The only problem is that since we're trying to make an algorithm to solve *general* graph optimization problems, we need to find a way to learn an efficient heuristic for our optimization problem.

In comes ... [Q-learning](https://en.wikipedia.org/wiki/Q-learning)! This model uses a structure2vec representation of a graph embedding, allowing us to create an *evaluation function* $Q$ which is analogous to a heuristic function in a standard greedy approach. This function will be learned through reinforcement learning on many graph instances, without having example optimal solutions. The purpose of the function is to take in a partially constructed solution and a proposed node in the graph that is not already in the partial solution, and return some measure of the quality this node would add to our solution. Then we can use the standard greedy approach and simply pick the node that maximizes our evaluation function.

![Greedy Node Addition](./blog_images/blog_combopt/greedy.png "Greedy Node Addition")

This naturally brings up a few questions:
1. How can we specify to our learning algorithm the constraints of our particular optimization problem?
2. How can we parametrize $Q$ so that it can use these constraints along with input data to find a sufficient evaluator function?
3. Once parametrized, how can we learn our $Q$ function without using data with ground truth labels?



We'll tackle these questions one at a time.

### Specifying a Problem

First, for any graph optimization problem, we can represent an instance of the problem as a graph $G$, which has a vertex set $V$, an edge set $E$, and a weight function $w : (V, V) \to \mathbb{R}$ where the input is a pair of vertices $(u, v) \in E$.

The we can specify an optimization problem by three components, a helper procedure $h$, a cost function $c$, and a termination criterion $t$. The purpose of the helper procedure is to take in a set of vertices $S$ and return a structure that is relevant to the problem. The cost function $c$ utilizes structures produced by $h$ to compute a metric of the quality of a partial solution $S$. The termination criterion $t$ determines when the algorithm can finally stop. Let's go through a couple examples of these functions for combinatorial problems we've already mentioned.

#### Minimum Vertex Cover
The helper function wouldn't need to create any helper structures, since the cost could directly be computed as $c(h(S), G) = -\lvert S \rvert$ which favors smaller vertex covers. The termination $t(h(S), G)$ criterion would require that every vertex is covered, i.e. for every edge $(u, v) \in E$ we have $u \in S$ or $v \in S$.

#### MAXCUT
For MAXCUT we would have $h(S)$ construct $\overline{S} = V / S$ (the complement of $S$) in order to construct a cut-set $C = \{(u, v) | (u, v) \in E, u \in S, v \in \overline{S}\}$ which facilitates computation of the cost of the cut. The cost itself is given by

$$c(h(S), G) = \sum_{(u, v) \in C} w(u, v)$$

which is the sum of all the edge weights which cross the cut. In this problem, there would be no termination criterion; instead the program would run until all every choice of proposed nodes to be added to the solution would not decrease the cut weight.

Now, let's move on to the next pressing question.

### Parametrizing an evaluator function

Using our previously mentioned technique of node embedding $\mu_v$, we can more rigorously specify our update step for the embedding:

$$\mu_v^{(t+1)} \leftarrow \text{relu}\left(\theta_1 x_v + \theta_2 \sum_{u \in \mathcal{N}(v)} \mu_u^{(t)} + \theta_3\sum_{u \in \mathcal{N}(v)} \text{relu}\left(\theta_4 w(v, u)\right)\right)$$

where denoting $p$ as the dimensionality of $\mu_v$, then we would have $\theta_1, \theta_4 \in \mathbb{R}^p$ and $\theta_2, \theta_3 \in \mathbb{R}^{p \times p}$ as the parameters to our embedding update step. Inside the [relu](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) function, the first term has a factor $x_v$, where $x_v = 1$ if $v \in S$, our partial solution set and $x_v = 0$ otherwise; this term accounts for the presence of the vertex in the solution set. The second term aggregates information about the adjacent vertices, which themselves contain information about their neighbors. The last term in the sum directly accounts for the weights of the edges connected to our vertex $v$. It's clear that after running this update procedure for $T$ steps, the embedding $\mu_v^{(T)}$ can contain information about vertices which are separated by up to $T$ edges away since we recursively sum over neighboring vertices' previous embeddings.

Now we can finally define our learned evaluator function $\hat{Q}$ as

$$\hat{Q}(h(S), v, \Theta) = \theta_5^{T}\text{relu}([\theta_6\sum_{u \in V} \mu_u^{(T)}, \theta_7 \mu_v^{(T)}])$$

where $\theta_5 \in \mathbb{R}^{2p}, \theta_6, \theta_7 \in \mathbb{R}^{p\times p}$. The collection of all $7$ $\theta$ variables form the total parametrization of the problem, given by $\Theta = \{\theta_i\}_{i=1}^7$. Inside the $\text{relu}$ layer, this formulation of $\hat{Q}$ concatenates the sum of all vertex embeddings with the embedding for a specific vertex $v$ ($[\cdot, \cdot]$ is the concatenation operator) allowing the evaluator to take into account both the state of the entire graph as well as the proposed vertex to be added to the partial solution. The variable $T$ is a hyperparameter of the algorithm, which as previously discussed, allows each vertex embedding to account for neighbors up to $T$ edges away.

This formulation looks pretty powerful! It contains a lot of aggregate information about the structure of the underlying graph, but it's still not clear how to find ideal parameters for our specific combinatorial problem.

### Q-Learning to the rescue!

Unlike previous work, our structure2vec formulation does not rely on ground truth labels. This means that we should have some reinforcement learning approach to training our parameters. Our reinforcement learning algorithm will have the following framework applicable to our problem formulation:

1. States: A collection $S$ of vertices in our proposed solution, with the goal of finding $\hat{S}$, the terminal solution. The vertices are represented by their embeddings.
2. Actions: The set of vertices $v \in V$ that are not part of our partial solution $S$. These vertices are also represented by their embeddings.
3. Transition: Adding a vertex $v \in V$ to our solution $S$, by changing $x_v$ from 0 to 1.
4. Rewards: We define the reward function in state $S$ of taking action $v$ as

$$r(S, v) = c(h(S \cup \{v \}), G) - c(h(S), G)$$

with $c(h(\varnothing), G) = 0$.

5. Policy: Choose $v$ such that $\pi(v | S) = \text{argmax}_{v' \in \overline{S}} \hat{Q}(h(S), v')$ which essentially chooses the vertex which maximizes the output of the evaluator function, indicating an optimal choice for the next vertex to add to the partial solution.

This framework defines a basic feedback loop that allows the learning algorithm, also called an agent, to learn to make decisions about what actions it should take to optimize its reward. It can change its behavior by changing the parameters in $\Theta$, which forces the algorithm to learn a good vertex embedding as well as a good evaluator function.

On an input graph $G$, the learning algorithm completes an *episode*, which is a complete solution $S$ after which no remaining action would increase the reward to the agent. The learning algorithm minimizes the squared loss by updating the parameters $\Theta$ according to the gradient of

$$[y - \hat{Q}(h(S_t), v_t, \Theta)]^2$$

when in a non-terminal partial solution given by $S_t$ after $t$ time steps. The target value $y$ is given by $\sum_{i=0}^{n-1} r(S_{t+i}, v_{t+i}) + \gamma \max_{v'} \hat{Q}(h(S_{t+n}), v'; \Theta)$. This paradigm is called $n$-step learning, since this target accounts for delayed rewards that the agent may not see immediately after every action. Both $n$ and $\gamma$, the factor by which to weigh the output of the evaluator function, are hyperparameters of this learning algorithm.

However, this formulation does not make clear which samples we train on. Do we train on a single $n$ length sequence from an episode or more? When does training occur? As the algorithm runs, it sores instances of its training steps as a tuple: $(S_t, a_t, R_{t, t+n}, S_{t+n})$ where $S_t$ and $S_{t+n}$ denote the state at time step $t$ and $t+n$ respectively, $a_t$ is the action taken at timestep $t$, and $R_{t, t+n}$ stores the cumulative reward in this timeframe: $R_{t, t+n} = \sum_{i=0}^{n-1} r(S_{t+i}, a_{t+i})$. Then batch stochastic gradient descent updates are performed on the parameters from a random sample of its *experiences* from $E$, a process which is called *experience replay*. After enough training on many episodes/experiences, we should find that the parameters in $\Theta$ reflect important aspects of the combinatorial optimization problem.

Wew, that was a lot of background. This entire paradigm is called Structure2Vec Deep Q-Learning, or S2V-DQN for short. Now that the idea of the research is laid out in front of us ...

### What's so good about this algorithm anyway?

Previous work done in machine learning based combinatorial optimization has been too restrictive for both scalability and application to a variety of problems. This network architecture in this novel method is general enough to be applied to different types of graph problems, and captures more of the combinatorial structure of the underlying graph compared to previous methods which used standard network architectures. Capturing the combinatorial structure by using the graph embedding also comes with the added benefit of scalability to larger graphs; whereas previous approaches needed to tailor the size of their input to match that of the network.

Another benefit comes from the process of $n$-step learning, allowing the evaluation function $Q$ to directly train on the objective function of the optimization problem, while still being able to account for delayed rewards that are not obvious to a standard greedy heuristic. Additionally, improved benefit is derived from the usage of experience replay, creating training samples with respect to the potential benefit of each node rather than the entire solution, allowing for many more sample instances to train on compared to a standard policy gradient approach, which only performs an update based on the entire solution.

![Model Performance](./blog_images/blog_combopt/performance.png "Model Performance")

These benefits are not just theoretical, the results speak for themselves! S2V-DQN was tested on TSP, MAXCUT, and MVC, and was even able to find a more optimal solution than a neural network based algorithm which was specifically designed to solve TSP; let alone being able to find near optimal solution for all 3 problems. The benefits don't stop there; the algorithm was empirically able to find near optimal solutions for graph sizes over 10 times larger than graphs it was trained on, and was able to do so slightly faster compared to other problem-specific algorithms which found similarly optimal solutions.
