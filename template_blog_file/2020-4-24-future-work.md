---
title: 'Learning Combinatorial Optimization over Graphs'
date: 2020-04-24
permalink: /posts/2020/04/graph_optimization/
tags:
  - combinatorial_optimization
  - graph_theory
---
## Future Work

Since the publication of the Dai et al. paper, several potential extensions and improvements have been suggested to further the capabilities of data-driven algorithmic learning for graph optimization. To inform further exploration of this topic, we have outlined some of the identified weaknesses of the original research and relevant supporting papers.

### Eliminating Human Dependencies

One major pitfall that we noticed in the algorithm was the use of the helper function $h$. This function takes in a set of vertices $S$ and returns a structure that is relevant to the problem, and due to the fact that it is both human-designed and problem specific, reliance on it is a major limitation of this model.

In exploring strategies for eliminating this dependency, we came across the paper [Attention, Learn to Solve Routing Problems!](https://arxiv.org/pdf/1803.08475.pdf). The paper modifies aspects of a model architecture known as the [transformer](https://arxiv.org/abs/1706.03762) in order to create a graph embedding. Transformers have been used very successfully in [natural language processing](https://en.wikipedia.org/wiki/Natural_language_processing), which is another problem field that benefits significantly from removing assumptions about the data's structure. The model is then trained using reinforcement learning. Despite the significant improvements in performance over the methods from our original paper, we still have some doubts about the efficacy of this method due to the fact that it was only trained and evaluated on the order of 100 nodes.

This brings us to the next potential limitation for the field of reinforcement learning based combinatorial optimization for graphs.

### Scalability

The instances of the original model trained on 100 nodes generalized decently to 1200 nodes, but these numbers are still nothing in comparison to the scale of real-world networking problems. We searched through the existing literature to assess the scalability of the learning approach to graph optimization, and found [Learning Heuristics over Large Graphs via Deep Reinforcement Learning](https://arxiv.org/pdf/1903.03332.pdf). The paper uses a [graph convolutional neural network (GCN)](https://arxiv.org/pdf/1901.00596.pdf) instead of structure2vec for the graph embedding process, which was then trained using the Deep-Q Learning method that Dai et al. also used. The models were evaluated on the order of millions of nodes, and were able to outperform not only the standard heuristic approaches but also the dominant supervised learning based graph optimization algorithm. However, like our original model, this model also utilized a human-coded helper function for training their model.

### Extensibility

#### Datasets
The original paper used three real-world datasets for examining each of the three graph theory problems.

1. Minimum Vertex Cover (MVC): The authors used the [MemeTracker](http://snap.stanford.edu/netinf/#data) digraph, which describes how phrases move from one website to another.
2. Maximum Cut (MAXCUT): The authors chose 10 instances of Ising spin glass models.
3. Traveling Salesman Problem (TSP): The algorithm was tested on the standard TSPLIB [library](http://elib.zib.de/pub/mp-testdata/tsp/tsplib/tsp/index.html) for instances ranging from 51 to 318 cities.

One potentially interesting avenue of study is examining differently structured datasets to see how we might modify the original algorithm to be more robust to different structures, without the hand-coded helper function.

#### Graph Optimization Problems
In both the original paper and the other papers we examined in our research, the learning algorithms were primarily used on three specific graph problems and their variations: minimum vertex cover (MVC), maximum cut (MAXCUT), and the traveling salesman problem (TSP).

While these are three of the most broadly applicable graph theory problems, we believe that there is value in determining if and how the theory can be extended to other problems in the field i.e. maximum matching.

![Maximum Matching Example](./blog_images/blog_combopt/maxmatch.png "Maximum Matching Example")

Green edges are the set of edges that allow the most vertices from the left side of the graph to be connected to the right.


### Final Thoughts
Based on our research, we believe that reinforcement learning algorithms show a lot of promise for providing significant computational benefits when applied to combinatorial optimization of graphs. Given the number of avenues for real-world applications of graphs, this computational benefit has the potential to convert to genuine financial and infrastructural improvement. Study in this domain is clearly still in the growth phase, but this means there is a lot of opportunity to expand and contribute to its development.
