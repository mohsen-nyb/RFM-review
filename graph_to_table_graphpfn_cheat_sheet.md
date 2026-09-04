# Graph-to-Table and Graph-PFN Foundation Models — Cheat Sheet

A concise reference for the progression from **graph-to-table adaptation** to **native prior-data fitted graph foundation models**.

Papers covered:

1. **Bringing Graphs to the Table (TAG)**
2. **Turning Tabular Foundation Models into Graph Foundation Models (G2T-FM)**
3. **GraphPFN: A Prior-Data Fitted Graph Foundation Model**
4. **NodePFN: Learning Posterior Predictive Distributions for Node Classification from Synthetic Graph Priors**

---

# 0. Core Question

Can we reuse the strong generalization ability of tabular foundation models for graph learning?

The progression is:

```text
Graph
-> hand-crafted graph-to-table features
-> pretrained TFM
```

then:

```text
Graph
-> compact graph adapter
-> pretrained TFM
```

then:

```text
Graph
-> learned graph reasoning inside a pretrained TFM
-> graph PFN
```

and finally:

```text
Graph
-> node-level PFN architecture
-> global ICL + local message passing
```

The central design problem is:

> **Where should graph structure enter the foundation model, and how much feature information should be preserved before graph reasoning?**

---

# 1. Bringing Graphs to the Table (TAG)

## Main contribution

TAG asks whether a graph can be converted into a table and solved by a strong pretrained **Tabular Foundation Model (TFM)**.

Core pipeline:

```text
Graph G=(A,X)
    |
    v
graph-derived node features
    |
    v
one tabular row per node
    |
    v
TFM
    |
    v
node prediction
```

The TFM itself does **not** directly see the adjacency matrix.

## Node Representation

For each node `v`, TAG constructs:

```text
raw node features
+
1-hop feature aggregation
+
2-hop feature aggregation
+
3-hop feature aggregation
+
4-hop feature aggregation
+
RandomWalkPE
+
LaplacianPE
+
GPSE
```

Feature propagation:

```text
X
A_hat X
A_hat^2 X
A_hat^3 X
A_hat^4 X
```

where:

```text
A_hat = D^-1 A
```

Conceptually:

```text
z_v = [X_v | (A_hat X)_v | (A_hat^2 X)_v | ... | RWPE_v | LapPE_v | GPSE_v]
```

## Why Separate Hop Features?

TAG does not force 1-hop, 2-hop, and 3-hop information into one hidden state. Instead, they remain separate feature groups. This is useful especially under **heterophily**, where blindly smoothing neighbor information can hurt.

## Structural Encoders

### RandomWalkPE
Encodes local/topological random-walk behavior.

### LaplacianPE
Uses graph Laplacian eigenvectors to provide global positional information.

### GPSE
Uses a pretrained structural graph encoder. TAG keeps GPSE frozen.

## Large-Table Handling

The graph-derived table may be too large for one TFM. TAG therefore samples multiple sub-tables with different row and column subsets. A typical setup uses around 10 TFM predictors with approximately 2,500 labeled rows and 400 columns per view.

## LinearGNNs

TAG also uses simple linear predictors over individual graph representations:

```text
W* = argmin_W ||H_L W - Y_L||^2
```

Then:

```text
Y_hat = H W*
```

These are called **LinearGNNs**, but they are much simpler than GCN, GAT, or GraphSAGE.

Full TAG uses roughly:

```text
10 TFM predictors
+
8 LinearGNN predictors
```

followed by ensemble selection.

## Key Strength

TAG shows that:

```text
good graph-to-table encoding
+
strong pretrained TFM
```

can outperform several task-specific GNNs and early GFMs.

## Main Limitation

Graph structure is compressed into fixed statistics **before** task reasoning. Once the graph becomes `z_v`, the TFM no longer knows which exact nodes are connected to which, except through the precomputed encodings.

---

# 2. G2T-FM — Turning Tabular Foundation Models into Graph Foundation Models

## Main contribution

G2T-FM simplifies and systematizes graph-to-table adaptation.

Core pipeline:

```text
Graph
 |
 v
Original Node Features
+
Neighborhood Feature Aggregation
+
Classic Structural Features
+
PEARL
 |
 v
one augmented node table
 |
 v
TFM
```

Compared with TAG:

```text
TAG -> many graph views + many predictors
G2T -> one more compact augmented table
```

## 2.1 Neighborhood Feature Aggregation (NFA)

For each node, G2T uses **1-hop neighbor statistics**.

For numerical features:

```text
mean
min
max
```

For categorical features:

```text
one-hot encode -> average over neighbors
```

This exposes local feature distributions rather than only mean propagation.

## 2.2 Classic Structural Features

G2T adds:

```text
degree
PageRank
Laplacian positional embeddings
```

A reported configuration uses 14 Laplacian dimensions plus degree and PageRank, giving 16 classic structural dimensions.

## 2.3 PEARL

PEARL provides additional structural embeddings:

```text
random node signals
    |
    v
GNN
    |
    v
structure-dependent embeddings
```

A reported implementation uses approximately a 3-layer GCN, hidden size 512, output dimension 16, averaged across 8 random initializations.

In pure ICL mode, the PEARL GNN is not task-trained.

## Final Node Vector

```text
z_v = [x_v | NFA_v | degree_v | PageRank_v | LapPE_v | PEARL_v]
```

Then:

```text
{(z_i, y_i)} + z_query -> TFM -> prediction
```

## ICL Mode

```text
graph features = fixed
TFM weights     = frozen
```

Adaptation happens through labeled context nodes.

## Fine-Tuning Mode

G2T can also fine-tune the TFM and PEARL on the target graph.

## Key Strength

G2T is modular:

```text
graph adapter + any strong TFM
```

It naturally inherits capabilities such as classification and regression from the TFM.

## Main Limitation

Most graph features remain hand-designed, so the model still does not learn from pretraining which graph aggregation operator should be used.

---

# 3. GraphPFN — A Prior-Data Fitted Graph Foundation Model

## Main contribution

GraphPFN moves graph reasoning **inside** the foundation model.

Instead of:

```text
Graph -> fixed graph features -> TFM
```

it does:

```text
Graph -> pretrained TFM + learned graph adapters -> prediction
```

GraphPFN uses **LimiX** as its pretrained tabular backbone.

## 3.1 What Is LimiX?

LimiX is a strong tabular / structured-data foundation model. Conceptually it reasons over both the feature dimension and the sample dimension, keeping a grid of feature tokens.

GraphPFN adds graph-aware attention to this structure.

## 3.2 Three Attention Axes

### 1. Feature Attention
Within one node:

```text
feature1 <-> feature2 <-> feature3
```

### 2. Global Sample Attention
Across examples, subject to the PFN context/query mask.

### 3. Graph Message-Passing Attention
Across connected nodes if `(u,v) in E`.

This gives:

```text
feature reasoning
+
global task inference
+
local graph reasoning
```

## 3.3 Per-Feature Graph Message Passing

GraphPFN does not let every feature token of one node attend to every feature token of a neighbor. Instead:

```text
age_A    <-> age_B
income_A <-> income_B
debt_A   <-> debt_B
```

for connected nodes.

Approximate cost:

```text
O(|E| F d)
```

rather than `O(|E| F^2 d)` for full cross-feature graph attention.

## Why Preserve Feature Identity?

Different features may need different graph propagation behavior. Keeping feature channels separate allows feature-specific relational reasoning.

Trade-off:

```text
more compute
<->
better feature-specific graph reasoning
```

## 3.4 Synthetic Graph Prior

GraphPFN pretrains on synthetic graph tasks that jointly generate:

```text
graph topology
+
node features
+
targets
```

### Topology Generator

Reported prior:

```text
multiple degree-corrected SBMs
+
preferential attachment
```

This creates communities, overlapping communities, heavy-tailed degree distributions, and core-periphery structure.

### Graph-Aware SCM

GraphPFN extends a TabICL-style MLP SCM. At each hidden layer, some latent variables use ordinary MLP transformations while others use graph operators such as:

```text
Mean
Max
Min
GCN
Graph Transformer
```

A random parameter controls how graph-dependent each synthetic task is.

So the prior spans:

```text
almost tabular
-> weak graph dependence
-> strong graph dependence
```

The model can therefore learn whether to trust node features, graph neighbors, or both.

## 3.5 Structural Variables

GraphPFN can also include degree and PageRank inside the synthetic data-generating process, allowing targets to depend directly on structural properties.

## 3.6 Pretraining Strategy

GraphPFN starts from pretrained LimiX.

During graph pretraining:

```text
LimiX backbone = frozen
graph adapters = trainable
```

Reported pretraining scale is about 1.6M synthetic graph datasets.

## 3.7 Losses

GraphPFN uses:

```text
supervised PFN node prediction
+
masked graph modeling
```

Total:

```text
L = L_supervised + 0.1 * L_MGM
```

Masked Graph Modeling removes edges, samples non-edges, and predicts edge vs non-edge.

## 3.8 Random Node Features

GraphPFN appends a small number of random features to each node to break graph symmetries and increase structural expressivity. A reported configuration uses 8 random node features.

## 3.9 Main Strength

GraphPFN replaces hand-engineered graph features with learned graph communication pretrained over synthetic graph tasks.

## 3.10 Important Prior-Ablation Lesson

The sophisticated multi-SBM + preferential-attachment prior is stronger than simplistic random graphs such as Erdős–Rényi, but a simpler degree-corrected SBM can be surprisingly competitive.

Therefore the strongest conclusion is:

```text
graph-aware synthetic pretraining matters
```

rather than:

```text
one exact topology generator is uniquely optimal
```

---

# 4. NodePFN — ICLR 2026

## Main contribution

NodePFN gives a different answer to the GraphPFN design choice.

Instead of keeping one token per feature, it first compresses all features into **one token per node**, then performs graph reasoning at the node level.

This is approximately:

```text
features -> row/node embedding -> graph message passing
```

## 4.1 Input Embedding

For labeled context nodes:

```text
h_v^0 = W_X x_v + W_Y y_v
```

For unlabeled query nodes:

```text
h_v^0 = W_X x_v
```

Thus all node features become one d-dimensional node token. A reported hidden dimension is `d = 512`.

## 4.2 Parallel Two-Branch Layer

Every NodePFN layer combines:

```text
global PFN attention
+
local GCN message passing
```

in parallel.

### Branch 1 — Global Context-Query Attention

For context nodes:

```text
SelfAttention(context)
```

For query nodes:

```text
CrossAttention(query, context)
```

This is the PFN / ICL branch.

### Branch 2 — Local GCN

```text
H_mpnn = GCN(H, A)
```

This branch learns local topology.

### Fusion

```text
H^(l+1) = LayerNorm(H^l + H_attn + H_mpnn)
```

So every layer performs:

```text
global task inference
+
local structural reasoning
```

## 4.3 Early Label Injection

Context labels are included directly in `h_v^0`. Therefore the graph branch can propagate label-conditioned information.

This lets the model learn both:

```text
homophily:   neighbor label ≈ my label
heterophily: neighbor label != my label
```

depending on the task.

## 4.4 Synthetic Prior

NodePFN uses a simpler graph prior than GraphPFN.

Reported topology mixture:

```text
50% Erdős-Rényi
+
50% contextual SBM
```

The contextual SBM spans a range of homophily values, so pretraining includes heterophilic, mixed, and homophilic graphs.

Feature and label relationships are generated using an MLP-based SCM.

## 4.5 Complexity

Node-level graph message passing:

```text
O(L |E| d)
```

Global PFN attention:

```text
O(N^2 d)
```

Total is roughly:

```text
O(L |E| d + N^2 d)
```

The graph branch itself is therefore much cheaper than GraphPFN's per-feature graph attention.

## 4.6 Main Trade-Off vs GraphPFN

### NodePFN

```text
X_v -> h_v -> graph reasoning
```

Advantages:

```text
cheap graph message passing
simple architecture
```

Disadvantage:

```text
feature identities are compressed early
```

### GraphPFN

```text
feature tokens -> graph reasoning per feature
```

Advantages:

```text
feature-specific graph propagation
preserves feature identity
```

Disadvantage:

```text
higher graph-attention cost
```

---

# 5. Homophily and Heterophily

## Homophily

Connected nodes tend to have similar labels/features.

```text
blue -- blue -- blue
```

Formally:

```text
(u,v) in E => y_u ≈ y_v
```

Classic GCN-style smoothing tends to work well.

## Heterophily

Connected nodes tend to have different labels/types.

```text
blue -- red -- blue -- red
```

Examples:

```text
buyer -- seller
student -- professor
customer -- merchant
fraudster -- victim
```

Heterophily does **not** mean graph structure is useless. It means the useful relation may be complementary/opposite rather than similarity-based.

---

# 6. TAG vs G2T-FM vs GraphPFN vs NodePFN

| Model | Graph Representation | Graph Pretraining | Graph Reasoning Location | Feature Identity During Graph Reasoning | Main Adaptation | Main Strength | Main Limitation |
|---|---|---:|---|---:|---|---|---|
| **TAG** | `X, AX, A²X, ... + PE` | No | Before TFM | Partially preserved as separate derived columns | TFM ICL | Very strong graph-to-table baseline | Heavy feature engineering + ensemble |
| **G2T-FM** | NFA + degree/PageRank/LapPE/PEARL | No | Before TFM | Mostly compressed into augmented columns | TFM ICL / FT | Simple modular graph adapter | Mostly hand-designed graph features |
| **GraphPFN** | Native adjacency + feature-token grid | **Yes** | Inside pretrained TFM | **Yes** | Graph-aware PFN ICL / FT | Learned graph reasoning + strong TFM backbone | More expensive per-feature graph attention |
| **NodePFN** | One node token + adjacency | **Yes** | Parallel with PFN attention | **No — compressed early** | Graph PFN ICL | Efficient node-level message passing | Early feature compression; weaker feature flexibility |

---

# 7. Evolution in One Line

```text
TAG
-> expose graph structure as many tabular features
```

```text
G2T-FM
-> build a cleaner graph-to-table adapter
```

```text
GraphPFN
-> learn graph communication inside a pretrained TFM
```

```text
NodePFN
-> learn graph PFN inference directly with one token per node
```

---

# 8. Three Graph-Reasoning Strategies

## Strategy 1 — Fixed Graph Features

Used by TAG and G2T-FM.

```text
Graph -> hand-designed structural features -> TFM
```

Examples:

```text
A X
A^2 X
degree
PageRank
LapPE
RandomWalkPE
```

Advantages:

```text
simple
modular
can reuse any TFM
```

Disadvantages:

```text
structure is compressed before task reasoning
human-selected graph operators
```

## Strategy 2 — Per-Feature Graph Attention

Used by GraphPFN.

```text
feature token grid + adjacency-masked attention
```

Approximate graph cost:

```text
O(E F d)
```

Advantages:

```text
preserves feature identity
feature-specific propagation
```

Disadvantage:

```text
more expensive
```

## Strategy 3 — Row/Node-Level Graph Reasoning

Used by NodePFN.

```text
all node features -> one node embedding -> graph message passing
```

Approximate graph cost:

```text
O(E d)
```

Advantages:

```text
efficient
simple
```

Disadvantage:

```text
feature information is compressed early
```

---

# 9. Possible Hybrid Design

A natural middle ground is:

```text
F feature tokens
->
K learned summary tokens per node
->
graph attention
```

with:

```text
1 < K << F
```

Then graph cost becomes:

```text
O(E K d)
```

This creates a spectrum:

```text
K = 1     -> NodePFN-like
1 < K < F -> hybrid
K = F     -> GraphPFN-like
```

This may preserve more feature-specific information than NodePFN while remaining cheaper than GraphPFN.

---

# 10. Another Possible Hybrid: Shared Row-Level Routing

Compute graph attention weights once from row summaries:

```text
alpha_vu = Attention(row_v, row_u)
```

then reuse them across feature channels:

```text
h_vj' = sum_{u in N(v)} alpha_vu h_uj
```

This gives:

```text
row-level routing
+
feature-level values
```

which may reduce cost while preserving feature-specific representations.

---

# 11. Important Complexity Comparison

Let:

```text
N = number of nodes
E = number of graph edges
F = number of feature tokens
d = hidden dimension
```

### Full cross-feature graph attention

```text
O(E F^2 d)
```

### GraphPFN-style per-feature graph attention

```text
O(E F d)
```

### NodePFN-style row-level graph reasoning

```text
O(E d)
```

### Global PFN sample attention

Still approximately:

```text
O(N^2 d)
```

So for very large graphs, **global ICL attention may become the dominant bottleneck**.

---

# 12. Key Design Trade-Off

The most important architectural question is:

> **When should we compress feature information relative to graph reasoning?**

### Early compression

```text
features -> row token -> graph
```

Example: NodePFN.

Benefit:

```text
cheap
```

Risk:

```text
task-relevant feature details may be lost
```

### Late compression

```text
feature tokens -> graph reasoning -> later aggregation
```

Example: GraphPFN.

Benefit:

```text
feature-specific graph learning
```

Risk:

```text
more compute
```

This mirrors the broader tabular-FM debate:

```text
task-aware compression
vs
task-agnostic compression
vs
preserving feature identity
```

---

# 13. Most Important Takeaways

1. **TAG shows that graph learning can be surprisingly strong after converting graph structure into ordinary table columns.**

2. **G2T-FM simplifies the graph-to-table strategy** using NFA + degree + PageRank + LapPE + PEARL.

3. Both TAG and G2T rely primarily on fixed graph extraction + pretrained tabular intelligence rather than native graph pretraining.

4. **GraphPFN moves graph reasoning inside the foundation model.**

5. GraphPFN uses three reasoning channels:

```text
feature attention
+
global PFN sample attention
+
adjacency-masked graph attention
```

6. GraphPFN preserves feature identity during graph message passing.

7. GraphPFN's synthetic prior teaches how much graph structure should matter.

8. **NodePFN compresses all features into one node token before graph reasoning.**

9. NodePFN combines global ICL + local GCN in parallel.

10. NodePFN is much cheaper for graph message passing than GraphPFN.

11. **Homophily** means connected nodes tend to be similar.

12. **Heterophily** means connected nodes tend to differ.

13. Heterophily does not mean graph structure is useless; it means simple smoothing may be the wrong inductive bias.

14. Synthetic graph priors should vary community structure, degree patterns, homophily, heterophily, and graph dependence of labels.

15. The broad evolution is:

```text
graph feature engineering
-> graph adapters
-> graph-aware pretraining
-> learned graph inference algorithms
```

---

# 14. Promising Research Directions

## 1. Learned Graph Compression

Instead of `F` feature tokens or `1` node token, learn `K` graph-relevant summary tokens per node.

## 2. Task-Adaptive Graph Depth

Different tasks may need 0-hop, 1-hop, 2-hop, or many-hop reasoning. A foundation model could learn how deep to propagate.

## 3. Task-Adaptive Graph Breadth

Rather than aggregating all neighbors, retrieve/select useful neighbors using attention, uncertainty, value of information, or semantic similarity.

## 4. Retrieval + Graph ICL

Large graphs make `O(N^2)` global PFN attention expensive.

```text
large graph
-> retrieve useful support nodes
-> local/global PFN inference
```

## 5. Subquadratic Global ICL

Possible replacements for full sample attention:

```text
sparse attention
inducing tokens
prototypes
hierarchical memory
state-space models
retrieval
```

## 6. Richer Synthetic Graph Priors

Synthetic tasks could include:

```text
heterogeneous node types
edge types
temporal graphs
dynamic topology
higher-order motifs
label delays
multiple graphs
multi-relational structure
```

## 7. Synthetic + Real Graph Pretraining

Combine synthetic breadth with real graph realism while avoiding benchmark leakage.

## 8. Native Link / Edge / Graph-Level PFNs

Future models should unify:

```text
node classification
node regression
edge prediction
link prediction
graph classification
graph regression
```

inside one graph foundation model.

## 9. Feature Semantics + Graph Structure

Future models could combine:

```text
node values
+
feature names
+
node-type semantics
+
edge-type semantics
+
natural-language descriptions
+
graph topology
```

## 10. Adaptive Graph-Feature Routing

Not every feature needs graph propagation. Learn gates:

```text
g_j in [0,1]
```

and:

```text
h_vj' = h_vj + g_j * GraphMessage_j
```

so the model can skip unnecessary relational computation.

---

# 15. Final Mental Model

```text
TAG
Graph -> many fixed graph features -> TFM
```

```text
G2T-FM
Graph -> compact fixed graph adapter -> TFM
```

```text
GraphPFN
Graph -> learned per-feature graph adapters inside TFM
```

```text
NodePFN
Graph -> one node token -> global PFN attention + local GCN
```

And the central open question is:

> **How much graph structure and feature identity should be preserved before a foundation model compresses information for inference?**
