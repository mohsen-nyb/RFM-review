# Cheat Sheet — Adapting Tabular Foundation Models for Graphs and Relational Data

This cheat sheet summarizes the papers most relevant to the goal:

> **Take a strong pretrained TFM such as TabICL, keep it frozen, and learn a lightweight relational interface that makes it useful for relational databases.**

---

# 1. Towards Pretraining Text Encoders for TabPFN

## Main idea
Inject a new modality into a frozen TFM by mapping external representations into a small number of native TFM tokens.

## Architecture

```text
Text cell
   ↓
Frozen text encoder
   ↓
Text embedding
   ↓
Small trainable adapter
   ↓
1–k TabPFN-space tokens
   ↓
Frozen TabPFN
   ↓
Prediction
```

$$
e_i = E_{\text{text}}(t_i)
$$

$$
R_i = A_\phi(e_i) \in \mathbb{R}^{k \times d}
$$

$$
\hat y_i =
f_{\theta}^{\text{TabPFN}}(x_i, R_i),
\qquad \theta \text{ frozen}
$$

## Frozen
- TabPFN
- Text encoder

## Trained
- Small adapter/projector

## Key design choices
- 1–10 external tokens
- Early vs late fusion
- Normalization
- Adapter initialization

## Important finding
External information can be injected into a frozen TFM through only a few learned tokens.

## Limitation
The adapter is mostly task-independent; it maps the same external representation the same way regardless of the downstream task.

## Most useful idea for us

Replace:

$$
\text{text} \rightarrow \text{adapter} \rightarrow \text{TFM tokens}
$$

with:

$$
\boxed{
\mathcal{N}_t(v)
\rightarrow
E_{\text{rel}}
\rightarrow
r_{v,1},\dots,r_{v,k}
\rightarrow
\text{Frozen TabICL}
}
$$

**Relevance: 9.5 / 10**

---

# 2. GraphPFN

## Main idea
Turn a pretrained tabular FM into a graph FM by inserting trainable graph-neighborhood attention adapters inside a frozen TFM backbone.

## Backbone
Uses **LimiX**, which already performs:
- feature-wise attention
- sample-wise attention

GraphPFN adds:
- neighbor-wise graph attention

## Architecture

```text
Frozen LimiX block
       ↓
Trainable graph adapter
       ↓
Frozen LimiX block
       ↓
Trainable graph adapter
       ↓
...
```

$$
H^{(\ell+1)}
=
F_\theta^{(\ell)}(H^{(\ell)})
+
A_\phi^{(\ell)}(H^{(\ell)}, G)
$$

with:

$$
\nabla_\theta = 0,
\qquad
\nabla_\phi \neq 0
$$

## Frozen
- Original LimiX layers

## Trained
- Graph-attention adapters
- Auxiliary graph components

## Pretraining
Graph adapters are pretrained across millions of synthetic graph tasks.

Synthetic graphs include:
- communities
- diverse graph structures
- causal feature/label generation
- graph-dependent prediction rules

## Additional objective
Masked graph modeling / edge reconstruction.

## Important result
Pretrained graph adapters substantially outperform randomly initialized graph adapters tuned only downstream.

## Key lesson
A strong TFM does **not** need to be retrained from scratch to gain structural reasoning.

## Limitation for RDBs
GraphPFN assumes ordinary graph structure. RDBs additionally require:
- table types
- relation types
- direction
- temporal validity
- heterogeneous schemas
- multi-hop FK paths

## Most useful idea for us
Learn the **missing structural capability around a frozen TFM**, rather than relearning tabular reasoning itself.

**Relevance: 10 / 10**

---

# 3. OpenRFM

## Main idea
Relational Transformer (RT) alone has weak global ICL because labeled examples may not be reachable in a target entity's relational neighborhood.

OpenRFM adds a TabICL-style batch-level ICL stage.

## Key diagnosis

RT effectively performs relation-level ICL:

$$
\hat y_v
\approx
\sum_{i \in \mathcal{S}(v)}
\alpha(v,i)y_i
$$

where $\mathcal{S}(v)$ contains labeled examples reachable through relational traversal.

If almost no labeled task rows are reachable, RT struggles.

## Architecture

```text
Entity A neighborhood → RT → z_A
Entity B neighborhood → RT → z_B
Entity C neighborhood → RT → z_C
                          ↓
                  TabICL / batch ICL
                          ↓
                     Prediction
```

$$
z_i =
E_{\text{RT}}(\mathcal{N}(v_i))
$$

then:

$$
\hat y_q
=
f_{\text{TabICL}}
\left(
\{(z_i,y_i)\}_{i \in \mathcal{C}},
z_q
\right)
$$

## Key integration experiments
They test:
1. Full TabICL after relational representation
2. Linear bridge directly into TabICL ICL layers
3. Interleaving TabICL ICL blocks and RT blocks

## Important findings
- Full TabICL pathway works better than a simple linear bridge
- Normalization of the relational representation matters
- More complicated interleaving does not consistently help
- Adding TabICL ICL gives one of the largest gains in the paper

## Homophily / heterophily
Homophily:

$$
P(y_u = y_v \mid u \sim v)
$$

is high.

Heterophily:

$$
P(y_u \neq y_v \mid u \sim v)
$$

is high.

Varying this during synthetic pretraining forces the model to infer the current relational rule from labeled context.

## Most useful idea for us

Relational encoding should ideally be **task dependent**:

$$
\boxed{
r_{v,\tau}
=
E_{\text{rel}}
\left(
\mathcal{N}(v),
q_\tau
\right)
}
$$

where $q_\tau$ summarizes the current task from the ICL demonstrations.

**Relevance: 10 / 10**

---

# 4. TFM-Retouche

## Main idea
Adapt a frozen TFM by slightly modifying its input instead of fine-tuning model weights.

## Architecture

$$
x'
=
(1-\alpha)\odot x
+
\alpha\odot \delta_\phi(x)
$$

Then:

$$
x'
\rightarrow
\text{Frozen TabICLv2}
\rightarrow
\hat y
$$

## Frozen
- TabICLv2

## Trained
- Small input-space adapter
- Per-feature residual gate $\alpha$

## Important design
Initialize:

$$
\alpha \approx 0
$$

so initially:

$$
x' \approx x
$$

The model begins nearly identical to vanilla TabICL and gradually learns useful deviations.

## Identity guard
Compare:
- Frozen TFM
- Adapter + frozen TFM

If the adapter hurts validation performance, fall back to the identity transformation.

## Important result
In their experiments, input adaptation is more reliable than:
- LoRA
- full fine-tuning
- BETA-style front-end transformation

## Most useful idea for us

$$
z_v =
E_{\text{rel}}(\mathcal{N}_t(v))
$$

$$
x_v'
=
x_v
+
\alpha P_\phi(z_v)
$$

This is probably the easiest first relational-adapter prototype.

## Limitation
Relational information may not align naturally with the original raw feature dimensions.

**Relevance: 9 / 10**

---

# 5. TuneTables

## Main idea
Compress a whole downstream dataset into a small learned synthetic context for frozen TabPFN.

Normal context:

$$
\mathcal{C}
=
\{(x_i,y_i)\}_{i=1}^{N}
$$

TuneTables learns:

$$
P_\tau
=
\{
(\tilde x_1,\tilde y_1),
\dots,
(\tilde x_K,\tilde y_K)
\}
$$

with:

$$
K \ll N
$$

These synthetic rows are optimized by gradient descent through frozen TabPFN.

## Optimization

$$
\mathcal{L}
=
\ell
\left(
f_{\theta}^{\text{TabPFN}}
(P_\tau,x_q),
y_q
\right)
$$

with:

$$
\nabla_\theta=0
$$

but:

$$
\nabla_{P_\tau}\neq0
$$

## Important clarification
For each new dataset/task:
- initialize synthetic context
- optimize it using the training split
- freeze it
- use it for all test predictions

It is **per-task prompt tuning**, not zero-shot adaptation.

## Main interpretation

$$
\boxed{
\text{Large dataset}
\rightarrow
\text{small learned synthetic context}
}
$$

## Relevance to our task representation idea

Instead of:

$$
q_\tau
=
\frac{1}{N}\sum_i e_i
$$

we could represent a task with several task tokens:

$$
P_\tau
=
[p_1,\dots,p_K]
$$

and amortize the compression:

$$
P_\tau
=
G_\psi
\left(
\{(x_i,y_i)\}
\right)
$$

Then condition the relational adapter:

$$
R_v
=
E_{\text{rel}}
\left(
\mathcal{N}(v),
P_\tau
\right)
$$

## Interesting extension
Use TuneTables-style optimized prompts as **teacher task representations**, then distill them into a task encoder.

**Relevance: 8.5 / 10**

---

# 6. G2T-FM

## Main idea
Convert graph structure into ordinary tabular columns, then use an existing TFM.

## Architecture

$$
\tilde x_v
=
[
x_v;
g_v^{\text{graph}}
]
$$

Then:

$$
\tilde x_v
\rightarrow
\text{TFM}
\rightarrow
\hat y_v
$$

## Main graph features

### Neighborhood Feature Aggregation (NFA)

$$
\operatorname{mean}_{u\in\mathcal{N}(v)} x_{u,j}
$$

$$
\operatorname{min}_{u\in\mathcal{N}(v)} x_{u,j}
$$

$$
\operatorname{max}_{u\in\mathcal{N}(v)} x_{u,j}
$$

### Structural features
- degree
- PageRank
- Laplacian positional features

### PEARL
Produces structural representations using graph propagation over randomized node features.

## Philosophy

$$
\boxed{
\text{Graph}
\rightarrow
\text{fixed graph features}
\rightarrow
\text{TFM}
}
$$

## Analogy to RDBLearn

G2T-FM:

$$
\text{Graph}
\rightarrow
\text{fixed graph aggregations}
\rightarrow
\text{TFM}
$$

RDBLearn:

$$
\text{RDB}
\rightarrow
\text{DFS / SQL aggregations}
\rightarrow
\text{TFM}
$$

## Ablation insight
Removing neighborhood aggregation or structural features hurts performance on many tasks. Structural features give large gains on some datasets.

## Important implication for our experiments
A stronger simple baseline may be:

$$
\boxed{
\text{RDBLearn}
+
\text{cheap topology features}
+
\text{TabICL}
}
$$

not just vanilla RDBLearn.

**Relevance: 9 / 10**

---

# 7. GTAlign

## Full title
**Surprisingly Simple and Effective Multi-Domain Graph Foundation Model through Graph-to-Table Alignment**

## Main idea
Train a graph encoder to produce node representations that a TFM can directly reason over.

## Architecture

$$
G
\rightarrow
E_\phi
\rightarrow
h_v
\rightarrow
\text{TFM}
\rightarrow
\hat y_v
$$

Unlike G2T-FM, the graph representation is learned rather than handcrafted.

## Stage 1 — Graph encoder pretraining
Different graph domains map input features to a shared dimensionality.

A graph encoder is pretrained with graph contrastive learning.

## Stage 2 — Graph-to-TFM alignment
Create pseudo classification tasks from graph communities using Louvain clustering:

$$
\tilde y_v
=
\operatorname{Community}(v)
$$

Then construct ICL episodes:

$$
\{(h_i,\tilde y_i)\}_{i\in\mathcal C}
+
h_q
\rightarrow
\text{TFM}
\rightarrow
\hat{\tilde y}_q
$$

During alignment, both the graph encoder and TFM are optimized.

## Stage 3 — New graph domain
On a target graph:
- freeze the TFM
- adapt only the graph encoder using a few labels
- run TFM ICL over adapted graph representations

## Important result
Target-domain graph-encoder adaptation contributes a large fraction of final performance.

Therefore distinguish:
- task-specific adaptation
- cross-domain transferred adapter
- zero-shot adapter

## Core lesson
A good graph representation is not automatically a good **TFM-compatible representation**. Explicit alignment matters.

## RDB analogue
Possible pseudo relational alignment tasks:
- relation prediction
- entity/table type prediction
- masked column prediction
- temporal next-event prediction

while optimizing only the relational encoder and keeping TabICL frozen.

**Relevance: 9.5 / 10**

---

# Overall Comparison

| Paper | What gets adapted? | TFM frozen? | Structural input? | Main lesson for us |
|---|---|---:|---:|---|
| **Text Adapter** | External signal → latent tokens | Yes | No, text | New information can enter frozen TFM as a few native tokens |
| **GraphPFN** | Internal graph adapters | Base TFM yes during graph pretraining | Graph | Structural reasoning can be added around frozen TFM blocks |
| **OpenRFM** | Relational backbone + TabICL ICL | Partly | RDB | Relational context and tabular ICL are complementary |
| **Retouche** | Input-space residual adapter | Yes | No | Adapt data rather than TFM weights |
| **TuneTables** | Synthetic ICL context | Yes | No | A task can be compressed into a small learned prompt |
| **G2T-FM** | Fixed graph-derived columns | Yes in ICL setting | Graph | Strong structural features may be enough |
| **GTAlign** | Learned graph embedding aligned to TFM | Frozen at target stage | Graph | Learned structural representations should be aligned with TFM space |

---

# Four Main Adaptation Families

## 1. Input-space adapter
Representative: **TFM-Retouche**

$$
x
\rightarrow
x+\Delta x
\rightarrow
\text{Frozen TFM}
$$

**Pros**
- simple
- black-box compatible
- few parameters

**Cons**
- relational information must fit original feature space

---

## 2. External latent tokens
Representative: **TabPFN Text Adapter**

$$
\mathcal{N}(v)
\rightarrow
E_{\text{rel}}
\rightarrow
r_1,\dots,r_K
\rightarrow
\text{Frozen TFM}
$$

**Pros**
- clean relational interface
- variable-size neighborhood → fixed-size representation
- preserves original row

**Cons**
- requires access to a suitable TFM token interface

**Current preferred direction: Yes**

---

## 3. Internal relational adapters
Representative: **GraphPFN**

$$
TFM^{(\ell)}
\rightarrow
A_{\text{rel}}^{(\ell)}
\rightarrow
TFM^{(\ell+1)}
$$

**Pros**
- relational information can influence reasoning at every layer

**Cons**
- invasive
- backbone-specific
- more complexity

---

## 4. Prompt / context adaptation
Representative: **TuneTables**

$$
\mathcal C_\tau
\rightarrow
P_\tau
\rightarrow
\text{Frozen TFM}
$$

Possible relational extension:

$$
P_{v,\tau}
=
G_\phi
\left(
\mathcal N(v),
\mathcal C_\tau
\right)
$$

**Pros**
- naturally aligned with ICL

**Cons**
- per-task prompt optimization is not zero-shot
- synthetic pseudo examples may be harder to train

---

# Recommended Architecture for Our Project

## Step 1 — Task encoder

From labeled ICL examples:

$$
\mathcal C_\tau
=
\{(x_i,y_i)\}_{i=1}^{N}
$$

produce task tokens:

$$
P_\tau
=
G_\psi(\mathcal C_\tau)
\in
\mathbb R^{K_t\times d}
$$

## Step 2 — Relational encoder

For target entity $v$:

$$
R_v
=
E_\phi
\left(
\mathcal N_t(v),
P_\tau
\right)
$$

This lets the task representation determine which relational evidence matters.

## Step 3 — Relational token projection

$$
\tilde R_v
=
\operatorname{LN}
\left(
P_{\text{proj}}(R_v)
\right)
$$

Optionally use a residual gate:

$$
\tilde R_v
=
\alpha_v P_{\text{proj}}(R_v)
$$

with $\alpha_v \approx 0$ at initialization.

## Step 4 — Frozen TabICL

Feed the original row plus relational tokens into:

$$
\boxed{\text{Frozen TabICL}}
$$

for prediction.

---

# Core Experimental Ladder

1. **TabICL** — row features only.
2. **RDBLearn / DFS + TabICL** — strong parameter-free relational baseline.
3. **RDBLearn + cheap graph/topological features + TabICL** — stronger simple structural baseline.
4. **Retouche-style input adapter + TabICL** — generic adaptation without relational information.
5. **DFS + LoRA-TabICL** — PEFT control.
6. **Relational encoder + linear prediction head** — tests whether the encoder itself solves the task.
7. **Task-conditioned relational tokens + Frozen TabICL** — main method.

---

# Main Scientific Questions

## RQ1
Does learned relational representation outperform fixed DFS aggregation?

## RQ2
Does the gain come from relational information or simply extra trainable parameters?

## RQ3
Does the frozen TFM contribute beyond the relational encoder itself?

## RQ4
How many relational tokens are required?

$$
K\in\{1,2,4,8\}
$$

## RQ5
Does task conditioning help?

Compare:

$$
E_{\text{rel}}(\mathcal N(v))
$$

against:

$$
E_{\text{rel}}(\mathcal N(v),P_\tau)
$$

## RQ6
Can one adapter generalize across tasks and databases?

Evaluate:
- per-task training
- multi-task training
- leave-one-database-out
- zero-shot transfer

---

# Most Important Lessons Across the Literature

1. **Do not retrain the strong TFM unless necessary.**
2. **External representations must be aligned with the TFM's expected representation distribution.**
3. **Normalization matters.**
4. **Near-identity / near-zero initialization helps preserve the pretrained prior.**
5. **A strong fixed-feature baseline is surprisingly hard to beat.**
6. **Task-specific adaptation can create large gains, but is weaker evidence of foundation-model behavior.**
7. **Cross-task/domain transfer is essential for a true RFM claim.**
8. **One relational vector may be too restrictive; test multiple latent tokens.**
9. **Task conditioning is likely important because different tasks require different relational evidence.**
10. **Separate ICL support size from relational neighborhood size.**

---

# Current Relevance Ranking

| Paper | Relevance |
|---|---:|
| GraphPFN | **10 / 10** |
| OpenRFM | **10 / 10** |
| TabPFN Text Adapter | **9.5 / 10** |
| GTAlign | **9.5 / 10** |
| TFM-Retouche | **9 / 10** |
| G2T-FM | **9 / 10** |
| TuneTables | **8.5 / 10** |

---

# Papers Still Worth Reviewing

Recommended next order:

1. **LoGIC — Budgeted Context Construction for Node-Level Graph ICL with TFMs**
2. **Parameter-Free Encoders Remain Viable for RDB Foundation Models**
3. **G-Retriever**
4. **Is One Token All It Takes? Graph Pooling Tokens for LLM-based GraphQA**
5. **BETA / TabPFN Unleashed**
6. **On Finetuning Tabular Foundation Models**
7. **Exploring Fine-Tuning for Tabular Foundation Models**
8. **RDB-PFN**
9. **Griffin**
10. **TEA-GLM**
11. **GraphAdapter**
12. **LLaGA**
