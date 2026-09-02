# Relational Foundation Models — Architecture & ICL Cheat Sheet

This cheat sheet covers the second major block of relational foundation-model papers:

```text
KumoRFM
-> KumoRFM-2
-> OpenRFM
-> Griffin
-> RDBLearn / JUICE
```

The central question is:

```text
How should a model adapt to a new relational database and task?
```

The major competing answers are:

```text
1. Native relational ICL
2. Relational representation + explicit cross-sample ICL
3. Pretrained graph model + fine-tuning
4. Parameter-free relational aggregation + tabular FM
```

---

# 1. KumoRFM

## Core idea

Treat each relational prediction example as:

```text
(relational subgraph, label)
```

instead of:

```text
(row, label)
```

So TabPFN-style ICL:

```text
(x_i, y_i)
```

becomes relational ICL:

```text
(G_i, y_i)
```

where `G_i` is the temporally valid relational neighborhood around the target entity.

## Pipeline

```text
Relational DB
    |
    v
historical relational subgraphs
    |
    v
table-invariant row encoder
    |
    v
Relational Graph Transformer
    |
    v
one vector z_i per relational example
    |
    v
ICL Transformer across (z_i, y_i)
    |
    v
prediction
```

In symbols:

```text
G_i -> z_i
```

then:

```text
{(z_i, y_i)} + z_* -> y_*
```

## Key points

- Table-invariant row encoder handles heterogeneous schemas and datatypes.
- Rows become graph nodes and PK/FK relationships become edges.
- Relational Graph Transformer uses structural information such as node type, hop distance, relative time, and local graph structure.
- A separate example-level ICL module learns across `(z_i, y_i)` pairs.
- Predictive Query Language can generate historical labeled relational examples automatically.

## Main limitation

Task labels enter relatively late:

```text
G_i
-> generic relational representation z_i
-> ICL with y_i
```

This may compress away task-relevant information before the model knows the task well.

---

# 2. KumoRFM-2

## Core idea

Inject task information earlier and jointly reason across four axes:

```text
columns
+
rows
+
foreign-key relations
+
context samples
```

## Four attention axes

| Axis | What attends to what? | Purpose |
|---|---|---|
| **Column attention** | Same feature across rows | Learn distributions/statistics |
| **Row attention** | Features within one row | Learn feature interactions |
| **FK / graph attention** | PK/FK-connected rows | Learn relational structure |
| **Cross-sample attention** | Labeled examples | Perform ICL |

## Simplified workflow

```text
raw cells
   |
   v
semantic cell embeddings
   |
   v
inject task-label/context information early
   |
   v
COLUMN ATTENTION
   |
   v
ROW ATTENTION
   |
   v
repeat
   |
   v
task-conditioned row embeddings
   |
   v
FK / GRAPH ATTENTION
   |
   v
CROSS-SAMPLE ATTENTION
   |
   v
prediction
```

## Tensor intuition

If table `T` has:

```text
N_T rows
C_T columns
hidden dimension d
```

then after cell encoding:

```text
N_T x C_T x d
```

Column and row attention preserve this shape.

After row compression:

```text
N_T x d
```

Rows across tables become relational node tokens.

FK attention works over:

```text
N_total_rows x d
```

and cross-sample attention works over:

```text
N_context_examples x d
```

## Why early task conditioning matters

Different tasks need different information.

For fraud:

```text
device + recent transactions
```

may matter.

For churn:

```text
purchase frequency + support history
```

may matter.

KumoRFM-2 lets the task context influence feature extraction before information is compressed.

## Relation to TabPFN

TabPFN reasons mainly across:

```text
features
+
samples
```

KumoRFM-2 extends this to:

```text
columns
+
rows
+
relations
+
samples
```

A useful mental model:

```text
Tabular ICL -> 2D
Relational ICL -> 4D
```

## Context selection

KumoRFM-2 combines:

```text
local / self-history context
+
global / recent context
```

and can include lagged targets.

This exposes a major RFM problem:

```text
Which support examples should be retrieved?
```

## Scaling insight

Large RDB support is fundamentally:

```text
large DB
-> retrieve relevant rows/subgraphs/examples
-> run FM on manageable context
```

So:

```text
database size != Transformer context size
```

## Main limitations

- support selection remains a bottleneck;
- exact pretrained corpus and some low-level implementation details are not fully disclosed;
- more relational hops are not always better;
- retrieval quality strongly affects performance.

---

# 3. OpenRFM

## Core idea

Explain why Relational Transformer-style ICL works or fails.

Main diagnosis:

```text
relation-level ICL can fail because useful labels may not be relationally reachable
```

Then fix it by combining:

```text
relational ICL
+
explicit batch/sample-level ICL
```

## Relation-level ICL

RT-style inference samples a relational neighborhood around the query.

If that neighborhood reaches another task row with a known label, the query can use it.

Thus support is selected by:

```text
relational reachability
```

rather than explicit support retrieval.

## Label reachability

Conceptually:

```text
rho = probability(query context reaches another labeled task row)
```

High `rho` means the relational context contains useful labels.

Low `rho` means the model may see almost no support labels.

RT performance correlates strongly with this quantity.

## Kernel-regression interpretation

RT can be approximated as:

```text
y_hat
≈
sum_i alpha(query, support_i) * y_i
```

where attention behaves like a learned kernel.

If only one or zero labels are reachable, this mechanism becomes weak.

## OpenRFM solution: dual-stage ICL

For each task example:

```text
relational neighborhood
-> RT-style relational encoder
-> z_i
```

Then:

```text
(z_1, y_1)
(z_2, y_2)
...
z_*
```

are passed through explicit batch-level ICL.

So OpenRFM combines:

```text
relational representation
+
Tabular-style cross-sample ICL
```

## TabICL connection

OpenRFM reuses the ICL-learning component from a pretrained tabular FM:

```text
RT backbone
-> relational representations
-> TabICL-style ICL
-> prediction
```

## Lazy vs feature-learning ICL

OpenRFM tests whether the model really uses:

```text
features <-> labels
```

by corrupting labels or features.

If performance barely changes:

```text
lazy / fixed-kernel regime
```

If performance drops strongly:

```text
task-adaptive feature-learning regime
```

## Synthetic-prior lesson

Large synthetic diversity is not enough.

The pretraining tasks should contain hidden relational rules that support labels help identify.

Example:

```text
homophily / heterophily
```

The support examples should reveal which relational rule is active.

## PFN connection

PFN:

```text
sample latent task z
-> generate demonstrations
-> infer z from demonstrations
```

OpenRFM says relational pretraining should similarly contain:

```text
support-identifiable relational latent variables
```

Possible latent factors:

```text
feature rule
schema rule
temporal rule
relational rule
task rule
```

## Main takeaway

A strong RFM needs:

```text
good architecture
+
good support selection
+
a pretraining prior that forces real ICL
```

---

# 4. Griffin

## Core idea

Griffin is best viewed as:

```text
a pretrained, schema-general, task-conditioned relational GNN
```

It is primarily:

```text
pretrain -> fine-tune
```

rather than:

```text
frozen model -> ICL
```

## Pipeline

```text
cells
-> universal cell encoders
-> task-conditioned intra-row attention
-> one node embedding per row
-> heterogeneous MPNN
-> universal decoder
-> prediction
```

## Universal cell encoders

### Numerical values

Use:

```text
normalization
+
pretrained numerical encoder
```

### Text / categorical values

Use a pretrained text encoder.

### Metadata

Column, table, relation, and task names are semantically encoded.

Thus Griffin combines:

```text
value meaning
+
schema meaning
```

## Task-conditioned row attention

Suppose a row contains:

```text
age
state
tenure
income
```

and the task is:

```text
predict churn
```

Griffin uses the task embedding as part of the attention query.

Conceptually:

```text
task + current node state -> Query
column metadata          -> Keys
cell representations     -> Values
```

This asks:

```text
Which columns matter for this task?
```

## Heterogeneous MPNN

Rows become graph nodes and PK/FK links become edges.

Griffin aggregates:

```text
Mean within relation type
-> Max across relation types
```

This reduces domination by high-degree relation types.

## Universal decoder

### Classification

Encode class names and compare them with the target representation.

### Regression

Use a shared numerical decoder.

This allows output spaces to change across tasks.

## Pretraining

### Stage 1

Masked-cell completion on single tables.

### Stage 2

Joint supervised training across tabular and relational tasks.

Downstream tasks are then generally:

```text
fine-tuned
```

## Key distinction from PFN/Kumo

Griffin mainly learns:

```text
a reusable representation / initialization
```

not:

```text
a frozen inference algorithm
```

So:

```text
Griffin ~ BERT-style pretrain + fine-tune
```

while:

```text
Kumo/OpenRFM ~ PFN-style in-context adaptation
```

## Main lesson

Task conditioning should happen **inside representation learning**, not only at the final prediction head.

---

# 5. RDBLearn / JUICE

## Core idea

Challenge the need for a native relational FM.

Instead:

```text
relational DB
-> parameter-free relational feature synthesis
-> flat table
-> existing tabular FM
```

The encoder principle is:

```text
JUICE = Just Use Intra Column Encodings
```

## Central claim

For ICL:

```text
compress within homogeneous columns
```

but avoid:

```text
premature cross-column mixing
```

before the downstream ICL model sees labels.

## Vertical compression

For a column like:

```text
Order.amount
```

with values:

```text
50
300
120
80
```

compute statistics such as:

```text
mean
sum
min
max
std
```

These preserve the semantic identity:

```text
Order.amount
```

## Horizontal compression

A learned representation such as:

```text
f(
  amount,
  quantity,
  category,
  time
)
```

mixes heterogeneous columns.

JUICE argues this can destroy information before the task is known through labels.

## Typical generated features

For a Customer:

```text
COUNT(Orders)
MEAN(Order.amount)
MAX(Order.amount)
SUM(Order.amount)
MEAN(Item.price via Orders)
MODE(Item.category via Orders)
...
```

This gives one fixed-width row:

```text
z_customer
```

Then:

```text
(z_i, y_i)
```

go directly into:

```text
TabPFN / LimiX
```

for ICL.

## Parameter-free encoder

Typical fixed operations:

### Numeric

```text
sum
mean
mode
min
max
std
```

### Categorical

```text
count
mode
```

No relational neural encoder is trained.

## Why it can work

The tabular FM learns how to combine synthesized relational features based on support labels.

So trainability is shifted from:

```text
relational encoder
```

to:

```text
pretrained tabular ICL model
```

## Main limitation

Column-wise aggregation can lose cross-column row alignment.

Example:

Entity A:

```text
A B
1 1
0 0
```

Entity B:

```text
A B
1 0
0 1
```

Both have:

```text
SUM(A)=1
SUM(B)=1
```

but only Entity A contains:

```text
A AND B
```

within one child row.

A native relational model can preserve this pattern; independent aggregation cannot.

## Key takeaway

```text
strong relational feature synthesis
+
strong tabular FM
```

is a very strong baseline and may eliminate the need to train a dedicated RFM for many tasks.

---

# Architecture Comparison

| Model | Basic Representation | Relational Reasoning | Adaptation | Task Conditioning | Main Philosophy |
|---|---|---|---|---|---|
| **KumoRFM** | Row/node subgraphs | Relational Graph Transformer | Explicit ICL | Mostly late | Relational subgraph as ICL example |
| **KumoRFM-2** | Cell -> row hierarchy | FK/graph attention | **Cross-sample ICL** | **Early labels/context** | Multi-axis relational ICL |
| **OpenRFM** | RT cell-based context | RT relational attention | **Tabular batch-level ICL** | Support labels | Relational ICL + sample ICL |
| **Griffin** | Cell -> row/node | **MPNN** | **Fine-tuning** | Task semantic embedding | Pretrained graph model |
| **RDBLearn** | Flat synthesized features | Parameter-free aggregation | **Tabular FM ICL** | Through support labels in TFM | No native RFM needed |

---

# ICL Comparison

## KumoRFM

```text
(G_i, y_i)
-> relational encoder
-> z_i
-> ICL
```

## KumoRFM-2

```text
(G_i, y_i)
-> task-conditioned row/column reasoning
-> FK attention
-> cross-sample attention
```

## OpenRFM

```text
relational neighborhood
-> RT
-> z_i
-> TabICL-style cross-sample ICL
```

## Griffin

```text
pretrained relational model
-> target-task fine-tuning
```

No central example-level ICL mechanism.

## RDBLearn

```text
RDB
-> fixed relational features
-> TabPFN/LimiX ICL
```

---

# Where Task Information Enters

| Model | Task signal |
|---|---|
| **KumoRFM** | Labeled support examples, mainly in later ICL stage |
| **KumoRFM-2** | **Injected early during row/column processing** |
| **OpenRFM** | Support labels in explicit batch-level ICL |
| **Griffin** | **Target/task semantic embedding inside row attention** |
| **RDBLearn** | No task-aware relational encoder; task learned downstream by TFM |

---

# Semantic Context

Several models use semantic information such as:

```text
table names
column names
relation names
task names
```

via text encoders.

The evidence suggests semantic metadata helps most in:

```text
zero-shot
cross-schema
low-label
```

settings.

A useful empirical hierarchy is:

```text
task demonstrations
>
task-conditioned representation
>
schema-name semantics alone
```

Semantic metadata is useful, but it is usually not the dominant signal by itself.

---

# Three Major Schools of RFM Design

## 1. Native Relational ICL

```text
KumoRFM
KumoRFM-2
OpenRFM
```

Goal:

```text
new DB + context labels
-> frozen model
-> prediction
```

## 2. Pretrained Graph Model

```text
Griffin
```

Goal:

```text
pretrain across DBs
-> fine-tune on new task
```

## 3. Relational Adapter + Tabular FM

```text
RDBLearn / JUICE
TabPFN-Rel-like systems
```

Goal:

```text
RDB
-> relational feature synthesis
-> strong TFM
```

No dedicated relational FM training.

---

# Most Important Conceptual Takeaways

## 1. Task conditioning should happen before aggressive compression

KumoRFM-2 and Griffin both support this principle.

If the task is unknown:

```text
generic representation compression
```

may discard useful signals.

## 2. Relational proximity != useful ICL support

OpenRFM shows that labels reachable by BFS are not necessarily enough.

RFMs likely need explicit support retrieval or cross-sample ICL.

## 3. RFM pretraining must teach a learning algorithm

A large synthetic corpus alone is not enough.

Pretraining tasks should contain latent rules that support labels help identify.

This is the relational analogue of PFN task priors.

## 4. Database scale is primarily a retrieval problem

You cannot attend over billions of rows.

Instead:

```text
retrieve relevant rows/subgraphs/examples
-> reason over selected context
```

## 5. Relational structure is sometimes essential

Native models can preserve:

```text
cross-column row alignment
motifs
shared identifiers
multi-hop relational patterns
```

that flat aggregation can lose.

## 6. Flattening is still a very strong baseline

RDBLearn demonstrates that:

```text
good feature synthesis
+
strong tabular FM
```

can compete with much more complicated native RFMs.

## 7. A future RFM may need hybrid reasoning

The strongest future design may combine:

```text
preserved relational structure
+
column-preserving aggregations
+
task-conditioned routing
+
explicit cross-sample ICL
+
retrieval
```

rather than committing entirely to flattening or full graph attention.

---

# Key Trade-Off: Preserve Structure vs Compress Structure

## Preserve structure

Advantages:

```text
retain relational motifs
retain row alignment
retain fine-grained temporal interactions
retain shared-entity patterns
```

Disadvantages:

```text
expensive
hard to scale
harder optimization
context-selection problem
```

Examples:

```text
KumoRFM-2
OpenRFM
Griffin
```

## Compress structure

Advantages:

```text
simple
fast
scalable
works with strong TFMs
easy ICL
```

Disadvantages:

```text
can lose relational motifs
can lose cross-column co-occurrence
can lose topology
```

Example:

```text
RDBLearn
```

---

# Promising Future Research Directions

## 1. Hybrid Native + Aggregated Context

Use both:

```text
column-preserving aggregated features
+
selected raw relational subgraphs
```

Example:

```text
TabPFN-like global summary
+
Kumo-like local relational context
```

This could combine efficiency and expressivity.

## 2. Task-Adaptive Relational Retrieval

Instead of fixed BFS/hops:

```text
query/task
+
schema semantics
+
time
+
uncertainty
+
label utility
-> select context
```

Possible criteria:

```text
graph proximity
semantic similarity
temporal recency
diversity
uncertainty
value-of-information
```

## 3. Learned Route Selection

Instead of hard-coded schema paths:

```text
task
+
schema semantics
+
context labels
-> choose useful PK/FK routes
```

## 4. Support Selection as a First-Class Component

Given millions of labeled examples:

```text
which k should enter ICL?
```

Possible solutions:

```text
retrieval
clustering
prototype memory
uncertainty sampling
graph-based support selection
```

## 5. Better Relational Synthetic Priors

Synthetic tasks should vary latent dimensions such as:

```text
homophily
heterophily
temporal dependence
long-range dependency
interaction rules
relation semantics
schema structure
label delay
```

The support examples should be necessary to infer these rules.

## 6. Semantic + Statistical Schema Understanding

Combine:

```text
column/table/relation names
+
empirical value distributions
+
relational position
```

instead of relying on only one source.

## 7. Preserve Cross-Column Row Alignment Selectively

RDBLearn shows that mixing columns too early can be harmful.

Kumo shows that losing row alignment can also be harmful.

A promising middle ground:

```text
keep raw row interactions only where useful
+
aggregate everything else
```

## 8. Adaptive Compute

Easy tasks:

```text
flat aggregated context
```

Hard relational tasks:

```text
deeper graph retrieval
+
more support examples
+
more attention layers
```

The model could learn how much relational compute each query deserves.

## 9. Open, Reproducible RFMs

Important needs:

```text
open pretrained checkpoints
open synthetic generators
open context retrieval pipelines
standardized RFM benchmarks
```

## 10. Distillation / Production

For high-volume prediction:

```text
large RFM
-> task-specific small model
```

Possible students:

```text
small GNN
MLP
tree model
compact relational Transformer
```

---

# Final Mental Map

```text
                    RELATIONAL DATABASE
                           |
         ---------------------------------------
         |                  |                  |
         v                  v                  v

   Native RFM         Graph-pretrained      Flatten first
     branch               branch              branch

 KumoRFM-2              Griffin            RDBLearn
 OpenRFM                                     |
     |                                       v
     |                                    TabPFN
     v
 relational ICL
```

The central research tension is:

```text
How much relational structure should be preserved before the model sees the task labels?
```

And the second major question is:

```text
How should a model choose the relational and labeled context needed for each prediction?
```

These two questions sit at the center of current relational foundation-model research.
