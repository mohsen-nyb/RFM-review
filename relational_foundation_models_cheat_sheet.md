# Relational Deep Learning & Relational Foundation Models — Cheat Sheet

A concise reference for the progression:

```text
RDL
-> RelBench
-> RelGNN
-> Relational Transformer
-> PluRel
```

The main theme is:

```text
relational database
-> preserve relational structure
-> learn relational representations
-> pretrain across databases
-> scale with synthetic relational worlds
```

---

# 1. Relational Deep Learning (RDL) — ICML 2024

**Paper:** https://arxiv.org/abs/2312.04615

## Main contribution

Defines a general framework for applying deep learning directly to relational databases.

Instead of:

```text
relational DB
-> SQL feature engineering
-> flat table
-> XGBoost
```

use:

```text
relational DB
-> relational entity graph
-> GNN
```

## Database-to-graph mapping

```text
table       -> node type
row         -> node
columns     -> node features
PK/FK       -> edges
timestamp   -> temporal constraint
```

## Training table

A prediction task is defined by:

```text
(entity IDs, seed time, target)
```

Example:

```text
customer_id | seed_time | churn
```

This separates the underlying database from the predictive task.

## Temporal correctness

For a prediction at time `t`:

```text
input  = data available at or before t
target = event/value after t
```

So the model operates on temporally valid relational context.

## Key leap

```text
manual relational feature engineering
->
learned graph message passing
```

## Limitation

RDL is **not a foundation model**. A new database/task still requires task-specific model training.

---

# 2. RelBench — NeurIPS 2024

**Paper:** https://arxiv.org/abs/2407.20060

## Main contribution

Turns RDL into a standardized benchmark.

Provides:

```text
relational databases
+
task tables
+
temporal splits
+
evaluation metrics
+
baseline models
```

## Main task types

### Entity classification

```text
(entity, time) -> class
```

### Entity regression

```text
(entity, time) -> continuous value
```

### Recommendation / link prediction

```text
(source entity, time)
-> ranked destination entities
```

## Temporal evaluation

Train/validation/test are split by time:

```text
past -> train
later -> validation
latest -> test
```

Entities may appear in multiple splits.

## Important empirical result

A standard RDL pipeline:

```text
tabular row encoder
+
heterogeneous GNN
```

can often match or beat manually engineered relational pipelines while requiring far less task-specific feature engineering.

## Key leap

```text
relational learning
->
standardized, leakage-aware benchmark
```

---

# 3. RelGNN — ICML 2025

**Paper:** https://arxiv.org/abs/2502.06784

## Main contribution

Shows that the physical graph induced by PK/FK relations is not necessarily the optimal neural communication graph.

Core insight:

```text
database storage topology
!=
optimal message-passing topology
```

## Problem with ordinary GNN message passing

Suppose:

```text
Customer
   |
Transaction
   |
Product
```

The semantic relation may be:

```text
Customer purchased Product
```

but normalization inserts `Transaction` as an intermediate node.

A standard GNN needs:

```text
Customer -> Transaction -> Product
```

over two layers, causing possible:

- redundancy;
- information imbalance;
- intermediate-node entanglement.

## Atomic routes

RelGNN automatically detects schema patterns where one table contains multiple foreign keys.

Example:

```text
Transaction
  customer_id -> Customer
  product_id  -> Product
```

This induces:

```text
Customer -> Transaction -> Product
Product  -> Transaction -> Customer
```

Important:

> RelGNN does **not** decide that the middle table is unimportant.

Its features remain part of the message.

## Composite message passing

For:

```text
src -> mid -> dst
```

construct:

```text
FUSE(h_src, h_mid)
```

and send it directly to `dst`.

Conceptually:

```text
Customer -----\
               FUSE ---> Product
Transaction --/
```

A two-hop semantic relation becomes one RelGNN communication step.

## Bridge vs hub structures

### Bridge

```text
A <- M -> B
```

induces routes between `A` and `B` through `M`.

### Hub

```text
       A
       |
B ---- H ---- C
```

induces pairwise routes among referenced entity types through `H`.

## Atomic routes are schema-driven

They come from PK/FK structure, not manually chosen semantic meta-paths.

## Key leap

```text
hop-by-hop GNN propagation
->
schema-aware composite communication
```

## Limitation

RelGNN is still task-specific, not a foundation model.

---

# 4. Relational Transformer (RT) — ICLR 2026

**Paper:** https://arxiv.org/abs/2510.06377

## Main contribution

Introduces a schema-agnostic Transformer pretrained across relational databases.

Goal:

```text
many databases/tasks
-> pretrain once
-> new database/task
-> zero-shot prediction
```

---

## Cell-level tokenization

RT represents each **cell** as a token containing:

```text
value
+
column name
+
table name
```

So the same scalar can mean different things depending on schema context.

## Schema semantics

Table and column names are embedded with a frozen text encoder.

Therefore RT uses:

```text
statistical structure
+
natural-language schema semantics
```

---

## Task-table prompting

Prediction is converted into masked-cell prediction.

Example:

```text
time | user_id | churn
Jan  | U1      | 0
Feb  | U1      | 0
Mar  | U1      | 1
Apr  | U1      | [MASK]
```

Different tasks become:

```text
recover the masked target cell
```

---

## Relational Attention

### Column attention

Same column across rows.

Purpose:

```text
learn empirical feature distributions
```

Conceptually related to TabPFN-3's distribution embedder.

### Feature attention

Allows a row to attend to:

```text
same-row cells
+
parent rows referenced by FKs
```

Direction:

```text
FK -> PK
```

### Neighbor attention

Allows parent entities to attend to child rows that reference them.

Direction:

```text
PK -> FK
```

### Full attention

Optional unrestricted attention over the sampled context.

Later work suggests the specialized relational masks are more important.

---

## Context construction

RT samples a bounded relational neighborhood around the masked task row via PK/FK traversal.

Future rows are excluded to maintain temporal correctness.

---

## Pretraining

Original RT mainly pretrains on real RelBench databases.

Typical evaluation:

```text
leave-one-database-out
```

This tests transfer to an unseen schema and unseen task.

---

## Important zero-shot caveat

RT can use **historical labels from the target task** when they appear in context.

Therefore zero-shot means:

```text
no target-task gradient training
```

not necessarily:

```text
no target-task labels in context
```

Ablations show these historical labels are a major source of performance.

## Key leap

```text
task-specific relational GNN
->
schema-agnostic pretrained relational Transformer
```

---

# 5. PluRel — ICML 2026

**Paper:** https://arxiv.org/abs/2602.04029

## Main contribution

Addresses RT's biggest weakness:

```text
too few real relational databases for pretraining
```

PluRel generates **entire synthetic relational databases**.

It is mainly:

```text
synthetic relational data generator
+
scaling-law study
```

rather than a new RFM architecture.

---

## Three levels of relational generation

A relational database requires:

```text
1. schema
2. PK/FK row connectivity
3. feature values
```

PluRel explicitly generates all three.

---

## 5.1 Schema generation

Sample a random schema graph over tables.

Randomizes:

- number of tables;
- table sizes;
- feature counts;
- schema topology.

Current generation focuses on DAG-style schemas.

---

## 5.2 PK/FK connectivity generation

Knowing that:

```text
Order.customer_id -> Customer
```

does not determine which customer each order references.

PluRel uses a **Hierarchical Stochastic Block Model (HSBM)** to generate structured connectivity.

Conceptually:

```text
latent entity groups
+
connection probabilities
->
realistic relational structure
```

---

## 5.3 Feature generation

Each table gets its own SCM.

Child-table SCMs can depend on features from referenced parent rows.

Example:

```text
Order.quantity
=
f(
    Customer.income,
    Product.price,
    noise
)
```

Thus features and relational structure are coupled.

---

## Temporal generation

PluRel can introduce:

```text
trend
+
seasonality/cycles
+
noise/fluctuation
```

so synthetic databases are not purely IID.

---

## Pretraining objective

PluRel mainly supplies synthetic databases to RT.

RT still uses:

```text
masked token prediction
```

over relational cells.

---

# Scaling Laws

PluRel studies two separate scaling axes.

## Database diversity

```text
N = number of distinct synthetic databases
```

## Token volume

```text
S = total pretraining tokens
```

Critical finding:

```text
N and S must scale together
```

Too few databases + too many tokens:

```text
-> overfitting to a small set of relational worlds
```

Too many databases + too few tokens:

```text
-> underfitting each world
```

## Foundation-model lesson

For RFMs, scaling is not only about tokens but also about:

```text
database diversity
+
schema diversity
+
task diversity
```

---

## Synthetic vs real training

The strongest recipe is generally:

```text
synthetic relational pretraining
->
continued real-data pretraining
```

Synthetic-only still shows a synthetic-to-real gap.

## Key leap

```text
few real relational databases
->
large prior over synthetic relational worlds
```

---

# Model / Paper Comparison

| Work | Main Contribution | Representation | Training Regime | Cross-DB Transfer | Foundation Model? |
|---|---|---|---|---|---|
| **RDL** | Relational DB -> temporal heterogeneous graph | Row/node-level | Task-specific supervised GNN | No | No |
| **RelBench** | Standardized relational benchmark | Model-agnostic | Benchmark infrastructure | N/A | No |
| **RelGNN** | Atomic routes + composite message passing | Schema-aware graph routes | Task-specific supervised GNN | No | No |
| **Relational Transformer** | Schema-agnostic relational Transformer | Cell tokens + relational attention | Multi-DB pretraining | **Yes** | **Yes** |
| **PluRel** | Synthetic relational DB generation + scaling laws | Generates schemas, edges, features | Synthetic RT pretraining | Enables stronger transfer | RFM data infrastructure |

---

# Evolution in One Line

```text
RDL
-> preserve relational structure
```

```text
RelBench
-> standardize relational learning
```

```text
RelGNN
-> improve relational message passing
```

```text
Relational Transformer
-> pretrain across databases
```

```text
PluRel
-> scale relational pretraining with synthetic worlds
```

---

# Most Important Takeaways

1. **RDL is primarily a problem formulation**, not a foundation model.

2. A relational database naturally maps to:

```text
table -> node type
row -> node
PK/FK -> edge
```

3. **Time must be first-class** to avoid leakage.

4. RelBench standardizes:

```text
database
+
task
+
seed time
+
temporal split
+
metrics
```

5. RelGNN shows that normalized DB schemas can create artificial multi-hop communication paths.

6. **Atomic routes** are inferred automatically from bridge/hub PK/FK patterns.

7. Atomic routes do not discard middle-table features; they **fuse them into composite messages**.

8. RT is the first true RFM in this sequence.

9. RT combines:

```text
cell values
+
schema semantics
+
column statistics
+
PK/FK relations
```

10. RT zero-shot performance relies substantially on **historical task labels in context**, so zero-shot must be interpreted carefully.

11. PluRel shows that **database diversity itself is a scaling variable**.

12. Synthetic relational pretraining helps most when followed by **real-data alignment / continued pretraining**.

---

# Promising Future Research Directions

## 1. Native Relational In-Context Learning

Combine:

```text
relation-level context
+
dataset-level labeled examples
```

Goal:

```text
new DB
+
small labeled task context
+
relational neighborhood
->
prediction
```

---

## 2. Learn Relational Routes Instead of Hard-Coding Them

RelGNN derives atomic routes purely from schema structure.

Future models could learn:

```text
schema topology
+
table/column semantics
+
task context
->
which relational routes are useful
```

---

## 3. Relation Semantics

Models should explicitly distinguish relations like:

```text
buyer_id  -> User
seller_id -> User
```

using:

```text
FK names
+
relation descriptions
+
directionality
+
semantic roles
```

---

## 4. Richer Synthetic Relational Priors

Extend PluRel beyond numeric/categorical features to:

- text;
- JSON;
- images;
- event streams;
- geospatial data;
- free-form metadata;
- nested structures.

---

## 5. Cyclic / Recursive Schemas

Support:

```text
Employee.manager_id -> Employee
```

and more general:

```text
cycles
self-relations
recursive hierarchies
```

---

## 6. Joint Scaling Laws

Study:

```text
model size
+
training tokens
+
number of databases
+
schema diversity
+
task diversity
```

jointly.

---

## 7. Synthetic + Real Curriculum

Instead of only:

```text
synthetic pretrain
-> real fine-tune
```

study adaptive mixtures of:

```text
synthetic
+
real
+
hard synthetic examples
```

throughout training.

---

## 8. Retrieval-Based Relational Context

Replace pure BFS sampling with learned retrieval based on:

```text
task relevance
+
schema relevance
+
graph distance
+
temporal relevance
+
uncertainty
+
value of information
```

---

## 9. Temporal Relational Foundation Models

Explicitly model:

```text
when a relation existed
when a value changed
event order
label delay
temporal causality
```

instead of using time mainly as a sampling constraint.

---

## 10. RFM -> Production Student

Distill:

```text
large relational foundation model
->
small task-specific model
```

Possible students:

- small GNN;
- relational MLP;
- tree ensemble;
- compact Transformer.

---

# Next Papers

Natural continuation:

```text
KumoRFM / KumoRFM-2
-> relational in-context learning architecture
```

then:

```text
OpenRFM
-> dissect why relational ICL works and where it fails
```

then:

```text
Griffin
-> graph-centric relational foundation modeling
```

The next major question is:

```text
What is the best architecture for a relational foundation model?
```
