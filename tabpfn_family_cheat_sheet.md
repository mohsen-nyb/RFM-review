# TabPFN Family Cheat Sheet

A concise reference for the evolution from **Prior-Data Fitted Networks (PFNs)** to **TabPFN-3**.

---

## 0. Core Idea

Traditional ML learns a dataset-specific predictor:

\[
D_{\text{train}} \rightarrow \text{optimize } \theta_D \rightarrow f_{\theta_D}(x)
\]

PFNs instead pretrain a model to act as a **learning algorithm**:

\[
(D_{\text{train}}, x_*) \rightarrow q_\theta(y_* \mid x_*, D_{\text{train}})
\]

At inference time, \(\theta\) is frozen; adaptation happens **in context**.

> **Key mental model:** TabPFN is closer to a learned Bayesian/AutoML algorithm than to “BERT for tables.”

---

# 1. PFN — *Transformers Can Do Bayesian Inference* (ICLR 2022)

**Paper:** https://arxiv.org/abs/2112.10510

### Main contribution
Introduces **Prior-Data Fitted Networks (PFNs)**.

### Core training idea
Sample entire synthetic learning tasks from a prior:

\[
t \sim p(t), \qquad D \sim p(D\mid t)
\]

Then train:

\[
q_\theta(y_* \mid x_*, D)
\]

with negative log-likelihood:

\[
\mathcal{L}
=
-\log q_\theta(y_* \mid x_*,D)
\]

Across many sampled tasks, minimizing NLL minimizes:

\[
D_{\mathrm{KL}}
\left(
p(y\mid x,D)
\;\|\;
q_\theta(y\mid x,D)
\right)
\]

So the network learns to approximate the **Bayesian posterior predictive distribution**.

### Important distinction
It does **not necessarily recover the full posterior** \(p(t\mid D)\).

It directly approximates:

\[
p(y\mid x,D)
\]

### Why Transformers?
Datasets are sets; without positional encodings the model can be permutation-invariant/equivariant over examples.

### Key leap
\[
\boxed{\text{Learn the inference algorithm, not one predictor}}
\]

### Takeaway
The quality of a PFN is strongly determined by its **prior over learning problems**.

---

# 2. TabPFN — ICLR 2023

**Paper:** https://arxiv.org/abs/2207.01848

### Main contribution
Adapts PFNs to small tabular classification by designing a tabular task prior.

### Synthetic task generators

#### BNN prior
Sample a random neural network:

\[
X \rightarrow \text{random NN} \rightarrow Y
\]

One sampled BNN generates one synthetic dataset.

#### SCM prior
Sample a structural causal model:

\[
z_i = f_i(z_{\text{parents}(i)}, \epsilon_i)
\]

Then choose some variables as features and another as target.

Important:

\[
X \rightarrow Y
\]

is **not required**.

The prior can contain:

\[
X_1 \rightarrow Y \rightarrow X_2
\]

so observed features may be causes **or effects** of the target.

### Dataset generation
For one fixed SCM/BNN:
- keep graph/functions fixed;
- resample noise/input values;
- generate many rows;
- then sample a new SCM/BNN for the next task.

### Training
A batch contains **many whole datasets**, not independent rows.

Example:

\[
512 \text{ datasets/batch}
\]

Each dataset is split into:
- context/training rows;
- query/test rows.

Loss:

\[
-\sum_{i \in \text{query}}
\log q_\theta(y_i\mid x_i,D_{\text{context}})
\]

### Architecture
- roughly one token per row;
- training rows attend to training rows;
- test rows attend to training rows;
- test rows do not attend to each other.

### Ensemble trick
Use multiple:
- feature-column rotations/permutations;
- class-label rotations;
- preprocessing variants;

then average predictions.

Why needed?
- row order is naturally permutation-invariant;
- feature coordinates inside a row are **not inherently permutation-invariant**.

### Key leap
\[
\boxed{
\text{PFN}
+
\text{SCM/BNN tabular prior}
=
\text{general-purpose small-data classifier}
}
\]

### Main limitation
Designed mainly for:
- \(\sim 1\)K rows;
- \(\sim 100\) features;
- classification.

---

# 3. TabPFNv2 — Nature 2025

**Paper:** https://www.nature.com/articles/s41586-024-08328-6

### Main contribution
Turns TabPFN into a much more practical tabular foundation model.

### Major architectural change
Move from row-centric representation to a **2D table representation**.

The model alternates:

\[
\text{feature attention}
\leftrightarrow
\text{sample attention}
\]

### Feature attention
Within each row:

\[
x_{i1}\leftrightarrow x_{i2}\leftrightarrow \cdots
\]

Captures feature interactions.

### Sample attention
Within each feature:

\[
x_{1j}\leftrightarrow x_{2j}\leftrightarrow \cdots
\]

Captures dataset-level/statistical structure.

### Explicit attention complexity
For \(n\) rows and \(d\) features:

\[
\boxed{
O(dn^2 + nd^2)
}
\]

because:
- sample attention costs \(O(n^2)\), repeated for \(d\) features;
- feature attention costs \(O(d^2)\), repeated for \(n\) rows.

### Stronger synthetic prior
Adds more realistic:
- missingness;
- skewed distributions;
- quantization;
- categorical-like behavior;
- irrelevant/noisy features;
- outliers;
- transformations.

### New capabilities
- regression;
- categorical data;
- missing values;
- larger tables;
- reusable embeddings;
- anomaly/density-related uses.

### Important idea
Inductive bias is often encoded via:

\[
\boxed{\text{the distribution of synthetic tasks}}
\]

rather than only architecture/regularization.

### Key leap
\[
\boxed{
\text{row-level ICL}
\rightarrow
\text{2D row/column reasoning}
}
\]

### Typical scale
Roughly:
- \(\sim 10\)K rows;
- \(\sim 500\) features.

---

# 4. TabPFN-2.5

**Paper:** https://arxiv.org/abs/2511.08667

### Main contribution
Scaling, stronger inference, and production-oriented deployment.

### Main changes

#### Deeper models
- classification: ~24 layers;
- regression: ~18 layers.

#### Feature grouping
Several features are grouped into one feature token.

Benefit:

\[
\text{fewer feature tokens}
\rightarrow
\text{cheaper feature attention}
\]

#### Thinking rows
Adds learned extra rows:

\[
T_1,\ldots,T_{64}
\]

These are not real samples.

They act as learned computational/workspace slots and possible attention sinks.

#### Larger preprocessing ensemble
Multiple:
- scalers;
- quantile transforms;
- soft clipping;
- feature permutations;
- SVD-derived features.

#### Feature subsampling
For very high-dimensional tables, individual ensemble members may see only subsets of features.

### Distillation
Use TabPFN as a strong teacher:

\[
(D,x)\rightarrow \text{TabPFN}
\]

and distill into a dataset-specific:

\[
x\rightarrow g_{\phi_D}(x)
\]

where \(g\) may be:
- MLP;
- tree ensemble.

Why?
- lower serving latency;
- lower memory;
- no full context at prediction time.

Tradeoff:
- loses in-context learning;
- usually preserves most, not all, teacher accuracy.

### Real-TabPFN-2.5
Synthetic-pretrained TabPFN further trained on **real datasets**.

Important empirical lesson:

\[
\boxed{
\text{synthetic + real}
>
\text{synthetic-only}
}
\]

for reported classification settings.

This suggests a remaining synthetic-to-real prior gap.

### Key leap
\[
\boxed{
\text{strong TFM}
\rightarrow
\text{scalable + deployable TFM}
}
\]

---

# 5. TabPFN-3

**Paper:** https://arxiv.org/abs/2605.13986

### Main contribution
Redesigns the architecture for much larger \(n\), reaching up to \(\sim 1\)M rows.

### Core pipeline

\[
\boxed{
\text{Distribution Embedder}
\rightarrow
\text{Feature Aggregator}
\rightarrow
\text{Row-level ICL Transformer}
}
\]

---

## 5.1 Distribution Embedder

Goal: represent each value relative to its **column distribution**.

Example:

\[
20
\]

means something different in:

\[
[18,19,20,21,22]
\]

versus:

\[
[1,2,3,4,20].
\]

So:

\[
E(20\mid D_1)
\neq
E(20\mid D_2)
\]

### Inducing points
Instead of full \(n^2\) attention within every column, use \(K\) learned latent summaries:

\[
I_1,\ldots,I_K
\]

with roughly \(K=128\).

Complexity:

\[
O(ndK)
\]

instead of:

\[
O(dn^2)
\]

for this distribution-modeling stage.

---

## 5.2 CLS tokens / Feature Aggregation

Each row contains distribution-aware feature embeddings:

\[
h_{i1},\ldots,h_{id}
\]

Add learned CLS summary tokens:

\[
C_1,\ldots,C_4
\]

These attend over the row's features.

After several layers, concatenate the CLS states:

\[
r_i
=
[C'_1\Vert C'_2\Vert C'_3\Vert C'_4]
\]

giving one fixed-dimensional row representation.

### Why important?
It collapses:

\[
d \text{ feature representations}
\rightarrow
1 \text{ row representation}
\]

before expensive ICL.

---

## 5.3 Row-level ICL

Now perform in-context learning over:

\[
r_1,\ldots,r_n
\]

rather than separately for every feature.

Dominant row-attention term becomes:

\[
\boxed{O(n^2)}
\]

instead of v2's:

\[
\boxed{O(dn^2)}
\]

### Rough comparison

TabPFNv2:

\[
O(dn^2 + nd^2)
\]

TabPFN-3:

\[
O(ndK + nd^2 + n^2)
\]

with fixed small \(K\).

### Main scaling leap
\[
\boxed{
dn^2
\rightarrow
n^2
}
\]

---

## 5.4 Retrieval-style classification decoder

Instead of a fixed \(K\)-class output head:

1. compare a test representation with training representations;
2. compute learned similarities;
3. aggregate attention weights by class label.

Conceptually:

\[
\text{test embedding}
\rightarrow
\text{similar training rows}
\rightarrow
P(Y)
\]

This supports many-class prediction more naturally.

---

## 5.5 Additional v3 ideas

- query-aware attention scaling for longer contexts;
- row chunking;
- KV-cache optimization / quantization;
- up to \(\sim 1\)M rows;
- synthetic-only core pretraining.

### Thinking mode
Public details are incomplete.

Known:
- spends additional fit/test-time compute;
- uses TabPFN itself;
- improves results.

Unknown:
- exact algorithm.

Ensembling/search may be involved, but it is **not publicly established that Thinking = just ensembling**.

---

# 6. Relational Extension: TabPFN-Rel

Important distinction:

\[
\boxed{
\text{TabPFN-Rel is not a native graph/relational Transformer}
}
\]

Pipeline:

\[
\text{relational DB}
\rightarrow
\text{deep feature synthesis / aggregation}
\rightarrow
\text{flat table}
\rightarrow
\text{TabPFN}
\]

Example:

\[
\text{Customer}
\rightarrow
\text{Transactions}
\rightarrow
\text{Merchant}
\]

becomes features such as:

- transaction count;
- mean transaction amount;
- maximum amount;
- number of unique merchants;
- deeper aggregated statistics.

### Key lesson
A very strong tabular FM + strong relational feature engineering is an extremely strong relational baseline.

---

# Model Comparison

| Model | Core innovation | Representation / Attention | Approx. scale | Synthetic prior | Regression | Key limitation |
|---|---|---|---:|---|---|---|
| **PFN** | Learn Bayesian posterior prediction | Dataset + query | Toy/general tasks | GP, BNN, etc. | Yes | Depends heavily on prior |
| **TabPFN** | Tabular SCM/BNN prior | Row tokens; row ICL | ~1K rows / ~100 feat. | SCM + BNN | Limited / no main support | Small numerical classification |
| **TabPFNv2** | 2D table Transformer | Feature + sample attention | ~10K / ~500 | Much richer synthetic prior | Yes | \(O(dn^2+nd^2)\) |
| **TabPFN-2.5** | Scaling + deployment | v2-style + deeper + thinking rows | ~50K–100K / ~2K | Broader synthetic tasks | Yes | Still expensive row attention |
| **TabPFN-3** | Distribution-aware encoding + row collapse | Distribution embedder → CLS aggregation → row ICL | up to ~1M rows | Synthetic-only core | Yes | \(O(n^2)\) row ICL still costly at extreme scale |

---

# Evolution in One Line

\[
\boxed{
\text{PFN}
\rightarrow
\text{learn inference}
}
\]

\[
\boxed{
\text{TabPFN}
\rightarrow
\text{learn tabular inference}
}
\]

\[
\boxed{
\text{TabPFNv2}
\rightarrow
\text{explicit 2D table reasoning}
}
\]

\[
\boxed{
\text{TabPFN-2.5}
\rightarrow
\text{scale + deployment}
}
\]

\[
\boxed{
\text{TabPFN-3}
\rightarrow
\text{distribution-aware features + scalable row-level ICL}
}
\]

---

# Most Important Takeaways

1. **PFNs learn a learning algorithm**, not one predictor.
2. **Synthetic task design is as important as architecture.**
3. SCMs are useful because real predictive features may be both **causes and effects** of the target.
4. TabPFN performs **in-context adaptation** with frozen weights.
5. v1 treated rows as tokens; v2 explicitly modeled **row and column structure**.
6. v2's real complexity is approximately:
   \[
   O(dn^2+nd^2)
   \]
7. v3's main scaling trick is:
   \[
   dn^2 \rightarrow n^2
   \]
   by collapsing features before row-level ICL.
8. The v3 distribution embedder makes a value meaningful **relative to its feature distribution**.
9. CLS tokens are learned **summary slots**, not target classes.
10. Distillation is mainly a **deployment strategy**, not the core PFN idea.
11. Real-data continued training improving v2.5 suggests the synthetic prior is still imperfect.
12. TabPFN-Rel demonstrates that:
   \[
   \text{relational feature engineering}
   +
   \text{strong TFM}
   \]
   can compete strongly with native relational models.

---

# Promising Research Directions

## 1. Native Relational PFNs
Replace:

\[
\text{relational DB}
\rightarrow
\text{flatten}
\rightarrow
\text{TabPFN}
\]

with:

\[
\boxed{
\text{relational graph/database}
\rightarrow
\text{native in-context relational inference}
}
\]

Questions:
- how to encode schemas, entities, edges, and foreign keys?
- how to generalize across unseen schemas?

---

## 2. Synthetic + Real Pretraining
Combine:

\[
\text{large synthetic task diversity}
+
\text{large real dataset corpora}
\]

rather than relying almost entirely on either one.

Key question:

> How should synthetic and empirical priors be mixed without benchmark leakage?

---

## 3. Retrieval-Based Context Selection
Current ICL still becomes expensive at large \(n\).

Instead:

\[
D
\rightarrow
\text{retrieve useful subset}
\rightarrow
\text{PFN inference}
\]

Potentially use:
- semantic retrieval;
- nearest neighbors;
- graph neighborhoods;
- uncertainty-aware retrieval;
- learned value-of-information selection.

---

## 4. Subquadratic Row ICL
TabPFN-3 removes the \(d\) factor but still has:

\[
O(n^2)
\]

row attention.

Promising directions:
- sparse attention;
- inducing tokens;
- hierarchical memory;
- clustering/prototypes;
- state-space models;
- retrieval + local attention.

---

## 5. Better Column Semantics
Vanilla TabPFN primarily learns from values/statistics.

Future models could combine:

\[
\text{column values}
+
\text{column names/descriptions}
+
\text{ontology/schema information}
\]

using LLM/text encoders.

---

## 6. Graph/Relational Synthetic Priors
Extend SCM priors from hidden data generators to **explicit relational task generators**.

Generate:
- heterogeneous schemas;
- temporal graphs;
- many-to-many relations;
- entity attributes;
- label delays;
- relational motifs.

Then train a PFN to infer directly over these structures.

---

## 7. Adaptive Test-Time Compute
TabPFN-3 Thinking suggests a broader direction:

\[
\boxed{
\text{easy dataset}
\rightarrow
\text{little inference compute}
}
\]

\[
\boxed{
\text{hard dataset}
\rightarrow
\text{more search / ensemble / retrieval / reasoning}
}
\]

A principled learned compute-allocation policy could be valuable.

---

## 8. Foundation Model → Production Student
Generalize TabPFN distillation:

\[
\text{large tabular/relational FM}
\rightarrow
\text{small task-specific model}
\]

Possible students:
- MLP;
- tree ensemble;
- small GNN;
- compact relational Transformer.

Important research question:

> How much of the teacher's prior knowledge can be preserved in a latency-constrained student?

---

# Suggested Next Reading

After TabPFN, the natural next topic is **Relational Foundation Models**, especially methods that directly model:

- relational databases;
- heterogeneous graphs;
- multi-table schemas;
- cross-table in-context learning.

The central comparison to keep in mind is:

\[
\boxed{
\text{flatten relational structure}
\quad\text{vs}\quad
\text{model relational structure natively}
}
\]
