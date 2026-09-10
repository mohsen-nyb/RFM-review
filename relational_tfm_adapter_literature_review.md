# Literature Review: Lightweight Relational Adaptation of Frozen Tabular Foundation Models

## Executive conclusion

The exact combination **heterogeneous/temporal RDB neighborhood → small learned relational encoder/adapter → frozen general-purpose TFM → prediction** does not appear, in the papers surveyed through September 10, 2026, to have a direct published match.

The closest lines of work are:

1. **Towards Pretraining Text Encoders for TabPFN (2026):** freezes TabPFN and maps external text representations through a lightweight adapter into a small number of native TabPFN tokens. This is almost the same interface as relational-neighborhood → relational-token → frozen TFM, except the external modality is text rather than an RDB neighborhood.
2. **GraphPFN (ICML 2026):** augments the pretrained TFM LimiX with graph-neighborhood attention adapters, freezes the original TFM layers, and trains graph-specific adapters on a synthetic graph prior.
3. **TFM-Retouche (2026) and BETA/TabPFN Unleashed (ICML 2025):** place small learned adapters/encoders in front of a frozen TFM. These are especially relevant to a TabICL implementation, but the adapter only transforms the existing tabular row.
4. **OpenRFM (2026):** combines a relational transformer with a batch-level ICL component lifted from pretrained TabICL. It is the closest RDB-specific precedent, but the direction is reversed: the learned relational backbone remains the main model and TabICL supplies the ICL stage.
5. **TuneTables (NeurIPS 2024):** provides a TFM precedent for continuous prompt/context optimization.
6. **G-Retriever, TEA-GLM, LLaGA and GraphAdapter:** show the graph→encoder/projector→continuous-token→frozen-foundation-model pattern in the LLM literature.

The most promising research direction is therefore a **conditional relational token adapter**: an entity- and timestamp-specific relational encoder produces one or a few tokens in the frozen TFM's native representation space.

## Closest papers and their relationship to the proposed idea

| Paper | Frozen general TFM/FM? | Structural/RDB input? | What is trained? | Similarity to proposed method |
|---|---|---|---|---|
| **Towards Pretraining Text Encoders for TabPFN (2026)** | Yes | External text, not relations | Tiny token adapter | Extremely close interface |
| **GraphPFN (ICML 2026)** | Original LimiX layers frozen during graph-adapter pretraining | Graph | Neighborhood-attention adapters | Extremely close structurally |
| **OpenRFM (2026)** | Reuses pretrained TabICL ICL component | RDB | Relational backbone + integration | Closest RDB-specific work |
| **TFM-Retouche (2026)** | Yes, TabICLv2 | No new relational evidence | Input-space residual adapter | Very close implementation pattern |
| **BETA / TabPFN Unleashed (ICML 2025)** | Yes, TabPFN | No | Front-end encoder | Very close adapter pattern |
| **TuneTables (NeurIPS 2024)** | Mostly frozen PFN | No | Continuous context/prompt | Soft-prompt precedent |
| **G2T-FM (2025)** | Uses pretrained TabPFN | Graph via fixed features | No learned structural adapter | Graph analogue of RDBLearn+TFM |
| **RDBLearn (2026)** | Yes | RDB via fixed DFS/SQL aggregations | No relational parameters | Primary baseline/foil |
| **G-Retriever / TEA-GLM** | Yes/mostly frozen LLM | Graph | Graph encoder/projector | Strong architectural analogue |
| **RDB-PFN / Griffin** | No: dedicated relational pretraining/model | RDB | Full relational model | Alternative full-RFM strategy |

## 1. Towards Pretraining Text Encoders for TabPFN

**Mustafa Tajjar, Alexander Pfefferle, Lennart Purucker, Frank Hutter. arXiv:2606.04876 (2026).**  
https://arxiv.org/abs/2606.04876

The paper freezes both TabPFN and a sentence encoder, then trains only a lightweight adapter that maps each text representation into a short sequence of tokens in TabPFN's native embedding space. It explores roughly 1–10 tokens and studies token count, fusion depth, normalization, and initialization.

The proposed relational version is almost a direct substitution:

**text embedding → adapter → TabPFN tokens**

becomes

**relational neighborhood → relational encoder/projector → TFM tokens**.

The missing technical problem—and therefore potential novelty—is encoding variable-size, heterogeneous, typed, temporally valid RDB neighborhoods rather than a single text embedding.

## 2. GraphPFN: A Prior-Data Fitted Graph Foundation Model

**Dmitry Eremeev et al., ICML 2026 / arXiv:2509.21489.**  
https://arxiv.org/abs/2509.21489

GraphPFN starts from pretrained LimiX and inserts graph-neighborhood attention adapters after transformer blocks. The adapters perform sparse attention over one-hop neighbors. During graph-specific pretraining, the original TFM layers are frozen and only the graph adapters are updated.

This is the strongest precedent for the statement:

> A pretrained tabular FM can acquire structural reasoning by training only added structural modules.

GraphPFN differs from the proposed RDB setting because an RDB contains heterogeneous row/table types, typed foreign-key relations, asymmetric paths, temporal constraints, and multiple relation semantics. It is also more invasive because adapters are inserted repeatedly inside the backbone.

A key GraphPFN ablation is that graph-pretrained adapters outperform randomly initialized adapters tuned only downstream. This makes **task-specific adapter training vs cross-task relational-adapter pretraining** a crucial experiment.

## 3. OpenRFM: Dissecting Relational In-Context Learning

**Zhikai Chen et al., arXiv:2606.04320 (2026).**  
https://arxiv.org/abs/2606.04320

OpenRFM diagnoses limitations of relation-level ICL in the Relational Transformer and adds a batch-level ICL layer lifted from pretrained TabICL.

It explores several integration strategies:
- relational representation passed into a TabICL pipeline,
- a linear bridge into TabICL's ICL stage,
- interleaving TabICL ICL layers with relational-transformer blocks.

The paper finds that integration details, especially normalization and preservation of the pretrained TabICL pathway, matter substantially. This is highly relevant because the proposed method reverses the direction:

**OpenRFM:** relational backbone → TabICL ICL component  
**Proposed:** frozen TabICL backbone → compact learned relational interface

Thus OpenRFM does not eliminate the need for a learned relational backbone, while the proposed work asks whether that backbone can be replaced with a small adapter.

## 4. TFM-Retouche: A Lightweight Input-Space Adapter for TFMs

**Duong Nguyen, Mohammed Jawhar, Nicolas Chesneau. arXiv:2605.06047 (2026).**  
https://arxiv.org/abs/2605.06047

Retouche freezes a TFM, including experiments with TabICLv2, and trains a lightweight near-identity input adapter. Conceptually:

\[
x'=(1-lpha)x+lpha\Delta_\phi(x).
\]

A relational extension could use:

\[
z_v^{rel}=E_\phi(\mathcal N_t(v)), \qquad
x_v'=x_v+lpha P_\phi(z_v^{rel}),
\]

then feed \(x'_v\) to frozen TabICL.

This is likely the easiest first implementation. Its main limitation is that Retouche itself does not add external information; it only reshapes the existing row.

## 5. BETA / TabPFN Unleashed

**Siyang Liu, Han-Jia Ye. ICML 2025.**

BETA trains a small encoder before a frozen TabPFN. It is another strong precedent for a learned front-end around a fixed TFM. A relational version would replace the row-only encoder with an encoder that receives RDB context.

## 6. TuneTables

**Benjamin Feuer et al. NeurIPS 2024.**

TuneTables applies continuous context/prompt optimization to PFNs. It is the most relevant TFM precedent for soft prompting.

A static task prompt is not sufficient for relational prediction because every target row has a different neighborhood. The stronger extension is a **conditional prompt**:

\[
P_{v,t}=g_\phi(\mathcal N_t(v)),
\]

or, more ambitiously,

\[
P_{v,t}=g_\phi(\mathcal N_t(v),q_{	au}),
\]

where \(q_{	au}\) summarizes the task inferred from ICL demonstrations.

## 7. G2T-FM: Turning Tabular Foundation Models into Graph Foundation Models

**Dmitry Eremeev et al. arXiv:2508.20906 (2025).**  
https://arxiv.org/abs/2508.20906

G2T-FM computes fixed graph/neighborhood features and gives them to TabPFN. It forms a useful analogy:

| Graph setting | RDB setting |
|---|---|
| G2T-FM: fixed graph features + TFM | RDBLearn: fixed DFS/SQL features + TFM |
| GraphPFN: learned graph adapter + frozen TFM | **Proposed: learned relational adapter + frozen TFM** |

This is a particularly clean way to motivate the project.

## 8. LoGIC: Budgeted Context Construction for Graph ICL with TFMs

**Mingqi Yang, Zidong Guo, Jihui Yang, Wenming Zuo. arXiv:2609.05955 (September 9, 2026).**  
https://arxiv.org/abs/2609.05955

LoGIC studies how to construct graph ICL contexts for systems such as G2T-FM and GraphPFN. It separates the labeled ICL budget from an unlabeled graph halo available to structural adapters.

This suggests that a relational-TFM system should distinguish:
- **ICL support budget:** labeled target rows shown to TabICL,
- **relational evidence budget:** unlabeled/historical neighbors seen by the relational adapter.

These should be controlled separately in experiments.

## 9. RDBLearn and Parameter-Free Encoders

**Linjie Xu et al. (2026)** and **Linjie Xu, David Wipf, arXiv:2607.05476 (2026).**  
https://arxiv.org/abs/2607.05476

RDBLearn is the main scientific foil: it argues that simple parameter-free vertical aggregations can remain highly competitive when paired with strong single-table TFMs.

A learned relational adapter therefore needs to show benefits specifically where fixed aggregations are limited:
- cross-column interactions,
- task-dependent relation relevance,
- nonlinear neighbor interactions,
- temporal/order-sensitive patterns,
- typed path semantics.

The July 2026 follow-up also sharpens the argument that fixed/task-independent relational representations have theoretical limitations. This motivates a **task-conditioned relational adapter** rather than merely a generic GNN embedding.

## 10. RDB-PFN and Griffin

**RDB-PFN (arXiv:2603.03805, 2026)** directly pretrains a PFN for relational ICL on millions of synthetic relational tasks.  
**Griffin (ICML 2025)** builds a dedicated graph-centric relational database foundation model.

These represent the high-complexity alternative: build/pretrain an RFM rather than adapt a general TFM. They are important positioning references but are less similar architecturally.

## 11. Graph-to-frozen-LLM analogues

### G-Retriever — NeurIPS 2024
Uses a graph encoder/projector to produce continuous graph tokens that condition a pretrained/frozen LLM.

### TEA-GLM — arXiv:2408.14512
Aligns GNN representations with the token embedding space of a frozen LLM, with a small projector and emphasis on cross-dataset/task transfer.

### LLaGA — ICML 2024
Maps structure-aware graph sequences to LLM-compatible tokens.

### GraphAdapter — arXiv:2402.12984
Uses a GNN as a lightweight structural adapter around an LLM.

These collectively establish the architectural pattern:

\[
	ext{structured neighborhood}
ightarrow
	ext{specialist encoder}
ightarrow
	ext{few continuous tokens}
ightarrow
	ext{frozen foundation model}.
\]

That pattern transfers naturally to TFMs.

## 12. TFM PEFT / fine-tuning papers

**On Finetuning Tabular Foundation Models** (Rubachev et al., arXiv:2506.08982) and  
**Exploring Fine-Tuning for Tabular Foundation Models** (Tanna et al., arXiv:2601.09654, WWW 2026)

show that full or parameter-efficient fine-tuning is not uniformly beneficial across TFMs/tasks.

This is why **LoRA should be an important baseline rather than the main method**. LoRA efficiently changes TFM weights, but it does not itself solve the core problem of how multi-table evidence is represented.

A meaningful comparison is:

\[
	ext{RDBLearn/DFS + frozen TabICL}
\]

vs.

\[
	ext{RDBLearn/DFS + LoRA-TabICL}
\]

vs.

\[
	ext{learned relational token + frozen TabICL}.
\]

## Recommended architecture ladder

### A. Relational residual input adapter

\[
z_v=E_\phi(\mathcal N_t(v)),\qquad
x'_v=x_v+lpha P_\phi(z_v),\qquad
\hat y=f_	heta^{TFM}(x'_v,\mathcal C),
\]

with \(	heta\) frozen.

**Pros:** simplest integration, near-identity initialization, TabICL-friendly.  
**Cons:** compresses all relational evidence into row-feature space.

### B. Conditional relational soft-token adapter — recommended main direction

\[
Z_v=E_\phi(\mathcal N_t(v))\in\mathbb R^{k	imes d},
\]

\[
R_v=\mathrm{LN}(P_\phi(Z_v)),
\]

\[
\hat y=f_	heta^{TFM}(x_v,R_v,\mathcal C),\qquad 	heta\ 	ext{frozen}.
\]

Variants:
- one global relational token,
- \(k\) latent tokens,
- relation-specific tokens,
- hop-specific tokens,
- task-conditioned latent tokens.

This best combines the ideas of the TabPFN Text Adapter, G-Retriever/TEA-GLM, GraphPFN, and RDB-specific relational modeling.

### C. Internal GraphPFN-style relational adapters

\[
H^{(\ell+1)}
=
F_	heta^{(\ell)}(H^{(\ell)})
+
lpha_\ell A_\phi^{(\ell)}(H^{(\ell)},\mathcal N(v)).
\]

Freeze \(F_	heta\); train only \(A_\phi\).

**Pros:** relational evidence can influence reasoning at several depths.  
**Cons:** more invasive and backbone-specific; likely makes adapter pretraining more important.

### D. LoRA

Add low-rank updates to selected TabICL modules while providing relational features/tokens.

Use primarily as a PEFT control. LoRA alone does not expose RDB context.

### E. Static soft prompt

Learn global task/database prompt vectors. Useful as an extra-parameter control, but weaker than target-dependent relational prompts.

## Baselines that isolate the claim

| Model | Relational evidence | Learned adapter? | TFM tuned? | Purpose |
|---|---|---:|---:|---|
| TabICL | none | No | No | Row-only baseline |
| RDBLearn + TabICL | fixed DFS/SQL | No | No | Strong simple baseline |
| Static prompt + TabICL | none | Yes | No | Extra-parameter control |
| Retouche-style adapter + TabICL | row only | Yes | No | Generic adapter control |
| RDBLearn + LoRA TabICL | fixed DFS | Yes | LoRA | PEFT control |
| Relational encoder + linear head | raw RDB neighborhood | Yes | N/A | Does adapter solve task alone? |
| **Relational token + frozen TabICL** | raw RDB neighborhood | Yes | No | Main method |
| Internal relational adapters + frozen TabICL | raw RDB neighborhood | Yes | No | Higher-capacity variant |
| GraphSAGE / RelGT | raw relational graph | full model | Yes | Task-specific relational models |
| RT/OpenRFM/etc. | relational context | dedicated RFM | pretraining/full | RFM references |

## Critical ablations

1. **Token count:** \(k\in\{1,2,4,8\}\).
2. **Fusion point:** input/early, after row encoding, before global ICL, repeated internal adapters.
3. **Normalization:** none vs LayerNorm/standardization.
4. **Initialization:** random vs zero-output/near-identity.
5. **Relational corruption:** correct neighbors vs random neighbors, shuffled relation types, shuffled timestamps.
6. **Adapter-only probe:** relational encoder + linear head vs relational encoder + frozen TFM.
7. **Training scope:** per-task adapter vs multi-task adapter vs leave-one-database-out transfer.
8. **Neighborhood budget:** separate ICL support size from relational-neighbor budget.

## Recommended novelty positioning

Avoid the broad claim:

> We add a graph/relational adapter to a tabular foundation model.

GraphPFN is too close to that statement.

A stronger and more defensible question is:

> **Can a general pretrained tabular ICL model be converted into a relational predictor through a small conditional interface that maps heterogeneous, temporally valid RDB neighborhoods into its native representation space, without retraining the tabular foundation model?**

The differentiating ingredients are:
1. RDB rather than ordinary graph structure.
2. Frozen general TFM rather than dedicated relational pretraining.
3. Target-specific relational evidence rather than static prompt tuning.
4. Variable RDB neighborhood → compact native TFM tokens rather than fixed DFS statistics.
5. Potential task conditioning from ICL demonstrations.
6. Cross-task/database reusable adapter rather than per-task training.

## Recommended research sequence

1. **Benchmark:** TabICL; RDBLearn/DFS + TabICL.
2. **Generic adaptation controls:** row-only Retouche-style adapter; DFS + LoRA; static prompt if useful.
3. **Core method:** local relational encoder → one projected relational token → frozen TabICL.
4. **Capacity:** 2/4/8 tokens; relation/hop-aware tokens; task-conditioned selection.
5. **Foundation claim:** multi-task adapter training and leave-one-database-out transfer.
6. **Only if needed:** internal GraphPFN-style adapters.

## Bottom line

The best first main method is **relational encoder → 1–k conditional soft tokens → frozen TabICL**.

It is:
- more expressive than RDBLearn's fixed DFS aggregation,
- less invasive than GraphPFN-style internal modification,
- directly motivated by the TabPFN Text Adapter and graph-to-token literature,
- compatible with a strong "preserve the tabular prior" story,
- and leaves a clear path from task-specific adaptation to a reusable relational foundation adapter.

### References

- Tajjar et al. *Towards Pretraining Text Encoders for TabPFN*. 2026. https://arxiv.org/abs/2606.04876
- Eremeev et al. *GraphPFN: A Prior-Data Fitted Graph Foundation Model*. ICML 2026. https://arxiv.org/abs/2509.21489
- Chen et al. *OpenRFM: Dissecting Relational In-Context Learning*. 2026. https://arxiv.org/abs/2606.04320
- Nguyen et al. *TFM-Retouche: A Lightweight Input-Space Adapter for Tabular Foundation Models*. 2026. https://arxiv.org/abs/2605.06047
- Liu & Ye. *TabPFN Unleashed: A Scalable and Effective Solution to Tabular Classification Problems*. ICML 2025.
- Feuer et al. *TuneTables: Context Optimization for Scalable Prior-Data Fitted Networks*. NeurIPS 2024.
- Eremeev et al. *Turning Tabular Foundation Models into Graph Foundation Models*. 2025. https://arxiv.org/abs/2508.20906
- Yang et al. *LoGIC: Budgeted Context Construction for Node-Level Graph In-Context Learning with Tabular Foundation Models*. 2026. https://arxiv.org/abs/2609.05955
- Xu & Wipf. *Parameter-Free Encoders Remain Viable for RDB Foundation Models*. 2026. https://arxiv.org/abs/2607.05476
- Wang et al. *RDB-PFN: Relational In-Context Learning via Synthetic Pre-training with Structural Prior*. 2026. https://arxiv.org/abs/2603.03805
- Wang et al. *Griffin: Towards a Graph-Centric Relational Database Foundation Model*. ICML 2025.
- He et al. *G-Retriever*. NeurIPS 2024.
- Wang et al. *LLMs as Zero-shot Graph Learners: Alignment of GNN Representations with LLM Token Embeddings*. 2024. https://arxiv.org/abs/2408.14512
- Chen et al. *LLaGA*. ICML 2024.
- Huang et al. *GraphAdapter: Can GNN be a Good Adapter for LLMs?* 2024. https://arxiv.org/abs/2402.12984
- Grover et al. *Is One Token All It Takes? Graph Pooling Tokens for LLM-based GraphQA*. 2026. https://arxiv.org/abs/2604.00342
- Rubachev et al. *On Finetuning Tabular Foundation Models*. 2025. https://arxiv.org/abs/2506.08982
- Tanna et al. *Exploring Fine-Tuning for Tabular Foundation Models*. WWW 2026. https://arxiv.org/abs/2601.09654
