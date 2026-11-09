# Leakage and experimental-integrity audit

**Project:** MQTT/IoT intrusion detection with graph neural networks
**Audited:** 2026-09-18 · **Training performed during this audit:** none
**Machine-readable evidence:** `results/audit/leakage_audit.json`, `results/audit/split_overlaps.csv`
**Reproduce:** `python -m experiments.leakage_audit`

---

## 0. One correction to the premise, before anything else

The brief states that EndpointLookup reports ~99–100%. It does not, and the
correction matters because this baseline is the main diagnostic instrument.

| Protocol | EndpointLookup macro-F1 | accuracy | MCC |
|---|---|---|---|
| A — random flow-level | 0.8558 | 0.9561 | 0.8946 |
| B — duplicate-aware | 0.8531 | 0.9542 | 0.8915 |
| C — endpoint-grouped | 0.2267 | 0.8295 | **0.0000** |
| CS — source-grouped | 0.3085 | 0.8615 | **0.0000** |

A lookup table keyed on `(ip_src, ip_dst)` using **no traffic features at all**
reaches 0.956 accuracy under a random split and collapses to an MCC of exactly
zero once endpoints are withheld. That collapse is the single clearest piece of
evidence in this audit, and it is evidence about the *protocol*, not about any
model.

---

## A. Current experimental pipeline

```
data/raw/*.zip
  └─ data_loader.load_representation()          extract, concat 5 captures
       ├─ binary_label   ← is_attack (verbatim)
       ├─ multiclass_label ← is_attack AND capture   [see §E]
       └─ capture_scenario_label, scenario, source_file  (provenance only)
            │
            ▼
  experiments/common.py :: PreparedData          ← ONE object per experimental cell
       1. feature_columns(df)                    drop labels, IPs, provenance
       2. make_split(protocol, …)                  split FIRST
       3. FlowPreprocessor.fit(df.iloc[train])     THEN fit preprocessing
       4. .transform(df)                           apply to all rows
       5. assert_no_duplicate_rows_across_splits() measure, do not enforce
       6. build_flow_graph(train_idx=train)        node feats + msg mask = train
            │
            ▼
  run_classical / run_neural                     identical splits for every model
            │
            ▼
  _finalise → RunRecord → results/raw/*.json     metrics, config, seed, hashes
```

The ordering in steps 2–4 is the important one: the split is constructed before
any fitted transformation exists.

---

## B. Current split methodology

| Protocol | Construction | Unit held out |
|---|---|---|
| A | stratified random 70/15/15 | individual flows |
| B | group by exact predictive feature vector, then stratified split of groups | duplicate groups |
| C | withhold whole **destination** endpoints, constrained draw | victim endpoints |
| CS | withhold whole **source** endpoints | source endpoints |
| L | leave-one-endpoint-out over 20 candidate destinations, model seed pinned | one endpoint per fold |

Protocol C/CS/L accept a draw only if the test split has ≥3 classes (≥2 for L),
holds ≤35% of rows, and contains no class absent from training; otherwise the
protocol declares itself not applicable rather than degrading.

---

## C. Current graph construction

Nodes are unique IP endpoints (24 for both representations). Each flow record is
one directed edge; flows sharing an ordered pair are kept as parallel edges.
Node attributes are 25 behavioural aggregates; edge attributes are the
preprocessed flow features.

Measured, not assumed (`leakage_audit.json → graph`):

- `node_index` is built over **all rows**, so the node set is 24 in every
  protocol regardless of split.
- Message passing uses **training edges only** (`message_edge_mask`).
- Node features aggregate **training edges only**. Divergence between
  train-only and all-flow aggregation is large (max absolute difference
  5.3 × 10⁸ under protocol A, affecting all 24 nodes and all 25 features),
  which confirms the restriction is real rather than cosmetic.
- Under protocol C, node 23 appears only in test and carries **all-zero**
  training features, as expected.

---

## D. Feature list and leakage classification

**No banned, target-derived, capture-identifying or endpoint-identifying column
is present in the feature matrix of either representation.** Verified
independently of the pipeline's own guard.

| Representation | Columns total | In feature matrix | Violations |
|---|---|---|---|
| uni-flow | 24 | 16 (→ 27 after row-wise derivations) | 0 |
| bi-flow | 37 | 29 (→ 40 after derivations) | 0 |

Excluded and why: `ip_src`, `ip_dst` (endpoint identity — used only as graph
node identity), `is_attack`, `binary_label`, `multiclass_label` (targets),
`scenario`, `capture_scenario_label`, `source_file` (capture identity,
provenance only).

Feature sets A/B/C/D requested in Phase 3 are **identical here**, because there
is no leaky column to remove. The endpoint-masked and scenario-masked conditions
therefore have to be realised by *evaluation protocol*, not by dropping a
column — endpoint identity reaches the model only through graph structure.

One feature deserves explicit comment rather than removal: **`prt_dst`**. On
this testbed the destination port nearly determines the attack class (22 →
Sparta, 1883 → MQTT brute-force, random high ports → scans). It is not
target-derived and it *is* observable at deployment, so removing it would be
wrong. But it means a large part of the task is solvable by a port lookup, and
the paper should say so.

---

## E. Label generation

```
binary_label     = is_attack                                   (verbatim)
multiclass_label = 0                    if is_attack == 0
                 = capture_scenario_label  otherwise
```

This is deliberate and was a correction made during development. Every attack
capture also contains the testbed's benign background traffic — 204,350
uni-flow rows, **41.2% of the release**. Labelling by capture alone (the
obvious reading of "derive the class from the scenario file") would relabel all
204,350 benign flows as attacks. `is_attack` decides *whether*; the capture
decides *which*. Neither is used alone.

Consequence for Phase 5 Experiment C: because `multiclass_label` is by
construction a 1:1 function of the capture for attack rows, a **capture-disjoint
multi-class split is structurally impossible** — withholding a capture removes
that class from training entirely. Capture-disjointness is only meaningful as
*unseen-attack detection under the binary task*. See §I.

---

## F. Leakage-risk table (measured, uni-flow, multi-class, seed 42)

Shared between TRAIN and TEST:

| Protocol | flow IDs | src IPs | dst IPs | pairs | five-tuples | feature vectors | test rows affected | captures | endpoint-disjoint |
|---|---|---|---|---|---|---|---|---|---|
| A | 0 | 23 | 24 | 76 | 15,183 | 3,836 | **12.31%** | 5 | ✗ |
| B | 0 | 23 | 24 | 76 | 15,266 | **0** | **0.00%** | 5 | ✗ |
| C | 0 | 7 | **0** | **0** | **0** | 2,769 | 16.60% | 5 | ✗ |
| CS | 0 | **0** | 4 | **0** | **0** | 1,309 | 12.94% | 5 | ✗ |
| L | 0 | 3 | **0** | **0** | **0** | 2,240 | 13.47% | 4 | ✗ |

Protocol A's duplicate-vector overlap is stable across seeds: 12.31%, 12.32%,
12.34% of test rows for seeds 42, 123, 2024.

### Findings, in order of severity

**F1 — CONFIRMED DEFECT: GA-EBT feature selection leaks across seeds.**
`experiments/common.py:261-267`. The genetic search result is cached at

```
ga_search_{representation}_{task}_protocol{protocol}.json      ← no seed, no fold
```

and `search()` returns the cached genome whenever the file exists. The search
runs once, on **seed 42's** training split. For seeds 123, 2024, 3407 and 2026
the selected feature subset and ensemble hyperparameters were therefore chosen
using data that includes those seeds' **test** rows. This is textbook
feature-selection leakage and it affects every GA-EBT number for seeds other
than 42. It does not affect any other model. *This was introduced by me as a
compute-saving measure and its leakage implication was not thought through.*

**F2 — CONFIRMED DESIGN GAP: no protocol is endpoint-disjoint in the required
sense.** Phase 5 Experiment B asks for TRAIN endpoints ∩ TEST endpoints = ∅.
Protocol C withholds destinations only, so a withheld victim still appears as a
*source* in training; CS is the mirror image. The `endpoint_disjoint` column is
`False` everywhere.

Whether a fully disjoint split is even possible was measured: the endpoint graph
is a **single connected component** (24 nodes, 56 undirected pairs). Any
bipartition requires discarding every flow that crosses it. The best search over
400 random bipartitions gave 315,182 test flows and 27,387 train flows while
**discarding 152,956 flows (30.9%)**. So Experiment B is achievable but only by
throwing away roughly a third of the data and accepting a badly unbalanced
split. That trade-off is a decision for you, not for me.

**F3 — CONFIRMED: no protocol is capture-disjoint.** All five captures appear in
both train and test everywhere (four under protocol L only because one capture's
endpoint was withheld). See §E for why the multi-class form is impossible and
what the meaningful binary form would be.

**F4 — RESIDUAL TRANSDUCTIVE ELEMENT (low severity, by design).** The node index
spans all splits, so an endpoint appearing only in test exists as a node during
training. It carries all-zero behavioural features and no training edge, so no
test traffic and no test label reaches the model through it; what leaks is the
bare existence of an address. A strictly inductive variant that rebuilds the
index from training endpoints alone is straightforward and is recommended in §I.

**F5 — NOT A LEAK, but the actual explanation of the high numbers.** Endpoint
identity is very nearly sufficient for the label:

| | uni-flow | bi-flow |
|---|---|---|
| distinct ordered pairs | 106 | 57 |
| pairs carrying exactly one class | 86 | 40 |
| share of flows in single-class pairs | **85.1%** | **78.2%** |
| flow-weighted mean pair purity | **0.9565** | 0.9210 |

0.9565 is an upper bound on the accuracy of a detector that sees only the
endpoint pair, and it is essentially what EndpointLookup achieves (0.9561). Any
model given the endpoint pair can reach that number without learning anything
about traffic.

**F6 — Preprocessing scope: clean.** Split is constructed before any fitted
object exists (`common.py:88-104`); `FlowPreprocessor.transform` raises if
called before `fit`; winsorisation bounds, medians, the quantile transform and
the constant/near-zero-variance drop are all fitted on `df.iloc[train]` only.
Class weights come from training labels only (`losses.class_weights`). The node
scaler is fitted on train-derived node features only.

---

## G. Duplicate-risk analysis

| | uni-flow | bi-flow |
|---|---|---|
| rows | 495,525 | 259,379 |
| exact duplicate rows (all columns) | **0** | **0** |
| duplicate predictive feature vectors | 56,495 (11.4%) | 38,370 (14.8%) |
| largest duplicate group | 17 | 17 |
| duplicate five-tuples | 73,996 | 37,071 |

There are no exact duplicate rows; duplication appears only after IPs and labels
are dropped, which is expected for scripted scan traffic. The largest group is
17 rows, so no single flow signature dominates.

**The decisive test has already been run.** Protocol B removes duplicate overlap
entirely (0 shared feature vectors, 0.00% of test rows) and performance did
**not** drop: GAT 1.0000 → 1.0000, E-GAT-IDS 1.0000 → 0.99996, XGBoost
0.9979 → 0.9982, MLP 0.9875 → 0.9840 macro-F1. **Duplicates are therefore not
the cause of the near-perfect results.** No rows were removed, and none should
be: repeated scan flows are genuine traffic.

---

## H. Files and functions requiring modification

| # | File / function | Defect | Fix |
|---|---|---|---|
| 1 | `experiments/common.py:261-267` `run_classical` | GA-EBT cache key omits seed/fold (F1) | include seed in the cache path, or disable caching; re-run GA-EBT for seeds ≠ 42 |
| 2 | `src/split.py` | no fully endpoint-disjoint protocol (F2) | add protocol `E`: bipartition endpoints, discard cross-partition flows, report the discard fraction |
| 3 | `src/split.py` | no capture-disjoint protocol (F3) | add protocol `U`: binary unseen-attack detection, train on normal + k−1 attack captures, test on the held-out capture |
| 4 | `src/graph_builder.py` `build_node_index` | node index spans all splits (F4) | add `node_index_source="train"` for a strictly inductive variant; keep current behaviour as the reported default with the contrast measured |
| 5 | `src/preprocessing.py` `assert_no_duplicate_rows_across_splits` | measures but never enforces | keep as measurement; make the quality-control script fail when protocol A's overlap exceeds a declared threshold |
| 6 | `experiments/quality_control.py` | does not check endpoint or capture disjointness | add both as explicit checks per protocol |

Nothing in this list is a model change. No model, loss, architecture or
hyperparameter requires modification on the evidence gathered.

---

## I. Recommended experiment sequence

1. **Fix F1 and re-run GA-EBT only** for seeds 123, 2024, 3407, 2026. Cheap,
   and the current GA-EBT multi-seed numbers cannot be reported until it is done.
2. **Add protocol E (fully endpoint-disjoint)** and report the 30.9% discard
   explicitly as the price of the guarantee. Run the full model set.
3. **Add protocol U (unseen-attack, binary)** — five folds, each withholding one
   attack capture. This is the only scientifically meaningful form of
   capture-disjointness here and it answers a question no protocol currently
   does: does the detector flag an attack type it never saw?
4. **Finish leave-one-endpoint-out** (20 folds × 13 models, uni-flow,
   multi-class). Already started; 7 of 260 records complete.
5. **Run the inductive node-index contrast** (F4) on protocol C to bound the
   residual transductive effect.
6. Only then: ablation, statistics, efficiency, manuscript.

### What the evidence already supports

The near-perfect results are **not** produced by a coding leak. Features are
clean, preprocessing is correctly scoped, message passing and node features are
train-only, and removing every duplicate (protocol B) changes nothing. They are
produced by two properties of the dataset: the task is close to linearly
separable from the destination port and protocol, and endpoint identity alone
fixes 85% of the labels. Under endpoint-grouped evaluation performance drops
sharply and becomes wildly unstable — the same GCN scored 0.4756, 0.5440 and
0.9927 macro-F1 across three draws — which is why leave-one-endpoint-out was
added.

Per the brief: that drop is **not** a failure. It is the measurement that the
random-split protocol was never capable of making.

---

## Audit limitations

This audit inspected the pipeline in this container, which is the same code
delivered to `C:\Udaya\MITS\Research Article\Article 3\research_project`. It did
not train models; all performance figures quoted are from run records produced
before the audit and are identified as such. Near-duplicate (as opposed to
exact-duplicate) analysis was not performed — at 495,525 rows a pairwise
similarity search is expensive, and the exact-duplicate result plus protocol B
already bound the memorisation contribution. No timestamp columns exist, so
temporal split feasibility (Experiment D) is nil and was not investigated
further.
