**Executive Summary**  
This SFT plan trains a biological multiplex foundation model—a world model / digital twin for systems biology interpretation—that internalizes the full-brain multiplex as a versioned, layer-aware, entity-aligned graph atlas. The curriculum moves from closed-book identity/schema learning to open-book matrix/vector/table QA and then structured tool-call supervision, using exact, validator-checkable questions over Ensembl IDs, gene symbols, multiplex/layer tags, local topology, global distance matrices, RWR-LOE rank vectors, MENTOR-EV/RWR-LOE module set algebra, cohesion statistics, calibration negatives, and provenance. This supports the broader MENTOR-RL goal of mechanistic exploration over large biological multiplexes, while leaving richer free-form interpretation primarily for DPO and later RL.

**Training Curriculum/Staging Summary**

* **Stage 1:** **closed-book entity/schema grounding** teaches the model the coordinate system of the multiplex: Ensembl IDs, gene symbols, layer tags, multiplex IDs, module IDs, and distance/rank conventions; the rationale is that a world model cannot store graph structure reliably unless entities and graph contexts are represented consistently.  
* **Stage 2: closed-book atlas priors** adds layer families, module-source distinctions, global topology vocabulary, and calibration rules; the rationale is to build broad multiplex context before asking for exact computation.   
* **Stage 3: open-book table/vector/matrix QA** trains exact extraction and comparison from neighbor tables, RWR-LOE vectors, distance shards, and module tables; the rationale is that many full-multiplex facts should be usable as context.  
* **Stage 4: global module and cohesion QA** trains set algebra, overlap, containment, within-vs-random distance, clustering ratio, and layer/cell-type specificity; the rationale is to make the representation module-aware and globally contextual.  
* **Stage 5: structured tool-call SFT** teaches when and how to call graph/RWR tools with biological arguments rather than raw file paths; the rationale is to prepare the model for DPO trajectory generation using deterministic, auditable tools. This staging matches the broader SFT→ DPO→ GRPO Reinforcement Learning plan.

**Question Type Summaries**

| Question type | Contribution to the main goal |
| ----- | ----- |
| **Entity normalization / Ensembl-symbol alignment** | Teaches the model that Ensembl IDs are the canonical graph keys and symbols are aliases, preventing entity drift across rank vectors, module tables, and graph tools. |
| **Multiplex identifiers and standardized tags** | Teaches the model to distinguish graph versions, layer scopes, module sources, and layer families, which is essential for a true multiplex world model rather than a flat graph model. |
| **Layer metadata and layer membership** | Teaches node presence, layer coverage, layer family, and layer provenance, so the model understands that evidence can be layer-specific and incomplete. |
| **Local topology: edges, neighbors, paths, subgraphs, components** | Provides atomic graph facts that ground later reasoning; these are the “grammar” of the multiplex. |
| **Path layer counts and layer specificity** | Teaches that paths and edges are supported by different biological evidence layers, preventing overclaims such as treating coexpression as physical interaction. |
| **RWR-LOE rank/vector QA** | Teaches global propagation-based proximity from seed genes and supports recovery/refinement behavior by identifying topologically close candidates beyond immediate neighbors. RWRtoolkit is designed for this kind of multiplex exploration and ranks genes relative to seed sets across multiple lines of evidence. |
| **Sharded distance-matrix QA** | Trains the model to use global all-gene context, not just local neighborhoods, by comparing pairwise distances, distance percentiles, and closest entities from matrix rows. |
| **MENTOR-EV / RWR-LOE module set algebra** | Teaches module identity and module relationships: subset, superset, intersection, difference, Jaccard, containment, parent/child clades, and cross-source agreement. |
| **Global cohesion and null comparison** | Teaches whether modules are topologically meaningful relative to random expectation using within-clade distance, clustering ratio, density, conductance, and cell-type-specific cohesion. This aligns with the MENTOR-eval goal of testing whether MENTOR clades are more cohesive than chance. |
| **Calibration negatives** | Teaches the model not to overclaim from absent edges, absent genes, top-k truncation, hub proximity, or layer-specific evidence gaps. |
| **Structured tool-call QA** | Teaches schema-valid action construction and evidence parsing so the model can interact with deterministic tools during later DPO/RL without seeing raw CLI details. |
| **Structured state updates from evidence** | Bridges SFT to DPO by teaching evidence-backed updates to `predicted_gene_ids`, `relationship_status`, and continuation state while avoiding long-form interpretation overtraining. |

**Numerical Output and Multiplex-Difficulty Policy**

SFT should teach numerical literacy over the multiplex, not force the model to memorize arbitrary floating-point values from model weights. Closed-book examples should emphasize identifiers, layer/module semantics, rank/distance directionality, qualitative bins, and selected stable facts. Exact numerical outputs such as RWR scores, distances, empirical p-values, edge weights, and full matrix entries should usually be supplied through open-book contexts or structured tool observations, then extracted, sorted, compared, rounded, or combined by the model. Integer ranks, counts, and set membership can be exact; floating-point fields should use a fixed precision in the answer schema and tolerance-aware validators.

Recommended numeric scoring rules:

* **Ranks, counts, IDs, and set membership:** exact match.
* **Floating-point values:** absolute or relative tolerance after fixed rounding, usually 3–5 significant figures.
* **Top-k/ranked lists:** precision@k, recall@k, ordering accuracy, and exact exclusion of seeds when required.
* **Derived metrics:** recompute from the returned sets/tables and compare with tolerance.
* **Calibration bins:** classify into `unusually_close`, `typical`, `far`, `not_in_top_k`, or `insufficient_context` rather than requiring memorized raw values.
* **Closed-book numeric questions:** use sparingly, and prefer coarse bins or stable integer facts over exact floats.

The model should learn that exact graph numbers live in rank-vector caches, distance-matrix shards, module tables, and deterministic tools. The checkpoint should store the coordinate system, global topology priors, and tool-use policy; it should not be expected to store every edge weight, RWR score, distance-matrix entry, or p-value in its parameters.

**Multiplex Learning Evaluation, Scaling, and Compute Plan**

Evaluate whether the model is learning the multiplex with held-out, validator-checkable suites across identity alignment, layer metadata, local topology, global distance/rank structure, module set algebra, calibration negatives, and structured tool calls. Add representation probes such as nearest-neighbor consistency, layer-family separability, module membership clustering, distance-bin prediction, and corrupted-multiplex contrast tests, where the model must distinguish true multiplex facts from shuffled layers, rewired edges, swapped module IDs, or mismatched rank-vector caches. For smaller-model extrapolation, train a ladder of models and data scales on the same curriculum, fit scaling curves on exact QA accuracy, tool-call validity, rank/vector ordering, module overlap accuracy, and downstream exact-positive trajectory yield, then use the 120B run only after the curves show non-saturating gains. With abundant compute, spend it on systematic ablations: data mixture sweeps, curriculum-stage sweeps, closed-book vs open-book ratios, numeric tolerance choices, model-size scaling, repeated seeds, teacher/strong-model distillation, hard-negative generation, and large held-out evaluations rather than one monolithic run.

**Difficulty Annotation Legend**

Questions marked `[DIFFICULT: ...]` are expected to be hard for multiplex training because they require exact memorization, large context extraction, global graph geometry, multi-set algebra, numerical computation, tool-schema selection, or evidence-to-state updates. These should be introduced later in the curriculum, emphasized in open-book/tool-call modes, and evaluated with validators rather than raw string matching. Unmarked questions are still useful but should be easier or more foundational.

**Difficult Question Inventory**

| Difficulty source | Most affected questions | Mitigation |
| ----- | ----- | ----- |
| Closed-book exact graph facts | 8, 10–15, 18, 21 | Prefer open-book/tool-call variants; reserve closed-book for coarse facts and sampled stable examples. |
| Numerical extraction and comparison | 23–38, 46–47, 56–60, 75–76 | Provide rank vectors, matrix shards, or tool outputs; use fixed rounding and tolerance-aware validators. |
| Large module set algebra | 41–53, 55, 77 | Provide module tables/overlap matrices; validate exact sets and recomputed Jaccard/containment. |
| Global multiplex context | 31–36, 55–63, 81 | Use distance shards, distribution summaries, and module-module distance tables rather than memorized all-pairs values. |
| Tool choice and schema control | 69–74, 78–79 | Train structured biological arguments only; reject raw CLI/file paths. |
| SFT-to-DPO bridge behavior | 80 | Keep outputs schema-valid and evidence-backed; do not reward unsupported exact membership. |

**Training Set Breakdown**  
---

* 8% entity normalization, Ensembl/gene-symbol alignment, and schema tags  
* 10% multiplex identifiers, layer tags, layer membership, and layer metadata  
* 12% edge existence, monoplex/multiplex neighbors, and local topology  
* 8% shortest paths and path layer counts  
* 10% induced subgraphs, components, shared neighbors, degree/hubness  
* 15% RWR-LOE rank/vector lookup, vector comparison, and sharded distance-matrix QA  
* 20% MENTOR-EV / RWR-LOE module set algebra: subsets, supersets, intersections, overlap, containment  
* 10% global multiplex context, module cohesion, calibration negatives, and null/random comparison  
* 7% open-book/tool-call QA over structured tables, rank vectors, distance shards, and provenance

**Core Multiplex QA**

1. **Standard Example Metadata / Tags**  
   Every question should carry standardized metadata fields so the model learns which multiplex, layer family, module source, and identifier namespace the question belongs to.  
   Example tags:

{  
  "schema\_version": "multiplex\_sft\_v1",  
  "book\_mode": "closed\_book | open\_book | tool\_call",  
  "question\_family": "edge\_existence | rwr\_loe\_vector | module\_set\_algebra | distance\_matrix | calibration",  
  "multiplex\_id": "full\_brain\_multiplex\_v1",  
  "store\_id": "full\_brain\_multiplex\_store",  
  "flist\_id": "full\_brain\_flist",  
  "layer\_scope": "single\_layer | layer\_subset | all\_layers",  
  "layer\_ids": \["HumanNetV3:string\_ppi:global:all:v3"\],  
  "layer\_families": \["ppi", "coexpression", "tf\_target", "bulk\_pen", "scpen"\],  
  "entity\_namespace": "ensembl\_gene\_id\_primary",  
  "module\_source": "mentor\_ev | rwr\_loe | none",  
  "answer\_format": "json"  
}

Design rule: Ensembl IDs are the canonical graph keys. Gene symbols are secondary display aliases.

1. **Entity Normalization: Gene Symbol to Ensembl ID**  
   Question: “Normalize gene symbol `A` for `full_brain_multiplex_v1`. Return the canonical graph identity.”  
   Answer shape:

{  
  "status": "resolved",  
  "gene\_id": "ENSG...",  
  "gene\_symbol": "A",  
  "canonical\_entity": "\<GENE:ENSG...|A\>",  
  "multiplex\_id": "full\_brain\_multiplex\_v1"  
}

Negative answer:

{  
  "status": "not\_found",  
  "input": "A",  
  "allowed\_claim": "No graph lookup should be performed until the gene is resolved to a canonical Ensembl ID."  
}

2. **Entity Normalization: Ensembl ID to Gene Symbol**  
   Question: “What display symbol is associated with `ENSG_A` in this graph version?”  
   Answer shape:

{  
  "gene\_id": "ENSG\_A",  
  "gene\_symbol": "A",  
  "canonical\_entity": "\<GENE:ENSG\_A|A\>"  
}

3. **Ambiguous Symbol / Alias Resolution**  
   Question: “The symbol `A` maps to multiple Ensembl IDs. What should the model do before graph lookup?”  
   Answer shape:

{  
  "status": "ambiguous",  
  "candidate\_gene\_ids": \["ENSG\_A1", "ENSG\_A2"\],  
  "action": "ask\_for\_disambiguation\_or\_use\_context",  
  "allowed\_claim": "Do not perform graph lookup until the canonical Ensembl gene ID is resolved."  
}

4. **Cross-Context Entity Alignment**  
   Question: “The RWR vector uses `ENSG_A`, but the module table uses symbol `A`. Are these the same graph entity?”  
   Answer shape:

{  
  "same\_entity": true,  
  "gene\_id": "ENSG\_A",  
  "gene\_symbol": "A",  
  "reason": "The symbol maps to the same canonical Ensembl gene identifier used in the vector."  
}

5. **Multiplex Identifier Parsing**  
   Question: “Parse multiplex identifier `full_brain_multiplex_v1`. What graph context does this refer to?”  
   Answer shape:

{  
  "multiplex\_id": "full\_brain\_multiplex\_v1",  
  "species": "Homo sapiens",  
  "context": "full\_brain",  
  "version": "v1",  
  "graph\_type": "biological\_multiplex"  
}

6. **Layer Tag Parsing**  
   Question: “Parse layer tag `HumanNetV3:string_ppi:global:all:v3` into source, modality, tissue/context, cell type, and version.”  
   Answer shape:

{  
  "layer\_id": "HumanNetV3:string\_ppi:global:all:v3",  
  "source": "HumanNetV3",  
  "modality": "string\_ppi",  
  "context": "global",  
  "cell\_type\_or\_region": "all",  
  "version": "v3",  
  "layer\_family": "ppi"  
}

7. **Layer Family Classification**  
   Question: “Is layer `bulkPEN:GTEx_v9:brain:hippocampus:v1` a PPI, coexpression, TF-target, bulk-PEN, scPEN, or other layer?”  
   Answer shape:

{  
  "layer\_id": "bulkPEN:GTEx\_v9:brain:hippocampus:v1",  
  "layer\_family": "bulk\_pen",  
  "tissue": "brain",  
  "region": "hippocampus"  
}

8. **Gene Layer Membership** [DIFFICULT: closed-book layer presence]  
    Difficulty note: Hardest as a closed-book fact. Prefer open-book/tool-call examples unless evaluating sampled memorized layer-presence facts.
   Question: “Which layers contain gene `ENSG_A`?”  
   Answer shape:

{  
  "gene\_id": "ENSG\_A",  
  "gene\_symbol": "A",  
  "multiplex\_id": "full\_brain\_multiplex\_v1",  
  "layer\_count": 4,  
  "layers": \["layer\_1", "layer\_2", "layer\_3", "layer\_4"\]  
}

9. **Nodes By Layer**  
   Question: “Which of these genes are present in `bulkPEN:GTEx_v9:brain:hippocampus:v1`?”  
   Answer shape:

{  
  "layer\_id": "bulkPEN:GTEx\_v9:brain:hippocampus:v1",  
  "present\_gene\_ids": \["ENSG\_A", "ENSG\_C"\],  
  "absent\_gene\_ids": \["ENSG\_B"\],  
  "present\_count": 2,  
  "absent\_count": 1  
}

10. **Layer-Specific Edge Existence** [DIFFICULT: closed-book exact graph fact]  
    Difficulty note: Closed-book edge existence can become memorization of sparse graph facts. Prefer open-book/tool-call variants for most examples.
    Question: “In layer `HumanNetV3:string_ppi:global:all:v3`, is there an edge between gene A and gene B? If so, return the edge.”  
    Answer shape:

{  
  "edge\_exists": true,  
  "source\_gene\_id": "ENSG\_A",  
  "source\_gene\_symbol": "A",  
  "target\_gene\_id": "ENSG\_B",  
  "target\_gene\_symbol": "B",  
  "layer\_id": "HumanNetV3:string\_ppi:global:all:v3",  
  "edge": \["ENSG\_A", "ENSG\_B"\],  
  "weight": 0.84  
}

Negative answer:

{  
  "edge\_exists": false,  
  "source\_gene\_id": "ENSG\_A",  
  "target\_gene\_id": "ENSG\_B",  
  "layer\_id": "HumanNetV3:string\_ppi:global:all:v3",  
  "allowed\_claim": "No edge is recorded between these genes in this layer and graph version."  
}

11. **Multiplex Edge Existence** [DIFFICULT: multiplex layer aggregation]  
    Difficulty note: Requires aggregating layer support across the multiplex; evaluate layer lists exactly when context is provided.
    Question: “Across all full-brain multiplex layers, is gene A connected to gene B? List the layers.”  
    Answer shape:

{  
  "edge\_exists\_any\_layer": true,  
  "source\_gene\_id": "ENSG\_A",  
  "target\_gene\_id": "ENSG\_B",  
  "supporting\_layers": \["layer\_1", "layer\_2"\],  
  "supporting\_layer\_count": 2  
}

**This teaches that an edge can exist in some layers but not others.**

12. **Layer-Specific Direct Neighbors** [DIFFICULT: closed-book neighbor list]  
    Difficulty note: Full neighbor lists are high-cardinality facts. Prefer supplied neighbor tables or tool outputs.
    Question: “What are gene A’s direct neighbors in `HumanNetV3:coexpression:global:all:v3`?”  
    Answer shape:

{  
  "query\_gene\_id": "ENSG\_A",  
  "layer\_id": "HumanNetV3:coexpression:global:all:v3",  
  "neighbor\_count": 3,  
  "neighbors": \[  
    {"gene\_id": "ENSG\_B", "gene\_symbol": "B", "weight": 0.91},  
    {"gene\_id": "ENSG\_C", "gene\_symbol": "C", "weight": 0.73},  
    {"gene\_id": "ENSG\_D", "gene\_symbol": "D", "weight": 0.66}  
  \]  
}

**Open-book version gives a neighbor table and asks for filtering/deduplication.**

13. **Multiplex Direct Neighbors** [DIFFICULT: multiplex neighbor-to-layer map]  
    Difficulty note: Requires deduplication plus layer provenance; use open-book/tool-call supervision.
    Question: “What are gene A’s unique direct neighbors across all layers, and which layers support each neighbor?”  
    Answer shape:

{  
  "query\_gene\_id": "ENSG\_A",  
  "unique\_neighbor\_count": 3,  
  "neighbors": \[  
    {  
      "gene\_id": "ENSG\_B",  
      "gene\_symbol": "B",  
      "supporting\_layers": \["layer\_1", "layer\_2"\],  
      "supporting\_layer\_count": 2  
    },  
    {  
      "gene\_id": "ENSG\_C",  
      "gene\_symbol": "C",  
      "supporting\_layers": \["layer\_3"\],  
      "supporting\_layer\_count": 1  
    }  
  \]  
}

**Answer should be a neighbor-to-layer map, not just a flat list.**

14. **Shortest Path, Monoplex** [DIFFICULT: path search]  
    Difficulty note: Shortest paths should usually be computed by a tool or supplied as context, then parsed by the model.
    Question: “What is the shortest path from gene A to gene B in `HumanNetV3:string_ppi:global:all:v3`? Return nodes and edges.”  
    Answer shape:

{  
  "path\_exists": true,  
  "layer\_id": "HumanNetV3:string\_ppi:global:all:v3",  
  "source\_gene\_id": "ENSG\_A",  
  "target\_gene\_id": "ENSG\_B",  
  "path\_gene\_ids": \["ENSG\_A", "ENSG\_X", "ENSG\_B"\],  
  "path\_edges": \[  
    \["ENSG\_A", "ENSG\_X"\],  
    \["ENSG\_X", "ENSG\_B"\]  
  \],  
  "hop\_count": 2  
}

Negative answer:

{  
  "path\_exists": false,  
  "layer\_id": "HumanNetV3:string\_ppi:global:all:v3",  
  "allowed\_claim": "No path is recorded in this layer."  
}

15. **Shortest Path, Multiplex** [DIFFICULT: multiplex path provenance]  
    Difficulty note: Requires path extraction plus per-edge layer support; use structured outputs and validators.
    Question: “What is the shortest path from gene A to gene B across the aggregate multiplex? Return nodes, edges, and edge layers.”  
    Answer shape:

{  
  "path\_exists": true,  
  "multiplex\_id": "full\_brain\_multiplex\_v1",  
  "path\_gene\_ids": \["ENSG\_A", "ENSG\_X", "ENSG\_B"\],  
  "hop\_count": 2,  
  "path\_edges": \[  
    {  
      "edge": \["ENSG\_A", "ENSG\_X"\],  
      "supporting\_layers": \["coexpression", "bulk\_pen"\]  
    },  
    {  
      "edge": \["ENSG\_X", "ENSG\_B"\],  
      "supporting\_layers": \["string\_ppi"\]  
    }  
  \]  
}

16. **Path Layer Decomposition**  
    Question: “This shortest path has edges on `coexpression`, `string_ppi`, and `TFs:brain`. How many path edges are supported by each layer?”  
    Answer shape:

{  
  "path\_edge\_count": 4,  
  "layer\_edge\_counts": {  
    "coexpression": 2,  
    "string\_ppi": 1,  
    "TFs:brain": 1  
  }  
}

17. **Compare Monoplex vs Multiplex Paths**  
    Question: “Gene A and gene B have no path in `HumanNetV3:string_ppi` but do have a multiplex path. What does that mean?”  
    Answer shape:

{  
  "monoplex\_path\_exists": false,  
  "multiplex\_path\_exists": true,  
  "allowed\_claim": "The relationship is not supported in that single layer, but a cross-layer network route is recorded in the aggregate graph.",  
  "disallowed\_claim": "Do not claim direct physical interaction from a multiplex path alone."  
}

18. **Induced Subgraph** [DIFFICULT: subgraph extraction]  
    Difficulty note: Induced subgraphs over arbitrary gene sets should be open-book/tool-generated rather than memorized.
    Question: “Given genes A, B, C, D, return all recorded edges among them in layer X.”  
    Answer shape:

{  
  "query\_gene\_ids": \["ENSG\_A", "ENSG\_B", "ENSG\_C", "ENSG\_D"\],  
  "layer\_id": "layer\_X",  
  "present\_gene\_ids": \["ENSG\_A", "ENSG\_B", "ENSG\_C", "ENSG\_D"\],  
  "missing\_gene\_ids": \[\],  
  "edge\_count": 2,  
  "edges": \[  
    {"source": "ENSG\_A", "target": "ENSG\_B", "weight": 0.72},  
    {"source": "ENSG\_C", "target": "ENSG\_D", "weight": 0.65}  
  \]  
}

**Also ask for `combined_edge_count` across all layers.**

19. **Connected Component**  
    Question: “Do these genes fall in one connected component in layer X? If not, list components.”  
    Answer shape:

{  
  "single\_component": false,  
  "component\_count": 2,  
  "components": \[  
    \["ENSG\_A", "ENSG\_B"\],  
    \["ENSG\_C", "ENSG\_D"\]  
  \]  
}

Useful for insufficient-support behavior.

20. **Shared Neighbor / Common-Neighbor QA**  
    Question: “What direct neighbors are shared by gene A and gene B in layer X?”  
    Answer shape:

{  
  "gene\_a": "ENSG\_A",  
  "gene\_b": "ENSG\_B",  
  "layer\_id": "layer\_X",  
  "shared\_neighbor\_count": 2,  
  "shared\_neighbors": \["ENSG\_C", "ENSG\_D"\]  
}

**This teaches local topology beyond pairwise edges.**

21. **Degree / Hub Bias** [DIFFICULT: hubness percentile/numeric]  
    Difficulty note: Degree can be exact, but percentiles should come from supplied distributions and be scored with tolerance.
    Question: “What is gene A’s degree in layer X or across the multiplex? Is it hub-like relative to the sampled distribution?”  
    Answer shape:

{  
  "gene\_id": "ENSG\_A",  
  "scope": "all\_layers",  
  "degree": 1842,  
  "degree\_percentile": 99.2,  
  "hub\_like": true,  
  "caveat": "High-degree genes can appear proximal to many genes, so proximity claims should be interpreted with hub bias in mind."  
}

22. **Layer Specificity**  
    Question: “Gene A-B is connected in `coexpression` but not `string_ppi`. What claim is allowed?”  
    Answer shape:

{  
  "allowed\_claim": "The pair has coexpression support in this graph version.",  
  "disallowed\_claim": "Do not claim direct physical protein interaction unless a PPI/string layer supports it."  
}

---

**RWR-LOE, Distance Matrix, and Global Multiplex QA**

23. **RWR-LOE Rank Lookup** [DIFFICULT: numeric RWR output]  
    Difficulty note: Exact ranks can be supervised directly, but scores/distances should come from context/tool output with rounded numeric validation.
    Question: “For seed gene A, what is gene B’s RWR-LOE rank and score?”  
    Answer shape:

{  
  "seed\_gene\_ids": \["ENSG\_A"\],  
  "target\_gene\_id": "ENSG\_B",  
  "rank": 12,  
  "score": 0.0041,  
  "distance": 0.021,  
  "rule": "Lower rank and lower distance indicate stronger RWR-LOE proximity; higher score indicates stronger RWR support."  
}

**Closed-book for selected facts; open-book for rank-vector tables.**

24. **RWR-LOE Compare Two Genes in One Rank Vector** [DIFFICULT: rank-vector comparison]  
    Difficulty note: Good open-book task; the model should compare ranks/scores rather than recall them from weights.
    Question: “In seed gene A’s RWR-LOE rank vector, which gene is closer by RWR-LOE: gene B or gene C?”  
    Answer shape:

{  
  "seed\_gene\_ids": \["ENSG\_A"\],  
  "closer\_gene\_id": "ENSG\_B",  
  "comparison": \[  
    {"gene\_id": "ENSG\_B", "rank": 12, "score": 0.0041, "distance": 0.021},  
    {"gene\_id": "ENSG\_C", "rank": 84, "score": 0.0012, "distance": 0.087}  
  \],  
  "rule": "Lower rank indicates stronger RWR-LOE proximity."  
}

25. **RWR-LOE Closest Entities From Rank Vector** [DIFFICULT: top-k vector extraction]  
    Difficulty note: Requires sorting/filtering a rank vector and excluding seeds; use precision@k/recall@k evaluation.
    Question: “Given this RWR-LOE rank vector for seed gene A, list the closest 10 non-seed entities.”  
    Answer shape:

{  
  "seed\_gene\_ids": \["ENSG\_A"\],  
  "exclude\_seed\_genes": true,  
  "top\_k": 10,  
  "closest\_genes": \[  
    {"gene\_id": "ENSG\_B", "gene\_symbol": "B", "rank": 1, "score": 0.0092},  
    {"gene\_id": "ENSG\_C", "gene\_symbol": "C", "rank": 2, "score": 0.0087}  
  \]  
}

26. **RWR-LOE Query Gene Filtering** [DIFFICULT: query-filtered rank vector]  
    Difficulty note: Requires joining query genes to a rank vector and preserving ordering.
    Question: “Given seed set S and query genes Q, which query genes are most proximal by RWR-LOE?”  
    Answer shape:

{  
  "seed\_gene\_ids": \["ENSG\_A", "ENSG\_B"\],  
  "query\_gene\_ids": \["ENSG\_C", "ENSG\_D", "ENSG\_E"\],  
  "ranked\_query\_genes": \[  
    {"gene\_id": "ENSG\_D", "rank": 5, "score": 0.0062},  
    {"gene\_id": "ENSG\_C", "rank": 19, "score": 0.0031},  
    {"gene\_id": "ENSG\_E", "rank": 204, "score": 0.0004}  
  \]  
}

27. **RWR-LOE Elbow Cutoff Membership** [DIFFICULT: elbow membership]  
    Difficulty note: Requires applying an exact cutoff rule; validate retained/excluded sets.
    Question: “Which genes are retained if the RWR-LOE module cutoff is `rank < elbow_rank_cutoff`?”  
    Answer shape:

{  
  "seed\_gene\_id": "ENSG\_A",  
  "elbow\_rank\_cutoff": 8,  
  "membership\_rule": "rank \< elbow\_rank\_cutoff",  
  "retained\_gene\_ids": \["ENSG\_B", "ENSG\_C", "ENSG\_D"\],  
  "excluded\_gene\_ids": \["ENSG\_E", "ENSG\_F"\]  
}

**This teaches the RWR-LOE module definition.**

28. **RWR-LOE Rank Gap / Elbow Reasoning** [DIFFICULT: elbow detection]  
    Difficulty note: Requires numeric curve reasoning; introduce after simpler rank/cutoff tasks.
    Question: “Given these ordered RWR-LOE ranks and scores, where is the geometric elbow, and which genes are on the high-score side?”  
    Answer shape:

{  
  "elbow\_rank\_cutoff": 7,  
  "high\_score\_side\_gene\_ids": \["ENSG\_A", "ENSG\_B", "ENSG\_C"\],  
  "low\_score\_side\_gene\_ids": \["ENSG\_D", "ENSG\_E"\],  
  "membership\_rule": "Retain genes with rank lower than the elbow cutoff."  
}

29. **RWR-LOE Vector Intersection** [DIFFICULT: vector set algebra]  
    Difficulty note: Requires top-k extraction plus intersection/Jaccard; evaluate with recomputed metrics.
    Question: “Do seed sets S1 and S2 produce overlapping top-50 RWR-LOE neighborhoods? Return intersection and Jaccard.”  
    Answer shape:

{  
  "seed\_set\_a": \["ENSG\_A"\],  
  "seed\_set\_b": \["ENSG\_B"\],  
  "top\_k": 50,  
  "intersection\_gene\_ids": \["ENSG\_X", "ENSG\_Y"\],  
  "intersection\_size": 2,  
  "union\_size": 98,  
  "jaccard": 0.0204  
}

30. **RWR-LOE Leave-One-Out Support** [DIFFICULT: leave-one-out support]  
    Difficulty note: Requires interpreting LOO RWR support; important for refinement but should be open-book/tool-backed.
    Question: “Given leave-one-out RWR ranks for this gene set, which gene is least supported by the rest of the set?”  
    Answer shape:

{  
  "gene\_set": \["ENSG\_A", "ENSG\_B", "ENSG\_C", "ENSG\_D"\],  
  "least\_supported\_gene\_id": "ENSG\_D",  
  "support\_table": \[  
    {"held\_out\_gene\_id": "ENSG\_A", "loo\_rank": 3, "recommendation": "keep"},  
    {"held\_out\_gene\_id": "ENSG\_D", "loo\_rank": 9425, "recommendation": "drop\_candidate"}  
  \],  
  "interpretation": "ENSG\_D is the weakest topological fit under leave-one-out RWR support."  
}

Useful for refinement.

31. **Sharded Distance Matrix Pair Lookup** [DIFFICULT: matrix shard lookup]  
    Difficulty note: Use open-book matrix shards; exact floats should be tolerance-scored.
    Question: “Given this RWR distance matrix shard, what is the distance between gene A and gene D?”  
    Answer shape:

{  
  "context\_type": "rwr\_distance\_matrix\_shard",  
  "multiplex\_id": "full\_brain\_multiplex\_v1",  
  "distance\_metric": "spearman\_distance",  
  "lower\_is\_closer": true,  
  "gene\_a": "ENSG\_A",  
  "gene\_b": "ENSG\_D",  
  "distance": 0.87  
}

32. **Sharded Distance Matrix Comparison** [DIFFICULT: numeric matrix comparison]  
    Difficulty note: The key skill is lower-is-closer comparison, not memorization of matrix values.
    Question: “According to this distance matrix row, is gene B closer to gene A than gene C is?”  
    Answer shape:

{  
  "anchor\_gene\_id": "ENSG\_A",  
  "candidate\_a": {"gene\_id": "ENSG\_B", "distance": 0.12},  
  "candidate\_b": {"gene\_id": "ENSG\_C", "distance": 0.44},  
  "closer\_gene\_id": "ENSG\_B",  
  "rule": "Lower distance is closer."  
}

33. **Closest Entities From Distance Matrix Row** [DIFFICULT: global nearest-neighbor row]  
    Difficulty note: Requires sorting a global row/shard; evaluate top-k ordering and seed/self exclusion.
    Question: “Given this distance row for gene A, list the 10 closest genes excluding gene A itself.”  
    Answer shape:

{  
  "anchor\_gene\_id": "ENSG\_A",  
  "distance\_metric": "spearman\_distance",  
  "exclude\_self": true,  
  "top\_k": 10,  
  "closest\_genes": \[  
    {"gene\_id": "ENSG\_B", "distance": 0.03},  
    {"gene\_id": "ENSG\_C", "distance": 0.05}  
  \]  
}

34. **Cross-Shard Distance Lookup** [DIFFICULT: shard routing]  
    Difficulty note: Requires knowing the sharding/indexing convention; use metadata-rich examples.
    Question: “The A-B pair is not in shard `rows_000100_000199`. Which shard should be queried?”  
    Answer shape:

{  
  "requested\_pair": \["ENSG\_A", "ENSG\_B"\],  
  "current\_shard\_contains\_pair": false,  
  "next\_shard\_id": "rows\_000300\_000399",  
  "reason": "The row gene ENSG\_A belongs to that row shard."  
}

35. **Distance Percentile Calibration** [DIFFICULT: numeric percentile calibration]  
    Difficulty note: Best learned with supplied quantiles and bin labels; avoid closed-book exact percentiles.
    Question: “Given global distance quantiles, is the A-B distance unusually close, typical, or far?”  
    Answer shape:

{  
  "gene\_a": "ENSG\_A",  
  "gene\_b": "ENSG\_B",  
  "distance": 0.04,  
  "distance\_percentile": 2.1,  
  "classification": "unusually\_close",  
  "rule": "Lower percentile indicates closer-than-typical network proximity."  
}

36. **Rank Vector vs Distance Matrix Consistency** [DIFFICULT: cross-context consistency]  
    Difficulty note: Requires checking multiplex/cache/metric identity across multiple artifacts.
    Question: “Gene B is rank 5 in gene A’s RWR-LOE vector, but the matrix distance d(A,B) is high. What should be checked?”  
    Answer shape:

{  
  "checks": \[  
    "same\_multiplex\_id",  
    "same\_flist\_hash",  
    "same\_distance\_metric",  
    "same\_seed\_set",  
    "same\_layer\_scope",  
    "same RWR/RWR-LOE version",  
    "same rank-vector cache version"  
  \],  
  "allowed\_claim": "Do not treat the values as contradictory until metric and context identity are confirmed."  
}

37. **Layer Ablation** [DIFFICULT: ablation interpretation]  
    Difficulty note: Requires interpreting a perturbation of graph context, not just a static edge fact.
    Question: “If removing `HumanNetV3:string_ppi` worsens A-B proximity, what caveat should be added?”  
    Answer shape:

{  
  "pair": \["ENSG\_A", "ENSG\_B"\],  
  "ablated\_layer": "HumanNetV3:string\_ppi",  
  "effect": "proximity\_worsened",  
  "caveat": "The A-B topological relationship is layer-sensitive and depends partly on this layer."  
}

38. **Node Perturbation / Seed Essentiality** [DIFFICULT: perturbation interpretation]  
    Difficulty note: Requires causal-style reading of rank-vector changes; use structured tool observations.
    Question: “If removing gene A from the seed set changes the top RWR candidates sharply, how should A be described?”  
    Answer shape:

{  
  "gene\_id": "ENSG\_A",  
  "effect": "large\_rank\_vector\_shift",  
  "description": "seed\_essential",  
  "allowed\_claim": "Gene A is influential for this RWR signal."  
}

---

**MENTOR-EV / RWR-LOE Module Set Algebra**

39. **MENTOR-EV Module Membership**  
    Question: “Which genes are members of MENTOR-EV module M?”  
    Answer shape:

{  
  "module\_id": "mentor\_ev:full\_brain\_multiplex\_v1:gw\_dendrogram\_v2:clade\_0001842",  
  "module\_source": "mentor\_ev",  
  "gene\_count": 4,  
  "gene\_ids": \["ENSG\_A", "ENSG\_B", "ENSG\_C", "ENSG\_D"\]  
}

40. **RWR-LOE Module Membership**  
    Question: “Which genes are members of RWR-LOE module R?”  
    Answer shape:

{  
  "module\_id": "rwr\_loe:full\_brain\_multiplex\_v1:seed\_ENSG\_A:geometric\_elbow\_v1",  
  "module\_source": "rwr\_loe",  
  "seed\_gene\_id": "ENSG\_A",  
  "membership\_rule": "rank \< elbow\_rank\_cutoff",  
  "gene\_count": 3,  
  "gene\_ids": \["ENSG\_A", "ENSG\_B", "ENSG\_C"\]  
}

41. **MENTOR-EV / RWR-LOE Intersection**  
    Question: “What is the intersection between MENTOR-EV module M and RWR-LOE module R?”  
    Answer shape:

{  
  "module\_a": "mentor\_ev:...",  
  "module\_b": "rwr\_loe:...",  
  "intersection\_gene\_ids": \["ENSG\_A", "ENSG\_B"\],  
  "intersection\_size": 2  
}

42. **Set Difference**  
    Question: “Which genes are in MENTOR-EV module M but not in RWR-LOE module R?”  
    Answer shape:

{  
  "module\_a": "mentor\_ev:...",  
  "module\_b": "rwr\_loe:...",  
  "set\_difference": "module\_a\_minus\_module\_b",  
  "gene\_ids": \["ENSG\_C", "ENSG\_D"\],  
  "count": 2  
}

43. **RWR-LOE Subset of MENTOR-EV**  
    Question: “Is RWR-LOE module R a subset of MENTOR-EV module M?”  
    Answer shape:

{  
  "subset": true,  
  "candidate\_subset": "rwr\_loe:...",  
  "candidate\_superset": "mentor\_ev:...",  
  "violating\_gene\_ids": \[\],  
  "containment\_fraction": 1.0  
}

44. **MENTOR-EV Superset of RWR-LOE**  
    Question: “Is MENTOR-EV module M a superset of RWR-LOE module R?”  
    Answer shape:

{  
  "superset": true,  
  "candidate\_superset": "mentor\_ev:...",  
  "candidate\_subset": "rwr\_loe:...",  
  "extra\_genes\_in\_superset": \["ENSG\_C", "ENSG\_D"\]  
}

45. **Near-Subset With Violating Genes** [DIFFICULT: near-subset errors]  
    Difficulty note: Requires exact containment plus violation detection; validate returned violating genes.
    Question: “RWR-LOE module R is not an exact subset of MENTOR-EV module M. Which genes violate containment?”  
    Answer shape:

{  
  "exact\_subset": false,  
  "containment\_fraction": 0.86,  
  "violating\_gene\_ids": \["ENSG\_X"\],  
  "allowed\_claim": "R is a near-subset of M with one violating gene."  
}

46. **Jaccard Overlap** [DIFFICULT: computed Jaccard]  
    Difficulty note: Compute from sets or supplied counts; score with numeric tolerance.
    Question: “Compute Jaccard overlap between MENTOR-EV module M and RWR-LOE module R.”  
    Answer shape:

{  
  "module\_a": "mentor\_ev:...",  
  "module\_b": "rwr\_loe:...",  
  "intersection\_size": 5,  
  "union\_size": 12,  
  "jaccard": 0.4167  
}

47. **Containment Coefficients** [DIFFICULT: computed containment]  
    Difficulty note: Requires direction-aware containment; recompute values from counts/sets.
    Question: “What fraction of RWR-LOE module R is contained in MENTOR-EV module M, and what fraction of M is contained in R?”  
    Answer shape:

{  
  "module\_a": "mentor\_ev:...",  
  "module\_b": "rwr\_loe:...",  
  "intersection\_size": 5,  
  "module\_a\_size": 10,  
  "module\_b\_size": 6,  
  "fraction\_of\_a\_in\_b": 0.5,  
  "fraction\_of\_b\_in\_a": 0.8333  
}

This teaches subset/superset direction, not just Jaccard.

48. **Best Matching Module** [DIFFICULT: best-overlap retrieval]  
    Difficulty note: Requires ranking many modules; use overlap tables and ranking metrics.
    Question: “Which RWR-LOE module best matches MENTOR-EV module M by containment or Jaccard?”  
    Answer shape:

{  
  "query\_module": "mentor\_ev:...",  
  "ranking\_metric": "jaccard",  
  "best\_match\_module": "rwr\_loe:...",  
  "best\_match\_score": 0.72,  
  "ranked\_matches": \[  
    {"module\_id": "rwr\_loe:...", "jaccard": 0.72, "containment": 0.91},  
    {"module\_id": "rwr\_loe:...", "jaccard": 0.41, "containment": 0.55}  
  \]  
}

49. **Module Overlap Ranking** [DIFFICULT: overlap ranking]  
    Difficulty note: Requires sorting module candidates by overlap; evaluate order and metrics.
    Question: “Rank these candidate modules by overlap with module M.”  
    Answer shape:

{  
  "query\_module": "mentor\_ev:...",  
  "ranked\_modules": \[  
    {"module\_id": "rwr\_loe:module\_1", "intersection\_size": 8, "jaccard": 0.67},  
    {"module\_id": "rwr\_loe:module\_2", "intersection\_size": 3, "jaccard": 0.21}  
  \]  
}

50. **Parent/Child Clade Relation**  
    Question: “In the MENTOR-EV dendrogram, is module M\_child nested inside module M\_parent?”  
    Answer shape:

{  
  "child\_module": "mentor\_ev:...:clade\_child",  
  "parent\_module": "mentor\_ev:...:clade\_parent",  
  "is\_nested": true,  
  "child\_gene\_count": 12,  
  "parent\_gene\_count": 48  
}

51. **Sibling Modules**  
    Question: “Which modules share the same immediate parent as M?”  
    Answer shape:

{  
  "query\_module": "mentor\_ev:...:clade\_0001842",  
  "parent\_module": "mentor\_ev:...:clade\_0001800",  
  "sibling\_modules": \[  
    "mentor\_ev:...:clade\_0001843",  
    "mentor\_ev:...:clade\_0001844"  
  \]  
}

52. **Multi-Module Intersection** [DIFFICULT: multi-set intersection]  
    Difficulty note: Requires intersection across more than two modules; validate exact set output.
    Question: “Which genes appear in all of modules M1, M2, and R1?”  
    Answer shape:

{  
  "modules": \["mentor\_ev:M1", "mentor\_ev:M2", "rwr\_loe:R1"\],  
  "intersection\_gene\_ids": \["ENSG\_A", "ENSG\_B"\],  
  "intersection\_size": 2  
}

53. **Unique Genes By Module Source** [DIFFICULT: source-specific set difference]  
    Difficulty note: Requires joining multiple RWR-LOE modules against one MENTOR-EV module.
    Question: “Which genes are unique to the MENTOR-EV module and absent from all listed RWR-LOE modules?”  
    Answer shape:

{  
  "mentor\_ev\_module": "mentor\_ev:M",  
  "rwr\_loe\_modules": \["rwr\_loe:R1", "rwr\_loe:R2"\],  
  "mentor\_ev\_unique\_gene\_ids": \["ENSG\_X", "ENSG\_Y"\],  
  "count": 2  
}

54. **Module Provenance**  
    Question: “Which source produced this module: MENTOR-EV dendrogram or RWR-LOE elbow expansion?”  
    Answer shape:

{  
  "module\_id": "rwr\_loe:full\_brain\_multiplex\_v1:seed\_ENSG\_A:geometric\_elbow\_v1",  
  "module\_source": "rwr\_loe",  
  "construction\_rule": "seed-centered RWR-LOE rank vector with geometric elbow cutoff"  
}

55. **Set Overlap vs Topological Distance** [DIFFICULT: set + distance integration]  
    Difficulty note: Requires combining set overlap with global distance context; introduce late.
    Question: “Module M and RWR-LOE module R overlap by only 2 genes, but their mean inter-module distance is in the closest 1% globally. Are they set-overlapping, topologically close, both, or neither?”  
    Answer shape:

{  
  "set\_relationship": "weak\_overlap",  
  "topological\_relationship": "globally\_close",  
  "basis": {  
    "intersection\_size": 2,  
    "distance\_percentile": 1.0  
  }  
}

---

**Global Cohesion, Calibration, and Negatives**

56. **Within-Clade Distance** [DIFFICULT: numeric aggregation]  
    Difficulty note: Mean distances should be computed from supplied tables and tolerance-scored.
    Question: “Given all pairwise distances inside module M, compute mean within-module distance.”  
    Answer shape:

{  
  "module\_id": "mentor\_ev:M",  
  "pair\_count": 6,  
  "mean\_within\_distance": 0.082,  
  "distance\_metric": "spearman\_distance",  
  "lower\_is\_closer": true  
}

57. **Within-Clade vs Random Distribution** [DIFFICULT: empirical p-value]  
    Difficulty note: Requires counting over random-set summaries; avoid exact closed-book p-values.
    Question: “Given the observed mean within-clade distance and 500 random-set means, compute empirical p-value.”  
    Answer shape:

{  
  "observed\_mean\_within\_distance": 0.082,  
  "random\_set\_count": 500,  
  "empirical\_p\_value": 0.006,  
  "interpretation": "The module is more topologically cohesive than expected by random same-size sets."  
}

58. **Distance Ratio / Clustering Ratio** [DIFFICULT: derived ratio]  
    Difficulty note: Compute from supplied distances; evaluate with tolerance.
    Question: “Given mean within-module distance and mean outside-module distance, compute the clustering ratio.”  
    Answer shape:

{  
  "mean\_within\_distance": 0.08,  
  "mean\_outside\_distance": 0.40,  
  "clustering\_ratio": 5.0,  
  "interpretation": "High ratio indicates stronger within-module topological cohesion."  
}

59. **Subgraph Density** [DIFFICULT: edge-density arithmetic]  
    Difficulty note: Requires arithmetic over subgraph counts; evaluate recomputed density.
    Question: “Given edge count and possible edge count for module M in layer X, compute edge density.”  
    Answer shape:

{  
  "module\_id": "mentor\_ev:M",  
  "layer\_id": "layer\_X",  
  "edge\_count": 12,  
  "possible\_edge\_count": 45,  
  "edge\_density": 0.2667  
}

60. **Conductance / Boundary Ratio** [DIFFICULT: boundary-ratio arithmetic]  
    Difficulty note: Requires arithmetic and calibration relative to a null when available.
    Question: “Given internal and boundary edge counts, is the module well separated?”  
    Answer shape:

{  
  "internal\_edge\_count": 40,  
  "boundary\_edge\_count": 5,  
  "boundary\_ratio": 0.1111,  
  "separation": "well\_separated",  
  "caveat": "Interpret relative to a same-size random module distribution when available."  
}

61. **Cell-Type-Specific Cohesion** [DIFFICULT: cell-type-specific context]  
    Difficulty note: Requires layer/context distinction across cell types; use late-stage global-context examples.
    Question: “Module M is cohesive in astrocyte scPEN but not neuron scPEN. What topological statement is supported?”  
    Answer shape:

{  
  "module\_id": "mentor\_ev:M",  
  "cohesive\_contexts": \["scPEN:brain:astrocyte"\],  
  "noncohesive\_contexts": \["scPEN:brain:neuron"\],  
  "allowed\_claim": "The module shows cell-type-specific topological cohesion in astrocyte-associated layers.",  
  "disallowed\_claim": "Do not claim universal brain-wide cohesion from one cell-type-specific layer."  
}

62. **Layer-Sensitive Cohesion** [DIFFICULT: layer-sensitive context]  
    Difficulty note: Requires interpreting ablation output as a caveat, not overclaiming biology.
    Question: “If removing layer X destroys module M’s cohesion, what caveat is supported?”  
    Answer shape:

{  
  "module\_id": "mentor\_ev:M",  
  "ablated\_layer": "layer\_X",  
  "effect": "cohesion\_decreases",  
  "caveat": "The module’s topological cohesion is sensitive to layer X."  
}

63. **Global Nearest Modules** [DIFFICULT: global module retrieval]  
    Difficulty note: Requires module-module distance ranking; should be open-book over tables.
    Question: “Given a module-module distance table, which modules are globally nearest to module M?”  
    Answer shape:

{  
  "query\_module": "mentor\_ev:M",  
  "top\_k": 5,  
  "nearest\_modules": \[  
    {"module\_id": "mentor\_ev:M2", "distance": 0.04},  
    {"module\_id": "rwr\_loe:R7", "distance": 0.06}  
  \]  
}

64. **No-Edge / No-Path Calibration**  
    Question: “Gene A and gene B have no recorded edge/path in this graph version. What can and cannot be concluded?”  
    Answer shape:

{  
  "allowed\_claim": "No edge or path is recorded between these genes in this graph version and layer scope.",  
  "disallowed\_claims": \[  
    "The genes are biologically unrelated in all possible contexts.",  
    "No interaction exists in biology.",  
    "The relationship has been experimentally disproven."  
  \]  
}

This should appear often to prevent biological overclaiming.

65. **No Direct Edge But Multiplex Proximity Exists**  
    Question: “Gene A and gene B have no direct edge, but RWR-LOE ranks B highly from seed A. What claim is allowed?”  
    Answer shape:

{  
  "allowed\_claim": "The genes are topologically proximal under RWR-LOE in this multiplex.",  
  "disallowed\_claim": "Do not claim a direct recorded edge or direct physical interaction."  
}

66. **Gene Absent From Layer**  
    Question: “Gene A is absent from layer X. What should the graph agent do?”  
    Answer shape:

{  
  "gene\_id": "ENSG\_A",  
  "layer\_id": "layer\_X",  
  "status": "gene\_absent\_from\_layer",  
  "allowed\_action": "Use another valid layer scope or report that this layer cannot support a claim about the gene.",  
  "disallowed\_action": "Do not invent layer-specific edges for an absent node."  
}

67. **RWR Rank Outside Top-K** [DIFFICULT: top-k truncation calibration]  
    Difficulty note: Requires distinguishing not-in-top-k from absent-from-vector; important negative training.
    Question: “Gene B is not present in the top-100 RWR-LOE results for seed A. What can be concluded?”  
    Answer shape:

{  
  "seed\_gene\_id": "ENSG\_A",  
  "target\_gene\_id": "ENSG\_B",  
  "top\_k": 100,  
  "status": "not\_in\_top\_k",  
  "allowed\_claim": "Gene B is not among the top-100 returned genes in this RWR-LOE query.",  
  "disallowed\_claim": "Do not conclude that gene B has no finite rank unless the full vector was checked."  
}

68. **String-Based Interpretation De-Emphasis** [DATASET RULE: not primary SFT QA]  
    Difficulty note: Use as a dataset policy or calibration example; do not over-sample as ordinary QA.
    Question: “Given only a plausible biological phrase but no graph evidence, should the model claim that a module is validated?”  
    Answer shape:

{  
  "validated\_group": false,  
  "relationship\_status": "unknown",  
  "reason": "SFT should prioritize exact graph facts, set relations, and structured evidence. Free-form mechanistic style can be shaped during DPO."  
}

---

**Open-Book / Tool-Call QA**

69. **Choose the Correct Tool: RWR-LOE Rank Vector** [DIFFICULT: tool selection can overuse RWR]  
    Difficulty note: Teach the cheapest specific rank/vector tool when available; avoid making full RWR the default for every question.
    Question: “You need to compare gene B and gene C in gene A’s RWR-LOE rank vector. Which structured tool should be called?”  
    Answer shape:

{  
  "preferred\_tool\_name": "get\_rank\_vector\_summary",  
  "arguments": {  
    "seed\_genes": \["ENSG\_A"\],  
    "top\_k": 100  
  },  
  "reason": "The task requires a compact rank-vector view from seed gene A without forcing a full fresh RWR call when a cached summary is available.",  
  "alternative\_tools": \[  
    {  
      "tool\_name": "get\_rank",  
      "use\_case": "single source-target rank lookup",  
      "arguments": {"source\_gene": "ENSG\_A", "target\_gene": "ENSG\_B"}  
    },  
    {  
      "tool\_name": "rwr\_loe",  
      "use\_case": "seed/query RWR-LOE ranking when a rank-vector summary is unavailable or query genes must be filtered",  
      "arguments": {"seed\_genes": \["ENSG\_A"\], "query\_genes": \["ENSG\_B", "ENSG\_C"\], "top\_k": 100}  
    }  
  \]  
}

70. **Choose the Correct Tool: Pairwise Distance** [DIFFICULT: tool schema + metric semantics]  
    Difficulty note: Requires metric-aware arguments; validate schema exactly.
    Question: “What tool call retrieves RWR++ distance between gene A and gene B?”  
    Answer shape:

{  
  "tool\_name": "get\_distance",  
  "arguments": {  
    "gene\_a": "ENSG\_A",  
    "gene\_b": "ENSG\_B",  
    "distance\_metric": "spearman"  
  }  
}

71. **Choose the Correct Tool: Layer Membership**  
    Question: “What tool call checks which multiplex layers contain gene A?”  
    Answer shape:

{  
  "tool\_name": "get\_gene\_layers",  
  "arguments": {  
    "gene": "ENSG\_A"  
  }  
}

72. **Choose the Correct Tool: Component Summary**  
    Question: “What tool call checks whether a gene set falls into one connected component?”  
    Answer shape:

{  
  "tool\_name": "get\_component\_summary",  
  "arguments": {  
    "genes": \["ENSG\_A", "ENSG\_B", "ENSG\_C"\],  
    "max\_components": 10  
  }  
}

73. **Choose the Correct Tool: Induced Subgraph**  
    Question: “What tool call extracts all intra-set edges among genes A, B, C, and D?”  
    Answer shape:

{  
  "tool\_name": "induce\_subgraph",  
  "arguments": {  
    "genes": \["ENSG\_A", "ENSG\_B", "ENSG\_C", "ENSG\_D"\]  
  }  
}

74. **Choose the Correct Tool: Layer Ablation** [DIFFICULT: tool schema may vary]  
    Difficulty note: Keep arguments aligned with the implemented wrapper; do not teach unsupported fields.
    Question: “What tool call tests whether proximity is dependent on layer X?”  
    Answer shape:

{  
  "tool\_name": "get\_layer\_ablation",  
  "arguments": {  
    "seed\_genes": \["ENSG\_A", "ENSG\_B"\],  
    "distance\_metric": "spearman",  
    "top\_k": 50  
  },  
  "schema\_note": "Only include fields supported by the current wrapper. If a specific layer must be tested, use the implemented layer-selection field name from the runtime schema rather than inventing CLI-style arguments."  
}

75. **Parse RWR-LOE Tool Result** [DIFFICULT: tool-output parsing]  
    Difficulty note: Requires parsing and sorting structured RWR output; score top-k ordering.
    **Question: “Given this RWR-LOE observation, return the closest three non-seed genes.”**  
    **Answer shape:**

{  
  "seed\_gene\_ids": \["ENSG\_A"\],  
  "closest\_non\_seed\_genes": \[  
    {"gene\_id": "ENSG\_B", "rank": 1, "score": 0.0092},  
    {"gene\_id": "ENSG\_C", "rank": 2, "score": 0.0087},  
    {"gene\_id": "ENSG\_D", "rank": 3, "score": 0.0081}  
  \]  
}

76. **Parse Distance Matrix Shard** [DIFFICULT: matrix-output parsing]  
    Difficulty note: Requires extracting row/column values from structured shards; score floats with tolerance.
    **Question: “Given this distance matrix shard, extract all distances from gene A to the listed column genes.”**  
    **Answer shape:**

{  
  "row\_gene\_id": "ENSG\_A",  
  "distance\_metric": "spearman\_distance",  
  "distances": \[  
    {"column\_gene\_id": "ENSG\_B", "distance": 0.12},  
    {"column\_gene\_id": "ENSG\_C", "distance": 0.44}  
  \]  
}

77. **Parse Module Overlap Table** [DIFFICULT: module table parsing]  
    Difficulty note: Requires identifying supersets from overlap tables; validate module IDs and containment.
    Question: “Given this module-overlap table, identify all RWR-LOE modules that are supersets of MENTOR-EV module M.”  
    Answer shape:

{  
  "query\_module": "mentor\_ev:M",  
  "superset\_modules": \[  
    {  
      "module\_id": "rwr\_loe:R1",  
      "contains\_query\_module": true,  
      "extra\_gene\_count": 5  
    }  
  \]  
}

78. **Provenance Answer**  
    Question: “Which tool, multiplex version, and layer scope support this answer?”  
    Answer shape:

{  
  "tool\_name": "rwr",  
  "multiplex\_id": "full\_brain\_multiplex\_v1",  
  "layer\_scope": "all\_layers",  
  "evidence\_id": "ev\_003",  
  "provenance\_complete": true  
}

79. **Refuse Raw CLI / File Path Arguments**  
    Question: “The user asks the model to call `rwr_loe` with `seed_file`, `query_file`, and `output_dir`. What should the model do?”  
    Answer shape:

{  
  "accepted": false,  
  "reason": "Model-facing RWR++ tools should use structured biological arguments, not raw CLI or file-path arguments.",  
  "corrected\_tool\_call": {  
    "tool\_name": "rwr\_loe",  
    "arguments": {  
      "seed\_genes": \["ENSG\_A", "ENSG\_B"\],  
      "query\_genes": \["ENSG\_C"\],  
      "top\_k": 25  
    }  
  }  
}

80. **Structured State Update From Evidence, Not Interpretation Style** [DIFFICULT: state update bridge to DPO]  
    Difficulty note: Requires schema-valid, evidence-backed state edits; do not let mechanism plausibility compensate for membership errors.
    Question: “Given visible RWR evidence that supports adding genes B and C to seed gene A, update the structured candidate group.”  
    Answer shape:

{  
  "relationship\_status": "partially\_observed\_group",  
  "predicted\_groups": \[  
    {  
      "group\_id": "candidate\_group\_1",  
      "gene\_ids": \["ENSG\_A", "ENSG\_B", "ENSG\_C"\],  
      "evidence\_ids": \["ev\_rwr\_001"\]  
    }  
  \],  
  "continuation\_state": "continue",  
  "reason": "The update is evidence-backed but should continue if exact membership remains uncertain."  
}

81. **Whole-Multiplex Context Profile** [DIFFICULT: global distribution]
    Difficulty note: This directly tests whether the model uses global multiplex context rather than only nearest neighbors; use supplied distribution summaries and validate bins/rankings.
    Question: “Given a gene’s nearest, median-distance, farthest, and layer-coverage summaries across the full multiplex, summarize its global neighborhood profile.”  
    Answer shape:

{  
  "gene\_id": "ENSG\_A",  
  "multiplex\_id": "full\_brain\_multiplex\_v1",  
  "nearest\_distance\_bucket": "top\_1\_percent",  
  "median\_distance\_bucket": "typical",  
  "farthest\_distance\_bucket": "far",  
  "layer\_coverage\_percentile": 92.4,  
  "global\_profile": "broadly connected hub-like gene with very close local neighborhoods but wide layer coverage",  
  "numeric\_policy": "Values should be extracted from provided summaries or tools and scored with tolerance; do not require closed-book exact float recall."  
}

The key design rule: SFT topology, set-algebra, matrix, and tool-call questions should have exact, validator-checkable answers. Long string-based biological interpretation should be minimized in this SFT list and handled mainly during DPO, where grounded vs overclaiming interpretations can be compared under the same evidence context. Numerical outputs should be treated as extracted or computed artifacts with explicit metric semantics, rounding, and tolerance-aware validators; do not expect the model to memorize arbitrary full-multiplex floating-point values.

