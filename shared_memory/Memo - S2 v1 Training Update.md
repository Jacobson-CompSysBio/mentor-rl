# **Memo \- S2 v1 Training Update**

	Training for the S2 v1 corpus ran successfully. This memo will provide a breakdown of training details, analyze failure cases in training, and then detail the next steps to improve performance.

## **Training Breakdown**

	Training and evaluation completed successfully on the v1 training and inner test corpora. There were 13 training families (S2.1-S2.13) with 394 total question types. Each question type received 512 training examples, 64 validation examples, and 64 test examples. Additionally, training added 37,824 S1 and 12,608 S0 replay rows, so the training corpus had 252,160 examples total. 

	Training ran on 184 nodes for 1 epoch. With parallelism applied, this yielded 1,371 optimizer updates on a global batch size of 184 and a learning rate of 1e-4. These settings were re-used from the successful S1 v10 job that passed the S1 gate. During evaluation, **20,320 out of 25,216 test rows were scored as correct on all fields, giving a structured accuracy of 80.58%.** V1 was scored in report-only mode, so a gate has not been set for the S2 corpus yet. My proposed gate score for S2 is 93%; I believed that this slightly relaxed score will help us balance model performance with faster training. Table 1 provides a breakdown of accuracies by envelope section:

| Field | Correct/Total | Rate |
| :---: | :---: | :---: |
| Structured (all fields) | 20,320 / 25,216 | 80.58% |
| Status | 23,250 / 25,216 | 92.20% |
| Result | 20,782 / 25,216 | 82.42% |
| Continuation Decision | 22,755 / 25,216 | 90.24% |
| Provenance | 22,804 / 25,216 | 90.43% |
| Blocked Reason Present (When Required) | 23,219 / 25,216 | 92.08% |
| Exact Ordered Tool Calls | 22,690 / 25,216 | 89.98% |
| Valid Trajectory | 23,735 / 25,216 | 94.13% |

Table 1: Breakdown of accuracies by envelope section.

Additionally, Table 2 provides a breakdown of accuracies by question family:

| Family | Name | Correct | Rows | Rate |
| :---: | ----- | :---: | :---: | :---: |
| S2.1 | Layer-specific edge existence & directionality | 2,114 | 2,624 | 80.56% |
| S2.2 | Multiplex edge existence | 1,157 | 1,600 | 82.17% |
| S2.3 | Layer-specific direct neighbors | 1,544 | 1,600 | 96.50% |
| S2.4 | Multiplex direct neighbors | 1,096 | 1,536 | 71.35% |
| S2.5 | Gene layer membership | 1,510 | 1,600 | 94.38% |
| S2.6 | Nodes by layer | 1,539 | 1,600 | 96.19% |
| S2.7 | Shared neighbors | 1,210 | 1,856 | 65.19% |
| S2.8 | Degree and hubness | 1,491 | 1,856 | 80.33% |
| S2.9 | Induced subgraph | 1,468 | 1,920 | 76.46% |
| S2.10 | Connected components | 1,704 | 2,176 | 78.31% |
| S2.11 | Calibration (no edge and path) | 1,853 | 2,304 | 80.43% |
| S2.12 | Calibration (gene absent from layer) | 1,940 | 2,304 | 84.20% |
| S2.13 | Calibration (layer specificity) | 1,694 | 2,432 | 69.65% |

Table 2: S2 accuracies by question family

## **General Failure Modes**

## Examining failures by field, valid trajectory accuracy exceeded the gate threshold, while all other fields were below it. Status, continuation decision, provenance, and blocked reason present all achieved greater than 90% accuracy, which is promising, while exact ordered tool calling achieved 89.98% accuracy and result achieved 82% accuracy. The largest failure groups were a wrong result with every other field correct, an answer given before expected tool calls, and a tool call when none was expected. None of the trajectories failed mainly on formatting. This makes sense, as results and tool calling are more variable across question types than other fields, so they are harder to learn. Additionally, the model was specifically deficient when dealing with prior context (failed on 109 types, 2,963 of 4,896 failed rows) These numbers also suggest that the result and tool-calling fields field should get a majority of focus, while other non-passing fields could also use some reworking across question types. Additionally, prior context should get a targeted focus. Because the result field varies by question type, we need to examine the accuracy breakdown by question types to proceed. 

	Performance across question families was highly variable, ranging from 65% to 96%. Questions such as gene layer membership, nodes by layer, and layer-specific neighbors achieved greater than 90% accuracy. However, questions such as calibration (layer specificity), shared neighbors, multiplex direct neighbors, induced subgraph, and connected components scored in the 70s or lower. As we will cover in “Next Steps,” the lowest-scoring sections will serve as a focus for the next iteration of corpus improvements. 

	Stratifying questions further across the 394 subtypes, we see that several questions have catastrophic failures, a term which I am using to refer to less than 50% accuracy. Accuracy is bimodal – 220 types score at or above 90%, while 16 score below 10%. These failures are fairly evenly distributed across all question families, and they all require the most human examination to rework reasoning traces. Many question types, especially in S2.3, S2.5, and S2.6, S2.8, and S2.9, achieve greater than 95% accuracy. These well-performing question types require less human scrutiny, since they already perform well.

## **Failing Question Types**

	Next, we will discuss failure cases for specific question types. In total, 208 of 394 question types failed the gate. Failure modes can be categorized into three groups:

* **Negative:** at least half of the keyed answers are an absence or unsupported claim (no edge, no path, gene absent, not found, claim not supported)

* **Wrong answer:** wrong final answer fields, or wrong, extra, or missing tool calls

* **Formatting or parsing:** a malformed turn or answer object

* **Hallucinations/fabrications:** filling answer outputs with invented values instead of nulls

Negatives accounted for failures across 75 types, mostly in the calibration families S2.13, S2.12, and S2.11, plus S2.7. Negatives fail at the same rate that other rows fail: 80.27% of negatives are correct, compared to 80.73% of other rows. Therefore, negatives fail for the same reasons as other types, so absence by itself isn’t the hard part.

	Formatting and parsing is the main failure not linked to a question type. It accounts for a quarter of the failures in S2.5.16, S2.7.4, S2.7.22, S2.7.24, S2.7.26, and S2.12.30. Most errors in this class are not strict JSON or have an invalid status value.

## **Reasoning Failures**

	When examining reasoning failures, we see that prior context contributes the most to failure cases: 109 of the 208 failing types, or over 50% of failure cases. Examining reasoning failures more closely, we see the following breakdown:

1. **Prior context (2,639 rows):** the model misjudges evidence that is supplied by the harness.   
   1. It answers before making calls that the context still needs  
   2. It misreads a count, list, or flag from supplied observations  
   3. It calls a tool when supplied observations already contain the necessary information  
   4. Other, smaller groups  
      1. Extra calls after the evidence is complete  
      2. Repeated supplied observation  
2. **Question asked (1,426 rows):** The model answers a different question.   
   1. Answer has the wrong shape, such as a missing claim card or the wrong type of query  
   2. The first call uses the wrong tool or arguments  
   3. A value is misread from the tool results  
   4. A tool is called on a request that needs no call, such as a directed query on an undirected layer  
   5. No call is made when one is needed  
3. **Fabricated tool outputs:**  the model hallucinates/fabricates on answers  
   1. Premature answers are often filled with invented values, not nulls. The examples given in Dan’s feedback were S2.1.19, S2.2.4, S2.3.15, and S2.11.13  
   2. This failure case is the most consequential, because the output is well-formed and confident.  
4. **Continuation Decision (517 rows):** The calls are right, but the decision, blocker, or status is wrong, sometimes with the result. A typical case is missing that a prerequisite in the user’s wider requests blocks the next step  
5. **Ambiguity (9 types, 137 failed rows):** types with symbols that the supplied registry context resolves. Rows are counted in the groups above. The model gets the identity mapping or its registry evidence wrong.  
6. **Formatting (172 rows):** falls under no component and is covered in the previous section.

Additionally, v1 training traces show why prior context breaks. First, counts are stated and not derived. For example, in S2.8.16, the training reasoning says “Its exact distinct-neighbor degree is five,” and the rule is vague, saying “Count each ID once across layers.” However, the union’s members are never listed, and all 64 test rows are miscounted. Additionally, no-call types open with a stock sentence, such as “the supplied observations are sufficient; no call is needed.” In S2.9.6, the model opened with this sentence when a single input was present, and then skipped the necessary call in 56/64 rows. 

## **Next Steps**

	Based on our written and verbal discussion, we will implement several changes to improve the 208 failing types. Each change covers a single component of the reasoning while holding everything else constant. The recommended changes are:

* **Evidence inventory and visible derivation:** before any call, the reasoning should list all supplied observations and their scope. It notes whether observations are complete, bounded, truncated, or failed, what the observations imply, and what the requests still need. This should replace the stock sentence opener. Before a count, union, rank, or intersection, the reasoning should list its members, overlaps, or pages, and then state the value. 

* **Restate the request:** The first reasoning line should name the operation, its scope (layer, direction, representation), the query kind, and the answer fields required, such as each claim card. A directed query on an undirected layer, for example, is answered as “invalid input” with no call.

* **One stated reason for the decision:** The final reasoning should give the decision’s reason. “Stop” means the answer ends the standalone request; “continue” means it feeds a named later step; “defer” means a named prerequisite or blocker holds, with its blocked rewrite. 

* **Resolve identities first:** Each symbol is mapped to its supplied registry record, and the registry evidence is carried beside the graph evidence.

* **Oversample rare branches:** ensure that rare branches achieve parity within each family.

* **Add contrastive minimal pairs:** contrastive minimal pairs differ only in cue words for mode, edge query, stop versus continue, and directed versus undirected. Remove “The requested mode is complete” from the thinking template.

* **Verify derived values by code at generation time.**

* **Harmonize gold conventions:** gene ID order (S2.1.37), “all” scope reporting (S2.2.6, S2.2.16), and layer-ID case (S2.13.19, S2.13.20).

* **Add a schema validator and evidence-ID check at inference:** ensure that evidence and schema are valid during inference.

All of the prior changes have been implemented in a v2 training corpus, which is now in the queue. Furthermore, our discussion during Friday’s LLM meeting yielded several additional changes that can be made to our work process:

* **Accuracy distribution on the dashboard:** Add a density plot/histogram to show the per-type distribution with standard error, so that aggregate scoring isn’t misleading. Each type has 64 test rows, so its accuracy carries around ±3 points of standard error near 93%. Ensuring that accuracy is consistent across types is crucial for analyzing performance

* **Systematized corpus construction:** We have several options for systematizing the corpus construction. Three that I recommend are shared validators, declarative case specs, and solver-generated traces

  * Shared validators: add generation-time checks that ensure every count lists its members, every cited evidence ID exists, and conventions are consistent. 

  * Declarative case specs: each of the 394 types becomes a data record (tool sequence, evidence state, branch, derivation, result schema) one renderer composes the reasoning from component modules

  * Solver-generated traces: a reference policy runs each case against tool fixtures, and the trace is rendered from its execution log. 

* **Balancing ahead of time:** Here are some suggestions for balancing the corpus ahead of time:

  * Balance decisions, not types: stratify each family so call/reuse/diagnose, stop/continue/defer, and each mode value are near parity

  * Rule-based multipliers: give more rows to types with a derived value, supplied prior context, several calls, or a rare schema variant.

  * Fit difficulty on v1: regress per-type accuracy on previously mentioned features, and then predict difficulty for new types. This will show which types are hard, but not necessarily how many rows will fix them from a single sample point.

* **Varied reasoning text:** We also have several options to vary reasoning text

  * Detection: mask IDs and numbers, and then count how often each sentence repeats across traces. We can display the repeat rate per type on the dashboard

  * Conditional sentences: The worst v1 sentences were false for their row, such as “No compatible observation is supplied” when one was present. Render each sentence from the row’s actual state.

  * Paraphrase bank: write several reviewed variants per sentence slot, and sample them with a seeded generator. This is cheap, deterministic, and auditable. 

  * Structural variation: vary step order where steps are independent, and vary the level of detail. This will break the fixed sentence sequence, which wording changes alone won’t do.

  * LLM rewrite per row. Lock IDs, numbers, field names, and tool calls as placeholders, rewrite the prose, and exact-match the placeholders afterwords. This gives the most variety, but it needs caching and spot-checking in case meanings drift. 

