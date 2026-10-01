**F1: bad status**
`Answer:`
`{"status": "totally_made_up", "provenance": null, "result": {}, "continuation_decision": "stop", "blocked_reason": null, "summary": "x"}`

**F2: defer without blocked_reason**
`Answer:`
`{"status": "ambiguous", "provenance": null, "result": {}, "continuation_decision": "defer", "summary": "x"}`

**F3: blocked_reason matches no template**
`Answer:`
`{"status": "ambiguous", "provenance": null, "result": {}, "continuation_decision": "defer", "blocked_reason": "because_reasons", "summary": "x"}`

**F4: continuation_decision not in vocabulary**
`Answer:`
`{"status": "resolved", "provenance": null, "result": {}, "continuation_decision": "proceed", "blocked_reason": null, "summary": "x"}`

**F5: tool not in inventory**
`Tool call:`
`lookup_gene_thing({"gene_symbol": "TP53"})`
`Answer:`
`{"status": "resolved", "provenance": null, "result": {}, "continuation_decision": "stop", "blocked_reason": null, "summary": "x"}`

**F6: layer tag wrong arity for its family**
`Answer:`
`{"status": "resolved", "provenance": null, "result": {"layer_id": "HumanNetV3:string_ppi:global:v3", "layer_family": "ppi"}, "continuation_decision": "stop", "blocked_reason": null, "summary": "x"}`

**F7: layer tag unknown prefix**
`Answer:`
`{"status": "resolved", "provenance": null, "result": {"layer_id": "scPENX:single_cell_pen:brain:hippocampus:astrocyte:v1"}, "continuation_decision": "stop", "blocked_reason": null, "summary": "x"}`

**F8: layer_family not in vocabulary**
`Answer:`
`{"status": "resolved", "provenance": null, "result": {"layer_family": "ppi_ish"}, "continuation_decision": "stop", "blocked_reason": null, "summary": "x"}`

**F9: claim not in vocabulary**
`Answer:`
`{"status": "resolved", "provenance": null, "result": {"supported_claims": ["proves_causation"]}, "continuation_decision": "stop", "blocked_reason": null, "summary": "x"}`

**F10: answer JSON is malformed**
`Answer:`
`{"status": "resolved" "provenance": null, "result": {}, "continuation_decision": "stop", "blocked_reason": null, "summary": "x"}`

**F11: bad tag hidden in result.input behind status resolved**
Answer:
`{"status": "resolved", "provenance": null, "result": {"input": "scPEN:single_cell_pen:brain:hippocampus:astrocyte"}, "continuation_decision": "stop", "blocked_reason": null, "summary": "x"}`

**F12: content inherited by reference instead of inlined**
Answer:
`(the same answer as trace 1)`

**F13: this turn's answer is bad even though the earlier turn is fine**
[Earlier turn: in context, masked, no thinking]
Answer:
`{"status": "resolved", "provenance": null, "result": {}, "continuation_decision": "stop", "blocked_reason": null, "summary": "x"}`

[This turn: trained]
Answer:
`{"status": "definitely", "provenance": null, "result": {}, "continuation_decision": "stop", "blocked_reason": null, "summary": "x"}`

**F14: tool-backed answer carries no provenance**
Tool call:
`lookup_human_gene_symbol({"gene_symbol": "TP53"})`
Tool result:
`{"status": "resolved", "provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_f14a01"}, "gene_id": "ENSG00000141510", "gene_symbol": "TP53"}`
Thinking:
`The registry maps TP53 to one ID.`
Answer:
`{"status": "resolved", "result": {"gene_id": "ENSG00000141510", "gene_symbol": "TP53"}, "continuation_decision": "stop", "blocked_reason": null, "summary": "x"}`

**F15: provenance disagrees with the tool result it claims to copy**
Tool call:
`lookup_human_gene_symbol({"gene_symbol": "TP53"})`
Tool result:
`{"status": "resolved", "provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_f15a01"}, "gene_id": "ENSG00000141510", "gene_symbol": "TP53"}`
Thinking:
`The registry maps TP53 to one ID.`
Answer:
`{"status": "resolved", "provenance": {"registry": "ensembl", "release": "999", "evidence_id": "ev_reg_f15a01"}, "result": {"gene_id": "ENSG00000141510", "gene_symbol": "TP53"}, "continuation_decision": "stop", "blocked_reason": null, "summary": "x"}`

**F16: provenance present but missing a required field**
Tool call:
`lookup_human_gene_symbol({"gene_symbol": "TP53"})`
Tool result:
`{"status": "resolved", "provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_f16a01"}, "gene_id": "ENSG00000141510", "gene_symbol": "TP53"}`
Thinking:
`The registry maps TP53 to one ID.`
Answer:
`{"status": "resolved", "provenance": {"registry": "ensembl", "release": "116"}, "result": {"gene_id": "ENSG00000141510", "gene_symbol": "TP53"}, "continuation_decision": "stop", "blocked_reason": null, "summary": "x"}`

**F17: an answer inlined as context reports an identifier without provenance**
[Earlier turn: in context, masked, no thinking]
Answer:
`{"status": "resolved", "provenance": null, "result": {"gene_id": "ENSG00000141510"}, "continuation_decision": "stop", "blocked_reason": null, "summary": "x"}`

[This turn: trained]
Answer:
`{"status": "resolved", "provenance": null, "result": {"gene_id": "ENSG00000141510"}, "continuation_decision": "stop", "blocked_reason": null, "summary": "x"}`

**F22: a tool that S1 does not call**
Tool call:
`resolve_context_ontology({"context_id": "brain:hippocampus:hippocampus:astrocyte:v1"})`
Answer:
`{"status": "resolved", "provenance": null, "result": {}, "continuation_decision": "stop", "blocked_reason": null, "summary": "x"}`

**F23: a blocked_reason template that fits another status**
Answer:
`{"status": "ambiguous", "provenance": null, "result": {"gene_symbol": "LINC02170"}, "continuation_decision": "defer", "blocked_reason": "The registry has no gene with the symbol LINC02170 in Ensembl release 116.", "summary": "x"}`

**F24: a blocked_reason sentence on an answer that does not defer**
Answer:
`{"status": "resolved", "provenance": null, "result": {"gene_symbol": "LINC02170"}, "continuation_decision": "stop", "blocked_reason": "The registry has no gene with the symbol LINC02170 in Ensembl release 116.", "summary": "x"}`

**F25: a blocked_reason sentence not filled from the answer**
Answer:
`{"status": "not_found", "provenance": null, "result": {"gene_symbol": "LINC02170"}, "continuation_decision": "defer", "blocked_reason": "The registry has no gene with the symbol HER2 in Ensembl release 116.", "summary": "x"}`

**F26: an evidence_id that differs from the lookup it rests on**
Tool call:
`lookup_human_gene_symbol({"gene_symbol": "TP53"})`
Tool result:
`{"status": "resolved", "provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_f26a01"}, "gene_id": "ENSG00000141510", "gene_symbol": "TP53"}`
Thinking:
`The registry maps TP53 to one ID.`
Answer:
`{"status": "resolved", "provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_f26b02"}, "result": {"gene_id": "ENSG00000141510", "gene_symbol": "TP53"}, "continuation_decision": "stop", "blocked_reason": null, "summary": "x"}`

**F27: a near-miss answer that copies the harness block, not the lookup**
Context (from the harness):
`Registry entries, Ensembl release 116:`
`{"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_f27a01"}, "entries": [{"gene_id": "ENSG00000154274", "gene_symbol": "PGCKA1"}]}`
Tool call:
`lookup_human_gene_symbol({"gene_symbol": "VARS1"})`
Tool result:
`{"status": "resolved", "provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_f27b02"}, "gene_id": "ENSG00000204394", "gene_symbol": "VARS1"}`
Thinking:
`The registry maps VARS1 to one ID.`
Answer:
`{"status": "resolved", "provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_f27a01"}, "result": {"gene_id": "ENSG00000204394", "gene_symbol": "VARS1"}, "continuation_decision": "continue", "blocked_reason": null, "summary": "x"}`

**F28: a convention answer that invents a provenance block**
Answer:
`{"status": "resolved", "provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_f28a01"}, "result": {"metric": "rank", "input": "RWR rank", "lower_is_stronger": true, "meaning": "proximity", "rule": "lower_rank_closer", "statement_true": null}, "continuation_decision": "stop", "blocked_reason": null, "summary": "x"}`

**F29: a layer tag whose family part disagrees with its prefix**
Answer:
`{"status": "resolved", "provenance": null, "result": {"layer_id": "bulkPEN:tf_target:brain:amygdala:v1", "layer_family": "bulk_pen"}, "continuation_decision": "stop", "blocked_reason": null, "summary": "x"}`

**F30: a TFs tag with the wrong arity**
Answer:
`{"status": "resolved", "provenance": null, "result": {"layer_id": "TFs:tf_target:brain:v1", "layer_family": "tf_target"}, "continuation_decision": "stop", "blocked_reason": null, "summary": "x"}`

**F31: a layer tag with a malformed version**
Answer:
`{"status": "resolved", "provenance": null, "result": {"layer_id": "HumanNetV3:string_ppi:global:all:3", "layer_family": "ppi"}, "continuation_decision": "stop", "blocked_reason": null, "summary": "x"}`

**F32: a tool result that is not JSON**
Tool call:
`lookup_human_gene_symbol({"gene_symbol": "TP53"})`
Tool result:
`{"status": "resolved" "provenance": null}`
Thinking:
`The result cannot be read.`
Answer:
`{"status": "resolved", "provenance": null, "result": {}, "continuation_decision": "stop", "blocked_reason": null, "summary": "x"}`

**F33: a harness registry block that is not JSON**
Context (from the harness):
`Registry entries, Ensembl release 116:`
`{"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_f33a01"} "entries": []}`
Answer:
`{"status": "resolved", "provenance": null, "result": {}, "continuation_decision": "stop", "blocked_reason": null, "summary": "x"}`

**D34: the trace that F34 repeats (not a declared fixture)**
User:
`Find the human Ensembl gene ID for PGCKA1.`
Tool call:
`lookup_human_gene_symbol({"gene_symbol": "PGCKA1"})`
Tool result:
`{"status": "resolved", "provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_d34a01"}, "gene_id": "ENSG00000154274", "gene_symbol": "PGCKA1"}`
Thinking:
`The registry maps PGCKA1 to one ID.`
Answer:
`{"status": "resolved", "provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_d34a01"}, "result": {"gene_id": "ENSG00000154274", "gene_symbol": "PGCKA1"}, "continuation_decision": "stop", "blocked_reason": null, "summary": "x"}`

**F34: a trace that repeats another trace's input and tool outcomes**
User:
`Find the human Ensembl gene ID for PGCKA1.`
Tool call:
`lookup_human_gene_symbol({"gene_symbol": "PGCKA1"})`
Tool result:
`{"status": "resolved", "provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_f34a01"}, "gene_id": "ENSG00000154274", "gene_symbol": "PGCKA1"}`
Thinking:
`The registry maps PGCKA1 to one ID.`
Answer:
`{"status": "resolved", "provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_f34a01"}, "result": {"gene_id": "ENSG00000154274", "gene_symbol": "PGCKA1"}, "continuation_decision": "stop", "blocked_reason": null, "summary": "x"}`

**F35: a tool result with no Thinking step after it**
Tool call:
`lookup_human_gene_symbol({"gene_symbol": "TP53"})`
Tool result:
`{"status": "resolved", "provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_f35a01"}, "gene_id": "ENSG00000141510", "gene_symbol": "TP53"}`
Answer:
`{"status": "resolved", "provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_f35a01"}, "result": {"gene_id": "ENSG00000141510", "gene_symbol": "TP53"}, "continuation_decision": "stop", "blocked_reason": null, "summary": "x"}`

**F36: a lookup result that carries a membership evidence_id**
Tool call:
`lookup_human_gene_symbol({"gene_symbol": "TP53"})`
Tool result:
`{"status": "resolved", "provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_mem_f36a01"}, "gene_id": "ENSG00000141510", "gene_symbol": "TP53"}`
Thinking:
`The registry maps TP53 to one ID.`
Answer:
`{"status": "resolved", "provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_mem_f36a01"}, "result": {"gene_id": "ENSG00000141510", "gene_symbol": "TP53"}, "continuation_decision": "stop", "blocked_reason": null, "summary": "x"}`

**F37: an answer with no provenance key**
Answer:
`{"status": "resolved", "result": {}, "continuation_decision": "stop", "blocked_reason": null, "summary": "x"}`

**F38: an evidence_id that another row already used**
Tool call:
`lookup_human_gene_symbol({"gene_symbol": "TP53"})`
Tool result:
`{"status": "resolved", "provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_f14a01"}, "gene_id": "ENSG00000141510", "gene_symbol": "TP53"}`
Thinking:
`The registry maps TP53 to one ID.`
Answer:
`{"status": "resolved", "provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_f14a01"}, "result": {"gene_id": "ENSG00000141510", "gene_symbol": "TP53"}, "continuation_decision": "stop", "blocked_reason": null, "summary": "x"}`

**F39: an anchor type that is not in the anchor list**
Answer:
`{"status": "resolved", "provenance": null, "result": {}, "continuation_decision": "stop", "blocked_reason": null, "summary": "The gene is a node in <MULTIPLEX:full_brain_multiplex_v1>."}`

**F40: a claim list out of vocabulary order**
Answer:
`{"status": "resolved", "provenance": null, "result": {"supported_claims": ["context_predictive_expression"], "unsupported_claims": ["regulatory_link_recorded", "physical_interaction_recorded", "regulation_direction", "global_coexpression", "cofunctional_association", "holds_in_other_contexts", "causal_effect"]}, "continuation_decision": "stop", "blocked_reason": null, "summary": "x"}`

**F41: a summary that asks for input on an answer that does not defer**
Tool call:
`lookup_human_gene_symbol({"gene_symbol": "TP53"})`
Tool result:
`{"status": "resolved", "provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_f41a01"}, "gene_id": "ENSG00000141510", "gene_symbol": "TP53"}`
Thinking:
`The registry maps TP53 to one ID.`
Answer:
`{"status": "resolved", "provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_f41a01"}, "result": {"gene_id": "ENSG00000141510", "gene_symbol": "TP53"}, "continuation_decision": "stop", "blocked_reason": null, "summary": "TP53 is <GENE:ENSG00000141510|TP53>. Give one Ensembl gene ID to continue."}`

**F42: a defer summary that asks for nothing**
Tool call:
`lookup_human_gene_symbol({"gene_symbol": "LINC02170"})`
Tool result:
`{"status": "not_found", "provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_f42a01"}, "gene_symbol": "LINC02170", "candidate_gene_ids": []}`
Thinking:
`The registry has no match for LINC02170.`
Answer:
`{"status": "not_found", "provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_f42a01"}, "result": {"gene_symbol": "LINC02170"}, "continuation_decision": "defer", "blocked_reason": "The registry has no gene with the symbol LINC02170 in Ensembl release 116.", "summary": "LINC02170 has no match in Ensembl release 116."}`

**F43: a summary anchor for a gene that is not in the result**
Tool call:
`lookup_human_gene_symbol({"gene_symbol": "TP53"})`
Tool result:
`{"status": "resolved", "provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_f43a01"}, "gene_id": "ENSG00000141510", "gene_symbol": "TP53"}`
Thinking:
`The registry maps TP53 to one ID.`
Answer:
`{"status": "resolved", "provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_f43a01"}, "result": {"gene_id": "ENSG00000141510", "gene_symbol": "TP53"}, "continuation_decision": "stop", "blocked_reason": null, "summary": "TP53 is <GENE:ENSG00000141736|ERBB2> in Ensembl release 116."}`

**F44: failed_tool on an answer that rests on failed calls alone**
Tool call:
`lookup_human_gene_symbol({"gene_symbol": "TP53"})`
Tool result:
`{"status": "error", "provenance": null, "error_code": "BACKEND_UNAVAILABLE", "message": "registry unreachable"}`
Thinking:
`The call failed, so I have no result for TP53.`
Answer:
`{"status": "error", "provenance": null, "result": {"failed_tool": "lookup_human_gene_symbol", "error_code": "BACKEND_UNAVAILABLE", "gene_symbol": "TP53"}, "continuation_decision": "defer", "blocked_reason": "The call to lookup_human_gene_symbol failed with BACKEND_UNAVAILABLE, so there is no registry result for TP53.", "summary": "The registry lookup for TP53 failed. Retry the lookup to continue."}`

**F45: a tool result whose keys are out of order**
Tool call:
`lookup_human_gene_symbol({"gene_symbol": "TP53"})`
Tool result:
`{"status": "resolved", "gene_id": "ENSG00000141510", "provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_f45a01"}, "gene_symbol": "TP53"}`
Thinking:
`The registry maps TP53 to one ID.`
Answer:
`{"status": "resolved", "provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_f45a01"}, "result": {"gene_id": "ENSG00000141510", "gene_symbol": "TP53"}, "continuation_decision": "stop", "blocked_reason": null, "summary": "x"}`

# S1.8: Direction Conventions

**F18: trace label disagrees with its enclosing family heading**
Answer:
`{"status": "resolved", "provenance": null, "result": {}, "continuation_decision": "stop", "blocked_reason": null, "summary": "x"}`

**F19: a claim appears in neither claim set**
Answer:
`{"status": "resolved", "provenance": null, "result": {"supported_claims": ["context_predictive_expression"], "unsupported_claims": ["physical_interaction_recorded", "regulatory_link_recorded", "regulation_direction", "global_coexpression", "cofunctional_association", "holds_in_other_contexts"]}, "continuation_decision": "stop", "blocked_reason": null, "summary": "x"}`

**F20: a claim appears in both claim sets**
Answer:
`{"status": "resolved", "provenance": null, "result": {"supported_claims": ["context_predictive_expression"], "unsupported_claims": ["context_predictive_expression", "physical_interaction_recorded", "regulatory_link_recorded", "regulation_direction", "global_coexpression", "cofunctional_association", "holds_in_other_contexts", "causal_effect"]}, "continuation_decision": "stop", "blocked_reason": null, "summary": "x"}`

**F21: a tool failure reported as a clean negative**
Tool call:
`lookup_human_gene_symbol({"gene_symbol": "TP53"})`
Tool result:
`{"status": "error", "provenance": null, "error_code": "BACKEND_UNAVAILABLE", "message": "registry unreachable"}`
Thinking:
`The call failed.`
Answer:
`{"status": "not_found", "provenance": null, "result": {"gene_symbol": "TP53"}, "continuation_decision": "defer", "blocked_reason": "The registry has no gene with the symbol TP53 in Ensembl release 116.", "summary": "x"}`
