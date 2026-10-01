**P1: clean resolved**
`Answer:`
`{"status": "resolved", "provenance": null, "result": {}, "continuation_decision": "stop", "blocked_reason": null, "summary": "x"}`

**P2: clean defer with a filled sentence**
Tool call:
`lookup_human_gene_symbol({"gene_symbol": "LINC02170"})`
Tool result:
`{"status": "not_found", "provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_p02a01"}, "gene_symbol": "LINC02170", "candidate_gene_ids": []}`
Thinking:
`The registry has no match for LINC02170.`
Answer:
`{"status": "not_found", "provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_p02a01"}, "result": {"gene_symbol": "LINC02170"}, "continuation_decision": "defer", "blocked_reason": "The registry has no gene with the symbol LINC02170 in Ensembl release 116.", "summary": "LINC02170 has no match in Ensembl release 116. Give a current symbol or an Ensembl gene ID to continue."}`

**P3: all four layer grammars**
`Answer:`
`{"status": "resolved", "provenance": null, "result": {"a": "HumanNetV3:string_ppi:global:all:v3", "b": "bulkPEN:bulk_pen:brain:amygdala:v1", "c": "scPEN:single_cell_pen:brain:hippocampus:astrocyte:v1", "d": "TFs:tf_target:brain:all:v1", "layer_family": "scpen", "supported_claims": ["context_predictive_expression"]}, "continuation_decision": "stop", "blocked_reason": null, "summary": "<LAYER:HumanNetV3:string_ppi:global:all:v3>, <LAYER:bulkPEN:bulk_pen:brain:amygdala:v1>, <LAYER:scPEN:single_cell_pen:brain:hippocampus:astrocyte:v1>, and <LAYER:TFs:tf_target:brain:all:v1> parse."}`

**P4: a lookup and a membership check, the answer copies the lookup**
Tool call:
`lookup_human_gene_symbol({"gene_symbol": "PPP2R3B"})`
Tool result:
`{"status": "ambiguous", "provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_p04a01"}, "gene_symbol": "PPP2R3B", "candidate_gene_ids": ["ENSG00000167393", "ENSG00000292327"], "candidate_details": {"ENSG00000167393": {"chromosome": "X", "biotype": "protein_coding"}, "ENSG00000292327": {"chromosome": "Y", "biotype": "protein_coding"}}}`
Thinking:
`The registry maps PPP2R3B to two IDs.`
Tool call:
`check_multiplex_membership({"multiplex_id": "full_brain_multiplex_v1",`
`"gene_ids": ["ENSG00000167393", "ENSG00000292327"]})`
Tool result:
`{"status": "resolved", "provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_mem_p04b02"}, "multiplex_id": "full_brain_multiplex_v1", "present_gene_ids": ["ENSG00000167393"], "absent_gene_ids": ["ENSG00000292327"]}`
Thinking:
`Only one candidate is a node.`
Answer:
`{"status": "resolved", "provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_p04a01"}, "result": {"gene_id": "ENSG00000167393", "gene_symbol": "PPP2R3B", "candidate_gene_ids": ["ENSG00000167393", "ENSG00000292327"], "multiplex_id": "full_brain_multiplex_v1", "present_gene_ids": ["ENSG00000167393"]}, "continuation_decision": "continue", "blocked_reason": null, "summary": "x"}`

**P5: a genuinely rejected tag in result.input is not parsed**
Answer:
`{"status": "invalid_input", "provenance": null, "result": {"input": "scPEN:single_cell_pen:brain:hippocampus:astrocyte", "expected": "layer_tag"}, "continuation_decision": "defer", "blocked_reason": "The input scPEN:single_cell_pen:brain:hippocampus:astrocyte is not a well-formed layer tag.", "summary": "The tag scPEN:single_cell_pen:brain:hippocampus:astrocyte has no version part. Give the full tag to continue."}`

**P6: a masked earlier turn is context, not this trace's answer**
[Earlier turn: in context, masked, no thinking]
Answer:
`{"status": "ambiguous", "provenance": null, "result": {}, "continuation_decision": "defer", "blocked_reason": "x", "summary": "x"}`

[This turn: trained]
Answer:
`{"status": "resolved", "provenance": null, "result": {}, "continuation_decision": "stop", "blocked_reason": null, "summary": "x"}`

**P7: provenance present and faithfully copied**
Tool call:
`lookup_human_gene_symbol({"gene_symbol": "TP53"})`
Tool result:
`{"status": "resolved", "provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_p07a01"}, "gene_id": "ENSG00000141510", "gene_symbol": "TP53"}`
Thinking:
`The registry gives one ID for TP53.`
Answer:
`{"status": "resolved", "provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_p07a01"}, "result": {"gene_id": "ENSG00000141510", "gene_symbol": "TP53"}, "continuation_decision": "continue", "blocked_reason": null, "summary": "TP53 is <GENE:ENSG00000141510|TP53> in Ensembl release 116. The analysis continues with this ID."}`

**P8: provenance carried forward from an earlier turn**
[Earlier turn: in context, masked, no thinking]
Answer:
`{"status": "ambiguous", "provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_p08a01"}, "result": {"gene_symbol": "FAM197Y5", "candidate_gene_ids": ["ENSG00000225516", "ENSG00000290330"], "candidate_details": {"ENSG00000225516": {"chromosome": "Y", "biotype": "transcribed_unprocessed_pseudogene"}, "ENSG00000290330": {"chromosome": "Y", "biotype": "lncRNA"}}}, "continuation_decision": "defer", "blocked_reason": "The symbol FAM197Y5 matches 2 genes in Ensembl release 116, and nothing in the task chooses one.", "summary": "x"}`

[This turn: trained]
Answer:
`{"status": "resolved", "provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_p08a01"}, "result": {"gene_id": "ENSG00000290330", "gene_symbol": "FAM197Y5", "candidate_gene_ids": ["ENSG00000225516", "ENSG00000290330"]}, "continuation_decision": "continue", "blocked_reason": null, "summary": "x"}`

**P9: the two claim sets partition the vocabulary exactly**
Answer:
`{"status": "resolved", "provenance": null, "result": {"supported_claims": ["context_predictive_expression"], "unsupported_claims": ["physical_interaction_recorded", "regulatory_link_recorded", "regulation_direction", "global_coexpression", "cofunctional_association", "holds_in_other_contexts", "causal_effect"]}, "continuation_decision": "stop", "blocked_reason": null, "summary": "x"}`

**P10: a failed tool call is not evidence, so no provenance is due**
Tool call:
`lookup_human_gene_symbol({"gene_symbol": "TP53"})`
Tool result:
`{"status": "error", "provenance": null, "error_code": "BACKEND_UNAVAILABLE", "message": "registry unreachable"}`
Thinking:
`The call failed, so I have no result for TP53.`
Answer:
`{"status": "error", "provenance": null, "result": {"tool": "lookup_human_gene_symbol", "error_code": "BACKEND_UNAVAILABLE", "gene_symbol": "TP53"}, "continuation_decision": "defer", "blocked_reason": "The call to lookup_human_gene_symbol failed with BACKEND_UNAVAILABLE, so there is no registry result for TP53.", "summary": "The registry lookup for TP53 failed because the backend was unreachable. Retry the lookup to continue."}`

**P11: a harness registry block, no call, the answer copies the block**
Context (from the harness):
`Registry entries, Ensembl release 116:`
`{"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_p11a01"}, "entries": [{"gene_id": "ENSG00000154274", "gene_symbol": "PGCKA1"}]}`
Answer:
`{"status": "resolved", "provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_p11a01"}, "result": {"gene_id": "ENSG00000154274", "gene_symbol": "PGCKA1"}, "continuation_decision": "stop", "blocked_reason": null, "summary": "x"}`

**P12: a near-miss block and a lookup, the answer copies the lookup**
Context (from the harness):
`Registry entries, Ensembl release 116:`
`{"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_p12a01"}, "entries": [{"gene_id": "ENSG00000154274", "gene_symbol": "PGCKA1"}]}`
Tool call:
`lookup_human_gene_symbol({"gene_symbol": "VARS1"})`
Tool result:
`{"status": "resolved", "provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_p12b02"}, "gene_id": "ENSG00000204394", "gene_symbol": "VARS1"}`
Thinking:
`The registry maps VARS1 to one ID.`
Answer:
`{"status": "resolved", "provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_p12b02"}, "result": {"gene_id": "ENSG00000204394", "gene_symbol": "VARS1"}, "continuation_decision": "continue", "blocked_reason": null, "summary": "x"}`

**P13: a partial failure keeps the lookup and fits the lookup status**
Tool call:
`lookup_human_gene_symbol({"gene_symbol": "GOLGA2P9"})`
Tool result:
`{"status": "ambiguous", "provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_p13a01"}, "gene_symbol": "GOLGA2P9", "candidate_gene_ids": ["ENSG00000269332", "ENSG00000290587"], "candidate_details": {"ENSG00000269332": {"chromosome": "19", "biotype": "transcribed_unprocessed_pseudogene"}, "ENSG00000290587": {"chromosome": "19", "biotype": "lncRNA"}}}`
Thinking:
`The registry maps GOLGA2P9 to two IDs.`
Tool call:
`check_multiplex_membership({"multiplex_id": "full_brain_multiplex_v1", "gene_ids": ["ENSG00000269332", "ENSG00000290587"]})`
Tool result:
`{"status": "error", "provenance": null, "error_code": "BACKEND_UNAVAILABLE", "message": "graph service unreachable"}`
Thinking:
`The membership check failed.`
Answer:
`{"status": "ambiguous", "provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_p13a01"}, "result": {"gene_symbol": "GOLGA2P9", "candidate_gene_ids": ["ENSG00000269332", "ENSG00000290587"], "candidate_details": {"ENSG00000269332": {"chromosome": "19", "biotype": "transcribed_unprocessed_pseudogene"}, "ENSG00000290587": {"chromosome": "19", "biotype": "lncRNA"}}, "multiplex_id": "full_brain_multiplex_v1", "failed_tool": "check_multiplex_membership", "error_code": "BACKEND_UNAVAILABLE"}, "continuation_decision": "defer", "blocked_reason": "The lookup found 2 candidates, but the call to check_multiplex_membership failed with BACKEND_UNAVAILABLE, so no candidate is checked in full_brain_multiplex_v1.", "summary": "GOLGA2P9 matches 2 genes in Ensembl release 116: <GENE:ENSG00000269332|GOLGA2P9> and <GENE:ENSG00000290587|GOLGA2P9>. Retry the check, or give one Ensembl gene ID to continue."}`

**P14: a wrong-type sentence fills its label slot from metadata**
Answer:
`{"status": "invalid_input", "provenance": null, "result": {"input": "brain:hippocampus:hippocampus:astrocyte:v1", "expected": "layer_tag"}, "continuation_decision": "defer", "blocked_reason": "The input brain:hippocampus:hippocampus:astrocyte:v1 is a context ID, not a layer tag.", "summary": "The input is a context ID, not a layer tag. Give a layer tag to continue."}`

**P15: an S1.4 sentence reads the side that blocks the answer**
Context (from the harness):
`Module group_0, gene_ids: ["ENSG00000141736", "ENSG00000141510"]`
Tool call:
`lookup_human_gene_symbol({"gene_symbol": "HER2"})`
Tool result:
`{"status": "not_found", "provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_p15a01"}, "gene_symbol": "HER2", "candidate_gene_ids": []}`
Thinking:
`The registry has no match for HER2.`
Answer:
`{"status": "not_found", "provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_p15a01"}, "result": {"first": {"item": "abstract", "input": "HER2", "gene_id": null}, "second": {"item": "module", "input": "ENSG00000141736", "gene_id": "ENSG00000141736"}, "same_gene": null}, "continuation_decision": "defer", "blocked_reason": "The registry has no gene with the symbol HER2 in Ensembl release 116.", "summary": "HER2 has no match in Ensembl release 116, so this answer does not say whether it names the same gene as <GENE:ENSG00000141736>. Give the gene symbol or Ensembl ID for HER2 to continue."}`

**P16: a success row and its failure row share a prompt (minimal pair)**
User:
`Find the human Ensembl gene ID for VARS1.`
Tool call:
`lookup_human_gene_symbol({"gene_symbol": "VARS1"})`
Tool result:
`{"status": "resolved", "provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_p16a01"}, "gene_id": "ENSG00000204394", "gene_symbol": "VARS1"}`
Thinking:
`The registry maps VARS1 to one ID.`
Answer:
`{"status": "resolved", "provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_p16a01"}, "result": {"gene_id": "ENSG00000204394", "gene_symbol": "VARS1"}, "continuation_decision": "stop", "blocked_reason": null, "summary": "VARS1 is <GENE:ENSG00000204394|VARS1> in Ensembl release 116."}`

**P17: the failure row of the P16 minimal pair**
User:
`Find the human Ensembl gene ID for VARS1.`
Tool call:
`lookup_human_gene_symbol({"gene_symbol": "VARS1"})`
Tool result:
`{"status": "error", "provenance": null, "error_code": "BACKEND_UNAVAILABLE", "message": "registry unreachable"}`
Thinking:
`The call failed, so I have no result for VARS1.`
Answer:
`{"status": "error", "provenance": null, "result": {"tool": "lookup_human_gene_symbol", "error_code": "BACKEND_UNAVAILABLE", "gene_symbol": "VARS1"}, "continuation_decision": "defer", "blocked_reason": "The call to lookup_human_gene_symbol failed with BACKEND_UNAVAILABLE, so there is no registry result for VARS1.", "summary": "The registry lookup for VARS1 failed because the backend was unreachable. Retry the lookup to continue."}`
