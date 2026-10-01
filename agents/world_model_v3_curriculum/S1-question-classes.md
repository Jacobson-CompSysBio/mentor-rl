> Updated 2026-09-30 to the S1 contract in `S1-coordinate-system.md`. Changes from the advisor's version: `provenance` on every tool result and answer (`null` without registry evidence), a per-call `evidence_id`, the registry block as JSON, `blocked_reason` as a sentence on every `defer` answer (`null` otherwise), the frozen bulkPEN and TF tag form, and the agreed trace fixes. New traces: S1.6.7 and S1.7.9. S0.7 still uses a placeholder ID. Updated 2026-10-01: the S1.3.10 result names its multiplex, the S0.5 summary uses digits, the scPEN traces use the six-field lattice tags, and S1.3.6 uses BMS1P4 (trace fixes 12 to 15).

# S0: Identity Worked Traces

**Overall guidance:**

* Case  
  * Lookup tools should ignore case and return the canonical form. The tool argument should copy user text exactly, and the answer should use the tool’s canonical form  
  * User input case should be perturbed to simulate real-world case issues  
* Real Examples Only  
  * Every question should be generated from and grounded in Ensembl release 116  
* Extended Registry  
  * All ensembl 116 release genes should be included, named or unnamed  
  * Tool calls should distinguish between unnamed genes and genes that don’t exist  
* Task Framing  
  * Examples should be split \~50/50 between steps in a larger task and a direct question; this will induce both `continue` and `stop` results

**S0.1: Named symbol as a step in a larger task**  
`User:`  
`Resolve the human gene symbol TP53 for the next analysis step.`

`Thinking:`  
`The next step needs the Ensembl ID for TP53. I will look up the symbol in the registry.`

`Tool call:`  
`lookup_human_gene_symbol({"gene_symbol": "TP53"})`

`Tool result:`  
`{"status": "resolved",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_l2n1p4"},`  
 `"gene_id": "ENSG00000141510",`  
 `"gene_symbol": "TP53"}`

`Thinking:`  
`The registry gives one ID for TP53, so the next step can use it.`

`Answer:`  
`{"status": "resolved",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_l2n1p4"},`  
 `"result": {"gene_id": "ENSG00000141510", "gene_symbol": "TP53"},`  
 `"continuation_decision": "continue",`  
 `"blocked_reason": null,`  
 `"summary": "TP53 is <GENE:ENSG00000141510|TP53> in Ensembl release 116."}`

**S0.2: Named ID in a direct question**  
`User:`  
`What is the gene symbol for ENSG00000141510?`

`Thinking:`  
`I need the symbol for this Ensembl ID. I will look it up in the registry.`

`Tool call:`  
`lookup_human_ensembl_gene_id({"gene_id": "ENSG00000141510"})`

`Tool result:`  
`{"status": "resolved",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_ibf12p"},`  
 `"gene_id": "ENSG00000141510",`  
 `"gene_symbol": "TP53"}`

`Thinking:`  
`The registry gives one symbol for this ID. The question asks only for the`  
`symbol, so the task is complete.`

`Answer:`  
`{"status": "resolved",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_ibf12p"},`  
 `"result": {"gene_id": "ENSG00000141510", "gene_symbol": "TP53"},`  
 `"continuation_decision": "stop",`  
 `"blocked_reason": null,`  
 `"summary": "<GENE:ENSG00000141510|TP53> has the symbol TP53 in Ensembl release 116."}`

**S0.3: Lowercase input**  
`User:`  
`Look up tp53 before the next analysis step.`

`Thinking:`  
`The task gives the symbol in lowercase. The lookup tool matches without`  
`regard to case, so I will pass the text as given.`

`Tool call:`  
`lookup_human_gene_symbol({"gene_symbol": "tp53"})`

`Tool result:`  
`{"status": "resolved",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_he480b"},`  
 `"gene_id": "ENSG00000141510",`  
 `"gene_symbol": "TP53"}`

`Thinking:`  
`The registry matches tp53 to one ID and gives the canonical symbol TP53. The`  
`next step can use the ID.`

`Answer:`  
`{"status": "resolved",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_he480b"},`  
 `"result": {"gene_id": "ENSG00000141510", "gene_symbol": "TP53"},`  
 `"continuation_decision": "continue",`  
 `"blocked_reason": null,`  
 `"summary": "The input tp53 is <GENE:ENSG00000141510|TP53> in Ensembl release 116."}`

**S0.4: Unnamed gene as a step in a task**  
`User:`  
`Give the symbol for ENSG00000299624 before the next analysis step.`

`Thinking:`  
`I need the symbol for this Ensembl ID. I will look it up in the registry.`

`Tool call:`  
`lookup_human_ensembl_gene_id({"gene_id": "ENSG00000299624"})`

`Tool result:`  
`{"status": "resolved",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_bhspfg"},`  
 `"gene_id": "ENSG00000299624",`  
 `"gene_symbol": null}`

`Thinking:`  
`The registry has this ID but no symbol for it. The ID is still valid, so the`  
`analysis can continue with it.`

`Answer:`  
`{"status": "resolved",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_bhspfg"},`  
 `"result": {"gene_id": "ENSG00000299624", "gene_symbol": null},`  
 `"continuation_decision": "continue",`  
 `"blocked_reason": null,`  
 `"summary": "<GENE:ENSG00000299624> is a human gene in Ensembl release 116 with no known symbol. The analysis continues with the ID."}`

**S0.5: Ambiguous symbol**  
`User:`  
`Resolve the human gene symbol GOLGA2P9 for the next analysis step.`

`Thinking:`  
`The next step needs one Ensembl ID for GOLGA2P9. I will look up the symbol`  
`in the registry.`

`Tool call:`  
`lookup_human_gene_symbol({"gene_symbol": "GOLGA2P9"})`

`Tool result:`  
`{"status": "ambiguous",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_fqucik"},`  
 `"gene_symbol": "GOLGA2P9",`  
 `"candidate_gene_ids": ["ENSG00000269332", "ENSG00000290587"],`  
 `"candidate_details": {"ENSG00000269332": {"chromosome": "19", "biotype": "transcribed_unprocessed_pseudogene"},`  
                       `"ENSG00000290587": {"chromosome": "19", "biotype": "lncRNA"}}}`

`Thinking:`  
`The registry maps GOLGA2P9 to two IDs and cannot choose one. I will keep`  
`every candidate and ask for one ID.`

`Answer:`  
`{"status": "ambiguous",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_fqucik"},`  
 `"result": {"gene_symbol": "GOLGA2P9",`  
            `"candidate_gene_ids": ["ENSG00000269332", "ENSG00000290587"],`  
            `"candidate_details": {"ENSG00000269332": {"chromosome": "19", "biotype": "transcribed_unprocessed_pseudogene"},`  
                                  `"ENSG00000290587": {"chromosome": "19", "biotype": "lncRNA"}}},`  
 `"continuation_decision": "defer",`  
 `"blocked_reason": "The symbol GOLGA2P9 matches 2 genes in Ensembl release 116, and nothing in the task chooses one.",`  
 `"summary": "The symbol GOLGA2P9 matches 2 genes in Ensembl release 116: <GENE:ENSG00000269332|GOLGA2P9>, <GENE:ENSG00000290587|GOLGA2P9>. Give one Ensembl gene ID to continue."}`

**S0.6: Absent symbol**  
`User:`  
`Resolve the human gene symbol LINC02170.`

`Thinking:`  
`I need the Ensembl ID for LINC02170. I will look it up in the registry.`

`Tool call:`  
`lookup_human_gene_symbol({"gene_symbol": "LINC02170"})`

`Tool result:`  
`{"status": "not_found",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_icfu38"},`  
 `"gene_symbol": "LINC02170",`  
 `"candidate_gene_ids": []}`

`Thinking:`  
`The registry has no match for LINC02170. I will not replace it with a name`  
`from memory.`

`Answer:`  
`{"status": "not_found",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_icfu38"},`  
 `"result": {"gene_symbol": "LINC02170"},`  
 `"continuation_decision": "defer",`  
 `"blocked_reason": "The registry has no gene with the symbol LINC02170 in Ensembl release 116.",`  
 `"summary": "LINC02170 has no match in Ensembl release 116. Give a current symbol or an Ensembl gene ID to continue."}`

**S0.7: ID not in release 116**  
`User:`  
`What is the gene symbol for ENSG00000XXXXXX?`

`Thinking:`  
`I need the symbol for this Ensembl ID. I will look it up in the registry.`

`Tool call:`  
`lookup_human_ensembl_gene_id({"gene_id": "ENSG00000XXXXXX"})`

`Tool result:`  
`{"status": "not_found",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_x7lj0h"},`  
 `"gene_id": "ENSG00000XXXXXX",`  
 `"gene_symbol": null}`

`Thinking:`  
`The registry has no gene with this ID in release 116. I will not replace it`  
`with an ID or a symbol from memory.`

`Answer:`  
`{"status": "not_found",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_x7lj0h"},`  
 `"result": {"gene_id": "ENSG00000XXXXXX"},`  
 `"continuation_decision": "defer",`  
 `"blocked_reason": "The registry has no gene with the ID ENSG00000XXXXXX in Ensembl release 116.",`  
 `"summary": "ENSG00000XXXXXX is not a gene ID in Ensembl release 116. Give a current Ensembl gene ID to continue."}`

# S1.1: Symbol-to-Ensembl Mapping

**Overall guidance**

* Skip the call only on exact match  
  * If only the only no-tool rows have a block containing the symbol, the model learns “block present \= skip”  
  * Add near miss rows: listing other genes, coming from another release, etc, where a tool still must be called  
  * This trains the no-call action of the model  
* Absent-symbol pool  
  * Previous HGNC symbols and aliases that do not match a current registry symbol, without regard to case  
* Choose the tool based on the input form  
  * Models must recognize that ensembl \= ensembl tool and symbol \= symbol tool  
* Species  
  * Because tools ignore case now, there may be some overlap between species names (mouse Gapdh vs. human GAPDH)  
  * If the task names another species, give `invalid_input` \+ no call  
  * If the task names no species, default to treating it as human  
* Wording  
  * Use “thinking” not “assistant explanation” to match oss-harmony format  
  * Don’t say “verified” in reasoning, since that’s not what the tool is doing – it’s mapping  
  * Vary the prompt wording; the cue to skip a tool call shouldn’t be dependent on a fixed phrase  
* Case Variants  
  * \~10% of rows use lowercase/mixed case to simulate human error

**S1.1.1: Direct Resolution**  
`User:`  
`Find the human Ensembl gene ID for VARS1.`

`Thinking:`  
`The task gives only the symbol VARS1. I will look it up in the registry.`

`Tool call:`  
`lookup_human_gene_symbol({"gene_symbol": "VARS1"})`

`Tool result:`  
`{"status": "resolved",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_44k6e0"},`  
 `"gene_id": "ENSG00000204394",`  
 `"gene_symbol": "VARS1"}`

`Thinking:`  
`The registry maps VARS1 to one ID, ENSG00000204394. The task asks only for`  
`this ID, so the task is complete.`

`Answer:`  
`{"status": "resolved",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_44k6e0"},`  
 `"result": {"gene_id": "ENSG00000204394", "gene_symbol": "VARS1"},`  
 `"continuation_decision": "stop",`  
 `"blocked_reason": null,`  
 `"summary": "VARS1 is <GENE:ENSG00000204394|VARS1> in Ensembl release 116."}`

**S1.1.2: Resolution within task**  
`User:`  
`Prepare the Ensembl gene ID for the seed WDR95P so the larger analysis can continue.`

`Thinking:`  
`The next step needs the Ensembl ID for the seed WDR95P. I will look it up in the registry.`

`Tool call:`  
`lookup_human_gene_symbol({"gene_symbol": "WDR95P"})`

`Tool result:`  
`{"status": "resolved",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_2snfjr"},`  
 `"gene_id": "ENSG00000231894",`  
 `"gene_symbol": "WDR95P"}`

`Thinking:`  
`The registry maps WDR95P to one ID, ENSG00000231894. The analysis can`  
`continue with this ID.`

`Answer:`  
`{"status": "resolved",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_2snfjr"},`  
 `"result": {"gene_id": "ENSG00000231894", "gene_symbol": "WDR95P"},`  
 `"continuation_decision": "continue",`  
 `"blocked_reason": null,`  
 `"summary": "The seed WDR95P is <GENE:ENSG00000231894|WDR95P> in Ensembl release 116. The analysis continues with this ID."}`

**S1.1.3: Absent Symbol (Previous Symbol or Alias)**  
`User:`  
`Resolve this human gene symbol: LINC02170.`

`Thinking:`  
`The task gives only the symbol LINC02170. I will look it up in the registry.`

`Tool call:`  
`lookup_human_gene_symbol({"gene_symbol": "LINC02170"})`

`Tool result:`  
`{"status": "not_found",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_9rb4vl"},`  
 `"gene_symbol": "LINC02170",`  
 `"candidate_gene_ids": []}`

`Thinking:`  
`The registry has no match for LINC02170. I will not replace it with a symbol or an ID from memory, so I must defer.`

`Answer:`  
`{"status": "not_found",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_9rb4vl"},`  
 `"result": {"gene_symbol": "LINC02170"},`  
 `"continuation_decision": "defer",`  
 `"blocked_reason": "The registry has no gene with the symbol LINC02170 in Ensembl release 116.",`  
 `"summary": "LINC02170 has no match in Ensembl release 116, so no graph lookup can use it. Give a current symbol or an Ensembl gene ID to continue."}`

**S1.1.4: Already resolved (Harness block lists the symbol)**  
`Context (from the harness):`  
`Registry entries, Ensembl release 116:`  
`{"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_7bdwsi"},`  
 `"entries": [{"gene_id": "ENSG00000154274", "gene_symbol": "PGCKA1"}]}`

`User:`  
`Return the Ensembl gene ID for PGCKA1.`

`Thinking:`  
`The registry entries from the harness list PGCKA1 for Ensembl release 116.`  
`The lookup is already done, so I will not call the tool.`

`Answer:`  
`{"status": "resolved",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_7bdwsi"},`  
 `"result": {"gene_id": "ENSG00000154274", "gene_symbol": "PGCKA1"},`  
 `"continuation_decision": "stop",`  
 `"blocked_reason": null,`  
 `"summary": "PGCKA1 is <GENE:ENSG00000154274|PGCKA1> in Ensembl release 116."}`

**S1.1.5: Block without the symbol**  
`Context (from the harness):`  
`Registry entries, Ensembl release 116:`  
`{"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_jb83mp"},`  
 `"entries": [{"gene_id": "ENSG00000154274", "gene_symbol": "PGCKA1"}]}`

`User:`  
`Before the next step, give the Ensembl gene ID for the seed VARS1.`

`Thinking:`  
`The registry entries list PGCKA1 but not VARS1. I will look up VARS1 in the`  
`registry.`

`Tool call:`  
`lookup_human_gene_symbol({"gene_symbol": "VARS1"})`

`Tool result:`  
`{"status": "resolved",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_iwcjay"},`  
 `"gene_id": "ENSG00000204394",`  
 `"gene_symbol": "VARS1"}`

`Thinking:`  
`The registry maps VARS1 to one ID, ENSG00000204394. The next step can use`  
`it.`

`Answer:`  
`{"status": "resolved",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_iwcjay"},`  
 `"result": {"gene_id": "ENSG00000204394", "gene_symbol": "VARS1"},`  
 `"continuation_decision": "continue",`  
 `"blocked_reason": null,`  
 `"summary": "The seed VARS1 is <GENE:ENSG00000204394|VARS1> in Ensembl release 116. The analysis continues with this ID."}`

**S1.1.6: Ensembl ID given as input**  
`User:`  
`Find the human Ensembl gene ID for the seed ENSG00000231894.`

`Thinking:`  
`The seed is already an Ensembl gene ID, not a symbol. The symbol tool cannot`  
`match it, so I will check the ID with the ID tool.`

`Tool call:`  
`lookup_human_ensembl_gene_id({"gene_id": "ENSG00000231894"})`

`Tool result:`  
`{"status": "resolved",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_47ehzo"},`  
 `"gene_id": "ENSG00000231894",`  
 `"gene_symbol": "WDR95P"}`

`Thinking:`  
`The ID is in release 116 and has the symbol WDR95P. The task asks only for`  
`the ID, so the task is complete.`

`Answer:`  
`{"status": "resolved",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_47ehzo"},`  
 `"result": {"gene_id": "ENSG00000231894", "gene_symbol": "WDR95P"},`  
 `"continuation_decision": "stop",`  
 `"blocked_reason": null,`  
 `"summary": "The seed is already an Ensembl gene ID: <GENE:ENSG00000231894|WDR95P> in Ensembl release 116."}`

**S1.1.7: Not a human gene symbol**  
`User:`  
`Find the human Ensembl gene ID for the mouse gene Gapdh.`

`Thinking:`  
`The task names a mouse gene. The registry holds human genes only, and the`  
`lookup ignores case, so it could match a human symbol with the same letters.`  
`I will not call the tool.`

`Answer:`  
`{"status": "invalid_input",`  
 `"provenance": null,`  
 `"result": {"input": "Gapdh", "expected": "human_gene_symbol"},`  
 `"continuation_decision": "defer",`  
 `"blocked_reason": "The input Gapdh is a mouse gene symbol, and the registry holds human genes only.",`  
 `"summary": "Gapdh is named as a mouse gene, and the registry holds human genes only. Give a human gene symbol to continue."}`

**S1.1.8: Lookup tool unavailable**  
`User:`  
`Find the human Ensembl gene ID for VARS1.`

`Thinking:`  
`The task gives only the symbol VARS1. I will look it up in the registry.`

`Tool call:`  
`lookup_human_gene_symbol({"gene_symbol": "VARS1"})`

`Tool result:`  
`{"status": "error", "provenance": null, "error_code": "BACKEND_UNAVAILABLE", "message": "registry unreachable"}`

`Thinking:`  
`The call failed because the registry could not be reached. This does not show`  
`that VARS1 is absent, so I have no result for it. I must defer.`

`Answer:`  
`{"status": "error",`  
 `"provenance": null,`  
 `"result": {"tool": "lookup_human_gene_symbol", "error_code": "BACKEND_UNAVAILABLE", "gene_symbol": "VARS1"},`  
 `"continuation_decision": "defer",`  
 `"blocked_reason": "The call to lookup_human_gene_symbol failed with BACKEND_UNAVAILABLE, so there is no registry result for VARS1.",`  
 `"summary": "The registry lookup for VARS1 failed because the backend was unreachable. This does not mean VARS1 is absent. Retry the lookup to continue."}`

# S1.2: Ensembl-to-Symbol Mapping

**Overall guidance**

* Normalization  
  * Should be built into the tool  
  * Trim spaces, ignore cases, strip “.” suffix  
* Format check  
  * “ENSG followed by 11 digits” is an exact check, so it should be done via tool, not by the model  
  * Check inputs with non-human/non-gene ensembl prefixes  
  * Inputs with ENSG get called, but may return invalid\_input  
  * The model should not repair an ID  
* No symbol ≠ failure  
* Tool \> Memory  
  * Focus some questions on renamed genes to ensure that the model uses a tool to match the gene to the latest item in the registry  
  * Same with “retired” ids

**S1.2.1: Direct Lookup**

`User:`  
`What is the human gene symbol for ENSG00000255121?`

`Thinking:`  
`I need the symbol for this Ensembl ID. I will look it up in the registry.`

`Tool call:`  
`lookup_human_ensembl_gene_id({"gene_id": "ENSG00000255121"})`

`Tool result:`  
`{"status": "resolved",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_6afmyv"},`  
 `"gene_id": "ENSG00000255121",`  
 `"gene_symbol": "CENATAC-DT"}`

`Thinking:`  
`The registry gives the symbol CENATAC-DT for this ID in release 116. The`  
`question asks only for the symbol, so the task is complete.`

`Answer:`  
`{"status": "resolved",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_6afmyv"},`  
 `"result": {"gene_id": "ENSG00000255121", "gene_symbol": "CENATAC-DT"},`  
 `"continuation_decision": "stop",`  
 `"blocked_reason": null,`  
 `"summary": "<GENE:ENSG00000255121|CENATAC-DT> has the symbol CENATAC-DT in Ensembl release 116."}`

**S1.2.2: Lookup Inside Task**  
`User:`  
`The next operation needs a readable label. Resolve seed ENSG00000177830 to`  
`its gene symbol.`

`Thinking:`  
`The next operation needs the symbol for the seed ID. I will look it up in the`  
`registry.`

`Tool call:`  
`lookup_human_ensembl_gene_id({"gene_id": "ENSG00000177830"})`

`Tool result:`  
`{"status": "resolved",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_kd3gli"},`  
 `"gene_id": "ENSG00000177830",`  
 `"gene_symbol": "CHID1"}`

`Thinking:`  
`The registry gives the symbol CHID1 for the seed ID. The next operation can use it.`

`Answer:`  
`{"status": "resolved",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_kd3gli"},`  
 `"result": {"gene_id": "ENSG00000177830", "gene_symbol": "CHID1"},`  
 `"continuation_decision": "continue",`  
 `"blocked_reason": null,`  
 `"summary": "The seed <GENE:ENSG00000177830|CHID1> has the symbol CHID1 in Ensembl release 116. The analysis continues with this gene."}`

**S1.2.3: Version Suffix**  
`User:`  
`Give the gene symbol for human Ensembl gene ID ENSG00000099204.23.`

`Thinking:`  
`The ID has a version suffix. The tool ignores the suffix, so I will pass the ID as given.`

`Tool call:`  
`lookup_human_ensembl_gene_id({"gene_id": "ENSG00000099204.23"})`

`Tool result:`  
`{"status": "resolved",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_t2ojwk"},`  
 `"gene_id": "ENSG00000099204",`  
 `"gene_symbol": "ABLIM1"}`

`Thinking:`  
`The registry maps the ID to ABLIM1. The version suffix does not change the`  
`gene. The question asks only for the symbol, so the task is complete.`

`Answer:`  
`{"status": "resolved",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_t2ojwk"},`  
 `"result": {"gene_id": "ENSG00000099204", "gene_symbol": "ABLIM1"},`  
 `"continuation_decision": "stop",`  
 `"blocked_reason": null,`  
 `"summary": "ENSG00000099204.23 is <GENE:ENSG00000099204|ABLIM1> in Ensembl release 116. The version suffix does not change the gene."}`

**S1.2.4: No symbol: inside a task**  
`User:`  
`Prepare the gene symbol for seed ENSG00000300082, so the larger analysis can`  
`continue.`

`Thinking:`  
`The analysis needs the symbol for the seed ID. I will look it up in the`  
`registry.`

`Tool call:`  
`lookup_human_ensembl_gene_id({"gene_id": "ENSG00000300082"})`

`Tool result:`  
`{"status": "resolved",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_vtnntb"},`  
 `"gene_id": "ENSG00000300082",`  
 `"gene_symbol": null}`

`Thinking:`  
`The registry has this ID but no symbol for it. The ID is still valid, so the`  
`analysis can continue with it.`

`Answer:`  
`{"status": "resolved",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_vtnntb"},`  
 `"result": {"gene_id": "ENSG00000300082", "gene_symbol": null},`  
 `"continuation_decision": "continue",`  
 `"blocked_reason": null,`  
 `"summary": "<GENE:ENSG00000300082> is a human gene in Ensembl release 116 with no known symbol. The analysis continues with the ID."}`

**S1.2.5: Not in release 116**  
`User:`  
`Resolve this value as a human Ensembl gene ID, and give its symbol:`  
`ENSG00000182378.14_PAR_Y.`

`Thinking:`  
`I need the symbol for this Ensembl ID. I will pass the ID to the registry`  
`tool as given.`

`Tool call:`  
`lookup_human_ensembl_gene_id({"gene_id": "ENSG00000182378.14_PAR_Y"})`

`Tool result:`  
`{"status": "not_found",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_ug1qg6"},`  
 `"gene_id": "ENSG00000182378.14_PAR_Y",`  
 `"gene_symbol": null}`

`Thinking:`  
`The registry has no gene with this ID in release 116. The _PAR_Y suffix marks`  
`the chromosome Y copy of a PAR gene in an older GENCODE release. I will not`  
`map it to another gene from memory, so I must defer.`

`Answer:`  
`{"status": "not_found",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_ug1qg6"},`  
 `"result": {"gene_id": "ENSG00000182378.14_PAR_Y"},`  
 `"continuation_decision": "defer",`  
 `"blocked_reason": "The registry has no gene with the ID ENSG00000182378.14_PAR_Y in Ensembl release 116.",`  
 `"summary": "ENSG00000182378.14_PAR_Y is not a gene ID in Ensembl release 116. Give the release 116 gene ID to continue."}`

**S1.2.6: Malformed ENSG**  
`User:`  
`Resolve this human Ensembl gene ID to its symbol: ENSG0000023134.`

`Thinking:`  
`The input starts with ENSG, so it is meant as a human gene ID. The tool`  
`checks the format, so I will pass it as given.`

`Tool call:`  
`lookup_human_ensembl_gene_id({"gene_id": "ENSG0000023134"})`

`Tool result:`  
`{"status": "invalid_input",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_54ns0g"},`  
 `"input": "ENSG0000023134",`  
 `"expected": "human_gene_id"}`

`Thinking:`  
`The tool reports that the input is not a well-formed human gene ID. I will`  
`not guess a missing digit, so I must defer.`

`Answer:`  
`{"status": "invalid_input",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_54ns0g"},`  
 `"result": {"input": "ENSG0000023134", "expected": "human_gene_id"},`  
 `"continuation_decision": "defer",`  
 `"blocked_reason": "The input ENSG0000023134 is not a well-formed human gene ID.",`  
 `"summary": "ENSG0000023134 is not a well-formed human gene ID. A human gene ID is ENSG followed by 11 digits. Give the full ID to continue."}`

**S1.2.7: Already Resolved (no call)**  
`Context (from the harness):`  
`Registry entries, Ensembl release 116:`  
`{"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_e3t9t1"},`  
 `"entries": [{"gene_id": "ENSG00000230860", "gene_symbol": "CCNB1IP1P2"}]}`

`User:`  
`What is the human gene symbol for ENSG00000230860?`

`Thinking:`  
`The registry entries from the harness list this ID for Ensembl release 116,`  
`with the symbol CCNB1IP1P2. The lookup is already done, so I will not call`  
`the tool.`

`Answer:`  
`{"status": "resolved",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_e3t9t1"},`  
 `"result": {"gene_id": "ENSG00000230860", "gene_symbol": "CCNB1IP1P2"},`  
 `"continuation_decision": "stop",`  
 `"blocked_reason": null,`  
 `"summary": "<GENE:ENSG00000230860|CCNB1IP1P2> has the symbol CCNB1IP1P2 in Ensembl release 116."}`

**S1.2.8: Block without the ID**  
`Context (from the harness):`  
`Registry entries, Ensembl release 116:`  
`{"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_g8xju3"},`  
 `"entries": [{"gene_id": "ENSG00000230860", "gene_symbol": "CCNB1IP1P2"}]}`

`User:`  
`Before the next analysis step, give the symbol for seed gene`  
`ENSG00000177830.`

`Thinking:`  
`The registry entries list ENSG00000230860 but not ENSG00000177830. I will`  
`look up the seed ID in the registry.`

`Tool call:`  
`lookup_human_ensembl_gene_id({"gene_id": "ENSG00000177830"})`

`Tool result:`  
`{"status": "resolved",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_ngvk8f"},`  
 `"gene_id": "ENSG00000177830",`  
 `"gene_symbol": "CHID1"}`

`Thinking:`  
`The registry gives the symbol CHID1 for the seed ID. The next step can use`  
`it.`

`Answer:`  
`{"status": "resolved",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_ngvk8f"},`  
 `"result": {"gene_id": "ENSG00000177830", "gene_symbol": "CHID1"},`  
 `"continuation_decision": "continue",`  
 `"blocked_reason": null,`  
 `"summary": "The seed <GENE:ENSG00000177830|CHID1> has the symbol CHID1 in Ensembl release 116. The analysis continues with this gene."}`

**S1.2.9: Symbol given as input**  
`User:`  
`Give the gene symbol for human Ensembl gene ID CENATAC-DT.`

`Thinking:`  
`The input CENATAC-DT is a gene symbol, not an Ensembl ID. The ID tool cannot`  
`match it, so I will check it with the symbol tool.`

`Tool call:`  
`lookup_human_gene_symbol({"gene_symbol": "CENATAC-DT"})`

`Tool result:`  
`{"status": "resolved",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_evl2jm"},`  
 `"gene_id": "ENSG00000255121",`  
 `"gene_symbol": "CENATAC-DT"}`

`Thinking:`  
`CENATAC-DT is a symbol in release 116, with the ID ENSG00000255121. The input`  
`is already the symbol, so the task is complete.`

`Answer:`  
`{"status": "resolved",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_evl2jm"},`  
 `"result": {"gene_id": "ENSG00000255121", "gene_symbol": "CENATAC-DT"},`  
 `"continuation_decision": "stop",`  
 `"blocked_reason": null,`  
 `"summary": "The input is already a gene symbol: <GENE:ENSG00000255121|CENATAC-DT> in Ensembl release 116."}`

**S1.2.10: Not a human ID (no call)**  
`User:`  
`Before the next analysis step, give the symbol for seed gene`  
`ENSMUSG00000112237.`

`Thinking:`  
`The ID starts with ENSMUSG, which marks a mouse gene. The registry holds`  
`human genes only, so I will not call the tool.`

`Answer:`  
`{"status": "invalid_input",`  
 `"provenance": null,`  
 `"result": {"input": "ENSMUSG00000112237", "expected": "human_gene_id"},`  
 `"continuation_decision": "defer",`  
 `"blocked_reason": "The input ENSMUSG00000112237 is a mouse gene ID, and the registry holds human genes only.",`  
 `"summary": "ENSMUSG00000112237 is a mouse gene ID, and the registry holds human genes only. Give a human Ensembl gene ID to continue."}`

**S1.2.11: Lookup tool unavailable**  
`User:`  
`Give the symbol for ENSG00000141510 before the next analysis step.`

`Thinking:`  
`I need the symbol for this Ensembl ID, so I will look it up in the registry.`

`Tool call:`  
`lookup_human_ensembl_gene_id({"gene_id": "ENSG00000141510"})`

`Tool result:`  
`{"status": "error", "provenance": null, "error_code": "BACKEND_UNAVAILABLE", "message": "registry unreachable"}`

`Thinking:`  
`The call failed because the registry could not be reached. This does not show`  
`that the ID is absent from release 116. The next step needs the symbol, so I`  
`must defer.`

`Answer:`  
`{"status": "error",`  
 `"provenance": null,`  
 `"result": {"tool": "lookup_human_ensembl_gene_id", "error_code": "BACKEND_UNAVAILABLE", "gene_id": "ENSG00000141510"},`  
 `"continuation_decision": "defer",`  
 `"blocked_reason": "The call to lookup_human_ensembl_gene_id failed with BACKEND_UNAVAILABLE, so there is no registry result for ENSG00000141510.",`  
 `"summary": "The registry lookup for ENSG00000141510 failed because the backend was unreachable. This does not show that the ID is invalid. Retry the lookup to continue."}`

# S1.3: Ambiguous Symbol Resolution

**Overall guidance**

* Thinking  
  * First thinking step should be similar to s1.1 – the model doesn’t know a symbol will be ambiguous before a tool is called  
* Ambiguity classes  
  * PAR copies on chromosome X and Y  
  * A pseudogene and lncRNA with the same name on the same chromosome  
  * Multicopy families (small RNAs from s0)  
* Long lists  
  * Keep the s0 cap on the length of a candidate list  
  * If \>5 candidates, the summary anchors none and gives the count  
* Candidate details  
  * Ambiguous answer copies the candidate details to the output  
  * Allows user to examine evidence and make a choice  
* No choice from memory  
  * The model chooses a candidate in two cases: membership check leaves exactly one, or the user’s reply selects exactly one  
* Membership check  
  * Call the membership check only when task/harness names a multiplex, and defer otherwise  
  * Pass the exact candidate set  
* Basis  
  * A resolved answer names its basis; either the multiplex id or the user’s choice  
* Use next turn, not an ask user tool  
  * Let `defer` end the turn, and then let the user reply start the next user turn  
* Multi-turn render  
  * Render earlier turns just like the harness will do at inference  
  * User message \+ final answer, with no thinking

**S1.3.1: Direct ambiguous symbol, no multiplex**  
`User:`  
`Convert this human gene symbol to its Ensembl gene ID: GOLGA2P9.`

`Thinking:`  
`The task gives only the symbol GOLGA2P9. I will look it up in the registry.`

`Tool call:`  
`lookup_human_gene_symbol({"gene_symbol": "GOLGA2P9"})`

`Tool result:`  
`{"status": "ambiguous",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_d5bjht"},`  
 `"gene_symbol": "GOLGA2P9",`  
 `"candidate_gene_ids": ["ENSG00000269332", "ENSG00000290587"],`  
 `"candidate_details": {"ENSG00000269332": {"chromosome": "19", "biotype": "transcribed_unprocessed_pseudogene"},`  
                       `"ENSG00000290587": {"chromosome": "19", "biotype": "lncRNA"}}}`

`Thinking:`  
`The registry maps GOLGA2P9 to two IDs on chromosome 19: a transcribed`  
`pseudogene and a lncRNA. No multiplex is named, so nothing chooses between`  
`them. I will keep both and ask for one ID.`

`Answer:`  
`{"status": "ambiguous",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_d5bjht"},`  
 `"result": {"gene_symbol": "GOLGA2P9",`  
            `"candidate_gene_ids": ["ENSG00000269332", "ENSG00000290587"],`  
            `"candidate_details": {"ENSG00000269332": {"chromosome": "19", "biotype": "transcribed_unprocessed_pseudogene"},`  
                                  `"ENSG00000290587": {"chromosome": "19", "biotype": "lncRNA"}}},`  
 `"continuation_decision": "defer",`  
 `"blocked_reason": "The symbol GOLGA2P9 matches 2 genes in Ensembl release 116, and nothing in the task chooses one.",`  
 `"summary": "GOLGA2P9 matches 2 genes in Ensembl release 116: <GENE:ENSG00000269332|GOLGA2P9> (transcribed pseudogene) and <GENE:ENSG00000290587|GOLGA2P9> (lncRNA). Give one Ensembl gene ID to continue."}`

**S1.3.2: Ambiguous symbol inside a task**

`User:`  
`Prepare the Ensembl gene ID for seed FAM197Y5, so the larger analysis can`  
`continue.`

`Thinking:`  
`The next step needs the Ensembl ID for the seed FAM197Y5. I will look it up`  
`in the registry.`

`Tool call:`  
`lookup_human_gene_symbol({"gene_symbol": "FAM197Y5"})`

`Tool result:`  
`{"status": "ambiguous",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_yebrx6"},`  
 `"gene_symbol": "FAM197Y5",`  
 `"candidate_gene_ids": ["ENSG00000225516", "ENSG00000290330"],`  
 `"candidate_details": {"ENSG00000225516": {"chromosome": "Y", "biotype": "transcribed_unprocessed_pseudogene"},`  
                       `"ENSG00000290330": {"chromosome": "Y", "biotype": "lncRNA"}}}`

`Thinking:`  
`The registry maps FAM197Y5 to two IDs on chromosome Y: a transcribed`  
`pseudogene and a lncRNA. No multiplex is named, so nothing chooses between`  
`them. The next step needs one ID, so I will ask for it.`

`Answer:`  
`{"status": "ambiguous",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_yebrx6"},`  
 `"result": {"gene_symbol": "FAM197Y5",`  
            `"candidate_gene_ids": ["ENSG00000225516", "ENSG00000290330"],`  
            `"candidate_details": {"ENSG00000225516": {"chromosome": "Y", "biotype": "transcribed_unprocessed_pseudogene"},`  
                                  `"ENSG00000290330": {"chromosome": "Y", "biotype": "lncRNA"}}},`  
 `"continuation_decision": "defer",`  
 `"blocked_reason": "The symbol FAM197Y5 matches 2 genes in Ensembl release 116, and nothing in the task chooses one.",`  
 `"summary": "FAM197Y5 matches 2 genes in Ensembl release 116: <GENE:ENSG00000225516|FAM197Y5> (transcribed pseudogene) and <GENE:ENSG00000290330|FAM197Y5> (lncRNA). Give one Ensembl gene ID to continue."}`

**S1.3.3: Already-Marked Ambiguity (no call)**

`Context (from the harness):`  
`Registry entries, Ensembl release 116:`  
`{"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_327imr"},`  
 `"entries": [{"status": "ambiguous",`  
              `"gene_symbol": "PPP2R3B",`  
              `"candidate_gene_ids": ["ENSG00000167393", "ENSG00000292327"],`  
              `"candidate_details": {"ENSG00000167393": {"chromosome": "X", "biotype": "protein_coding"},`  
                                    `"ENSG00000292327": {"chromosome": "Y", "biotype": "protein_coding"}}}]}`

`User:`  
`Before the next analysis step, give the Ensembl gene ID for the seed PPP2R3B.`

`Thinking:`  
`The registry entries from the harness mark PPP2R3B as ambiguous in release 116, with one candidate on chromosome X and one on chromosome Y. The lookup is already done, and no multiplex is named, so I will not call a tool. I will ask for one ID.`

`Answer:`  
`{"status": "ambiguous",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_327imr"},`  
 `"result": {"gene_symbol": "PPP2R3B",`  
            `"candidate_gene_ids": ["ENSG00000167393", "ENSG00000292327"],`  
            `"candidate_details": {"ENSG00000167393": {"chromosome": "X", "biotype": "protein_coding"},`  
                                  `"ENSG00000292327": {"chromosome": "Y", "biotype": "protein_coding"}}},`  
 `"continuation_decision": "defer",`  
 `"blocked_reason": "The symbol PPP2R3B matches 2 genes in Ensembl release 116, and nothing in the task chooses one.",`  
 `"summary": "PPP2R3B matches 2 genes in Ensembl release 116: <GENE:ENSG00000167393|PPP2R3B> (chromosome X) and <GENE:ENSG00000292327|PPP2R3B> (chromosome Y). Give one Ensembl gene ID to continue."}`

**S1.3.4: Block for another symbol**  
`Context (from the harness):`  
`Registry entries, Ensembl release 116:`  
`{"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_zerly9"},`  
 `"entries": [{"status": "ambiguous",`  
              `"gene_symbol": "PPP2R3B",`  
              `"candidate_gene_ids": ["ENSG00000167393", "ENSG00000292327"],`  
              `"candidate_details": {"ENSG00000167393": {"chromosome": "X", "biotype": "protein_coding"},`  
                                    `"ENSG00000292327": {"chromosome": "Y", "biotype": "protein_coding"}}}]}`

`User:`  
`Convert this human gene symbol to its Ensembl gene ID: GOLGA2P9.`

`Thinking:`  
`The registry entries mark PPP2R3B, but they do not list GOLGA2P9. I will look`  
`up GOLGA2P9 in the registry.`

`Tool call:`  
`lookup_human_gene_symbol({"gene_symbol": "GOLGA2P9"})`

`Tool result:`  
`{"status": "ambiguous",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_7of8cf"},`  
 `"gene_symbol": "GOLGA2P9",`  
 `"candidate_gene_ids": ["ENSG00000269332", "ENSG00000290587"],`  
 `"candidate_details": {"ENSG00000269332": {"chromosome": "19", "biotype": "transcribed_unprocessed_pseudogene"},`  
                       `"ENSG00000290587": {"chromosome": "19", "biotype": "lncRNA"}}}`

`Thinking:`  
`The registry maps GOLGA2P9 to two IDs on chromosome 19: a transcribed`  
`pseudogene and a lncRNA. No multiplex is named, so nothing chooses between`  
`them. I will keep both and ask for one ID.`

`Answer:`  
`{"status": "ambiguous",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_7of8cf"},`  
 `"result": {"gene_symbol": "GOLGA2P9",`  
            `"candidate_gene_ids": ["ENSG00000269332", "ENSG00000290587"],`  
            `"candidate_details": {"ENSG00000269332": {"chromosome": "19", "biotype": "transcribed_unprocessed_pseudogene"},`  
                                  `"ENSG00000290587": {"chromosome": "19", "biotype": "lncRNA"}}},`  
 `"continuation_decision": "defer",`  
 `"blocked_reason": "The symbol GOLGA2P9 matches 2 genes in Ensembl release 116, and nothing in the task chooses one.",`  
 `"summary": "GOLGA2P9 matches 2 genes in Ensembl release 116: <GENE:ENSG00000269332|GOLGA2P9> (transcribed pseudogene) and <GENE:ENSG00000290587|GOLGA2P9> (lncRNA). Give one Ensembl gene ID to continue."}`

**S1.3.5: Multiplex Keeps One**

`Context (from the harness):`  
`Active multiplex: full_brain_multiplex_v1`

`User:`  
`Prepare the Ensembl gene ID for seed PPP2R3B, so the larger analysis can`  
`continue.`

`Thinking:`  
`The next step needs the Ensembl ID for the seed PPP2R3B. I will look it up`  
`in the registry.`

`Tool call:`  
`lookup_human_gene_symbol({"gene_symbol": "PPP2R3B"})`

`Tool result:`  
`{"status": "ambiguous",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_77ew41"},`  
 `"gene_symbol": "PPP2R3B",`  
 `"candidate_gene_ids": ["ENSG00000167393", "ENSG00000292327"],`  
 `"candidate_details": {"ENSG00000167393": {"chromosome": "X", "biotype": "protein_coding"},`  
                       `"ENSG00000292327": {"chromosome": "Y", "biotype": "protein_coding"}}}`

`Thinking:`  
`The registry maps PPP2R3B to two IDs, one on chromosome X and one on`  
`chromosome Y. The harness names the active multiplex, so I will check which`  
`candidates are nodes in it.`

`Tool call:`  
`check_multiplex_membership({"multiplex_id": "full_brain_multiplex_v1",`  
                            `"gene_ids": ["ENSG00000167393", "ENSG00000292327"]})`

`Tool result:`  
`{"status": "resolved",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_mem_qo6i6g"},`  
 `"multiplex_id": "full_brain_multiplex_v1",`  
 `"present_gene_ids": ["ENSG00000167393"],`  
 `"absent_gene_ids": ["ENSG00000292327"]}`

`Thinking:`  
`Only ENSG00000167393 is a node in this multiplex. The analysis runs on this`  
`multiplex, so it can continue with this ID.`

`Answer:`  
`{"status": "resolved",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_77ew41"},`  
 `"result": {"gene_id": "ENSG00000167393",`  
            `"gene_symbol": "PPP2R3B",`  
            `"candidate_gene_ids": ["ENSG00000167393", "ENSG00000292327"],`  
            `"multiplex_id": "full_brain_multiplex_v1",`  
            `"present_gene_ids": ["ENSG00000167393"]},`  
 `"continuation_decision": "continue",`  
 `"blocked_reason": null,`  
 `"summary": "PPP2R3B matches 2 genes in Ensembl release 116. Only <GENE:ENSG00000167393|PPP2R3B> (chromosome X) is a node in full_brain_multiplex_v1, so the analysis continues with this ID in this multiplex."}`

**S1.3.6: Multiplex Keeps Several**

`Context (from the harness):`  
`Active multiplex: full_brain_multiplex_v1`

`User:`  
`Find the human Ensembl gene ID for BMS1P4.`

`Thinking:`  
`The task gives only the symbol BMS1P4. I will look it up in the registry.`

`Tool call:`  
`lookup_human_gene_symbol({"gene_symbol": "BMS1P4"})`

`Tool result:`  
`{"status": "ambiguous",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_53xfkm"},`  
 `"gene_symbol": "BMS1P4",`  
 `"candidate_gene_ids": ["ENSG00000242338", "ENSG00000271816"],`  
 `"candidate_details": {"ENSG00000242338": {"chromosome": "10", "biotype": "transcribed_unprocessed_pseudogene"},`  
                       `"ENSG00000271816": {"chromosome": "10", "biotype": "lncRNA"}}}`

`Thinking:`  
`The registry maps BMS1P4 to two IDs on chromosome 10: a transcribed`  
`pseudogene and a lncRNA. The harness names the active multiplex, so I will`  
`check which candidates are nodes in it.`

`Tool call:`  
`check_multiplex_membership({"multiplex_id": "full_brain_multiplex_v1",`  
                            `"gene_ids": ["ENSG00000242338", "ENSG00000271816"]})`

`Tool result:`  
`{"status": "resolved",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_mem_ihtjof"},`  
 `"multiplex_id": "full_brain_multiplex_v1",`  
 `"present_gene_ids": ["ENSG00000242338", "ENSG00000271816"],`  
 `"absent_gene_ids": []}`

`Thinking:`  
`Both candidates are nodes in this multiplex, so membership does not choose`  
`one. I will keep both and ask for one ID.`

`Answer:`  
`{"status": "ambiguous",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_53xfkm"},`  
 `"result": {"gene_symbol": "BMS1P4",`  
            `"candidate_gene_ids": ["ENSG00000242338", "ENSG00000271816"],`  
            `"candidate_details": {"ENSG00000242338": {"chromosome": "10", "biotype": "transcribed_unprocessed_pseudogene"},`  
                                  `"ENSG00000271816": {"chromosome": "10", "biotype": "lncRNA"}},`  
            `"multiplex_id": "full_brain_multiplex_v1",`  
            `"present_gene_ids": ["ENSG00000242338", "ENSG00000271816"]},`  
 `"continuation_decision": "defer",`  
 `"blocked_reason": "The symbol BMS1P4 matches 2 genes, and 2 of them are nodes in full_brain_multiplex_v1, so the multiplex does not choose one.",`  
 `"summary": "BMS1P4 matches 2 genes in Ensembl release 116, and both are nodes in full_brain_multiplex_v1: <GENE:ENSG00000242338|BMS1P4> (transcribed pseudogene) and <GENE:ENSG00000271816|BMS1P4> (lncRNA). Give one Ensembl gene ID to continue."}`

**S1.3.7: Multiplex Keeps None**

`Context (from the harness):`  
`Active multiplex: full_brain_multiplex_v1`

`User:`  
`Before the next analysis step, give the Ensembl gene ID for seed gene`  
`FAM197Y5.`

`Thinking:`  
`The next step needs the Ensembl ID for the seed FAM197Y5. I will look it up`  
`in the registry.`

`Tool call:`  
`lookup_human_gene_symbol({"gene_symbol": "FAM197Y5"})`

`Tool result:`  
`{"status": "ambiguous",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_6bm40m"},`  
 `"gene_symbol": "FAM197Y5",`  
 `"candidate_gene_ids": ["ENSG00000225516", "ENSG00000290330"],`  
 `"candidate_details": {"ENSG00000225516": {"chromosome": "Y", "biotype": "transcribed_unprocessed_pseudogene"},`  
                       `"ENSG00000290330": {"chromosome": "Y", "biotype": "lncRNA"}}}`

`Thinking:`  
`The registry maps FAM197Y5 to two IDs on chromosome Y. The harness names the`  
`active multiplex, so I will check which candidates are nodes in it.`

`Tool call:`  
`check_multiplex_membership({"multiplex_id": "full_brain_multiplex_v1",`  
                            `"gene_ids": ["ENSG00000225516", "ENSG00000290330"]})`

`Tool result:`  
`{"status": "resolved",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_mem_ecxsfm"},`  
 `"multiplex_id": "full_brain_multiplex_v1",`  
 `"present_gene_ids": [],`  
 `"absent_gene_ids": ["ENSG00000225516", "ENSG00000290330"]}`

`Thinking:`  
`Neither candidate is a node in this multiplex. A choice between them cannot`  
`help, because no lookup in this multiplex can use either one. I must defer.`

`Answer:`  
`{"status": "not_found",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_6bm40m"},`  
 `"result": {"gene_symbol": "FAM197Y5",`  
            `"candidate_gene_ids": ["ENSG00000225516", "ENSG00000290330"],`  
            `"multiplex_id": "full_brain_multiplex_v1",`  
            `"present_gene_ids": []},`  
 `"continuation_decision": "defer",`  
 `"blocked_reason": "No candidate for FAM197Y5 is a node in full_brain_multiplex_v1.",`  
 `"summary": "FAM197Y5 matches 2 genes in Ensembl release 116, but neither is a node in full_brain_multiplex_v1. Choose another gene or another multiplex to continue."}`

**S1.3.8: User Chooses at Next Turn**

`[Earlier turn: in context, masked, no thinking]`  
`User:`  
`Prepare the Ensembl gene ID for seed FAM197Y5, so the larger analysis can`  
`continue.`  
`Answer:`  
`{"status": "ambiguous",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_ok7ewm"},`  
 `"result": {"gene_symbol": "FAM197Y5",`  
            `"candidate_gene_ids": ["ENSG00000225516", "ENSG00000290330"],`  
            `"candidate_details": {"ENSG00000225516": {"chromosome": "Y", "biotype": "transcribed_unprocessed_pseudogene"},`  
                                  `"ENSG00000290330": {"chromosome": "Y", "biotype": "lncRNA"}}},`  
 `"continuation_decision": "defer",`  
 `"blocked_reason": "The symbol FAM197Y5 matches 2 genes in Ensembl release 116, and nothing in the task chooses one.",`  
 `"summary": "FAM197Y5 matches 2 genes in Ensembl release 116: <GENE:ENSG00000225516|FAM197Y5> (transcribed pseudogene) and <GENE:ENSG00000290330|FAM197Y5> (lncRNA). Give one Ensembl gene ID to continue."}`

`[This turn: trained]`  
`User:`  
`Use the lncRNA.`

`Thinking:`  
`My previous answer lists two candidates for FAM197Y5. Only ENSG00000290330`  
`is a lncRNA, so the reply chooses it. The analysis can continue with this`  
`ID.`

`Answer:`  
`{"status": "resolved",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_ok7ewm"},`  
 `"result": {"gene_id": "ENSG00000290330", "gene_symbol": "FAM197Y5", "candidate_gene_ids": ["ENSG00000225516", "ENSG00000290330"]},`  
 `"continuation_decision": "continue",`  
 `"blocked_reason": null,`  
 `"summary": "The user chose <GENE:ENSG00000290330|FAM197Y5>, the lncRNA. The analysis continues with this ID."}`

**S1.3.9: Reply does not choose at next turn**  
`[Earlier turn: in context, masked, no thinking]`  
`User:`  
`Prepare the Ensembl gene ID for seed FAM197Y5, so the larger analysis can`  
`continue.`  
`Answer:`  
`{"status": "ambiguous",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_hsx3jt"},`  
 `"result": {"gene_symbol": "FAM197Y5",`  
            `"candidate_gene_ids": ["ENSG00000225516", "ENSG00000290330"],`  
            `"candidate_details": {"ENSG00000225516": {"chromosome": "Y", "biotype": "transcribed_unprocessed_pseudogene"},`  
                                  `"ENSG00000290330": {"chromosome": "Y", "biotype": "lncRNA"}}},`  
 `"continuation_decision": "defer",`  
 `"blocked_reason": "The symbol FAM197Y5 matches 2 genes in Ensembl release 116, and nothing in the task chooses one.",`  
 `"summary": "FAM197Y5 matches 2 genes in Ensembl release 116: <GENE:ENSG00000225516|FAM197Y5> (transcribed pseudogene) and <GENE:ENSG00000290330|FAM197Y5> (lncRNA). Give one Ensembl gene ID to continue."}`

`[This turn: trained]`  
`User:`  
`Use the protein-coding one.`

`Thinking:`  
`My previous answer lists two candidates for FAM197Y5: a transcribed`  
`pseudogene and a lncRNA. Neither is protein-coding, so the reply does not choose one. I will ask again.`

`Answer:`  
`{"status": "ambiguous",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_hsx3jt"},`  
 `"result": {"gene_symbol": "FAM197Y5",`  
            `"candidate_gene_ids": ["ENSG00000225516", "ENSG00000290330"],`  
            `"candidate_details": {"ENSG00000225516": {"chromosome": "Y", "biotype": "transcribed_unprocessed_pseudogene"},`  
                                  `"ENSG00000290330": {"chromosome": "Y", "biotype": "lncRNA"}}},`  
 `"continuation_decision": "defer",`  
 `"blocked_reason": "The reply does not choose one of the 2 candidates for FAM197Y5.",`  
 `"summary": "Neither FAM197Y5 candidate is protein-coding: <GENE:ENSG00000225516|FAM197Y5> is a transcribed pseudogene, and <GENE:ENSG00000290330|FAM197Y5> is a lncRNA. Give one of these IDs to continue."}`

**S1.3.10: Membership check fails after a successful lookup**  
`User:`  
`Prepare the Ensembl gene ID for seed GOLGA2P9 in multiplex`  
`full_brain_multiplex_v1, so the larger analysis can continue.`

`Thinking:`  
`The next step needs the Ensembl ID for the seed GOLGA2P9. I will look it up in`  
`the registry.`

`Tool call:`  
`lookup_human_gene_symbol({"gene_symbol": "GOLGA2P9"})`

`Tool result:`  
`{"status": "ambiguous",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_fzob5t"},`  
 `"gene_symbol": "GOLGA2P9",`  
 `"candidate_gene_ids": ["ENSG00000269332", "ENSG00000290587"],`  
 `"candidate_details": {"ENSG00000269332": {"chromosome": "19", "biotype": "transcribed_unprocessed_pseudogene"},`  
                       `"ENSG00000290587": {"chromosome": "19", "biotype": "lncRNA"}}}`

`Thinking:`  
`The registry maps GOLGA2P9 to two IDs on chromosome 19: a transcribed`  
`pseudogene and a lncRNA. The task names a multiplex, so I will check which`  
`candidates are nodes in it.`

`Tool call:`  
`check_multiplex_membership({"multiplex_id": "full_brain_multiplex_v1", "gene_ids": ["ENSG00000269332", "ENSG00000290587"]})`

`Tool result:`  
`{"status": "error", "provenance": null, "error_code": "BACKEND_UNAVAILABLE", "message": "graph service unreachable"}`

`Thinking:`  
`The membership check failed, so I do not know which candidate is a node. A`  
`failed check does not show that a candidate is absent. I must defer.`

`Answer:`  
`{"status": "ambiguous",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_fzob5t"},`  
 `"result": {"gene_symbol": "GOLGA2P9",`  
            `"candidate_gene_ids": ["ENSG00000269332", "ENSG00000290587"],`  
            `"candidate_details": {"ENSG00000269332": {"chromosome": "19", "biotype": "transcribed_unprocessed_pseudogene"},`  
                                  `"ENSG00000290587": {"chromosome": "19", "biotype": "lncRNA"}},`  
            `"multiplex_id": "full_brain_multiplex_v1",`  
            `"failed_tool": "check_multiplex_membership",`  
            `"error_code": "BACKEND_UNAVAILABLE"},`  
 `"continuation_decision": "defer",`  
 `"blocked_reason": "The lookup found 2 candidates, but the call to check_multiplex_membership failed with BACKEND_UNAVAILABLE, so no candidate is checked in full_brain_multiplex_v1.",`  
 `"summary": "GOLGA2P9 matches 2 genes in Ensembl release 116: <GENE:ENSG00000269332|GOLGA2P9> (transcribed pseudogene) and <GENE:ENSG00000290587|GOLGA2P9> (lncRNA). The membership check in full_brain_multiplex_v1 failed, so neither candidate is ruled out. Retry the check, or give one Ensembl gene ID to continue."}`

# S1.4: Cross-Context Alignment

**Overall guidance**

* Identity only  
  * S1.4 just measures whether two mentions name a single gene, training semantic alignment  
  * Later set comparisons belong to s2.6 and s5.12  
  * Tool output parsing belongs to s4.17 and s9.4  
* Compare in ensembl ID space  
  * Because our networks use ensembl as the canonical identifier  
* ID and ID  
  * Two IDs name the same gene when they are equal after the .N suffix is removed and case is ignored  
  * Only well-formed IDs are considered here – malformed IDs are in s1.2  
* No mapping from memory  
  * A mapping from an alias must come from a tool result  
* Ambiguous symbols  
  * If the other ID is a candidate, same\_gene is null; if it is not a candidate, same\_gene is false  
* Calls  
  * Each side requires at most one lookup

**S1.4.1: Symbol and ID, match (step)**  
`Context (from the harness):`  
`Module group_0, gene_ids: ["ENSG00000112164", "ENSG00000204394"]`

`User:`  
`Before the next analysis step, check whether the typed seed GLP1R and`  
`ENSG00000112164 in module group_0 are the same gene.`

`Thinking:`  
`GLP1R is a symbol, and ENSG00000112164 is an Ensembl ID. I cannot compare`  
`them as strings, so I will look up GLP1R in the registry.`

`Tool call:`  
`lookup_human_gene_symbol({"gene_symbol": "GLP1R"})`

`Tool result:`  
`{"status": "resolved",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_tet3v1"},`  
 `"gene_id": "ENSG00000112164",`  
 `"gene_symbol": "GLP1R"}`

`Thinking:`  
`GLP1R maps to ENSG00000112164, the same ID as in the module. They are the`  
`same gene, so the next step can treat them as one.`

`Answer:`  
`{"status": "resolved",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_tet3v1"},`  
 `"result": {"first": {"item": "typed_seed_list", "input": "GLP1R", "gene_id": "ENSG00000112164"},`  
            `"second": {"item": "module", "input": "ENSG00000112164", "gene_id": "ENSG00000112164"},`  
            `"same_gene": true},`  
 `"continuation_decision": "continue",`  
 `"blocked_reason": null,`  
 `"summary": "The typed seed GLP1R and ENSG00000112164 in module group_0 are the same gene, <GENE:ENSG00000112164|GLP1R>."}`

**S1.4.2: Symbol and ID, no match (direct)**  
`Context (from the harness):`  
`Seed gene set:`  
`{"seed_gene_ids": ["ENSG00000198625", "ENSG00000141510"],`  
 `"seed_gene_symbols": ["MDM4", "TP53"]}`  
`Abstract (example text, not a real abstract):`  
`"MDM2 binds the transactivation domain of p53."`

`User:`  
`Is MDM2 in the abstract the same gene as the seed ENSG00000198625?`

`Thinking:`  
`The abstract gives the symbol MDM2, and the seed is an Ensembl ID. Two`  
`different symbols can still name one gene, so I will look up MDM2 in the`  
`registry.`

`Tool call:`  
`lookup_human_gene_symbol({"gene_symbol": "MDM2"})`

`Tool result:`  
`{"status": "resolved",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_1t1f2a"},`  
 `"gene_id": "ENSG00000135679",`  
 `"gene_symbol": "MDM2"}`

`Thinking:`  
`MDM2 maps to ENSG00000135679, not to ENSG00000198625. Different Ensembl IDs`  
`name different genes, so the answer is no.`

`Answer:`  
`{"status": "resolved",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_1t1f2a"},`  
 `"result": {"first": {"item": "abstract", "input": "MDM2", "gene_id": "ENSG00000135679"},`  
            `"second": {"item": "seed_gene_set", "input": "ENSG00000198625", "gene_id": "ENSG00000198625"},`  
            `"same_gene": false},`  
 `"continuation_decision": "stop",`  
 `"blocked_reason": null,`  
 `"summary": "MDM2 in the abstract is <GENE:ENSG00000135679|MDM2>. The seed <GENE:ENSG00000198625|MDM4> is a different gene."}`

**S1.4.3: ID and ID, notation only (direct, no call)**  
`User:`  
`Do ENSG00000204394.15 in the seed gene set and ensg00000204394 in module`  
`group_0 name the same gene?`

`Thinking:`  
`Both inputs are Ensembl gene IDs. They are equal after I remove the version`  
`suffix and ignore case, so they are the same key. No lookup is needed.`

`Answer:`  
`{"status": "resolved",`  
 `"provenance": null,`  
 `"result": {"first": {"item": "seed_gene_set", "input": "ENSG00000204394.15", "gene_id": "ENSG00000204394"},`  
            `"second": {"item": "module", "input": "ensg00000204394", "gene_id": "ENSG00000204394"},`  
            `"same_gene": true},`  
 `"continuation_decision": "stop",`  
 `"blocked_reason": null,`  
 `"summary": "ENSG00000204394.15 and ensg00000204394 are the same gene ID, <GENE:ENSG00000204394>. Only the version suffix and the case differ."}`

**S1.4.4: ID and ID, near miss (direct, no call)**  
`User:`  
`Is ENSG00000135679 in module group_0 the same gene as ENSG00000135678 in the`  
`seed gene set?`

`Thinking:`  
`Both inputs are well-formed Ensembl gene IDs. They differ in the last digit,`  
`so they are different keys. No lookup is needed.`

`Answer:`  
`{"status": "resolved",`  
 `"provenance": null,`  
 `"result": {"first": {"item": "module", "input": "ENSG00000135679", "gene_id": "ENSG00000135679"},`  
            `"second": {"item": "seed_gene_set", "input": "ENSG00000135678", "gene_id": "ENSG00000135678"},`  
            `"same_gene": false},`  
 `"continuation_decision": "stop",`  
 `"blocked_reason": null,`  
 `"summary": "<GENE:ENSG00000135679> and <GENE:ENSG00000135678> are different gene IDs, so they are not the same gene."}`

**S1.4.5: Alias or previous symbol (step)**  
`Context (from the harness):`  
`Module group_0, gene_ids: ["ENSG00000141736", "ENSG00000141510"]`  
`Abstract (example text, not a real abstract):`  
`"HER2 amplification drives proliferation in these cells."`

`User:`  
`Before the next analysis step, check whether HER2 in the abstract and`  
`ENSG00000141736 in module group_0 are the same gene.`

`Thinking:`  
`The abstract gives HER2, and the module gives an Ensembl ID. I will look up`  
`HER2 in the registry.`

`Tool call:`  
`lookup_human_gene_symbol({"gene_symbol": "HER2"})`

`Tool result:`  
`{"status": "not_found",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_9nilqk"},`  
 `"gene_symbol": "HER2",`  
 `"candidate_gene_ids": []}`

`Thinking:`  
`The registry has no match for HER2. I will not map it to a gene from memory,`  
`so I cannot say whether the two mentions name the same gene.`

`Answer:`  
`{"status": "not_found",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_9nilqk"},`  
 `"result": {"first": {"item": "abstract", "input": "HER2", "gene_id": null},`  
            `"second": {"item": "module", "input": "ENSG00000141736", "gene_id": "ENSG00000141736"},`  
            `"same_gene": null},`  
 `"continuation_decision": "defer",`  
 `"blocked_reason": "The registry has no gene with the symbol HER2 in Ensembl release 116.",`  
 `"summary": "HER2 has no match in Ensembl release 116, so this answer does not say whether it names the same gene as <GENE:ENSG00000141736> in module group_0. Give the gene symbol or Ensembl ID for HER2 to continue."}`

**S1.4.6: Ambiguous Symbol (direct)**  
`Context (from the harness):`  
`Module group_0, gene_ids: ["ENSG00000167393", "ENSG00000141510"]`

`User:`  
`Is the typed seed PPP2R3B the same gene as ENSG00000167393 in module`  
`group_0?`

`Thinking:`  
`PPP2R3B is a symbol, and the module gives an Ensembl ID. I will look up`  
`PPP2R3B in the registry.`

`Tool call:`  
`lookup_human_gene_symbol({"gene_symbol": "PPP2R3B"})`

`Tool result:`  
`{"status": "ambiguous",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_27y75a"},`  
 `"gene_symbol": "PPP2R3B",`  
 `"candidate_gene_ids": ["ENSG00000167393", "ENSG00000292327"],`  
 `"candidate_details": {"ENSG00000167393": {"chromosome": "X", "biotype": "protein_coding"},`  
                       `"ENSG00000292327": {"chromosome": "Y", "biotype": "protein_coding"}}}`

`Thinking:`  
`The registry maps PPP2R3B to two IDs, one on chromosome X and one on`  
`chromosome Y. ENSG00000167393 is one of them, but the symbol alone does not`  
`choose it. I cannot say whether they are the same gene.`

`Answer:`  
`{"status": "ambiguous",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_27y75a"},`  
 `"result": {"first": {"item": "typed_seed_list",`  
                      `"input": "PPP2R3B",`  
                      `"gene_id": null,`  
                      `"candidate_gene_ids": ["ENSG00000167393", "ENSG00000292327"],`  
                      `"candidate_details": {"ENSG00000167393": {"chromosome": "X", "biotype": "protein_coding"},`  
                                            `"ENSG00000292327": {"chromosome": "Y", "biotype": "protein_coding"}}},`  
            `"second": {"item": "module", "input": "ENSG00000167393", "gene_id": "ENSG00000167393"},`  
            `"same_gene": null},`  
 `"continuation_decision": "defer",`  
 `"blocked_reason": "The symbol PPP2R3B matches 2 genes in Ensembl release 116, and nothing in the task chooses one.",`  
 `"summary": "PPP2R3B matches 2 genes in Ensembl release 116: <GENE:ENSG00000167393|PPP2R3B> (chromosome X) and <GENE:ENSG00000292327|PPP2R3B> (chromosome Y). The module ID is one of them, so this answer does not say whether they are the same gene. Give one Ensembl gene ID for PPP2R3B to continue."}`

**S1.4.7: Registry block given (step, no call)**  
`Context (from the harness):`  
`Registry entries, Ensembl release 116:`  
`{"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_lph9k9"},`  
 `"entries": [{"gene_id": "ENSG00000141510", "gene_symbol": "TP53"}]}`  
`Module group_0, gene_ids: ["ENSG00000141510", "ENSG00000135679"]`

`User:`  
`Before the next analysis step, check whether the typed seed TP53 and`  
`ENSG00000141510 in module group_0 are the same gene.`

`Thinking:`  
`The registry entries from the harness list TP53 for Ensembl release 116, with`  
`the ID ENSG00000141510. The lookup is already done, so I will not call the`  
`tool. That ID is the same as the module ID.`

`Answer:`  
`{"status": "resolved",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_lph9k9"},`  
 `"result": {"first": {"item": "typed_seed_list", "input": "TP53", "gene_id": "ENSG00000141510"},`  
            `"second": {"item": "module", "input": "ENSG00000141510", "gene_id": "ENSG00000141510"},`  
            `"same_gene": true},`  
 `"continuation_decision": "continue",`  
 `"blocked_reason": null,`  
 `"summary": "The typed seed TP53 and ENSG00000141510 in module group_0 are the same gene, <GENE:ENSG00000141510|TP53>."}`

**S1.4.8: External ID from MyGene**  
`Context (from the harness):`  
`query_mygene result:`  
`{"query": "HER2", "requested_fields": ["symbol", "ensembl"],`  
 `"results": [{"symbol": "ERBB2", "ensembl": {"gene": "ENSG00000141736"}}],`  
 `"result_count": 1}`  
`Module group_0, gene_ids: ["ENSG00000141736", "ENSG00000141510"]`

`User:`  
`Before the next analysis step, check whether the gene that query_mygene gave`  
`for HER2 is the same gene as ENSG00000141736 in module group_0.`

`Thinking:`  
`The query_mygene result maps HER2 to ENSG00000141736. MyGene uses its own data release, so I will check this ID in the release 116 registry before I compare.`

`Tool call:`  
`lookup_human_ensembl_gene_id({"gene_id": "ENSG00000141736"})`

`Tool result:`  
`{"status": "resolved",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_4yyew3"},`  
 `"gene_id": "ENSG00000141736",`  
 `"gene_symbol": "ERBB2"}`

`Thinking:`  
`The ID is in release 116 with the symbol ERBB2, and it is the same ID as in the module. So the HER2 in the query_mygene result and the module gene are the same gene.`

`Answer:`  
`{"status": "resolved",`  
 `"provenance": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_4yyew3"},`  
 `"result": {"first": {"item": "query_mygene", "input": "HER2", "gene_id": "ENSG00000141736"},`  
            `"second": {"item": "module", "input": "ENSG00000141736", "gene_id": "ENSG00000141736"},`  
            `"same_gene": true},`  
 `"continuation_decision": "continue",`  
 `"blocked_reason": null,`  
 `"summary": "The query_mygene result maps HER2 to <GENE:ENSG00000141736|ERBB2>, which is in Ensembl release 116. It is the same gene as ENSG00000141736 in module group_0."}`

# S1.5: Context ID Resolution

**Overall guidance**

* This step is pending until context ontology data from Manesh and Erica is complete

# S1.6: Layer Tag Parsing

**Overall guidance**

* Read the tag only  
  * The tag contains all information about the layer in the multiplex  
  * Claims about edges, etc belong in s1.7  
* Thinking  
  * Read the tag in order: prefix, family, part order for the family, and each part

**S1.6.1: Direct Parse**  
`User:`  
`The next analysis step uses layer bulkPEN:bulk_pen:brain:amygdala:v1. Parse`  
`its tag first.`

`Thinking:`  
`The prefix bulkPEN marks a bulk predictive expression network layer, and the`  
`second part gives the family bulk_pen. In this grammar, the parts after the`  
`family are organ, region, and version: brain, amygdala, and v1. The tag gives`  
`no source, no network, and no cell type.`

`Answer:`  
`{"status": "resolved",`  
 `"provenance": null,`  
 `"result": {"layer_id": "bulkPEN:bulk_pen:brain:amygdala:v1",`  
            `"layer_family": "bulk_pen",`  
            `"source": null,`  
            `"network": null,`  
            `"organ": "brain",`  
            `"region": "amygdala",`  
            `"cell_type": null,`  
            `"version": "v1"},`  
 `"continuation_decision": "continue",`  
 `"blocked_reason": null,`  
 `"summary": "<LAYER:bulkPEN:bulk_pen:brain:amygdala:v1> is a bulk PEN layer for the brain, region amygdala, version v1. The tag gives no cell type."}`

**S1.6.2: One Field (direct)**  
`User:`  
`Which cell type does layer scPEN:single_cell_pen:brain:hippocampus:astrocyte:v1`  
`give?`

`Thinking:`  
`The prefix scPEN marks a single-cell PEN layer, and the second part,`  
`single_cell_pen, gives the family scpen. In this grammar, the parts after the`  
`family part are organ, region, cell type, and version: brain, hippocampus,`  
`astrocyte, and v1. The cell type is astrocyte. The tag gives no source.`

`Answer:`  
`{"status": "resolved",`  
 `"provenance": null,`  
 `"result": {"layer_id": "scPEN:single_cell_pen:brain:hippocampus:astrocyte:v1",`  
            `"layer_family": "scpen",`  
            `"source": null,`  
            `"network": null,`  
            `"organ": "brain",`  
            `"region": "hippocampus",`  
            `"cell_type": "astrocyte",`  
            `"version": "v1"},`  
 `"continuation_decision": "stop",`  
 `"blocked_reason": null,`  
 `"summary": "<LAYER:scPEN:single_cell_pen:brain:hippocampus:astrocyte:v1> gives the cell type astrocyte, in region hippocampus of the brain."}`

**S1.6.3: Direct parse, HumanNet Grammar (direct)**  
`User:`  
`What are the fields of layer HumanNetV3:string_ppi:global:all:v3?`

`Thinking:`  
`The prefix HumanNetV3 marks a HumanNet layer. In this grammar, the parts are`  
`source, network, organ, cell type, and version: HumanNetV3, string_ppi,`  
`global, all, and v3. The network is a PPI network, so the family is ppi. The`  
`tag gives no region.`

`Answer:`  
`{"status": "resolved",`  
 `"provenance": null,`  
 `"result": {"layer_id": "HumanNetV3:string_ppi:global:all:v3",`  
            `"layer_family": "ppi",`  
            `"source": "HumanNetV3",`  
            `"network": "string_ppi",`  
            `"organ": "global",`  
            `"region": null,`  
            `"cell_type": "all",`  
            `"version": "v3"},`  
 `"continuation_decision": "stop",`  
 `"blocked_reason": null,`  
 `"summary": "<LAYER:HumanNetV3:string_ppi:global:all:v3> is the HumanNetV3 string_ppi network, a ppi layer with organ global and cell type all, version v3."}`

**S1.6.4: List filter by field (direct)**  
`User:`  
`Which of these layers have the region amygdala?`  
`scPEN:single_cell_pen:brain:amygdaloid-complex:astrocyte:v1,`  
`bulkPEN:bulk_pen:brain:hippocampus:v1, HumanNetV3:string_ppi:global:all:v3`

`Thinking:`  
`I will parse each tag and compare its region field exactly. The scPEN tag`  
`gives the region amygdaloid-complex. Whether the amygdaloid complex is the`  
`amygdala is an ontology question, not a tag field, so it does not match. The`  
`bulkPEN tag gives the region hippocampus. The HumanNet tag gives no region. No`  
`layer matches.`

`Answer:`  
`{"status": "resolved",`  
 `"provenance": null,`  
 `"result": {"layers": [{"layer_id": "scPEN:single_cell_pen:brain:amygdaloid-complex:astrocyte:v1",`  
                        `"layer_family": "scpen",`  
                        `"source": null,`  
                        `"network": null,`  
                        `"organ": "brain",`  
                        `"region": "amygdaloid-complex",`  
                        `"cell_type": "astrocyte",`  
                        `"version": "v1"},`  
                       `{"layer_id": "bulkPEN:bulk_pen:brain:hippocampus:v1",`  
                        `"layer_family": "bulk_pen",`  
                        `"source": null,`  
                        `"network": null,`  
                        `"organ": "brain",`  
                        `"region": "hippocampus",`  
                        `"cell_type": null,`  
                        `"version": "v1"},`  
                       `{"layer_id": "HumanNetV3:string_ppi:global:all:v3",`  
                        `"layer_family": "ppi",`  
                        `"source": "HumanNetV3",`  
                        `"network": "string_ppi",`  
                        `"organ": "global",`  
                        `"region": null,`  
                        `"cell_type": "all",`  
                        `"version": "v3"}],`  
            `"filter": {"field": "region", "value": "amygdala"},`  
            `"matching_layer_ids": []},`  
 `"continuation_decision": "stop",`  
 `"blocked_reason": null,`  
 `"summary": "No layer has the region amygdala as written. <LAYER:scPEN:single_cell_pen:brain:amygdaloid-complex:astrocyte:v1> has the region amygdaloid-complex, <LAYER:bulkPEN:bulk_pen:brain:hippocampus:v1> has the region hippocampus, and <LAYER:HumanNetV3:string_ppi:global:all:v3> gives no region."}`

**S1.6.5: Wrong type (direct)**  
`User:`  
`Parse this layer tag:`  
`rwr_loe:full_brain_multiplex_v1:seed_ENSG00000112164:geometric_elbow_v1.`

`Thinking:`  
`This string starts with rwr_loe and names a multiplex, a seed gene, and an`  
`elbow rule. That is a module ID, not a layer tag, so it has no layer fields.`  
`I will ask for a layer tag.`

`Answer:`  
`{"status": "invalid_input",`  
 `"provenance": null,`  
 `"result": {"input": "rwr_loe:full_brain_multiplex_v1:seed_ENSG00000112164:geometric_elbow_v1", "expected": "layer_tag"},`  
 `"continuation_decision": "defer",`  
 `"blocked_reason": "The input rwr_loe:full_brain_multiplex_v1:seed_ENSG00000112164:geometric_elbow_v1 is a module ID, not a layer tag.",`  
 `"summary": "The input is a module ID, not a layer tag, so it has no layer fields. Give a layer tag to continue."}`

**S1.6.6: Malformed tag (direct)**  
`User:`  
`The next analysis step uses layer`  
`scPEN:single_cell_pen:brain:hippocampus:astrocyte. Parse its tag first.`

`Thinking:`  
`The prefix scPEN marks a single-cell PEN layer, and that grammar needs the`  
`family part, organ, region, cell type, and version. The tag ends after the`  
`cell type, with no version part. I will not guess the version, so I cannot`  
`parse the tag.`

`Answer:`  
`{"status": "invalid_input",`  
 `"provenance": null,`  
 `"result": {"input": "scPEN:single_cell_pen:brain:hippocampus:astrocyte", "expected": "layer_tag"},`  
 `"continuation_decision": "defer",`  
 `"blocked_reason": "The input scPEN:single_cell_pen:brain:hippocampus:astrocyte is not a well-formed layer tag.",`  
 `"summary": "The tag scPEN:single_cell_pen:brain:hippocampus:astrocyte has no version part, so it is not a valid layer tag. Give the full tag to continue."}`

**S1.6.7: Direct parse, TF layer (direct)**  
`User:`  
`Parse this layer tag: TFs:tf_target:brain:all:v1.`

`Thinking:`  
`The prefix TFs marks a transcription factor network layer, and the second part`  
`gives the family tf_target. In this grammar, the parts after the family are`  
`organ, region, and version: brain, all, and v1. The tag gives no source, no`  
`network, and no cell type.`

`Answer:`  
`{"status": "resolved",`  
 `"provenance": null,`  
 `"result": {"layer_id": "TFs:tf_target:brain:all:v1",`  
            `"layer_family": "tf_target",`  
            `"source": null,`  
            `"network": null,`  
            `"organ": "brain",`  
            `"region": "all",`  
            `"cell_type": null,`  
            `"version": "v1"},`  
 `"continuation_decision": "stop",`  
 `"blocked_reason": null,`  
 `"summary": "<LAYER:TFs:tf_target:brain:all:v1> is a transcription factor network layer for the brain, region all, version v1. The tag gives no cell type."}`

# S1.7: Layer Family and Supported Claims

**Overall guidance**

* Rules about layers, not genes  
  * S2.13 covers edges, s3.4 and s7.5 cover paths/projections  
* Family questions are in S1.7  
  * Needs to agree with s1.6 outputs  
* Thinking  
  * Name the part of the tag that sets the family, evidence type, and claim

**S1.7.1: Family of a layer (direct)**  
`User:`  
`What layer family is scPEN:single_cell_pen:brain:hippocampus:astrocyte:v1?`

`Thinking:`  
`The prefix scPEN and the family part single_cell_pen set the family to scpen,`  
`a single-cell predictive expression network layer. An edge in it is a`  
`predictive expression association in hippocampus astrocytes only.`

`Answer:`  
`{"status": "resolved",`  
 `"provenance": null,`  
 `"result": {"layer_id": "scPEN:single_cell_pen:brain:hippocampus:astrocyte:v1",`  
            `"layer_family": "scpen",`  
            `"evidence_type": "context_predictive_expression",`  
            `"supported_claims": ["context_predictive_expression"],`  
            `"unsupported_claims": ["physical_interaction_recorded", "regulatory_link_recorded", "regulation_direction",`  
                                   `"global_coexpression", "cofunctional_association", "holds_in_other_contexts", "causal_effect"]},`  
 `"continuation_decision": "stop",`  
 `"blocked_reason": null,`  
 `"summary": "<LAYER:scPEN:single_cell_pen:brain:hippocampus:astrocyte:v1> is in the scpen family. Its edges are predictive expression associations in hippocampus astrocytes."}`

**S1.7.2: List filter (direct)**  
`User:`  
`Which of these layers are PPI layers? HumanNetV3:string_ppi:global:all:v3,`  
`HumanNetV3:coexpression:global:all:v3, scPEN:single_cell_pen:brain:hippocampus:astrocyte:v1`

`Thinking:`  
`The first tag names a PPI source, so its family is ppi. The second tag names`  
`coexpression, and the third tag is a single-cell PEN layer. Only the first`  
`layer is a PPI layer.`

`Answer:`  
`{"status": "resolved",`  
 `"provenance": null,`  
 `"result": {"layer_family": "ppi",`  
            `"matching_layer_ids": ["HumanNetV3:string_ppi:global:all:v3"],`  
            `"other_layer_ids": ["HumanNetV3:coexpression:global:all:v3", "scPEN:single_cell_pen:brain:hippocampus:astrocyte:v1"]},`  
 `"continuation_decision": "stop",`  
 `"blocked_reason": null,`  
 `"summary": "Only <LAYER:HumanNetV3:string_ppi:global:all:v3> is a PPI layer. <LAYER:HumanNetV3:coexpression:global:all:v3> is a coexpression layer, and <LAYER:scPEN:single_cell_pen:brain:hippocampus:astrocyte:v1> is a single-cell PEN layer."}`

**S1.7.3: Claims of a layer (step)**  
`User:`  
`Before the next analysis step, list what an edge in`  
`HumanNetV3:coexpression:global:all:v3 can and cannot support.`

`Thinking:`  
`The tag names HumanNet coexpression, so the family is coexpression. An edge in it records coexpression across many datasets. It is global evidence, so it is not specific to this context, and it does not record binding or regulation.`

`Answer:`  
`{"status": "resolved",`  
 `"provenance": null,`  
 `"result": {"layer_id": "HumanNetV3:coexpression:global:all:v3",`  
            `"layer_family": "coexpression",`  
            `"evidence_type": "global_coexpression",`  
            `"supported_claims": ["global_coexpression"],`  
            `"unsupported_claims": ["context_predictive_expression", "physical_interaction_recorded", "regulatory_link_recorded",`  
                                   `"regulation_direction", "cofunctional_association", "holds_in_other_contexts", "causal_effect"]},`  
 `"continuation_decision": "continue",`  
 `"blocked_reason": null,`  
 `"summary": "An edge in <LAYER:HumanNetV3:coexpression:global:all:v3> supports global coexpression only. It does not show a physical interaction, a regulatory link, or an association specific to this context."}`

**S1.7.4: Check one claim (direct \+ hard negative)**  
`User:`  
`Can an edge in HumanNetV3:coexpression:global:all:v3 show that the two`  
`proteins bind?`

`Thinking:`  
`The family is coexpression, so an edge records coexpression across many`  
`datasets. Binding is a physical interaction, and only a ppi layer records`  
`that. The answer is no.`

`Answer:`  
`{"status": "resolved",`  
 `"provenance": null,`  
 `"result": {"layer_id": "HumanNetV3:coexpression:global:all:v3",`  
            `"layer_family": "coexpression",`  
            `"evidence_type": "global_coexpression",`  
            `"supported_claims": ["global_coexpression"],`  
            `"unsupported_claims": ["context_predictive_expression", "physical_interaction_recorded", "regulatory_link_recorded",`  
                                   `"regulation_direction", "cofunctional_association", "holds_in_other_contexts", "causal_effect"],`  
            `"claim": "physical_interaction_recorded",`  
            `"claim_supported": false},`  
 `"continuation_decision": "stop",`  
 `"blocked_reason": null,`  
 `"summary": "No. An edge in <LAYER:HumanNetV3:coexpression:global:all:v3> records global coexpression. It does not show that the two proteins bind."}`

**S1.7.5: Wrong type (step)**  
`User:`  
`Before the next analysis step, give the layer family of`  
`brain:hippocampus:CA1:astrocyte:adult:v1.`

`Thinking:`  
`This string is a context ID, not a layer tag. It has anatomy, cell type, and`  
`stage parts, and no layer source. A context has no layer family, so I will`  
`ask for a layer tag.`

`Answer:`  
`{"status": "invalid_input",`  
 `"provenance": null,`  
 `"result": {"input": "brain:hippocampus:CA1:astrocyte:adult:v1", "expected": "layer_tag"},`  
 `"continuation_decision": "defer",`  
 `"blocked_reason": "The input brain:hippocampus:CA1:astrocyte:adult:v1 is a context ID, not a layer tag.",`  
 `"summary": "The input is a context ID, not a layer tag, so it has no layer family. Give a layer tag to continue."}`

**S1.7.6: Claims of a PPI layer (step)**  
`User:`  
`Before the next analysis step, list what an edge in`  
`HumanNetV3:string_ppi:global:all:v3 can and cannot support.`

`Thinking:`  
`The prefix is HumanNetV3, so the network field sets the family. The network is`  
`string_ppi, a protein interaction network, so the family is ppi. An edge`  
`records that a source lists the two proteins as interacting. That is a record`  
`of an interaction, not an observation in this context, and it says nothing`  
`about expression or regulation.`

`Answer:`  
`{"status": "resolved",`  
 `"provenance": null,`  
 `"result": {"layer_id": "HumanNetV3:string_ppi:global:all:v3",`  
            `"layer_family": "ppi",`  
            `"evidence_type": "physical_interaction_recorded",`  
            `"supported_claims": ["physical_interaction_recorded"],`  
            `"unsupported_claims": ["context_predictive_expression", "regulatory_link_recorded", "regulation_direction",`  
                                   `"global_coexpression", "cofunctional_association", "holds_in_other_contexts", "causal_effect"]},`  
 `"continuation_decision": "continue",`  
 `"blocked_reason": null,`  
 `"summary": "An edge in <LAYER:HumanNetV3:string_ppi:global:all:v3> supports a recorded physical interaction only. It does not show expression in this context, a regulatory link, or a causal effect."}`

**S1.7.7: Claims of a bulk PEN layer (direct)**  
`User:`  
`What can an edge in bulkPEN:bulk_pen:brain:amygdala:v1 support?`

`Thinking:`  
`The prefix bulkPEN and the family part bulk_pen set the family to bulk_pen. An`  
`edge is a predictive expression association built from bulk amygdala`  
`expression, so it supports context predictive expression for that tissue. The`  
`tissue mixes cell types, so the edge says nothing about any one cell type in`  
`it, and it records no binding and no regulation.`

`Answer:`  
`{"status": "resolved",`  
 `"provenance": null,`  
 `"result": {"layer_id": "bulkPEN:bulk_pen:brain:amygdala:v1",`  
            `"layer_family": "bulk_pen",`  
            `"evidence_type": "context_predictive_expression",`  
            `"supported_claims": ["context_predictive_expression"],`  
            `"unsupported_claims": ["physical_interaction_recorded", "regulatory_link_recorded", "regulation_direction",`  
                                   `"global_coexpression", "cofunctional_association", "holds_in_other_contexts", "causal_effect"]},`  
 `"continuation_decision": "stop",`  
 `"blocked_reason": null,`  
 `"summary": "An edge in <LAYER:bulkPEN:bulk_pen:brain:amygdala:v1> supports predictive expression in bulk amygdala. The tissue mixes cell types, so the edge does not establish the association in any one of them."}`

**S1.7.8: Unrecognized network, family other (direct)**  
`User:`  
`What can an edge in HumanNetV3:cocitation:global:all:v3 support?`

`Thinking:`  
`The prefix is HumanNetV3, so the network field sets the family. The network is`  
`cocitation, which is not in the network list. The family is other, not a guess`  
`at the nearest named family. The layer supports no claim.`

`Answer:`  
`{"status": "resolved",`  
 `"provenance": null,`  
 `"result": {"layer_id": "HumanNetV3:cocitation:global:all:v3",`  
            `"layer_family": "other",`  
            `"evidence_type": null,`  
            `"supported_claims": [],`  
            `"unsupported_claims": ["context_predictive_expression", "physical_interaction_recorded", "regulatory_link_recorded",`  
                                   `"regulation_direction", "global_coexpression", "cofunctional_association",`  
                                   `"holds_in_other_contexts", "causal_effect"]},`  
 `"continuation_decision": "stop",`  
 `"blocked_reason": null,`  
 `"summary": "The network cocitation is not in the network list, so <LAYER:HumanNetV3:cocitation:global:all:v3> is in the other family and supports no claim."}`

**S1.7.9: Claims of a TF layer (step)**  
`User:`  
`Before the next analysis step, list what an edge in TFs:tf_target:brain:all:v1`  
`can and cannot support.`

`Thinking:`  
`The prefix TFs and the family part tf_target set the family to tf_target. An`  
`edge records a regulatory link between the two genes. The store keeps these`  
`edges undirected, so an edge does not show which gene regulates the other. It`  
`records no binding and no causal effect.`

`Answer:`  
`{"status": "resolved",`  
 `"provenance": null,`  
 `"result": {"layer_id": "TFs:tf_target:brain:all:v1",`  
            `"layer_family": "tf_target",`  
            `"evidence_type": "regulatory_link_recorded",`  
            `"supported_claims": ["regulatory_link_recorded"],`  
            `"unsupported_claims": ["context_predictive_expression", "physical_interaction_recorded", "regulation_direction",`  
                                   `"global_coexpression", "cofunctional_association", "holds_in_other_contexts", "causal_effect"]},`  
 `"continuation_decision": "continue",`  
 `"blocked_reason": null,`  
 `"summary": "An edge in <LAYER:TFs:tf_target:brain:all:v1> supports a recorded regulatory link only. It does not show which gene regulates the other, a physical interaction, or a causal effect."}`

# S1.8: Direction Conventions

**Overall guidance**

* Case and expansion variants  
  * RWR ←→ rwr ←→ random walk with restart  
* No assumptions  
  * If the model does not recognize a metric it should ask the user to clarify the convention  
* Positive and negative examples  
  * Questions should be answerable with true/false  
  * Equal distribution of true and false answers

**S1.8.1: Direction lookup, name variant (direct)**  
`User:`  
`For the random walk with restart rank, is a lower value stronger?`

`Thinking:`  
`"Random walk with restart rank" is the RWR rank. The rank orders genes by`  
`proximity to the seed, and the top rank is the closest gene. So a lower rank`  
`means a closer gene, and the answer is yes.`

`Answer:`  
`{"status": "resolved",`  
 `"provenance": null,`  
 `"result": {"metric": "rank",`  
            `"input": "random walk with restart rank",`  
            `"lower_is_stronger": true,`  
            `"meaning": "proximity",`  
            `"rule": "lower_rank_closer",`  
            `"statement_true": null},`  
 `"continuation_decision": "stop",`  
 `"blocked_reason": null,`  
 `"summary": "Yes. A lower RWR rank means the gene is closer to the seed."}`

**S1.8.2: True or false, true (direct)**  
`User:`  
`True or false: a higher Spearman distance between two genes' RWR vectors`  
`means the genes are farther apart.`

`Thinking:`  
`Spearman distance is 1 − ρ, so a lower value means the two RWR vectors are`  
`more alike and the genes are closer. A higher value therefore means farther`  
`apart. The statement is true.`

`Answer:`  
`{"status": "resolved",`  
 `"provenance": null,`  
 `"result": {"metric": "spearman_distance",`  
            `"input": "Spearman distance",`  
            `"lower_is_stronger": true,`  
            `"meaning": "proximity",`  
            `"rule": "lower_distance_closer",`  
            `"statement_true": true},`  
 `"continuation_decision": "stop",`  
 `"blocked_reason": null,`  
 `"summary": "True. Spearman distance is 1 − ρ, so a higher value means the genes are farther apart."}`

**S1.8.3: True or false, false (direct)**  
`User:`  
`True or false: a lower dot similarity means the two genes are more similar.`

`Thinking:`  
`Dot similarity is a similarity, so a higher value means the two RWR vectors`  
`are more alike. A lower value means less similar. The statement is false.`

`Answer:`  
`{"status": "resolved",`  
 `"provenance": null,`  
 `"result": {"metric": "dot_similarity",`  
            `"input": "dot similarity",`  
            `"lower_is_stronger": false,`  
            `"meaning": "similarity",`  
            `"rule": "higher_similarity_more_similar",`  
            `"statement_true": false},`  
 `"continuation_decision": "stop",`  
 `"blocked_reason": null,`  
 `"summary": "False. A higher dot similarity means the two genes are more similar."}`

**S1.8.4: True or false, false (step)**  
`User:`  
`Before the next analysis step, check this convention: a lower RWR score means`  
`a gene is closer to the seed. True or false?`

`Thinking:`  
`The RWR score is normalized to the seed, whose score is 1.0. A higher score`  
`means the walk reaches the gene more often, so the gene is closer. A lower`  
`score means farther. The statement is false.`

`Answer:`  
`{"status": "resolved",`  
 `"provenance": null,`  
 `"result": {"metric": "score",`  
            `"input": "RWR score",`  
            `"lower_is_stronger": false,`  
            `"meaning": "proximity",`  
            `"rule": "higher_score_closer",`  
            `"statement_true": false},`  
 `"continuation_decision": "continue",`  
 `"blocked_reason": null,`  
 `"summary": "False. A higher RWR score means a gene is closer to the seed."}`

**S1.8.5: Ambiguous name (direct)**  
`User:`  
`True or false: a lower percentile means a gene is closer to the seed.`

`Thinking:`  
`"Percentile" alone can be the distance percentile or the score percentile. A`  
`lower distance percentile means closer, but a lower score percentile means`  
`farther. The two candidates disagree, so I cannot answer. I will ask which`  
`percentile is meant.`

`Answer:`  
`{"status": "ambiguous",`  
 `"provenance": null,`  
 `"result": {"metric": null,`  
            `"input": "percentile",`  
            `"lower_is_stronger": null,`  
            `"meaning": null,`  
            `"rule": null,`  
            `"statement_true": null,`  
            `"candidate_metrics": ["distance_percentile", "score_percentile"]},`  
 `"continuation_decision": "defer",`  
 `"blocked_reason": "The name \"percentile\" can mean distance_percentile or score_percentile, and the two have opposite directions.",`  
 `"summary": "\"Percentile\" can mean the distance percentile, where lower is closer, or the score percentile, where higher is closer. Say which percentile you mean to continue."}`
**S1.8.6: Unknown Metric (direct)**  
`User:`  
`True or false: a lower diffusion state distance means stronger node proximity.`

`Thinking:`  
`Diffusion state distance is not in the convention set, so I have no rule for`  
`its direction. I will not assume a direction. I will ask for the convention.`

`Answer:`  
`{"status": "not_found",`  
 `"provenance": null,`  
 `"result": {"metric": null,`  
            `"input": "diffusion state distance",`  
            `"lower_is_stronger": null,`  
            `"meaning": null,`  
            `"rule": null,`  
            `"statement_true": null},`  
 `"continuation_decision": "defer",`  
 `"blocked_reason": "The metric \"diffusion state distance\" is not in the direction conventions.",`  
 `"summary": "Diffusion state distance is not a metric in the convention set, so this answer does not say which direction is stronger. Give its convention to continue."}`

# S1.9: Module ID Parsing

**Overall guidance**

* Need to retrieve the updated MENTOR-EV dendrogram files from Ken, which contain updated module IDs

