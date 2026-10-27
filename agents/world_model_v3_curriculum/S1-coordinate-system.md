# S1 coordinate-system training: v10

S1 v10 extends the S0 r32 uniform adapter with human identifier lookup,
ambiguity and membership handling, cross-context identity checks, layer tags,
layer-family claims, direction conventions, and module-ID parsing. The shared
`runtime/world_model/contract.yaml` and held-out panels remain frozen.

## Passing run

Training job **5768596** completed the pinned 32-node, one-epoch recipe:
55,276 training rows, batch 32, 1,728 optimizer steps, learning rate 1e-4,
LoRA rank 32, context 4,096. It starts from S0 job **5631883**. The corpus uses
276 deterministically selected S0 replay rows.

Evaluation job **5768597** passed `structured_95_v1`: **6,542 / 6,875
(95.1564%)**, with all 6,875 rows scored offline. The gate requires at least
95% overall structured-answer accuracy and offline execution. Summary wording
and blocked-reason sentence wording are reported separately. Model mistakes
count as failed rows; broken panel/checkpoint identities abort evaluation.

The compact receipt is `diagnostics/s1_v10_results/evaluation_summary.json`.
S0 retention job 5768598 was pending when that receipt was recorded; the S1
pass does not establish an S0 retention result.

## Retained entry points

| Purpose | Config / implementation |
| --- | --- |
| Training | `config/world_model_s1_120b_training_v10_32n.json`, `world_model_v2_s0.slurm` |
| S1 evaluation | `config/world_model_s1_120b_32n_test_v10_structured95_512n.json`, `world_model_s1_eval.slurm` |
| S0 retention | `config/world_model_s1_120b_s0_forgetting_test_v10_512n.json`, `world_model_v2_s0.slurm` in test mode |
| Corpus | Eight `runtime/world_model/s1_*_corpus.py` builders and `scripts/build_world_model_s1_bundle.py` |
| Data recipe | `runtime/world_model/s1_v10.yaml` |

The v10 overlay still pins the reviewed v4/v5 prompt banks, v6 retry policies,
and v9 choice replies. Their names identify immutable inputs, not alternate
training runs. Keep those files and the two retry helpers: their exact content
is included in the passing corpus provenance. The recipe also records segment
loss telemetry and checkpoint snapshots at steps 400, 800, 1200, and 1600.

## Data and reproduction

The active artifacts are under `data/world_model_v4/corpora/s1_v10` and
`data/world_model_v4/eval/s1_v10`. Reproduction also needs the pinned Ensembl
human/mouse GTFs and HGNC table under `data/world_model_v2/sources`, the stable-ID
history and source index under `data/world_model_v3`, the S0 v4 replay corpus,
the pinned MyGene cache at `data/runtime/alias_registry/mygene_query_cache.json`,
and the base tokenizer. Training requires the S0 final adapter and base model;
evaluation requires the completed S1 adapter. The S0 retention panel is
`data/world_model_v3/eval/s0_full_registry`.

To rebuild the current corpus, pass the v10 overlay explicitly to each builder:

```bash
for stage in s1_1 s1_2 s1_3 s1_4 s1_6 s1_7 s1_8 s1_9; do
  python -m runtime.world_model.${stage}_corpus --overlay runtime/world_model/s1_v10.yaml
done
python scripts/build_world_model_s1_bundle.py --overlay runtime/world_model/s1_v10.yaml
```

Run the rebuild in a separate checkout when preserving published artifacts.
S1.8 and S1.9 no longer hash the retired catalog and stage document; they still
record the builder, main's golden traces, and actual runtime/data inputs. Their rows reproduce exactly,
but a new build has updated provenance manifests. Keep the published v10 files
for the passing checkpoint; update config manifest pins before training on a
newly rebuilt corpus.
Check the existing inputs without submitting a job:

```bash
CONTRACT_ONLY=1 JOB_MODE=train \
RUN_CONFIG=config/world_model_s1_120b_training_v10_32n.json \
S0_METHOD_ID=oss120b-plain-base-tokenizer-lora-r32 \
bash world_model_v2_s0.slurm

CONTRACT_ONLY=1 \
RUN_CONFIG=config/world_model_s1_120b_32n_test_v10_structured95_512n.json \
S1_METHOD_ID=oss120b-plain-base-tokenizer-lora-r32 \
CHECKPOINT_PATH=data/world_model_v4/checkpoints/s1_v10_120b_32n/oss120b-plain-base-tokenizer-lora-r32/slurm-5768596 \
bash world_model_s1_eval.slurm
```

Generated corpora, model weights, optimizer states, and full evaluation reports
remain outside Git. Superseded S1 configs, smoke/robustness/retry diagnostic
entry points, historical reports, and their tests are omitted from this change.
