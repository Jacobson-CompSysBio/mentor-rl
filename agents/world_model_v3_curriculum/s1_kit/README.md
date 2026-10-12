# World-model contract validation

This kit checks rows against the shared S0/S1 world-model contract. The S0
corpus builder runs `validate.py` on every split before it publishes a corpus.

The directory keeps the name `s1_kit` because the S0 v4 corpus builder,
`runtime/world_model/s0_corpus.py`, calls `s1_kit/validate.py` by path. The v4
corpus manifest pins the SHA-256 value of that builder, so the path cannot
change without a new corpus version.

The sole machine-readable contract is
[`runtime/world_model/contract.yaml`](../../../runtime/world_model/contract.yaml).
The sole normative worked-example set is
[`runtime/world_model/golden_traces.jsonl`](../../../runtime/world_model/golden_traces.jsonl).

## Check the contract and the kit

Run from the repository root:

```bash
python -m runtime.world_model.contract check
python agents/world_model_v3_curriculum/s1_kit/validate.py \
  runtime/world_model/golden_traces.jsonl
agents/world_model_v3_curriculum/s1_kit/selftest.sh
agents/world_model_v3_curriculum/s1_kit/mutate.sh
```

- `contract check` validates `contract.yaml`, the golden rows, and the pinned
  HPC sources. It fails if a pinned source is not mounted.
- `validate.py` checks a JSONL corpus or a markdown question file.
- `selftest.sh` runs the validator on the fixtures. The fixtures cover 53
  known failures and must give no false positives on the clean input.
- `mutate.sh` disables each validation failure site in turn and checks that
  `selftest.sh` notices. A surviving mutant is a fixture gap.

To print the golden rows for review:

```bash
python -m runtime.world_model.contract render
```

## Build the source index

The corpus builders read one content-addressed source index. Build it once:

```bash
python -m runtime.world_model.source_index build
SOURCE_INDEX=data/world_model_v3/source_indexes/\
world_model_source_index_08d093d802562c0e0620e00b96ff94b4779a9d2118c48b4d741e9feabc7b0a9e.json.gz
python -m runtime.world_model.source_index check "$SOURCE_INDEX"
```

The index is generated and ignored by Git. It contains the release-116
registry fields, the leakage-safe HGNC historical-symbol and Ensembl
retired-ID pools, and the S1 facts (multiplex membership, native layer IDs,
and MENTOR-EV clade IDs). It does not copy graph edges or other derivable data.

The Ensembl stable-ID dump is also ignored generated input. Its official URL,
local path, and SHA-256 value are pinned together in `contract.yaml`. Fetch
that file before the first index build if it is not present.

## Build the S0 corpus

See [`../S0-human-gene-identifiers.md`](../S0-human-gene-identifiers.md).
