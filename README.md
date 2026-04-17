# MENTOR-RL

MENTOR-RL is a research repo for building CORUM-grounded biology tasks, generating
tool-using trajectories over a deterministic runtime, and training models with
supervised fine-tuning and GRPO-style reinforcement learning.

The current codebase is oriented around Frontier/Slurm workflows and local data
paths rather than a packaged pip install.

## Repo Layout

- `scripts/`: primary entry points for corpus building, trajectory generation, inference, and training
- `runtime/`: shared runtime/state/validation/scoring types used by the generator
- `cpp_runtime/`: lower-level runtime components
- `data/`: canonical CORUM corpora, generated trajectories, caches, and related artifacts
- `tests/`: unit and integration tests
- `generate_trajectories.slurm`, `train_sft.slurm`, `train_grpo.slurm`: cluster launchers

## Main Workflows

### 1. Build the CORUM corpus

`scripts/build_corum_corpus.py` turns the raw CORUM export into canonical task
JSONL files used by the rest of the pipeline.

```bash
python scripts/build_corum_corpus.py --help
```

Typical outputs live under `data/corum_corpus/`.

### 2. Generate trajectories

`scripts/generate_trajectories.py` generates shared-prefix trajectories from the
canonical tasks. It supports deterministic heuristic generation for testing and
model-backed generation through an OpenAI-compatible vLLM endpoint.

```bash
python scripts/generate_trajectories.py --help
```

For Frontier runs, use the Slurm launcher:

```bash
sbatch generate_trajectories.slurm
```

Generated artifacts are written under `data/corum_trajectories/`.

### 3. Train an SFT model

`scripts/train_sft.py` trains a supervised model over a local JSON dataset using
TRL's `SFTTrainer`.

```bash
python scripts/train_sft.py --help
```

Cluster runs typically go through:

```bash
sbatch train_sft.slurm
```

### 4. Train with GRPO

`scripts/train_grpo.py` contains the GRPO training path and patches TRL's
`VLLMClient` to work with a standard vLLM server.

```bash
python scripts/train_grpo.py --help
```

Cluster runs typically go through:

```bash
sbatch train_grpo.slurm
```

## Testing

Run the test suite with:

```bash
pytest
```

For quick script-level validation, the main Python entry points also expose
`--help`.

## Notes

- Many defaults assume the current ORNL/Frontier filesystem layout and large local model caches.
- The trajectory pipeline writes run artifacts and progress files into `data/`.
- Slurm launchers are the most complete examples of the intended production workflow.
