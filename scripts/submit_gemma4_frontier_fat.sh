#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  NODES=4 TASK_COUNT=288 TASK_CONCURRENCY=12 scripts/submit_gemma4_frontier_fat.sh

Common overrides:
  TASKS_PATH=/path/to/tasks.jsonl
  MODEL_PATH=/path/to/gemma-4-26B-A4B-it
  SIF=/path/to/vllm_openai_rocm_nightly.sif
  RUN_ROOT=/lustre/orion/syb114/scratch/$USER/mentor_rl_gemma4_runs/my_run
  QOS=debug WALLTIME=00:45:00 NODES=3 scripts/submit_gemma4_frontier_fat.sh
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd -P)}"
DEFAULT_DATA_ROOT="/lustre/orion/syb111/proj-shared/Personal/krusepi/projects/llms/mentor-rl"
DATA_ROOT="${DATA_ROOT:-${DEFAULT_DATA_ROOT}}"
SCRATCH="${SCRATCH:-/lustre/orion/syb114/scratch/${USER}}"
RUN_ID="${RUN_ID:-gemma4_26b_a4b_fat_$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${SCRATCH}/mentor_rl_gemma4_runs/${RUN_ID}}"
TASKS_PATH="${TASKS_PATH:-${DATA_ROOT}/data/module_corpus_full_brain_mixed/tasks.train.jsonl}"
TASK_COUNT="${TASK_COUNT:-64}"
TASK_OFFSET="${TASK_OFFSET:-0}"
NODES="${NODES:-4}"
ACCOUNT="${ACCOUNT:-SYB114}"
PARTITION="${PARTITION:-batch}"
QOS="${QOS:-}"
WALLTIME="${WALLTIME:-02:00:00}"
JOB_NAME="${JOB_NAME:-gemma4-fat}"
MODEL_PATH="${MODEL_PATH:-${GEMMA4_MODEL_PATH:-${SCRATCH}/hf_downloads/google/gemma-4-26B-A4B-it}}"
SIF="${SIF:-${GEMMA4_SIF:-${SCRATCH}/containers/vllm_openai_rocm_nightly-aa2b56ffb0c1d8b29d6468282a49d9b4a9f0dd1c.sif}}"
TASK_CONCURRENCY="${TASK_CONCURRENCY:-12}"
MAX_STEPS="${MAX_STEPS:-6}"
N_ACT="${N_ACT:-6}"
N_VER="${N_VER:-3}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-${MAX_MODEL_LEN}}"
GENERATOR_MAX_COMPLETION_TOKENS="${GENERATOR_MAX_COMPLETION_TOKENS:-4096}"
GENERATOR_ACTOR_RATIONALE_MAX_COMPLETION_TOKENS="${GENERATOR_ACTOR_RATIONALE_MAX_COMPLETION_TOKENS:-2048}"
GENERATOR_TIMEOUT_SECONDS="${GENERATOR_TIMEOUT_SECONDS:-3600}"

for numeric_name in TASK_COUNT TASK_OFFSET NODES TASK_CONCURRENCY MAX_STEPS N_ACT N_VER MAX_MODEL_LEN MAX_NUM_BATCHED_TOKENS; do
  numeric_value="${!numeric_name}"
  case "${numeric_value}" in (*[!0-9]*|'') echo "${numeric_name} must be an integer" >&2; exit 2;; esac
done
if (( TASK_COUNT < 1 )); then echo "TASK_COUNT must be >= 1" >&2; exit 2; fi
if (( NODES < 1 )); then echo "NODES must be >= 1" >&2; exit 2; fi
if (( TASK_COUNT < NODES )); then echo "TASK_COUNT must be >= NODES so every node receives work" >&2; exit 2; fi
if (( TASK_CONCURRENCY < 1 )); then echo "TASK_CONCURRENCY must be >= 1" >&2; exit 2; fi
if (( MAX_NUM_BATCHED_TOKENS < MAX_MODEL_LEN )); then
  echo "MAX_NUM_BATCHED_TOKENS must be >= MAX_MODEL_LEN for this vLLM build." >&2
  exit 2
fi

if [[ ! -f "${REPO_ROOT}/generate_trajectories.slurm" ]]; then
  echo "REPO_ROOT does not contain generate_trajectories.slurm: ${REPO_ROOT}" >&2
  exit 2
fi
if [[ ! -f "${TASKS_PATH}" ]]; then
  echo "TASKS_PATH not found: ${TASKS_PATH}" >&2
  exit 2
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Gemma model directory not found: ${MODEL_PATH}" >&2
  echo "Set MODEL_PATH=/path/to/gemma-4-26B-A4B-it if the model is staged elsewhere." >&2
  exit 2
fi
if [[ ! -f "${SIF}" ]]; then
  echo "vLLM Gemma container not found: ${SIF}" >&2
  echo "Set SIF=/path/to/the working vLLM ROCm nightly image." >&2
  exit 2
fi
if [[ "${DRY_RUN:-0}" != "1" ]] && ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is not on PATH. Run this on Frontier, or set DRY_RUN=1." >&2
  exit 2
fi

mkdir -p \
  "${RUN_ROOT}/inputs" \
  "${RUN_ROOT}/logs" \
  "${RUN_ROOT}/trajectories" \
  "${RUN_ROOT}/rwr_hpc_cache" \
  "${RUN_ROOT}/rwr_hpc_scratch" \
  "${RUN_ROOT}/hf_home"

TASKS_SELECTED_PATH="${RUN_ROOT}/inputs/tasks_selected.jsonl"
LANE_TASKS_DIR="${RUN_ROOT}/inputs/lane_tasks"
mkdir -p "${LANE_TASKS_DIR}"
python3 - "${TASKS_PATH}" "${TASKS_SELECTED_PATH}" "${TASK_COUNT}" "${TASK_OFFSET}" "${LANE_TASKS_DIR}" "${NODES}" <<'PY'
import sys
from pathlib import Path
src = Path(sys.argv[1])
out = Path(sys.argv[2])
count = int(sys.argv[3])
offset = int(sys.argv[4])
lane_dir = Path(sys.argv[5])
nodes = int(sys.argv[6])
rows = [line for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
selected = rows[offset: offset + count]
if len(selected) != count:
    raise SystemExit(f"requested {count} tasks at offset {offset}, but only found {len(selected)}")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(selected) + "\n", encoding="utf-8")
lane_dir.mkdir(parents=True, exist_ok=True)
lane_rows = [[] for _ in range(nodes)]
for index, row in enumerate(selected):
    lane_rows[index % nodes].append(row)
for lane_index, rows_for_lane in enumerate(lane_rows):
    if not rows_for_lane:
        raise SystemExit(f"lane {lane_index} would receive no tasks; reduce NODES or increase TASK_COUNT")
    (lane_dir / f"lane_{lane_index}.jsonl").write_text(
        "\n".join(rows_for_lane) + "\n",
        encoding="utf-8",
    )
PY

quote_env() {
  local name="$1"
  local value="${!name}"
  printf 'export %s=%q\n' "${name}" "${value}"
}

RUN_ENV_PATH="${RUN_ROOT}/run.env"
{
  for name in \
    RUN_ID RUN_ROOT REPO_ROOT DATA_ROOT SCRATCH TASKS_SELECTED_PATH LANE_TASKS_DIR TASK_COUNT NODES \
    MODEL_PATH SIF TASK_CONCURRENCY MAX_STEPS N_ACT N_VER MAX_NUM_BATCHED_TOKENS \
    MAX_MODEL_LEN GENERATOR_MAX_COMPLETION_TOKENS GENERATOR_ACTOR_RATIONALE_MAX_COMPLETION_TOKENS \
    GENERATOR_TIMEOUT_SECONDS; do
    quote_env "${name}"
  done
  printf 'export MODEL_PRESET=%q\n' "gemma4-26b-a4b-it"
  printf 'export GENERATOR_API_MODE=%q\n' "completions"
  printf 'export GENERATOR_REASONING_EFFORT=%q\n' ""
} > "${RUN_ENV_PATH}"

ENTRY_SCRIPT="${RUN_ROOT}/entry.sh"
cat > "${ENTRY_SCRIPT}" <<'ENTRY'
#!/usr/bin/env bash
set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT must be exported by the launch environment}"
if [[ -f "${RUN_ROOT}/run.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${RUN_ROOT}/run.env"
  set +a
fi

if [[ -z "${PYTHON:-}" ]]; then
  FRONTIER_PYTHON="${FRONTIER_PYTHON:-/lustre/orion/syb111/world-shared/environments/pytorch-rocm/bin/python}"
  if [[ -x "${FRONTIER_PYTHON}" ]]; then
    PYTHON="${FRONTIER_PYTHON}"
  else
    echo "FRONTIER_PYTHON is not executable for lane merge: ${FRONTIER_PYTHON}" >&2
    exit 2
  fi
  export PYTHON
fi

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/trajectories"

expand_slurm_nodelist() {
  local spec="$1"
  local group prefix inside part start end width value
  local groups=()
  local current="" char depth=0 i
  for ((i = 0; i < ${#spec}; i++)); do
    char="${spec:i:1}"
    case "${char}" in
      '[')
        depth=$((depth + 1))
        current+="${char}"
        ;;
      ']')
        depth=$((depth - 1))
        current+="${char}"
        ;;
      ',')
        if (( depth == 0 )); then
          groups+=("${current}")
          current=""
        else
          current+="${char}"
        fi
        ;;
      *)
        current+="${char}"
        ;;
    esac
  done
  if [[ -n "${current}" ]]; then
    groups+=("${current}")
  fi
  for group in "${groups[@]}"; do
    if [[ "${group}" != *"["* ]]; then
      printf '%s\n' "${group}"
      continue
    fi
    prefix="${group%%[*}"
    inside="${group#*[}"
    inside="${inside%%]*}"
    IFS=',' read -r -a parts <<< "${inside}"
    for part in "${parts[@]}"; do
      if [[ "${part}" == *"-"* ]]; then
        start="${part%-*}"
        end="${part#*-}"
        width="${#start}"
        for value in $(seq "$((10#${start}))" "$((10#${end}))"); do
          printf "%s%0${width}d\n" "${prefix}" "${value}"
        done
      else
        printf '%s%s\n' "${prefix}" "${part}"
      fi
    done
  done
}

mapfile -t ALLOC_NODES < <(expand_slurm_nodelist "${SLURM_JOB_NODELIST}")
LANE_COUNT="${#ALLOC_NODES[@]}"
printf '%s\n' "${ALLOC_NODES[@]}" > "${RUN_ROOT}/allocated_nodes.txt"

echo "FAT_GEMMA4_START_UTC=$(date -u +%FT%TZ)"
echo "RUN_ROOT=${RUN_ROOT}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "ALLOCATED_NODES=${ALLOC_NODES[*]}"

if (( LANE_COUNT != NODES )); then
  echo "Expected ${NODES} allocated nodes, got ${LANE_COUNT}: ${ALLOC_NODES[*]}" >&2
  exit 2
fi

printf 'lane_index\tnode\tout_dir\n' > "${RUN_ROOT}/lane_assignments.tsv"
for lane_index in "${!ALLOC_NODES[@]}"; do
  printf '%s\t%s\t%s\n' \
    "${lane_index}" \
    "${ALLOC_NODES[${lane_index}]}" \
    "${RUN_ROOT}/trajectories/node_${lane_index}" \
    >> "${RUN_ROOT}/lane_assignments.tsv"
done
cat "${RUN_ROOT}/lane_assignments.tsv"

run_lane() {
  local lane_index="$1"
  local node_name="$2"
  local lane_root="${RUN_ROOT}/trajectories/node_${lane_index}"
  local lane_tasks_path="${LANE_TASKS_DIR}/lane_${lane_index}.jsonl"
  mkdir -p "${lane_root}/logs"
  (
    set -euo pipefail
    export SLURM_JOB_NODELIST="${node_name}"
    export SLURM_NNODES=1
    export REPO_ROOT DATA_ROOT SCRATCH MODEL_PATH SIF MODEL_PRESET
    export TASKS_PATH="${lane_tasks_path}"
    export OUT_DIR="${lane_root}"
    export STORE_DIR="${DATA_ROOT}/data/runtime/full_brain_multiplex_store"
    export FULL_BRAIN_STORE_DIR="${DATA_ROOT}/data/runtime/full_brain_multiplex_store"
    export RWR_HPC_FLIST="${DATA_ROOT}/data/full_brain_flist.tsv"
    export RWR_HPC_BUILD_DIR="${DATA_ROOT}/external/rwr_hpc/build_frontier"
    export RWR_HPC_CACHE_DIR="${RUN_ROOT}/rwr_hpc_cache/node_${lane_index}"
    export RWR_HPC_SCRATCH_ROOT="${RUN_ROOT}/rwr_hpc_scratch/node_${lane_index}"
    export ENRICHMENT_BACKGROUND_PATH="${DATA_ROOT}/data/gw_dendrogram_corpus/modules.jsonl"
    export HF_HOME="${RUN_ROOT}/hf_home/node_${lane_index}"
    export LOCAL_NVME_ROOT="/mnt/bb/${USER}/${SLURM_JOB_ID}_gemma4_node_${lane_index}"
    export RUN_SMOKE_TEST=0
    export SMOKE_TEST_ONLY=0
    export PRECOMPUTE_SMOKE_CACHES=0
    export PREFETCH_MECHANISM_CACHE=0
    export MAX_TASKS=all
    export MAX_STEPS N_ACT N_VER TASK_CONCURRENCY
    export ACTOR_SAMPLING_STRATEGY=verbalized
    export GENERATOR_API_MODE GENERATOR_REASONING_EFFORT
    export GENERATOR_MAX_COMPLETION_TOKENS GENERATOR_ACTOR_RATIONALE_MAX_COMPLETION_TOKENS
    export GENERATOR_TIMEOUT_SECONDS
    export MAX_MODEL_LEN MAX_NUM_BATCHED_TOKENS
    export STAGE_MODEL_TO_NVME=1
    export VLLM_HEALTH_MAX_WAIT_SECONDS="${VLLM_HEALTH_MAX_WAIT_SECONDS:-1800}"
    export VLLM_HEALTH_POLL_SECONDS="${VLLM_HEALTH_POLL_SECONDS:-10}"
    export VLLM_HTTP_PORT=$(( 52000 + lane_index ))
    export RAY_PORT=$(( 53000 + lane_index ))
    export RAY_DASHBOARD_PORT=$(( 54000 + lane_index ))
    export VLLM_APPTAINER_FAKEROOT=0
    export VLLM_NO_USAGE_STATS=1
    export DO_NOT_TRACK=1
    export ALLOW_NETWORK_MYGENE=1
    export ALLOW_NETWORK_ENRICHMENT=1
    mkdir -p "${OUT_DIR}" "${RWR_HPC_CACHE_DIR}" "${RWR_HPC_SCRATCH_ROOT}" "${HF_HOME}"
    echo "LANE_START_UTC=$(date -u +%FT%TZ)"
    echo "LANE_INDEX=${lane_index} NODE=${node_name} OUT_DIR=${OUT_DIR}"
    echo "ENV_BEGIN"
    env | sort | grep -E '^(MODEL|SIF|MAX_|N_ACT|N_VER|TASK_|GENERATOR_|VLLM_|STAGE_|OUT_DIR|RUN_SMOKE|PREFETCH|RWR_HPC|REPO_ROOT|DATA_ROOT|SCRATCH|LOCAL_NVME_ROOT|SLURM_JOB_NODELIST|SLURM_NNODES)=' || true
    echo "ENV_END"
    /usr/bin/time -v bash "${REPO_ROOT}/generate_trajectories.slurm"
    echo "LANE_END_UTC=$(date -u +%FT%TZ)"
  ) > "${lane_root}/logs/lane.out" 2> "${lane_root}/logs/lane.err"
}

pids=()
: > "${RUN_ROOT}/lane_status.tsv"
echo -e "lane_index\tnode\tpid\tstatus\tfinished_utc" > "${RUN_ROOT}/lane_status.tsv"
for lane_index in "${!ALLOC_NODES[@]}"; do
  run_lane "${lane_index}" "${ALLOC_NODES[${lane_index}]}" &
  pids+=("$!")
done

fail_count=0
for lane_index in "${!pids[@]}"; do
  pid="${pids[${lane_index}]}"
  node_name="${ALLOC_NODES[${lane_index}]}"
  if wait "${pid}"; then status=0; else status=$?; fail_count=$((fail_count + 1)); fi
  echo -e "${lane_index}\t${node_name}\t${pid}\t${status}\t$(date -u +%FT%TZ)" >> "${RUN_ROOT}/lane_status.tsv"
done
cat "${RUN_ROOT}/lane_status.tsv"

if (( fail_count == 0 )); then
  "${PYTHON}" "${REPO_ROOT}/scripts/merge_trajectory_lanes.py" "${RUN_ROOT}" --expected-lanes "${LANE_COUNT}"
fi

find "${RUN_ROOT}" -maxdepth 3 -type f -printf "%TY-%Tm-%Td %TH:%TM:%TS %s %p\n" | sort > "${RUN_ROOT}/file_index.tsv"
echo "FAT_GEMMA4_END_UTC=$(date -u +%FT%TZ)"
exit "${fail_count}"
ENTRY
chmod +x "${ENTRY_SCRIPT}"

cat > "${RUN_ROOT}/submit_config.json" <<EOF_JSON
{
  "run_id": "${RUN_ID}",
  "run_root": "${RUN_ROOT}",
  "repo_root": "${REPO_ROOT}",
  "data_root": "${DATA_ROOT}",
  "tasks_path": "${TASKS_PATH}",
  "tasks_selected_path": "${TASKS_SELECTED_PATH}",
  "lane_tasks_dir": "${LANE_TASKS_DIR}",
  "lane_task_strategy": "round_robin_selected_tasks",
  "task_count": ${TASK_COUNT},
  "task_offset": ${TASK_OFFSET},
  "nodes": ${NODES},
  "task_concurrency_per_node": ${TASK_CONCURRENCY},
  "model_preset": "gemma4-26b-a4b-it",
  "model_path": "${MODEL_PATH}",
  "sif": "${SIF}",
  "max_model_len": "${MAX_MODEL_LEN}",
  "max_num_batched_tokens": "${MAX_NUM_BATCHED_TOKENS}"
}
EOF_JSON

cat > "${RUN_ROOT}/next_steps.txt" <<EOF_NEXT
Submitted Gemma 4 26B A4B Frontier fat-allocation rollout job.
Run root: ${RUN_ROOT}

Monitor:
  squeue -j <job_id>
  tail -f ${RUN_ROOT}/logs/slurm_<job_id>.out
  tail -f ${RUN_ROOT}/trajectories/node_*/logs/lane.out

Merged output after success:
  ${RUN_ROOT}/trajectories_merged

Audit:
  cd ${REPO_ROOT}
  python3 scripts/audit_trajectory_run.py \
    --run-dir ${RUN_ROOT}/trajectories_merged \
    --dpo-pair-gate \
    --required-task-types recovery,refinement \
    --required-evidence-modes graph,minimal
EOF_NEXT

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1; prepared fat-allocation run in ${RUN_ROOT}"
  echo "Entry script: ${ENTRY_SCRIPT}"
  exit 0
fi

sbatch_args=(
  -A "${ACCOUNT}"
  -p "${PARTITION}"
  -t "${WALLTIME}"
  -C nvme
  -N "${NODES}"
  --ntasks-per-node=1
  --gpus-per-node=8
  -J "${JOB_NAME}"
  -o "${RUN_ROOT}/logs/slurm_%j.out"
  -e "${RUN_ROOT}/logs/slurm_%j.err"
  --export=ALL
)
if [[ -n "${QOS}" ]]; then
  sbatch_args+=(-q "${QOS}")
fi

job_id=$(sbatch --parsable "${sbatch_args[@]}" "${ENTRY_SCRIPT}")
cat > "${RUN_ROOT}/job_id.txt" <<EOF_JOB
${job_id}
EOF_JOB
sed -i "1s/.*/Submitted Gemma 4 26B A4B Frontier fat-allocation rollout job: ${job_id}/" "${RUN_ROOT}/next_steps.txt"
cat "${RUN_ROOT}/next_steps.txt"
