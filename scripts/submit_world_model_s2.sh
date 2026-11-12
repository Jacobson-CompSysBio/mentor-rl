#!/bin/bash
# Validate S2 inputs or explicitly queue training and its CPU prerequisite jobs.
set -euo pipefail

die() { echo "[ERROR] $*" >&2; exit 2; }
usage() {
  echo "Usage: scripts/submit_world_model_s2.sh [--check|--prepare|--submit|--submit-pipeline] [--resume CHECKPOINT]"
}
mode=--check
resume_checkpoint=""
if [[ $# -gt 0 && "$1" =~ ^--(check|prepare|submit|submit-pipeline)$ ]]; then
  mode="$1"
  shift
fi
if [[ $# -gt 0 && "$1" == --resume ]]; then
  [[ $# -eq 2 ]] || { usage >&2; exit 2; }
  resume_checkpoint="$2"
  shift 2
fi
[[ $# -eq 0 ]] || { usage >&2; exit 2; }
[[ "${mode}" != --prepare || -z "${resume_checkpoint}" ]] || die "Prepare does not resume training."
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export RUN_CONFIG="config/world_model_s2_120b_training_v1_184n.json"
export S2_PYTHON="${S2_PYTHON:-/lustre/orion/syb111/world-shared/environments/pytorch-rocm/bin/python}"
export EXPECTED_RUN_CONFIG_SHA256="$(sha256sum "${RUN_CONFIG}" | awk '{print $1}')"
export S2_RESUME_CHECKPOINT="${resume_checkpoint}"
if [[ "${mode}" != --submit-pipeline ]]; then
  CONTRACT_ONLY=1 PREPARE_DATA_ONLY=0 bash world_model_s2.slurm
fi
if [[ "${mode}" == --check ]]; then
  echo "CPU contract checks passed. No job submitted."
  echo "Prepare: bash scripts/submit_world_model_s2.sh --prepare"
  echo "Submit:  bash scripts/submit_world_model_s2.sh --submit"
  echo "Queue all three Slurm jobs: bash scripts/submit_world_model_s2.sh --submit-pipeline"
  exit 0
fi
if [[ "${mode}" == --prepare ]]; then
  CONTRACT_ONLY=0 PREPARE_DATA_ONLY=1 bash world_model_s2.slurm
  exit 0
fi

if [[ "${mode}" == --submit ]]; then
  [[ -s data/world_model_v4/cache/s2_training_v1_184n/preparation/data_preparation.json ]] || \
    die "Run --prepare first, or use --submit-pipeline to schedule CPU preparation."
  "${S2_PYTHON}" - "data/world_model_v4/cache/s2_training_v1_184n/preparation/data_preparation.json" "${EXPECTED_RUN_CONFIG_SHA256}" <<'PYCACHE'
import json
from pathlib import Path
import sys
receipt=json.loads(Path(sys.argv[1]).read_text())
if (receipt.get('status')!='complete' or receipt.get('run_config_sha256')!=sys.argv[2]
        or not receipt.get('cache_files') or any(not Path(p).is_file() for p in receipt['cache_files'])):
    raise SystemExit('Prepared S2 cache is stale or incomplete; run --prepare again')
PYCACHE
fi
mkdir -p logs
# Preserve exactly the queued config/code bytes; the job also verifies their pins.
receipt_dir="$("${S2_PYTHON}" - "${REPO_ROOT}" "${RUN_CONFIG}" "${EXPECTED_RUN_CONFIG_SHA256}" "${resume_checkpoint}" "${mode}" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
root=Path(sys.argv[1])
config_path=Path(sys.argv[2])
config=json.loads((root/config_path).read_text())
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
directory=root/'data/world_model_v4/launch_reports/s2_v1_184n'/f'prepared-{stamp}-{os.getpid()}'
directory.mkdir(parents=True,exist_ok=False)
files=[config_path]+[Path(item['path']) for item in config['implementation_files']]
expected={item['path']:item['sha256'] for item in config['implementation_files']}
copies=[]
for relative in dict.fromkeys(files):
    if relative.is_absolute() or '..' in relative.parts:
        raise SystemExit('A snapshot input leaves the repository')
    source=root/relative
    target=directory/'sources'/relative
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,target)
    digest=hashlib.sha256(target.read_bytes()).hexdigest()
    if hashlib.sha256(source.read_bytes()).hexdigest()!=digest:
        raise SystemExit('An input changed while preparing submission')
    if str(relative) in expected and digest!=expected[str(relative)]:
        raise SystemExit('An implementation input differs from its config pin')
    copies.append({'path':str(relative),'sha256':digest,'snapshot':str(target.relative_to(directory))})
if copies[0]['sha256']!=sys.argv[3]:
    raise SystemExit('The config changed while preparing submission')
receipt={'schema_version':'mentor-rl-s2-submission-preparation-v1','created_utc':stamp,'run_config_sha256':sys.argv[3],'nodes':184,'ranks_per_node':8,'world_size':1472,'partition':'batch','walltime':'10:00:00','resume_checkpoint':sys.argv[4] or None,'files':copies,'submission_mode':sys.argv[5],'jobs':[],'submitted':False}
(directory/'preparation.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(directory)
PY
)"
export S2_SUBMISSION_RECEIPT="${receipt_dir}/preparation.json"
# A submitted allocation must train even when the caller used a CPU-only mode.
export CONTRACT_ONLY=0 PREPARE_DATA_ONLY=0

queue_job() {
  local response
  response="$(sbatch --parsable --export=ALL "$@")"
  [[ "${response}" =~ ^[0-9]+(\;[A-Za-z0-9_.-]+)?$ ]] || die "Unexpected sbatch response: ${response}"
  printf '%s\n' "${response%%;*}"
}

record_job() {
  "${S2_PYTHON}" - "${S2_SUBMISSION_RECEIPT}" "$1" "$2" "${3:-}" <<'PY'
import json
from pathlib import Path
import sys
path=Path(sys.argv[1])
data=json.loads(path.read_text())
data['jobs'].append({'role':sys.argv[2],'job_id':sys.argv[3],'dependency':sys.argv[4] or None})
data['submitted']=True
if sys.argv[2]=='train':
    data['job_id']=sys.argv[3]
path.write_text(json.dumps(data,indent=2)+'\n')
PY
}

training_dependency=()
training_dependency_value=""
if [[ "${mode}" == --submit-pipeline ]]; then
  # The check job validates large inputs; preparation runs only after it passes.
  check_job="$(queue_job world_model_s2_check.slurm)"
  record_job check "${check_job}"
  prepare_job="$(queue_job --dependency="afterok:${check_job}" world_model_s2_prepare.slurm)"
  record_job prepare "${prepare_job}" "afterok:${check_job}"
  training_dependency_value="afterok:${prepare_job}"
  training_dependency=(--dependency="${training_dependency_value}")
  echo "Queued S2 check ${check_job} and CPU preparation ${prepare_job}."
fi
job_id="$(queue_job "${training_dependency[@]}" --nodes=184 --ntasks=1472 \
  --ntasks-per-node=8 --gpus-per-node=8 --partition=batch --time=10:00:00 \
  world_model_s2.slurm)"
record_job train "${job_id}" "${training_dependency_value}"
printf '%s\n' "${job_id}" > "${receipt_dir}/job_id.txt"
echo "Queued S2 training job ${job_id}; receipt: ${receipt_dir}"
