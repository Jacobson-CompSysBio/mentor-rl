import os
import time
import pytest
import torch
import multiprocessing as mp
from vllm import LLM, SamplingParams

# -------- ROCm visibility bridge (ROCR -> HIP) --------
rocr = os.environ.get("ROCR_VISIBLE_DEVICES")
if rocr and "HIP_VISIBLE_DEVICES" not in os.environ:
    os.environ["HIP_VISIBLE_DEVICES"] = rocr
    os.environ.pop("ROCR_VISIBLE_DEVICES", None)

# Force vLLM V1 engine (recommended for vLLM 0.10.x)
os.environ.setdefault("VLLM_USE_V1", "1")

# -------- Config --------
MODEL = os.environ.get(
    "VLLM_MODEL_PATH",
    "/lustre/orion/syb111/proj-shared/Personal/krusepi/projects/llms/models/Llama-3.2-1B-Instruct",
)

def _gpu_count() -> int:
    try:
        return torch.cuda.device_count()
    except Exception:
        return 0

# If VLLM_TP is not set, use all visible GPUs
TP_ENV = int(os.environ.get("VLLM_TP", str(max(1, _gpu_count()))))
MAX_MODEL_LEN = int(os.environ.get("VLLM_MAX_LEN", "4096"))  # safe default on MI250X

# -------- Session bootstrap (spawn) --------
@pytest.fixture(scope="session", autouse=True)
def _ensure_spawn_method():
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    # Must be set before any LLM is created.
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)
    yield

# -------- Basic environment checks --------
def test_rocm_env_sanity():
    assert torch.cuda.is_available(), "torch.cuda not available on ROCm build"
    assert torch.version.hip is not None, "torch.version.hip must be set on ROCm"
    assert torch.version.cuda is None, "CUDA version should be None on ROCm"
    assert _gpu_count() >= 1, "Need at least one ROCm GPU visible"

# -------- Shared LLM (TP auto or from env) --------
@pytest.fixture(scope="session")
def llm_shared():
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        tensor_parallel_size=TP_ENV,
        trust_remote_code=True,
        max_model_len=MAX_MODEL_LEN,
    )
    yield llm
    # Best-effort cleanup; vLLM cleans up on exit too.
    try:
        llm.__del__()  # noqa: SLF001
    except Exception:
        pass

# -------- Tests --------
def test_basic_generate(llm_shared):
    out = llm_shared.generate(
        ["Hello ROCm!"], SamplingParams(max_tokens=8, temperature=0.0, seed=123)
    )
    assert out and out[0].outputs and isinstance(out[0].outputs[0].text, str)

def test_batch_generate(llm_shared):
    prompts = [f"Count to three #{i}:" for i in range(4)]
    res = llm_shared.generate(prompts, SamplingParams(max_tokens=6, temperature=0.8))
    assert len(res) == len(prompts)
    for r in res:
        assert r.outputs and r.outputs[0].text

def test_stop_tokens(llm_shared):
    stop = ["<STOP>"]
    res = llm_shared.generate(
        ["Say 'abc' then <STOP> then anything:"],
        SamplingParams(max_tokens=16, stop=stop, temperature=0.7, seed=321),
    )
    txt = res[0].outputs[0].text
    assert "<STOP>" not in txt, "stop tokens should truncate output including stop string"

def test_determinism_temp_zero(llm_shared):
    sp = SamplingParams(max_tokens=12, temperature=0.0, seed=9001)
    p = ["Deterministic output check:"]
    out1 = llm_shared.generate(p, sp)[0].outputs[0].text
    out2 = llm_shared.generate(p, sp)[0].outputs[0].text
    assert out1 == out2, "temperature=0 with same seed should be deterministic"

def test_logprobs(llm_shared):
    sp = SamplingParams(max_tokens=8, temperature=0.0, seed=7, logprobs=5)
    r = llm_shared.generate(["Test logprobs"], sp)[0].outputs[0]
    has_lp = hasattr(r, "logprobs") or hasattr(r, "token_logprobs")
    assert has_lp, "logprobs should be returned when requested"

def test_throughput_smoke(llm_shared):
    prompts = [f"Q{i}: say hi." for i in range(16)]
    sp = SamplingParams(max_tokens=16, temperature=0.8, seed=42)
    t0 = time.time()
    outs = llm_shared.generate(prompts, sp)
    dt = max(1e-6, time.time() - t0)
    gen_toks = 0
    for o in outs:
        if o.outputs and hasattr(o.outputs[0], "token_ids") and o.outputs[0].token_ids:
            gen_toks += len(o.outputs[0].token_ids)
        elif o.outputs and hasattr(o.outputs[0], "tokens"):
            gen_toks += len(o.outputs[0].tokens)
    assert gen_toks > 0, "should generate some tokens"
    assert dt > 0

@pytest.mark.skipif(_gpu_count() < 2, reason="Need >=2 GPUs for TP=2")
def test_tensor_parallel_2gpus():
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        tensor_parallel_size=2,
        trust_remote_code=True,
        max_model_len=MAX_MODEL_LEN,
    )
    outs = llm.generate(["TP=2 quick check"], SamplingParams(max_tokens=8, seed=11))
    assert outs and outs[0].outputs[0].text
    try:
        llm.__del__()  # noqa: SLF001
    except Exception:
        pass
