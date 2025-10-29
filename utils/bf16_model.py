# convert_oss_mxfp4_to_bf16.py
import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

SRC = "/lustre/orion/syb111/proj-shared/Personal/krusepi/projects/llms/models/gpt-oss-20b"
DST = "/lustre/orion/syb111/proj-shared/Personal/krusepi/projects/llms/models/gpt-oss-20b-bf16"

os.makedirs(DST, exist_ok=True)

# Tokenizer
tok = AutoTokenizer.from_pretrained(SRC, trust_remote_code=True)
tok.save_pretrained(DST)

# Load MXFP4, dequantize into BF16
quant_cfg = Mxfp4Config(dequantize=True)
model = AutoModelForCausalLM.from_pretrained(
    SRC,
    torch_dtype=torch.bfloat16,          # target dtype
    quantization_config=quant_cfg,
    low_cpu_mem_usage=True,
    offload_state_dict=True,
    offload_folder=os.path.join(DST, "offload_tmp"),
    attn_implementation="eager",
    trust_remote_code=True,
)

# ---- scrub quantization metadata from config (IMPORTANT) ----
cfg = model.config

# Helper: remove a key from all known internals
def _purge_key(obj, key: str):
    # attribute
    if hasattr(obj, key):
        try: delattr(obj, key)
        except Exception: pass
    # __dict__
    d = getattr(obj, "__dict__", None)
    if isinstance(d, dict):
        d.pop(key, None)
    # _internal_dict (HF stores config here)
    internal = getattr(obj, "_internal_dict", None)
    if isinstance(internal, dict):
        internal.pop(key, None)
    # Some versions stash init kwargs
    init_kwargs = getattr(obj, "init_kwargs", None)
    if isinstance(init_kwargs, dict):
        init_kwargs.pop(key, None)

# Nuke any quant-related keys outright
for k in ("quantization_config", "quantization_method", "quantization_bit_width"):
    _purge_key(cfg, k)

# Make dtype explicit in saved config (string is safest across versions)
cfg.torch_dtype = "bfloat16"
model.config = cfg
model.to(dtype=torch.bfloat16)

# Optional: tie weights (no-op if already tied)
try:
    model.tie_weights()
except Exception:
    pass

# Sanity: ensure the key is really gone
assert "quantization_config" not in getattr(model.config, "__dict__", {}), "quantization_config still present in __dict__"
if hasattr(model.config, "_internal_dict"):
    assert "quantization_config" not in model.config._internal_dict, "quantization_config still present in _internal_dict"

# Save clean BF16 model
model.save_pretrained(
    DST,
    safe_serialization=True,
    max_shard_size="10GB",
)
print("Saved BF16 model to:", DST)
