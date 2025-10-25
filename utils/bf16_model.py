# convert_oss_mxfp4_to_bf16.py
import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

src = "/lustre/orion/syb111/proj-shared/Personal/krusepi/projects/llms/models/gpt-oss-20b"          # your MXFP4 model dir (the one with /original under it)
dst = "/lustre/orion/syb111/proj-shared/Personal/krusepi/projects/llms/models/gpt-oss-20b-bf16"     # new BF16 directory to create

os.makedirs(dst, exist_ok=True)
tok = AutoTokenizer.from_pretrained(src)

quant_cfg = Mxfp4Config(dequantize=True)
model = AutoModelForCausalLM.from_pretrained(
    src,
    dtype=torch.bfloat16,          # dequantize to BF16
    quantization_config=quant_cfg,
    low_cpu_mem_usage=True,        # stream shards to lower peak CPU RAM
    offload_state_dict=True,       # spill to disk while loading to avoid DRAM spikes
    offload_folder=os.path.join(dst, "offload_tmp"),
    attn_implementation="eager",
)
tok.save_pretrained(dst)
model.save_pretrained(dst, safe_serialization=True, max_shard_size="5GB")
