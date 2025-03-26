import glob
import json
from safetensors.torch import load_file

model_path = '/lustre/orion/syb111/proj-shared/Personal/krusepi/llms/models/Llama-3.3-70B-Instruct/'

shard_files = sorted(glob.glob(model_path + "model-*-of-*.safetensors"))
for shard in shard_files:
    try:
        print(f"Checking {shard}...")
        _ = load_file(shard)
        print(f"{shard} is valid.")
    except Exception as e:
        print(f"Error in {shard}: {e}")


with open(model_path + "model.safetensors.index.json", "r") as f:
    index = json.load(f)

print(index.keys())
print("Weight map sample:", dict(list(index['weight_map'].items())[:5]))
