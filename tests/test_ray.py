import os, ray, torch

print("HIP_VISIBLE_DEVICES:", os.environ.get("HIP_VISIBLE_DEVICES"))
ngpu = len([x for x in os.environ.get("HIP_VISIBLE_DEVICES","").split(",") if x.strip()])
print("Detected GPUs:", ngpu)

# Tell Ray how many GPUs to register
ray.init(num_gpus=ngpu)

print("cluster_resources:", ray.cluster_resources())

@ray.remote(num_gpus=1)
def gpu_task(rank):
    return {
        "rank": rank,
        "visible": os.environ.get("HIP_VISIBLE_DEVICES"),
        "torch_is_available": torch.cuda.is_available(),
        "torch_device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

results = ray.get([gpu_task.remote(i) for i in range(ngpu)])
for r in results:
    print(r)

ray.shutdown()