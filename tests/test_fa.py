import flash_attn, torch, pathlib, importlib.metadata as m
print("flash-attn", m.version("flash_attn"))
print("GPU        :", torch.cuda.get_device_name(0))

# Make sure the compiled HIP extension is on disk
so = pathlib.Path(flash_attn.__file__).with_suffix(".so")
assert so.exists(), f"{so} missing"