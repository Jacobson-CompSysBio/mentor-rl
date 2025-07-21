import flash_attn, torch, pathlib, importlib.metadata as m
import pathlib, flash_attn

# look for any ROCm-style shared library
so = next(pathlib.Path(flash_attn.__file__).parent.glob("_flash_attn*.so"), None)
assert so is not None and so.exists(), "compiled ROCm extension missing"
print("Found:", so)