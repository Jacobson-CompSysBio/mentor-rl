#!/usr/bin/env python3
"""
quick_rope_flashattn_rocm_post1.py
Test Flash-Attention 2.7.4.post1 RoPE kernels (forward + backward) on ROCm.
Run:  python quick_rope_flashattn_rocm_post1.py
"""

import sys, torch
from flash_attn import __version__ as flash_version
from flash_attn.layers.rotary import RotaryEmbedding, apply_rotary_emb  # new API

# ---------- sanity checks ---------------------------------------------------
if torch.version.hip is None:
    sys.exit("‼️  This PyTorch build is NOT ROCm-enabled.")
if not torch.cuda.is_available():
    sys.exit("‼️  No HIP device visible (check HIP_VISIBLE_DEVICES).")
if not flash_version.startswith("2.7.4"):
    print(f"⚠️  Flash-Attention version is {flash_version}, expected 2.7.4.post1")

# ---------- fixed hyper-params ---------------------------------------------
BATCH, SEQLEN, HEADS, DIM = 1, 2048, 16, 64   # DIM per head (even)
DTYPE  = torch.float16
device = torch.device("cuda")

# (batch, seqlen, nheads, headdim) is the canonical flash-attn layout
q = torch.randn(BATCH, SEQLEN, HEADS, DIM, device=device, requires_grad=True)
k = torch.randn_like(q)

# ---------- fused RoPE ------------------------------------------------------
rope = RotaryEmbedding(DIM, interleaved=False, device=device)

# cached tables – no recomputation needed
cos, sin = rope._cos_cached[:SEQLEN], rope._sin_cached[:SEQLEN]

q_rot = apply_rotary_emb(q, cos, sin, interleaved=False)
k_rot = apply_rotary_emb(k, cos, sin, interleaved=False)

# ---------- naïve reference -------------------------------------------------
def naive_rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    rot_even = x1 * cos - x2 * sin
    rot_odd  = x1 * sin + x2 * cos
    return torch.stack((rot_even, rot_odd), dim=-1).flatten(-2)

q_ref = naive_rope(q.detach(), cos, sin)
k_ref = naive_rope(k.detach(), cos, sin)
torch.testing.assert_close(q_rot, q_ref, atol=1e-4, rtol=1e-4)
torch.testing.assert_close(k_rot, k_ref, atol=1e-4, rtol=1e-4)
print("✅ forward pass matches naïve reference")

# ---------- backward check --------------------------------------------------
loss = q_rot.sum() + k_rot.sum()
loss.backward()
print("✅ backward pass succeeded – gradients propagate on ROCm")