from ringX_attn.fused_attention import attention, HAS_FLASH
import torch

print(f"HAS_FLASH: {HAS_FLASH}")

q = torch.randn(1, 4, 1024, 128, dtype=torch.float16, device="xpu")
k = torch.randn(1, 4, 1024, 128, dtype=torch.float16, device="xpu")
v = torch.randn(1, 4, 1024, 128, dtype=torch.float16, device="xpu")

o, M = attention(q, k, v, False, 0.125)
print(f"o.shape: {o.shape}")  # expect (1, 4, 1024, 128)
print(f"M.shape: {M.shape}")  # expect (1, 4, 1024)
print("OK!")

from ringX_attn.ringX1_attn import ringX1_attn_func
print("import OK!")
