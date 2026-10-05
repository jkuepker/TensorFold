"""Check 4e driver: fp8 e4m3 mma.sync m16n8k32 vs torch on one tile."""
import ctypes
from pathlib import Path
import torch

lib = ctypes.CDLL(str(Path(__file__).parent / "libfp8.so"))
lib.gate_mma_fp8.argtypes = [ctypes.c_long] * 3
dev = "cuda"
torch.manual_seed(0)
torch.zeros(1, device=dev)
A = torch.randn(16, 32, device=dev).to(torch.float8_e4m3fn); B = torch.randn(32, 8, device=dev).to(torch.float8_e4m3fn)
Ab, Bb = A.view(torch.uint8).contiguous(), B.view(torch.uint8).contiguous()
C = torch.zeros(16, 8, device=dev, dtype=torch.float32)
torch.cuda.synchronize()
rc = lib.gate_mma_fp8(Ab.data_ptr(), Bb.data_ptr(), C.data_ptr()); torch.cuda.synchronize()
err = (C - A.float() @ B.float()).abs().max().item()
print(f"(e1) fp8 e4m3 mma.sync m16n8k32 via mma(): rc={rc} max_abs_err={err:.3e} {'PASS' if rc == 0 and err < 1e-3 else 'FAIL'}")
