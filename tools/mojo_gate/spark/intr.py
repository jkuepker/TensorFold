"""Check 4 driver: run each tensor-core kernel from libintr.so on torch tensors, compare numerically."""
import ctypes
from pathlib import Path
import torch

lib = ctypes.CDLL(str(Path(__file__).parent / "libintr.so"))
for nm in ("gate_mma", "gate_mma_ldm"):
    getattr(lib, nm).argtypes = [ctypes.c_long] * 3
dev, bf = "cuda", torch.bfloat16
torch.manual_seed(0)
torch.zeros(1, device=dev)


def run(fn, *ts):
    torch.cuda.synchronize()
    rc = fn(*[t.data_ptr() for t in ts]); torch.cuda.synchronize()
    return rc


A = torch.randn(16, 16, device=dev, dtype=bf); B = torch.randn(16, 8, device=dev, dtype=bf)
ref = A.float() @ B.float()
for label, fn in (("(a) mma.sync m16n8k16 bf16, fragments from global", lib.gate_mma),
                  ("(b,c,d) cp.async -> padded smem -> ldmatrix (x4 / x2.trans) -> mma", lib.gate_mma_ldm)):
    C = torch.zeros(16, 8, device=dev, dtype=torch.float32)
    rc = run(fn, A, B, C)
    err = (C - ref).abs().max().item()
    print(f"{label}: rc={rc} max_abs_err={err:.3e} {'PASS' if rc == 0 and err < 1e-4 else 'FAIL'}")
