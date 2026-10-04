"""Check 4 driver: run each intrinsic kernel from libintr.so on torch tensors, compare numerically."""
import ctypes
from pathlib import Path
import torch

lib = ctypes.CDLL(str(Path(__file__).parent / "libintr.so"))
for nm in ("gate_wmma", "gate_dot2"):
    getattr(lib, nm).argtypes = [ctypes.c_long] * 3
for nm in ("gate_loadtr", "gate_lds", "gate_nt"):
    getattr(lib, nm).argtypes = [ctypes.c_long] * 2
dev = "cuda"
bf = torch.bfloat16
torch.manual_seed(0)

def run(fn, *ts):
    torch.cuda.synchronize()
    rc = fn(*[t.data_ptr() for t in ts]); torch.cuda.synchronize()
    return rc

# (a) WMMA
A = torch.randn(16, 16, device=dev, dtype=bf); B = torch.randn(16, 16, device=dev, dtype=bf)
C = torch.zeros(16, 16, device=dev, dtype=torch.float32)
rc = run(lib.gate_wmma, A, B, C)
ref = A.float() @ B.float()
err = (C - ref).abs().max().item()
print(f"(a) wmma bf16 16x16x16 rc={rc} max_abs_err={err:.3e} {'PASS' if rc == 0 and err < 1e-2 else 'FAIL'}")

# (b) global.load.tr.b128 : input 32 lanes x 8 u16 = values 0..255 (exact in bf16)
src = torch.arange(256, device=dev, dtype=torch.float32).to(bf)
dst = torch.zeros_like(src)
rc = run(lib.gate_loadtr, src, dst)
d = dst.float().cpu().view(32, 8).int()
print("(b) load.tr.b128 rc", rc, "lane0..1 received:", d[0].tolist(), d[1].tolist())
# hypothesis: within each group of 8 lanes the 8x8 block of 16-bit elems is transposed
s8 = torch.arange(256).view(4, 8, 8)           # [group, lane, elem]
exp = s8.transpose(1, 2).reshape(32, 8)         # transposed per group
ok_t = torch.equal(d, exp.int())
perm = sorted(d.flatten().tolist()) == list(range(256))
print(f"    permutation of input: {perm}; equals per-8-lane 8x8 transpose: {ok_t}; identity: {torch.equal(d, s8.reshape(32,8).int())}")
print("    ", "PASS" if rc == 0 and perm and not torch.equal(d, s8.reshape(32,8).int()) else "FAIL")

# (c) LDS padded tile: 16x32 tile transposed through LDS
src = torch.randn(16, 32, device=dev, dtype=bf); dst = torch.zeros(32, 16, device=dev, dtype=bf)
rc = run(lib.gate_lds, src, dst)
print(f"(c) LDS padded tile rc={rc} equal-to-transpose: {torch.equal(dst, src.t())} {'PASS' if rc == 0 and torch.equal(dst, src.t()) else 'FAIL'}")

# (d) nontemporal loads
src = torch.randn(256, device=dev, dtype=bf); dst = torch.zeros_like(src)
rc = run(lib.gate_nt, src, dst)
print(f"(d) nontemporal load rc={rc} copy-equal: {torch.equal(src, dst)} {'PASS' if rc == 0 and torch.equal(src, dst) else 'FAIL'}")

# (e) fdot2
a = torch.randn(64, device=dev, dtype=bf); b = torch.randn(64, device=dev, dtype=bf)
o = torch.zeros(32, device=dev, dtype=torch.float32)
rc = run(lib.gate_dot2, a, b, o)
ref = (a.float().view(32, 2) * b.float().view(32, 2)).sum(1) + 0.5
err = (o - ref).abs().max().item()
print(f"(e) fdot2 bf16 rc={rc} max_abs_err={err:.3e} {'PASS' if rc == 0 and err < 1e-3 else 'FAIL'}")
