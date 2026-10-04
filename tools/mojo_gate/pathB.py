"""Check 3 path B: mojo build --emit shared-lib, C-ABI entry, ctypes + tensor.data_ptr()."""
import ctypes, statistics, time
from pathlib import Path
import torch

lib = ctypes.CDLL(str(Path(__file__).parent / "libgate.so"))
f = lib.gate_scale_add
f.argtypes = [ctypes.c_long] * 6   # handle, o, x, y, n, stream
f.restype = ctypes.c_long
lib.gate_init.restype = ctypes.c_long
H = lib.gate_init()
assert H
n = 1 << 16
x = torch.randn(n, device="cuda", dtype=torch.bfloat16)
y = torch.randn(n, device="cuda", dtype=torch.bfloat16)
o = torch.empty_like(x)
torch.cuda.synchronize()
rc = f(H, o.data_ptr(), x.data_ptr(), y.data_ptr(), n, 0)
ref = x * 2 + y
print("own-stream rc", rc, "correct:", torch.equal(o, ref))

# torch's current HIP stream
o.zero_()
stream = torch.cuda.current_stream().cuda_stream
rc = f(H, o.data_ptr(), x.data_ptr(), y.data_ptr(), n, stream)
torch.cuda.synchronize()
print("torch-current-stream rc", rc, "correct:", torch.equal(o, ref))

# ordering on a non-default stream behind queued torch work, no host sync
s = torch.cuda.Stream()
with torch.cuda.stream(s):
    a = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    for _ in range(50): a = a * 1.0001
    o2 = torch.empty_like(a)
    f(H, o2.data_ptr(), a.data_ptr(), y.data_ptr(), n, s.cuda_stream)
    r2 = a * 2 + y
s.synchronize()
print("non-default stream (no host sync) correct:", torch.equal(o2, r2))

def bench(fn, k=1000):
    ts = []
    for _ in range(k):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1e6
xs = torch.randn(256, device="cuda", dtype=torch.bfloat16)
ys = torch.randn(256, device="cuda", dtype=torch.bfloat16)
os_ = torch.empty_like(xs)
st = torch.cuda.current_stream().cuda_stream
args = (os_.data_ptr(), xs.data_ptr(), ys.data_ptr(), 256, st)
for _ in range(20): f(H, *args)
torch.cuda.synchronize()
us = bench(lambda: f(H, *args)); torch.cuda.synchronize()
tus = bench(lambda: torch.add(xs, ys, out=os_)); torch.cuda.synchronize()
print(f"launch overhead median us (1000 calls, torch stream): mojo shared-lib {us:.1f}  torch add {tus:.1f}")
args0 = args[:-1] + (0,)
us0 = bench(lambda: f(H, *args0), 200)
print(f"  own-stream+sync variant median us (200 calls): {us0:.1f}")
