"""Check 3 path A: max.experimental.torch.CustomOpLibrary + Mojo op on ROCm torch."""
import statistics, time
from pathlib import Path
import torch
from max.experimental.torch import CustomOpLibrary

lib = CustomOpLibrary(Path(__file__).parent / "ops")
op = lib.scale_add
n = 1 << 16
x = torch.randn(n, device="cuda", dtype=torch.bfloat16)
y = torch.randn(n, device="cuda", dtype=torch.bfloat16)
out = torch.empty_like(x)
op(out, x, y)
torch.cuda.synchronize()
ref = x * 2 + y
print("correct:", torch.equal(out, ref), "max abs err:", (out.float() - ref.float()).abs().max().item())

# stream behaviour: run on a non-default torch stream, check ordering against torch work
s = torch.cuda.Stream()
with torch.cuda.stream(s):
    a = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    for _ in range(50):
        a = a * 1.0001  # queued torch work on s
    o2 = torch.empty_like(a)
    op(o2, a, y)
    r2 = a * 2 + y
s.synchronize()
print("non-default stream correct:", torch.equal(o2, r2))

def bench(f, k=1000):
    ts = []
    for _ in range(k):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1e6
xs = torch.randn(256, device="cuda", dtype=torch.bfloat16)
ys = torch.randn(256, device="cuda", dtype=torch.bfloat16)
os_ = torch.empty_like(xs)
for _ in range(20): op(os_, xs, ys)
torch.cuda.synchronize()
mojo_us = bench(lambda: op(os_, xs, ys))
torch.cuda.synchronize()
torch_us = bench(lambda: torch.add(xs, ys, out=os_))
torch.cuda.synchronize()
print(f"launch overhead median us (async, 1000 calls): mojo op {mojo_us:.1f}  torch add {torch_us:.1f}")

# does the op call block the host / wait on torch's queued work (stream coupling)?
big = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize()
t = time.perf_counter()
for _ in range(20): big @ big
t_enq = time.perf_counter() - t
t = time.perf_counter(); op(os_, xs, ys); t_call = time.perf_counter() - t
t = time.perf_counter(); torch.cuda.synchronize(); t_sync = time.perf_counter() - t
print(f"queue 20 matmuls: enqueue {t_enq*1e3:.1f} ms; then op call {t_call*1e3:.1f} ms; remaining sync {t_sync*1e3:.1f} ms")
print("(op call ~ matmul-queue duration => op host-syncs/waits on torch stream; ~50us => async)")
