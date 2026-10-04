"""Check 5: stream 1 GB with 16-byte loads (Mojo) vs torch reading the same bytes."""
import ctypes
from pathlib import Path
import torch

lib = ctypes.CDLL(str(Path(__file__).parent / "libbw.so"))
lib.gate_bw.argtypes = [ctypes.c_long] * 6
NB = 1 << 30
x = torch.randint(0, 1 << 20, (NB // 4,), device="cuda", dtype=torch.int32)
torch.cuda.synchronize()
st = torch.cuda.current_stream().cuda_stream

def timed(fn, reps):
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    fn(2); torch.cuda.synchronize()
    e0.record(); fn(reps); e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / 1e3 / reps   # s per pass

best = None
for grid in (1024, 2048, 4096, 8192, 16384):
    part = torch.zeros(grid * 256, device="cuda", dtype=torch.int32)
    t = timed(lambda r: lib.gate_bw(x.data_ptr(), part.data_ptr(), NB, st, grid, r), 20)
    gbs = NB / t / 1e9
    print(f"mojo grid={grid:6d} block=256: {gbs:7.1f} GB/s")
    if best is None or gbs > best[0]: best, bpart, bgrid = (gbs, t), part, grid
# correctness of the reduction (mod 2^32 sum of all words)
lib.gate_bw(x.data_ptr(), bpart.data_ptr(), NB, st, bgrid, 1); torch.cuda.synchronize()
mojo_sum = int(bpart.to(torch.int64).sum().item() & 0xFFFFFFFF) if False else int((bpart.to(torch.int64) & 0xFFFFFFFF).sum().item() & 0xFFFFFFFF)
ref_sum = int(x.to(torch.int64).sum().item() & 0xFFFFFFFF)
print("reduction matches torch (mod 2^32):", mojo_sum == ref_sum)
tt = timed(lambda r: [x.sum() for _ in range(r)], 20)
print(f"torch x.sum() int32 1 GiB: {NB / tt / 1e9:7.1f} GB/s")
y = x.view(torch.bfloat16); tt2 = timed(lambda r: [y.float().sum() if False else y.sum(dtype=torch.float32) for _ in range(r)], 20)
print(f"torch bf16 sum(dtype=f32) same bytes: {NB / tt2 / 1e9:7.1f} GB/s")
print(f"best mojo {best[0]:.1f} GB/s (grid {bgrid}); card peak ~640 GB/s")
