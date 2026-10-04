"""Phase 1a: launch Mojo-compiled gfx1201 code objects from a torch C++ extension via
hipModuleLoadData + hipModuleLaunchKernel on torch's current HIP stream.
usage (in container, after mojo2hsaco.py kernels.mojo out): python bench.py out
"""
import json, os, statistics, struct, sys, time
import torch
from torch.utils.cpp_extension import load_inline

CPP = r"""
#include <torch/extension.h>
int64_t load_module(const std::string& path);
int64_t get_function(int64_t mod, const std::string& symbol);
void mojo_scale_add(int64_t fn, torch::Tensor o, torch::Tensor x, torch::Tensor y);
void mojo_wmma(int64_t fn, torch::Tensor a, torch::Tensor b, torch::Tensor c);
void mojo_generic(int64_t fn, int64_t gx, int64_t block, torch::Tensor kernarg_bytes);
void hip_scale_add(torch::Tensor o, torch::Tensor x, torch::Tensor y);
"""

HIP = r"""
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <fstream>
#include <sstream>

#define CK(x) do { hipError_t e_ = (x); TORCH_CHECK(e_ == hipSuccess, #x, ": ", hipGetErrorString(e_)); } while (0)

static std::vector<hipModule_t> g_mods;
static std::vector<hipFunction_t> g_fns;

int64_t load_module(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  TORCH_CHECK(f.good(), "cannot open ", path);
  std::stringstream ss; ss << f.rdbuf();
  std::string img = ss.str();
  hipModule_t m;
  CK(hipModuleLoadData(&m, img.data()));
  g_mods.push_back(m);
  return (int64_t)g_mods.size() - 1;
}

int64_t get_function(int64_t mod, const std::string& symbol) {
  hipFunction_t fn;
  CK(hipModuleGetFunction(&fn, g_mods.at(mod), symbol.c_str()));
  g_fns.push_back(fn);
  return (int64_t)g_fns.size() - 1;
}

// kernelParams form: only the explicit args; HIP fills the hidden block_count/group_size args itself.
void mojo_scale_add(int64_t fn, torch::Tensor o, torch::Tensor x, torch::Tensor y) {
  void* po = o.data_ptr(); void* px = x.data_ptr(); void* py = y.data_ptr();
  int32_t n = (int32_t)x.numel();
  void* args[] = {&po, &px, &py, &n};
  hipStream_t s = c10::hip::getCurrentHIPStream().stream();
  CK(hipModuleLaunchKernel(g_fns[fn], (n + 255) / 256, 1, 1, 256, 1, 1, 0, s, args, nullptr));
}

void mojo_wmma(int64_t fn, torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  void* pa = a.data_ptr(); void* pb = b.data_ptr(); void* pc = c.data_ptr();
  void* args[] = {&pa, &pb, &pc};
  hipStream_t s = c10::hip::getCurrentHIPStream().stream();
  CK(hipModuleLaunchKernel(g_fns[fn], 1, 1, 1, 32, 1, 1, 0, s, args, nullptr));
}

// kernarg_bytes: uint8 CPU tensor holding the exact kernarg buffer (HIP_LAUNCH_PARAM_BUFFER form)
void mojo_generic(int64_t fn, int64_t gx, int64_t block, torch::Tensor kb) {
  size_t sz = kb.numel();
  void* extra[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, kb.data_ptr(), HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
  hipStream_t s = c10::hip::getCurrentHIPStream().stream();
  CK(hipModuleLaunchKernel(g_fns[fn], gx, 1, 1, block, 1, 1, 0, s, nullptr, extra));
}

__global__ void hip_scale_add_k(__hip_bfloat16* o, const __hip_bfloat16* x, const __hip_bfloat16* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) o[i] = __float2bfloat16(2.f * __bfloat162float(x[i]) + __bfloat162float(y[i]));
}

void hip_scale_add(torch::Tensor o, torch::Tensor x, torch::Tensor y) {
  int n = (int)x.numel();
  hipStream_t s = c10::hip::getCurrentHIPStream().stream();
  hip_scale_add_k<<<(n + 255) / 256, 256, 0, s>>>((__hip_bfloat16*)o.data_ptr(), (const __hip_bfloat16*)x.data_ptr(), (const __hip_bfloat16*)y.data_ptr(), n);
  CK(hipGetLastError());
}
"""

d = sys.argv[1]
man = json.load(open(os.path.join(d, "manifest.json")))
K = {k["name"]: k for k in man["kernels"]}
os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx1201")
ext = load_inline("mojo_hipmodule", cpp_sources=CPP, cuda_sources=HIP,
                  functions=["load_module", "get_function", "mojo_scale_add", "mojo_wmma",
                             "mojo_generic", "hip_scale_add"], verbose=False)
p = torch.cuda.get_device_properties(0)
print("device", p.name, "pci_bus_id", p.pci_bus_id, p.gcnArchName, "torch", torch.__version__)
assert p.pci_bus_id == 7, "not card B"


def fn(name):
    k = K[name]
    m = ext.load_module(os.path.join(d, k["hsaco"]))
    return ext.get_function(m, k["symbol"])


f_sa, f_wm, f_ap = fn("scale_add"), fn("wmma_kernel"), fn("argprobe")
bf = torch.bfloat16
dev = "cuda"

for n in (1, 255, 256, 257, 4096, 1 << 20):
    x = torch.randn(n, device=dev, dtype=bf); y = torch.randn(n, device=dev, dtype=bf)
    o = torch.empty_like(x)
    ext.mojo_scale_add(f_sa, o, x, y); torch.cuda.synchronize()
    assert torch.equal(o, x * 2 + y), f"scale_add n={n} mismatch"
print("scale_add exact vs torch: PASS (n=1..1M, partial blocks; hidden args filled by HIP)")

a = torch.randn(16, 16, device=dev, dtype=bf); b = torch.randn(16, 16, device=dev, dtype=bf)
c = torch.empty(16, 16, device=dev, dtype=torch.float32)
ext.mojo_wmma(f_wm, a, b, c); torch.cuda.synchronize()
err = (c - a.float() @ b.float()).abs().max().item()
print(f"wmma 16x16x16 bf16 vs torch: max abs err {err:.3g}", "PASS" if err < 1e-4 else "FAIL")
assert err < 1e-4

dst = torch.zeros(4, device=dev, dtype=torch.int64)
buf = bytearray(K["argprobe"]["kernarg_size"])
struct.pack_into("<Q", buf, 0, dst.data_ptr()); struct.pack_into("<i", buf, 8, 11)
struct.pack_into("<q", buf, 16, 22); struct.pack_into("<i", buf, 24, 33); struct.pack_into("<q", buf, 32, 44)
ext.mojo_generic(f_ap, 1, 1, torch.frombuffer(buf, dtype=torch.uint8)); torch.cuda.synchronize()
print("argprobe (extra-buffer form, hand-built kernarg) ->", dst.tolist(), "PASS" if dst.tolist() == [11, 22, 33, 44] else "FAIL")
assert dst.tolist() == [11, 22, 33, 44]

# --- non-default stream: launch must queue behind work already on that stream
s = torch.cuda.Stream()
n = 1 << 16
x = torch.randn(n, device=dev, dtype=bf); y = torch.randn(n, device=dev, dtype=bf)
o = torch.zeros_like(x)
big = torch.randn(8192, 8192, device=dev, dtype=bf)
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(5): big @ big
    x2 = x * 1.0001
    ext.mojo_scale_add(f_sa, o, x2, y)
    ref = x2 * 2 + y
s.synchronize()
assert torch.equal(o, ref)
with torch.cuda.stream(s):
    ts_, te_ = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts_.record(); big @ big; ext.mojo_scale_add(f_sa, o, x, y); te_.record()
te_.synchronize()
print(f"non-default stream: PASS (consumed output of 5 matmuls + a scale on s; event span {ts_.elapsed_time(te_):.2f} ms covers matmul + launch)")


def bench(call, reps=1000):
    for _ in range(50): call()
    torch.cuda.synchronize()
    ts = []
    for i in range(reps):
        t0 = time.perf_counter(); call(); ts.append(time.perf_counter() - t0)
        if i % 100 == 99: torch.cuda.synchronize()
    ts.sort()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(reps): call()
    torch.cuda.synchronize(); tp = (time.perf_counter() - t0) / reps
    return statistics.median(ts) * 1e6, ts[int(.99 * reps)] * 1e6, tp * 1e6


xs = torch.randn(256, device=dev, dtype=bf); ys = torch.randn(256, device=dev, dtype=bf); os_ = torch.empty_like(xs)
cases = (("mojo hipModule", lambda: ext.mojo_scale_add(f_sa, os_, xs, ys)),
         ("hipcc <<<>>>", lambda: ext.hip_scale_add(os_, xs, ys)),
         ("torch.add", lambda: torch.add(xs, ys, out=os_)))
for rnd in range(2):
    for label, call in cases:
        m, p99, tp = bench(call)
        print(f"{label:16s} enqueue median {m:6.2f} us  p99 {p99:6.2f} us   back-to-back {tp:6.2f} us/call")

# --- hipGraph capture
x = torch.randn(256, device=dev, dtype=bf); y = torch.randn(256, device=dev, dtype=bf)
outs = [torch.zeros(256, device=dev, dtype=bf) for _ in range(10)]
sg = torch.cuda.Stream()
with torch.cuda.stream(sg):
    ext.mojo_scale_add(f_sa, outs[0], x, y)
sg.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    cur = x
    for i in range(10):
        ext.mojo_scale_add(f_sa, outs[i], cur, y)
        cur = outs[i]
for o_ in outs: o_.zero_()
g.replay(); torch.cuda.synchronize()
ref = x
for i in range(10):
    ref = ref * 2 + y
    assert torch.equal(outs[i], ref), f"graph launch {i} mismatch"
x.copy_(torch.randn(256, device=dev, dtype=bf)); g.replay(); torch.cuda.synchronize()
ref = x
for i in range(10):
    ref = ref * 2 + y
assert torch.equal(outs[9], ref)
print("hipGraph capture of 10 chained launches + replay (and replay with new input data): PASS")
