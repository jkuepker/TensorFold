// Host side of the Mojo prompt GEMM (prefill_rocm.mojo, TF_ROCM_PREFILL_GEMM=mojo, TF_ROCM_PREFILL8_GEMM=mojo): its
// code objects load once per GPU with hipModuleLoadData (a module belongs to the device current when it loads) and
// launch with hipModuleLaunchKernel on torch's current HIP stream of the tensors' GPU. The grid is the kernel's tile
// count; an output's bits never depend on the tile. No device code here.

#ifndef __HIPCC__
#error "prefill_rocm_mojo.cu is the ROCm Mojo prompt GEMM's launcher"
#endif

#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>

#include <atomic>
#include <fstream>
#include <iterator>
#include <mutex>
#include <string>
#include <vector>

namespace {

#define TF_HIP_CHECK(x)                                                                         \
    do {                                                                                        \
        const hipError_t e_ = (x);                                                              \
        TORCH_CHECK(e_ == hipSuccess, #x, ": ", hipGetErrorString(e_));                         \
    } while (0)

struct Kernel {
    hipFunction_t fn;
    int bm, bn, threads;
};

constexpr int MAX_GPUS = 16;

struct Kernels {
    std::vector<Kernel> k;
    std::atomic<bool> ready{false};
};

// one slot a GPU, loaded on its first use (a launch only indexes it); the slots never move
Kernels g_kernels[MAX_GPUS];
std::mutex g_load;

// ``device`` current for a scope: a module loads on, and a launch runs on, its tensors' GPU
struct OnDevice {
    int prev = -1;
    explicit OnDevice(int device) {
        TF_HIP_CHECK(hipGetDevice(&prev));
        if (prev == device) {
            prev = -1;      // already current: nothing to set, nothing to restore
        } else {
            TF_HIP_CHECK(hipSetDevice(device));
        }
    }
    ~OnDevice() {
        if (prev >= 0) (void)hipSetDevice(prev);
    }
};

int gpu_of(const at::Tensor& t) {
    const int device = t.get_device();
    TORCH_CHECK(device >= 0 && device < MAX_GPUS, "the Mojo prompt GEMM takes tensors on a GPU");
    return device;
}

const Kernel& kernel(int device, int which) {
    const Kernels& k = g_kernels[device];
    TORCH_CHECK(k.ready.load(std::memory_order_acquire), "the Mojo prompt GEMM's kernels are not loaded on GPU ",
                device);
    TORCH_CHECK(which >= 0 && which < static_cast<int>(k.k.size()), "the Mojo prompt GEMM has no kernel ", which);
    return k.k[which];
}

hipFunction_t load_one(const std::string& path, const std::string& symbol) {
    std::ifstream f(path, std::ios::binary);
    TORCH_CHECK(f.good(), "cannot read ", path);
    // the image stays alive with the module for the process (hipModuleLoadData may keep pointing at it)
    auto* image = new std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    hipModule_t module;
    TF_HIP_CHECK(hipModuleLoadData(&module, image->data()));
    hipFunction_t fn;
    TF_HIP_CHECK(hipModuleGetFunction(&fn, module, symbol.c_str()));
    return fn;
}

}  // namespace

// kernel i: (path, symbol, rows a block, outputs a block, threads a block)
void load_kernels(int device, const std::vector<std::string>& paths, const std::vector<std::string>& symbols,
                  const std::vector<int>& bm, const std::vector<int>& bn, const std::vector<int>& threads) {
    TORCH_CHECK(device >= 0 && device < MAX_GPUS, "GPU ", device, ": the Mojo prompt GEMM takes ", MAX_GPUS,
                " at most");
    TORCH_CHECK(symbols.size() == paths.size() && bm.size() == paths.size() && bn.size() == paths.size() &&
                threads.size() == paths.size(), "the Mojo prompt GEMM: one path, symbol and tile a kernel");
    Kernels& k = g_kernels[device];
    std::lock_guard<std::mutex> lock(g_load);
    if (k.ready.load(std::memory_order_relaxed)) return;
    OnDevice here(device);
    for (size_t i = 0; i < paths.size(); ++i) {
        k.k.push_back({load_one(paths[i], symbols[i]), bm[i], bn[i], threads[i]});
    }
    k.ready.store(true, std::memory_order_release);
}

// x (M, K) bf16 contiguous, w (N, K) bf16 contiguous -> out (M, N) bf16 or fp32 = x @ w.T with kernel ``which``.
void gemm(const at::Tensor& x, const at::Tensor& w, at::Tensor& out, int which) {
    const int device = gpu_of(x);
    const Kernel& kn = kernel(device, which);
    int m = x.size(0), n = w.size(0), kk = x.size(1);
    TORCH_CHECK(kk % 64 == 0 && w.size(1) == kk, "prompt GEMM: K a multiple of 64");
    if (m == 0 || n == 0) return;
    int f32 = out.scalar_type() == at::kFloat ? 1 : 0;
    void* xp = x.data_ptr();
    void* wp = w.data_ptr();
    void* op = out.data_ptr();
    void* args[] = {&xp, &wp, &op, &m, &n, &kk, &f32};
    OnDevice here(device);
    TF_HIP_CHECK(hipModuleLaunchKernel(kn.fn, (m + kn.bm - 1) / kn.bm, (n + kn.bn - 1) / kn.bn, 1, kn.threads, 1, 1, 0,
                                       at::hip::getCurrentHIPStream(device).stream(), args, nullptr));
}

// prefill_matmul8: x8 (M, K) e4m3 bytes, xs (M, K/64) bf16 group sums, a (M) fp32 row scales, w8 (N, K) e4m3 codes,
// tiled scales and biases -> out (M, N) bf16 or fp32 with kernel ``which``.
void gemm8(const at::Tensor& x8, const at::Tensor& xs, const at::Tensor& a, const at::Tensor& w8,
           const at::Tensor& scales, const at::Tensor& biases, at::Tensor& out, int which) {
    const int device = gpu_of(x8);
    const Kernel& kn = kernel(device, which);
    int m = x8.size(0), n = out.size(1), kk = x8.size(1);
    TORCH_CHECK(kk % 64 == 0 && w8.size(1) == kk && xs.size(1) == kk / 64, "FP8 prompt GEMM: K a multiple of 64");
    if (m == 0 || n == 0) return;
    int f32 = out.scalar_type() == at::kFloat ? 1 : 0;
    void* xp = x8.data_ptr();
    void* sp = xs.data_ptr();
    void* ap = a.data_ptr();
    void* wp = w8.data_ptr();
    void* scp = scales.data_ptr();
    void* bp = biases.data_ptr();
    void* op = out.data_ptr();
    void* args[] = {&xp, &sp, &ap, &wp, &scp, &bp, &op, &m, &n, &kk, &f32};
    OnDevice here(device);
    TF_HIP_CHECK(hipModuleLaunchKernel(kn.fn, (m + kn.bm - 1) / kn.bm, (n + kn.bn - 1) / kn.bn, 1, kn.threads, 1, 1, 0,
                                       at::hip::getCurrentHIPStream(device).stream(), args, nullptr));
}
