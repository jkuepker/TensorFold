// Host side of the Mojo prompt GEMM (prefill_rocm.mojo, TF_ROCM_PREFILL_GEMM=mojo): its code objects load once with
// hipModuleLoadData and launch with hipModuleLaunchKernel on torch's current HIP stream. The grid is the kernel's
// tile count; an output's bits never depend on the tile. No device code here.

#ifndef __HIPCC__
#error "prefill_rocm_mojo.cu is the ROCm Mojo prompt GEMM's launcher"
#endif

#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>

#include <fstream>
#include <iterator>
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

std::vector<Kernel>& kernels() {
    static std::vector<Kernel> k;
    return k;
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
void load_kernels(const std::vector<std::string>& paths, const std::vector<std::string>& symbols,
                  const std::vector<int>& bm, const std::vector<int>& bn, const std::vector<int>& threads) {
    auto& k = kernels();
    if (!k.empty()) return;
    std::vector<Kernel> loaded;
    for (size_t i = 0; i < paths.size(); ++i)
        loaded.push_back({load_one(paths[i], symbols[i]), bm[i], bn[i], threads[i]});
    k = loaded;
}

// x (M, K) bf16 contiguous, w (N, K) bf16 contiguous -> out (M, N) bf16 or fp32 = x @ w.T with kernel ``which``.
void gemm(const at::Tensor& x, const at::Tensor& w, at::Tensor& out, int which) {
    const auto& k = kernels();
    TORCH_CHECK(which >= 0 && which < static_cast<int>(k.size()), "the Mojo prompt GEMM's kernels are not loaded");
    const Kernel& kn = k[which];
    int m = x.size(0), n = w.size(0), kk = x.size(1);
    TORCH_CHECK(kk % 64 == 0 && w.size(1) == kk, "prompt GEMM: K a multiple of 64");
    if (m == 0 || n == 0) return;
    int f32 = out.scalar_type() == at::kFloat ? 1 : 0;
    void* xp = x.data_ptr();
    void* wp = w.data_ptr();
    void* op = out.data_ptr();
    void* args[] = {&xp, &wp, &op, &m, &n, &kk, &f32};
    TF_HIP_CHECK(hipModuleLaunchKernel(kn.fn, (m + kn.bm - 1) / kn.bm, (n + kn.bn - 1) / kn.bn, 1, kn.threads, 1, 1, 0,
                                       at::hip::getCurrentHIPStream().stream(), args, nullptr));
}
