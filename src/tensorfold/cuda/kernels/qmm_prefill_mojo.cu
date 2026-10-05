// Host side of the Mojo prompt matmuls (qmm_prefill.mojo, TF_CUDA_PREFILL_GEMM=mojo): their cubins load once with
// cuModuleLoadData and launch with cuLaunchKernel on torch's current CUDA stream, with qmm_prefill.cu's and
// qmm_prefill8.cu's grids (one block a 128 x 128 tile, row tiles in L2 bands of ``group``), so the Mojo and CUDA
// kernels give the same bits. No device code here.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>

#include <algorithm>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

constexpr int BM = 128, BN = 128, THREADS = 128;   // qmm_prefill.mojo's tile (prompt_tile's 9)

#define TF_CU_CHECK(x)                                                                          \
    do {                                                                                        \
        const CUresult e_ = (x);                                                                \
        const char* m_ = nullptr;                                                               \
        if (e_ != CUDA_SUCCESS) cuGetErrorString(e_, &m_);                                      \
        TORCH_CHECK(e_ == CUDA_SUCCESS, #x, ": ", m_ ? m_ : "unknown CUDA driver error");       \
    } while (0)

struct Kernels {
    CUfunction bf16 = nullptr, f32 = nullptr, fp8_bf16 = nullptr, fp8_f32 = nullptr;
};

Kernels& kernels() {
    static Kernels k;
    return k;
}

CUfunction load_one(const std::string& path, const std::string& symbol) {
    std::ifstream f(path, std::ios::binary);
    TORCH_CHECK(f.good(), "cannot read ", path);
    std::vector<char> image((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    CUmodule module;
    TF_CU_CHECK(cuModuleLoadData(&module, image.data()));
    CUfunction fn;
    TF_CU_CHECK(cuModuleGetFunction(&fn, module, symbol.c_str()));
    return fn;
}

}  // namespace

// (path, symbol) of each cubin: the bf16 kernels' bf16 and fp32 outputs, then the FP8 kernels'
void load_kernels(const std::vector<std::string>& paths, const std::vector<std::string>& symbols) {
    Kernels& k = kernels();
    if (k.bf16 != nullptr) return;
    TORCH_CHECK(paths.size() == 4 && symbols.size() == 4, "load_kernels: four (cubin, symbol) pairs");
    at::cuda::getCurrentCUDAStream();       // torch's primary context is current before the modules load
    k.f32 = load_one(paths[1], symbols[1]);
    k.fp8_bf16 = load_one(paths[2], symbols[2]);
    k.fp8_f32 = load_one(paths[3], symbols[3]);
    k.bf16 = load_one(paths[0], symbols[0]);
}

// qmm_prefill.cu's launch: x (M, K) bf16 rows (stride a multiple of 8, 16-byte aligned), tiled words, (K/64, npad)
// scales and biases -> out (M, N) bf16 or fp32.
void prefill(const at::Tensor& x, const at::Tensor& w, const at::Tensor& scales, const at::Tensor& biases,
             at::Tensor& out, int N) {
    const Kernels& k = kernels();
    TORCH_CHECK(k.bf16 != nullptr, "the Mojo prompt matmul's kernels are not loaded");
    int M = x.size(0), K = x.size(1);
    if (M == 0) return;
    const int rows_t = (M + BM - 1) / BM;
    // a group's inputs stay near 12 MB of L2 while its blocks sweep the column tiles
    int group = std::max(1, std::min(rows_t, static_cast<int>((12LL << 20) / (static_cast<long long>(BM) * K * 2))));
    int npad = static_cast<int>(scales.size(1));
    int ldx = M == 1 ? K : static_cast<int>(x.stride(0));
    void* px = x.data_ptr();
    void* pw = w.data_ptr();
    void* ps = scales.data_ptr();
    void* pb = biases.data_ptr();
    void* po = out.data_ptr();
    void* args[] = {&px, &pw, &ps, &pb, &po, &M, &N, &K, &npad, &ldx, &group};
    const unsigned grid = static_cast<unsigned>(rows_t * ((N + BN - 1) / BN));
    CUfunction fn = out.scalar_type() == at::kFloat ? k.f32 : k.bf16;
    TF_CU_CHECK(cuLaunchKernel(fn, grid, 1, 1, THREADS, 1, 1, 0,
                               reinterpret_cast<CUstream>(at::cuda::getCurrentCUDAStream().stream()), args, nullptr));
}

// qmm_prefill8.cu's launch (4-bit words, tile 0): x (M, K) e4m3 bytes in fragment order, xs (M, K/64) bf16 group
// sums over the row scale, a (M) fp32 row scales, tiled words, (K/64, npad) scales and biases -> out bf16 or fp32.
void prefill8(const at::Tensor& x, const at::Tensor& xs, const at::Tensor& a, const at::Tensor& w,
              const at::Tensor& scales, const at::Tensor& biases, at::Tensor& out, int N) {
    const Kernels& k = kernels();
    TORCH_CHECK(k.fp8_bf16 != nullptr, "the Mojo prompt matmul's kernels are not loaded");
    int M = x.size(0), K = x.size(1);
    if (M == 0) return;
    const int rows_t = (M + BM - 1) / BM;
    int group = std::max(1, std::min(rows_t, static_cast<int>((12LL << 20) / (static_cast<long long>(BM) * K))));
    int npad = static_cast<int>(scales.size(1));
    void* px = x.data_ptr();
    void* pxs = xs.data_ptr();
    void* pa = a.data_ptr();
    void* pw = w.data_ptr();
    void* ps = scales.data_ptr();
    void* pb = biases.data_ptr();
    void* po = out.data_ptr();
    void* args[] = {&px, &pxs, &pa, &pw, &ps, &pb, &po, &M, &N, &K, &npad, &group};
    const unsigned grid = static_cast<unsigned>(rows_t * ((N + BN - 1) / BN));
    CUfunction fn = out.scalar_type() == at::kFloat ? k.fp8_f32 : k.fp8_bf16;
    TF_CU_CHECK(cuLaunchKernel(fn, grid, 1, 1, THREADS, 1, 1, 0,
                               reinterpret_cast<CUstream>(at::cuda::getCurrentCUDAStream().stream()), args, nullptr));
}
