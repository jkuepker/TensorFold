// Host side of the Mojo tree attention (attention_rocm.mojo, TF_ROCM_TREE_KERNEL=mojo): its code objects load once
// with hipModuleLoadData and launch with hipModuleLaunchKernel on torch's current HIP stream, with the grids and
// blocks attention_rocm.cu's tree_shared and tree_tail use and a grid of its own for the merge (a row's bits never
// depend on it). No device code here.

#ifndef __HIPCC__
#error "attention_rocm_mojo.cu is the ROCm Mojo tree attention's launcher"
#endif

#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

constexpr int D = 256;
constexpr int MAXD = 128;
constexpr int LOADERS = 4;
constexpr int NL = 32 * LOADERS;
constexpr int MERGE_HEADS = 4;              // attention_rocm.mojo's MERGE_HEADS: (row, head) pairs a merge block

#define TF_HIP_CHECK(x)                                                                         \
    do {                                                                                        \
        const hipError_t e_ = (x);                                                              \
        TORCH_CHECK(e_ == hipSuccess, #x, ": ", hipGetErrorString(e_));                         \
    } while (0)

enum { SHARED_PIPE, SHARED_PIPE8, SHARED_FLAT, SHARED_FLAT8, TAIL16, TAIL8, MERGE, KERNELS };

std::vector<hipFunction_t>& kernels() {
    static std::vector<hipFunction_t> k;
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

hipFunction_t kernel(int which) {
    const auto& k = kernels();
    TORCH_CHECK(k.size() == KERNELS, "the Mojo tree attention's kernels are not loaded");
    return k[which];
}

void launch(hipFunction_t fn, dim3 grid, int block, void** args) {
    TF_HIP_CHECK(hipModuleLaunchKernel(fn, grid.x, grid.y, grid.z, block, 1, 1, 0,
                                       at::hip::getCurrentHIPStream().stream(), args, nullptr));
}

bool tree_supported(int heads, int kv_heads, int dim) {
    return dim == D && kv_heads > 0 && heads % kv_heads == 0 && heads / kv_heads <= 16;
}

}  // namespace

// paths and symbols in the order shared_pipe, shared_pipe8, shared_flat, shared_flat8, tail16, tail8, merge
void load_kernels(const std::vector<std::string>& paths, const std::vector<std::string>& symbols) {
    auto& k = kernels();
    if (!k.empty()) return;
    TORCH_CHECK(paths.size() == KERNELS && symbols.size() == KERNELS, "the Mojo tree attention has 7 kernels");
    std::vector<hipFunction_t> loaded;
    for (size_t i = 0; i < paths.size(); ++i) loaded.push_back(load_one(paths[i], symbols[i]));
    k = loaded;
}

// attention_rocm.cu's tree_shared: grid (KV heads, items), 65,535 items a launch.
void tree_shared(const at::Tensor& q, const at::Tensor& base, const at::Tensor& offs, const at::Tensor& streams,
                 const at::Tensor& items, at::Tensor& po, at::Tensor& pm, at::Tensor& pl, int hk, int cw, bool pipe,
                 double scale, bool kv8) {
    int w = q.size(0), h = q.size(1);
    const int n = items.size(0);
    TORCH_CHECK(tree_supported(h, hk, q.size(2)) && 1 <= cw && cw <= 8, "ROCm tree attention: head size 256");
    hipFunction_t fn = kernel(pipe ? (kv8 ? SHARED_PIPE8 : SHARED_PIPE) : (kv8 ? SHARED_FLAT8 : SHARED_FLAT));
    void* qp = q.data_ptr();
    void* bp = base.data_ptr();
    void* op = offs.data_ptr<int64_t>();
    void* sp = streams.data_ptr<int>();
    void* ip = items.data_ptr<int>();
    void* pop = po.data_ptr<float>();
    void* pmp = pm.data_ptr<float>();
    void* plp = pl.data_ptr<float>();
    int hkc = hk, g = h / hk;
    float sc = static_cast<float>(scale);
    for (int item0 = 0; item0 < n; item0 += 65535) {
        void* args[] = {&qp, &bp, &op, &sp, &ip, &pop, &pmp, &plp, &w, &h, &hkc, &g, &sc, &item0};
        launch(fn, dim3(hk, std::min(65535, n - item0)), 32 * (cw + (pipe ? LOADERS : 0)), args);
    }
}

// attention_rocm.cu's tree_tail's grid (W, KV heads, tails); a block of four loader waves and a folding one.
void tree_tail(const at::Tensor& q, const at::Tensor& kn, const at::Tensor& vn, const at::Tensor& base,
               const at::Tensor& offs, const at::Tensor& streams, const at::Tensor& rows, const at::Tensor& paths,
               const at::Tensor& depths, at::Tensor& po, at::Tensor& pm, at::Tensor& pl, int tails, double scale,
               bool kv8) {
    int w = q.size(0), h = q.size(1), hk = kn.size(1);
    TORCH_CHECK(tree_supported(h, hk, q.size(2)) && paths.size(1) == MAXD, "ROCm tree attention: head size 256");
    void* qp = q.data_ptr();
    void* knp = kn.data_ptr();
    void* vnp = vn.data_ptr();
    void* bp = base.data_ptr();
    void* op = offs.data_ptr<int64_t>();
    void* sp = streams.data_ptr<int>();
    void* rp = rows.data_ptr<int>();
    void* pp = paths.data_ptr<int>();
    void* dp = depths.data_ptr<int>();
    void* pop = po.data_ptr<float>();
    void* pmp = pm.data_ptr<float>();
    void* plp = pl.data_ptr<float>();
    int vs = static_cast<int>(vn.stride(0)), g = h / hk;
    float sc = static_cast<float>(scale);
    void* args[] = {&qp, &knp, &vnp, &bp, &op, &sp, &rp, &pp, &dp, &pop, &pmp, &plp, &w, &vs, &h, &hk, &g, &sc};
    if (w > 0) launch(kernel(kv8 ? TAIL8 : TAIL16), dim3(w, hk, tails), NL + 32, args);   // four loaders, a folder
}

// attention.py's _merge: po (chunks, W, H, 256), pm and pl (chunks, W, H) fp32 -> out (W, H, 256) bf16.
void tree_merge(const at::Tensor& po, const at::Tensor& pm, const at::Tensor& pl, at::Tensor& out,
                const at::Tensor& streams, const at::Tensor& rows) {
    int w = out.size(0), h = out.size(1);
    TORCH_CHECK(out.size(2) == D && po.size(3) == D && out.is_contiguous() && po.is_contiguous(),
                "ROCm tree attention merge: head size 256, contiguous");
    void* pop = po.data_ptr<float>();
    void* pmp = pm.data_ptr<float>();
    void* plp = pl.data_ptr<float>();
    void* outp = out.data_ptr();
    void* sp = streams.data_ptr<int>();
    void* rp = rows.data_ptr<int>();
    void* args[] = {&pop, &pmp, &plp, &outp, &sp, &rp, &w, &h};
    if (w > 0) launch(kernel(MERGE), dim3(w, (h + MERGE_HEADS - 1) / MERGE_HEADS), 64 * MERGE_HEADS, args);
}
