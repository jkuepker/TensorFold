// Host side of the Mojo lane matmul (qmm_rocm.mojo, TF_ROCM_LANE=mojo): its code objects load once per GPU with
// hipModuleLoadData (a module belongs to the device current when it loads) and launch with hipModuleLaunchKernel on
// torch's current HIP stream of the tensors' GPU, with qmm_rocm.cu's gemv_groups schedule (32-row passes, one or two
// 16-row tiles, shape-fixed K slices, persistent blocks), so the Mojo and HIP WMMA lanes give the same bits. No
// device code here.

#ifndef __HIPCC__
#error "qmm_rocm_mojo.cu is the ROCm Mojo lane's launcher"
#endif

#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <mutex>
#include <string>
#include <vector>

namespace {

constexpr int ROWS = 16;        // rows a 16-row tile takes; a pass takes two
constexpr int WARPS = 8;
constexpr int COLS = 16 * WARPS;
constexpr int SLAB = 4;

#define TF_HIP_CHECK(x)                                                                         \
    do {                                                                                        \
        const hipError_t e_ = (x);                                                              \
        TORCH_CHECK(e_ == hipSuccess, #x, ": ", hipGetErrorString(e_));                         \
    } while (0)

constexpr int MAX_GPUS = 16;

struct Kernels {
    hipFunction_t mt1 = nullptr, mt2 = nullptr, reduce = nullptr;
    int resident = 0, two = 0;  // persistent blocks the GPU holds at once (one-tile kernel; two-tile kernel)
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

int device_of(const at::Tensor& t) {
    const int device = t.get_device();
    TORCH_CHECK(device >= 0 && device < MAX_GPUS, "the Mojo lane matmul takes tensors on a GPU");
    return device;
}

const Kernels& kernels(int device) {
    const Kernels& k = g_kernels[device];
    TORCH_CHECK(k.ready.load(std::memory_order_acquire), "the Mojo lane matmul's kernels are not loaded on GPU ",
                device);
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

int per_gpu(hipFunction_t fn) {
    int per_cu = 0;
    TF_HIP_CHECK(hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(&per_cu, fn, 32 * WARPS, 0));
    return std::max(1, per_cu) * at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
}

// qmm_rocm.cu's persistent_grid: as many blocks as the GPU holds at once, fewer when that evens out their item
// counts; ``TF_ROCM_WMMA_GRID`` caps it. Only scheduling: an item's arithmetic never depends on the grid.
int persistent_grid(const Kernels& k, int items, int mt) {
    const int held = mt == 2 ? std::min(k.two, k.resident) : k.resident;
    const int rounds = (items + held - 1) / held;
    return (items + rounds - 1) / rounds;
}

void launch(hipFunction_t fn, int device, int grid, int block, void** args) {
    TF_HIP_CHECK(hipModuleLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0,
                                       at::hip::getCurrentHIPStream(device).stream(), args, nullptr));
}

}  // namespace

void load_kernels(int device, const std::string& mt1, const std::string& mt1_symbol, const std::string& mt2,
                  const std::string& mt2_symbol, const std::string& reduce, const std::string& reduce_symbol) {
    TORCH_CHECK(device >= 0 && device < MAX_GPUS, "GPU ", device, ": the Mojo lane matmul takes ", MAX_GPUS,
                " at most");
    Kernels& k = g_kernels[device];
    std::lock_guard<std::mutex> lock(g_load);
    if (k.ready.load(std::memory_order_relaxed)) return;
    OnDevice here(device);
    k.mt2 = load_one(mt2, mt2_symbol);
    k.reduce = load_one(reduce, reduce_symbol);
    hipFunction_t one = load_one(mt1, mt1_symbol);
    k.two = per_gpu(k.mt2);
    const char* cap = std::getenv("TF_ROCM_WMMA_GRID");
    const int all = per_gpu(one);
    k.resident = cap != nullptr && std::atoi(cap) > 0 ? std::min(all, std::atoi(cap)) : all;
    k.mt1 = one;
    k.ready.store(true, std::memory_order_release);
}

// qmm_rocm.cu's wmma_slices: the fewest K slices (1, 2, 4, 8 or 16) whose grid reaches ``fill`` blocks, each a
// whole number of slabs; a function of the weight's shape only.
int wmma_slices(int kg, int n, int fill) {
    const int cols = (n + COLS - 1) / COLS;
    int best = 1;
    for (int dks = 1; dks <= 16; dks *= 2) {
        if (kg % (dks * SLAB)) break;
        best = dks;
        if (cols * dks >= fill) break;
    }
    return best;
}

// qmm_rocm.cu's gemv_groups with wmma: x (M, K) bf16 rows (even stride, 16-byte aligned), xs (M, K/64) fp32 group
// sums, tiled words/scales/biases -> out (M, N) bf16 or fp32; part (slices * min(M, 32) * N) fp32 when K is split;
// counts: zeros, one per 128 outputs, left at zero.
void gemv_groups(const at::Tensor& x, const at::Tensor& xs, const at::Tensor& words, const at::Tensor& scales,
                 const at::Tensor& biases, int n, at::Tensor& out, at::Tensor& part, int fill, at::Tensor& counts) {
    const int device = device_of(x);
    const Kernels& k = kernels(device);
    OnDevice here(device);
    int kg = words.size(1);
    const int m = x.size(0);
    int slices = wmma_slices(kg, n, fill);
    int gps = kg / slices;
    int f32 = out.scalar_type() == at::kFloat ? 1 : 0;
    const int pass = 2 * ROWS;
    const int blocks = (n + COLS - 1) / COLS;
    for (int r0 = 0; r0 < m; r0 += pass) {
        int rows = std::min(pass, m - r0);
        void* p = slices > 1 ? part.data_ptr<float>() : nullptr;
        void* xr = reinterpret_cast<unsigned*>(x.data_ptr()) + static_cast<size_t>(r0) * (x.stride(0) / 2);
        int ld = static_cast<int>(x.stride(0) / 2);
        void* xsr = xs.data_ptr<float>() + static_cast<size_t>(r0) * kg;
        void* w = words.data_ptr();
        void* sc = scales.data_ptr();
        void* bi = biases.data_ptr();
        void* o = static_cast<char*>(out.data_ptr()) + static_cast<size_t>(r0) * n * out.element_size();
        void* cnt = counts.data_ptr<int>();
        const int mt = rows <= ROWS ? 1 : 2;
        int grid = persistent_grid(k, blocks * slices, mt);
        void* args[] = {&xr, &ld, &rows, &xsr, &kg, &gps, &slices, &w, &sc, &bi, &n, &p, &o, &f32,
                        &cnt, &grid};
        launch(mt == 1 ? k.mt1 : k.mt2, device, grid, 32 * WARPS, args);
    }
}

// part (slices, total) fp32 -> out (total) fp32 or bf16, slices added in order (qmm_rocm.cu's reduce_kernel).
void reduce_slices(const at::Tensor& part, int slices, at::Tensor& out) {
    const int device = device_of(out);
    const Kernels& k = kernels(device);
    OnDevice here(device);
    int64_t total = out.numel();
    int f32 = out.scalar_type() == at::kFloat ? 1 : 0;
    void* pp = part.data_ptr();
    void* o32 = f32 ? out.data_ptr() : nullptr;
    void* o16 = f32 ? nullptr : out.data_ptr();
    void* args[] = {&pp, &slices, &total, &o32, &o16, &f32};
    if (total > 0) launch(k.reduce, device, static_cast<int>((total + 255) / 256), 256, args);
}
