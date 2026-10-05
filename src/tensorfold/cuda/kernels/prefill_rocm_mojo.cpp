#include <torch/extension.h>

#include <string>
#include <vector>

void load_kernels(const std::vector<std::string>& paths, const std::vector<std::string>& symbols,
                  const std::vector<int>& bm, const std::vector<int>& bn, const std::vector<int>& threads);
void gemm(const at::Tensor& x, const at::Tensor& w, at::Tensor& out, int which);
void gemm8(const at::Tensor& x8, const at::Tensor& xs, const at::Tensor& a, const at::Tensor& w8,
           const at::Tensor& scales, const at::Tensor& biases, at::Tensor& out, int which);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("load_kernels", &load_kernels, "Load the Mojo prompt GEMM's code objects (.hsaco paths, symbols) once");
    m.def("gemm", &gemm, "ROCm prompt GEMM out = x @ w.T on bf16 rows and weights (Mojo WMMA kernels)");
    m.def("gemm8", &gemm8, "ROCm FP8 prompt GEMM on e4m3 rows and the weight's e4m3 codes (Mojo WMMA kernels)");
}
