#include <torch/extension.h>

#include <string>
#include <vector>

void load_kernels(const std::vector<std::string>& paths, const std::vector<std::string>& symbols,
                  const std::vector<int>& bm, const std::vector<int>& bn, const std::vector<int>& threads);
void gemm(const at::Tensor& x, const at::Tensor& w, at::Tensor& out, int which);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("load_kernels", &load_kernels, "Load the Mojo prompt GEMM's code objects (.hsaco paths, symbols) once");
    m.def("gemm", &gemm, "ROCm prompt GEMM out = x @ w.T on bf16 rows and weights (Mojo WMMA kernels)");
}
