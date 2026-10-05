#include <torch/extension.h>

#include <string>
#include <vector>

void load_kernels(const std::vector<std::string>& paths, const std::vector<std::string>& symbols);
void prefill(const at::Tensor& x, const at::Tensor& w, const at::Tensor& scales, const at::Tensor& biases,
             at::Tensor& out, int N);
void prefill8(const at::Tensor& x, const at::Tensor& xs, const at::Tensor& a, const at::Tensor& w,
              const at::Tensor& scales, const at::Tensor& biases, at::Tensor& out, int N);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("load_kernels", &load_kernels, "Load the Mojo prompt matmuls' cubins (paths, symbols) once");
    m.def("prefill", &prefill, "NVIDIA prompt matmul, bf16 or fp32 out (Mojo kernels, qmm_prefill.cu's bits)");
    m.def("prefill8", &prefill8, "NVIDIA FP8 prompt matmul, bf16 or fp32 out (Mojo kernels, qmm_prefill8.cu's bits)");
}
