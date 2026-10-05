#include <torch/extension.h>

#include <string>

void load_kernels(const std::string& bf16, const std::string& bf16_symbol, const std::string& f32,
                  const std::string& f32_symbol);
void prefill(const at::Tensor& x, const at::Tensor& w, const at::Tensor& scales, const at::Tensor& biases,
             at::Tensor& out, int N);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("load_kernels", &load_kernels, "Load the Mojo prompt matmul's cubins (path, symbol) once");
    m.def("prefill", &prefill, "NVIDIA prompt matmul, bf16 or fp32 out (Mojo kernels, qmm_prefill.cu's bits)");
}
