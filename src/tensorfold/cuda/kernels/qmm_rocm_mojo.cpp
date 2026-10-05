#include <torch/extension.h>

#include <string>

void load_kernels(int device, const std::string& mt1, const std::string& mt1_symbol, const std::string& mt2,
                  const std::string& mt2_symbol);
void gemv_groups(const at::Tensor& x, const at::Tensor& xs, const at::Tensor& words, const at::Tensor& scales,
                 const at::Tensor& biases, int n, at::Tensor& out, at::Tensor& part, int fill, at::Tensor& counts);
int wmma_slices(int kg, int n, int fill);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("load_kernels", &load_kernels,
          "Load the Mojo lane matmul's code objects (.hsaco path, symbol) on a GPU, once");
    m.def("gemv_groups", &gemv_groups, "ROCm decode matmul on group-major 4-bit words (Mojo WMMA kernels)");
    m.def("wmma_slices", &wmma_slices, "WMMA K slices of a (kg groups, n outputs) weight");
}
