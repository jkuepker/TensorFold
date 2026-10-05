#include <torch/extension.h>

#include <string>
#include <vector>

void load_kernels(const std::vector<std::string>& paths, const std::vector<std::string>& symbols);
void tree_shared(const at::Tensor& q, const at::Tensor& base, const at::Tensor& offs, const at::Tensor& streams,
                 const at::Tensor& items, at::Tensor& po, at::Tensor& pm, at::Tensor& pl, int hk, int cw, bool pipe,
                 double scale, bool kv8);
void tree_tail(const at::Tensor& q, const at::Tensor& kn, const at::Tensor& vn, const at::Tensor& base,
               const at::Tensor& offs, const at::Tensor& streams, const at::Tensor& rows, const at::Tensor& paths,
               const at::Tensor& depths, at::Tensor& po, at::Tensor& pm, at::Tensor& pl, int tails, double scale,
               bool kv8);
void tree_merge(const at::Tensor& po, const at::Tensor& pm, const at::Tensor& pl, at::Tensor& out,
                const at::Tensor& streams, const at::Tensor& rows);

bool prompt_supported(int heads, int kv_heads, int dim);
void prompt_attention(const at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& v_cache, at::Tensor& out,
                      int p0, double scale, int rb, bool pipe);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("load_kernels", &load_kernels, "Load the Mojo attention's code objects (.hsaco paths, symbols) once");
    m.def("shared", &tree_shared, "Tree attention over full committed chunks (Mojo WMMA kernels)");
    m.def("tail", &tree_tail, "Tree attention over each row's tail chunks: committed keys, then its path (Mojo)");
    m.def("prompt_supported", &prompt_supported, "Whether the Mojo prompt attention takes these heads");
    m.def("attention", &prompt_attention, "Prompt attention over the caches' keys (Mojo, attention_rocm.cu's bits)");
    m.def("merge", &tree_merge, "Chunk partials merged in key order into bf16 rows (Mojo, Triton _merge's bits)");
}
