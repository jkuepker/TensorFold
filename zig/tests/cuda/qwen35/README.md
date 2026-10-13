# Qwen3.6 MoE router oracle

`capture_router.py` runs the Python engine's router, top-k and combine launches (`tensorfold/cuda/moe.py`) over every row-count tile,
tie and signed-zero cases, and both combine input forms, with the Triton launch recorder (`tools/zig/triton_aot_manifest.py`) installed.
It writes `fixtures/` (inputs and outputs as raw bytes), `fixtures.json`, `launches.json` and `jit.json`.
`capture_router_engine.py` does the same inside a real checkpoint's generation, so the weights and activations are the model's own.

Pack the kernel set as for Nemotron (`nemotron/box/README.md` steps 2 and 3): `triton_aot_manifest.py`, then `nemotron/aot_pack.py`.
Run `tf-cuda-test qwen35-router <kernel set> <capture dir> [rounds]`: the replayed cubins must equal the Python bytes, and
`qwen_route.cu` must equal both. `Router` in `zig/src/families/qwen3_5/cuda/router.zig` takes the native kernels only where
`exact` is true.
