"""Sustained WMMA TFLOPS on this GPU (peak_wmma.mojo through the prompt GEMM's launcher). usage: PYTHONPATH=src
python3 tools/mojo_gate/peak/peak.py"""
from pathlib import Path

import torch

from tensorfold.cuda.build import hip_arch, load
from tensorfold.cuda.mojo import build_hsaco

here = Path(__file__).parent
kern = Path(__file__).parents[3] / "src/tensorfold/cuda/kernels"
built, manifest = build_hsaco(here / "peak_wmma.mojo", hip_arch())
ext = load(name="tensorfold_prefill_rocm_mojo_v1", sources=[str(kern / "prefill_rocm_mojo.cpp"),
                                                           str(kern / "prefill_rocm_mojo.cu")],
           extra_cuda_cflags=["-O3"], verbose=False)
k = {e["name"]: e for e in manifest["kernels"]}
names = ["peak_bf16", "peak_fp8"]
ext.load_kernels([str(built / k[n]["hsaco"]) for n in names], [k[n]["symbol"] for n in names], [1, 1], [1, 1],
                 [256, 256])
props = torch.cuda.get_device_properties(0)
for which, name in enumerate(names):
    for blocks, steps in ((props.multi_processor_count * 8, 16 * 4096), (props.multi_processor_count * 8, 16 * 65536)):
        x = torch.zeros((1, steps), dtype=torch.bfloat16, device="cuda")
        w = torch.zeros((1, steps), dtype=torch.bfloat16, device="cuda")
        out = torch.zeros((blocks, 1), dtype=torch.bfloat16, device="cuda")
        xs = torch.zeros((blocks, steps), dtype=torch.bfloat16, device="cuda")
        ext.gemm(xs, w, out, which)
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        ext.gemm(xs, w, out, which)
        b.record()
        b.synchronize()
        t = a.elapsed_time(b) / 1e3
        flops = blocks * 8 * 16 * (steps // 16) * 2 * 16 * 16 * 16
        print(f"{name} {blocks} blocks: {flops / t / 1e12:.1f} TFLOPS ({t * 1e3:.1f} ms)", flush=True)
