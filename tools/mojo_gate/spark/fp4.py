"""Check 4e driver: block-scaled FP4 mma.sync m16n8k64 on sm_121. Every nibble is e2m1 1.0 (0x2) and the four
ue4m3 scale blocks of A are (1,2,4,8) (0x38,0x40,0x48,0x50): D must be 16*(1+2+4+8) = 240 in every element
(each of the four 16-element k-blocks contributes 16 * scale_a * scale_b). Scaling B's blocks too, (1,1,1,1) -> (2,2,2,2): 480."""
import ctypes
from pathlib import Path
import torch

lib = ctypes.CDLL(str(Path(__file__).parent / "libfp4.so"))
lib.gate_mma_fp4.argtypes = [ctypes.c_long] * 3
dev = "cuda"
torch.zeros(1, device=dev)


def pack(*b): return b[0] | b[1] << 8 | b[2] << 16 | b[3] << 24


one, two, four, eight = 0x38, 0x40, 0x48, 0x50
for sfa, sfb, want in ((pack(one, two, four, eight), pack(one, one, one, one), 240.0),
                       (pack(one, two, four, eight), pack(two, two, two, two), 480.0),
                       (pack(one, one, one, one), pack(one, one, one, one), 64.0)):
    out = torch.zeros(32, 4, device=dev, dtype=torch.float32)
    torch.cuda.synchronize()
    rc = lib.gate_mma_fp4(sfa, sfb, out.data_ptr()); torch.cuda.synchronize()
    ok = rc == 0 and bool((out == want).all())
    print(f"(e2) fp4 e2m1 block-scaled mma.sync m16n8k64: rc={rc} D[0,:4]={out[0].tolist()} want {want} {'PASS' if ok else 'FAIL'}")
