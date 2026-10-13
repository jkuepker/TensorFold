"""Fixtures for `tf-cuda-test affine4-experts`: the Python engine's grouped MLX 4-bit experts (cuda.experts), SwiGLU and relu^2."""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from oracle import save  # noqa: E402

E, NI, D, SLOTS = 33, 512, 2048, 9      # experts (the shared one last), width, dims, routed picks and the shared one
DECODE, PROMPT = (1, 3, 17), (3, 300)   # rows a call: one item, a few, past a decode tile; a partial prompt tile, a wide plan
LIMIT = 0.4                             # a SwiGLU clamp that bites on these values


def matrix(n: int, k: int, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    words = torch.randint(-(2 ** 31), 2 ** 31, (E, n, k // 8), generator=g, dtype=torch.int32)
    s = torch.rand(E, n, k // 64, generator=g) * 0.02 + 0.004
    return words, s.bfloat16(), (-s * (4 + torch.rand(E, n, k // 64, generator=g) * 8)).bfloat16()


def picks(rows: int, g: torch.Generator) -> torch.Tensor:
    routed = [torch.randperm(E - 1, generator=g)[:SLOTS - 1] for _ in range(rows)]
    return torch.cat([torch.stack(routed), torch.full((rows, 1), E - 1)], dim=1).int().contiguous()


def main() -> None:
    from tensorfold.cuda import experts as grouped

    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    a = ap.parse_args()
    g = torch.Generator().manual_seed(23)
    gate, up, down = matrix(NI, D, g), matrix(NI, D, g), matrix(D, NI, g)
    cuda = lambda m: tuple(t.cuda() for t in m)  # noqa: E731
    layers = {"swiglu": grouped.make([cuda(gate), cuda(up)], cuda(down), 64),
              "limited": grouped.make([cuda(gate), cuda(up)], cuda(down), 64, limit=LIMIT),
              "relu2": grouped.make([cuda(gate)], cuda(down), 64)}
    arrays = {f"{name}_{i}": t for name, m in (("gate", gate), ("up", up), ("down", down))
              for i, t in zip(("words", "scales", "biases"), m)}
    arrays.update(packed_up=layers["swiglu"].up, packed_down=layers["swiglu"].down, packed_relu2_up=layers["relu2"].up)
    for mode, cases in (("d", DECODE), ("p", PROMPT)):
        for rows in cases:
            pk = picks(rows, g)
            x = (torch.randn(rows, D, generator=g) * 0.5).bfloat16()
            plan = grouped.Plan(rows, SLOTS, E, "cuda", prefill=mode == "p")
            grouped.route(pk.cuda(), plan, grouped.PREFILL_TILE if mode == "p" else grouped.TILE)
            tag = f"{mode}{rows}"
            arrays.update({f"{tag}_picks": pk, f"{tag}_x": x, f"{tag}_members": plan.members,
                           f"{tag}_items": plan.items, f"{tag}_counts": plan.counts})
            for name, ex in layers.items():
                act = torch.zeros((rows * SLOTS, NI), dtype=torch.bfloat16, device="cuda")
                grouped.gate_up(x.cuda(), ex, plan, act, rows)
                y = torch.zeros((rows * SLOTS, D), dtype=torch.bfloat16 if mode == "p" else torch.float32, device="cuda")
                grouped.down(act, ex, plan, y, rows)
                torch.cuda.synchronize()
                arrays.update({f"{tag}_{name}_act": act, f"{tag}_{name}_y": y})
    save(Path(a.out), {"experts": E, "width": NI, "dims": D, "slots": SLOTS, "limit": LIMIT,
                       "decode": ",".join(map(str, DECODE)), "prompt": ",".join(map(str, PROMPT))}, arrays)


if __name__ == "__main__":
    main()
