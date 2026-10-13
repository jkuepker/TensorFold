#!/usr/bin/env python3
"""Rebuild a captured Triton kernel set so every kernel waits on the grid before it, and mark it `pdl` in aot.json.

Programmatic dependent launch lets a kernel's launch be staged while the one before it still runs. The staged kernel
must begin with `griddepcontrol.wait` (it holds until the earlier grid has finished and flushed its writes), which
Triton kernels do not have. Each kernel's PTX is found in a Triton cache by its hash, `griddepcontrol.wait;` goes in
as the first instruction, and ptxas builds the cubin with Triton's own command line. Before patching, the unpatched
PTX is built too and must reproduce the captured cubin byte for byte; if it does not, the toolchain differs and the
kernel is refused.

  aot_pdl.py --set <kernel set> --cache <Triton cache> [--cache ...] --ptxas <ptxas> --out <new kernel set>
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

WAIT = "\tgriddepcontrol.wait;\n"


def find_ptx(caches, name, digest):
    """Every cached PTX of that kernel and hash (runs from different trees differ in their source paths)."""
    for root in caches:
        for d in os.listdir(root):
            meta = os.path.join(root, d, name + ".json")
            if not os.path.exists(meta):
                continue
            with open(meta) as f:
                m = json.load(f)
            if m.get("hash") == digest and os.path.exists(os.path.join(root, d, name + ".ptx")):
                yield os.path.join(root, d, name + ".ptx"), m


def patch(ptx):
    """The wait as the entry's first instruction: right after its last register declaration."""
    lines = ptx.splitlines(keepends=True)
    entry = next(i for i, l in enumerate(lines) if ".visible .entry" in l)
    brace = next(i for i in range(entry, len(lines)) if lines[i].startswith("{"))
    at = brace + 1
    while lines[at].lstrip().startswith(".reg") or lines[at].lstrip().startswith(".loc") or lines[at].startswith("$L__func_begin") or not lines[at].strip():
        at += 1
    # the first line that is not a declaration, label or location: its instruction runs after the wait
    return "".join(lines[:at]) + WAIT + "".join(lines[at:])


def build(ptxas, ptx_text, arch, work):
    src = os.path.join(work, "k.ptx")
    out = os.path.join(work, "k.cubin")
    with open(src, "w") as f:
        f.write(ptx_text)
    subprocess.run([ptxas, "-lineinfo", "-v", "--regAllocOptLevel=2", "--gpu-name=" + arch, src, "-o", out], check=True, stderr=subprocess.DEVNULL)
    with open(out, "rb") as f:
        return f.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--cache", action="append", required=True)
    ap.add_argument("--ptxas", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(os.path.join(a.set, "aot.json")) as f:
        aot = json.load(f)
    os.makedirs(os.path.join(a.out, "cubins"), exist_ok=True)
    bad = 0
    with tempfile.TemporaryDirectory() as work:
        for k in aot["kernels"]:
            old = os.path.join(a.set, "cubins", k["hash"] + ".cubin")
            with open(old, "rb") as f:
                captured = f.read()
            text = None
            for ptx, meta in find_ptx(a.cache, k["name"], k["hash"]):
                arch = "sm_%d%s" % (meta["target"]["arch"], "a" if meta["target"]["arch"] >= 90 else "")
                with open(ptx) as f:
                    cand = f.read()
                if build(a.ptxas, cand, arch, work) == captured:
                    text = cand
                    break
            if text is None:
                print("no cached PTX that this ptxas builds into the captured cubin:", k["name"], k["hash"][:12], file=sys.stderr)
                bad += 1
                continue
            with open(os.path.join(a.out, "cubins", k["hash"] + ".cubin"), "wb") as f:
                f.write(build(a.ptxas, patch(text), arch, work))
            k["pdl"] = True
            print("patched", k["name"], k["hash"][:12])
    if bad:
        sys.exit("%d kernels not rebuilt" % bad)
    with open(os.path.join(a.out, "aot.json"), "w") as f:
        json.dump(aot, f, indent=1)


if __name__ == "__main__":
    main()
