#!/usr/bin/env python3
"""Mojo GPU kernels -> loadable gfx1201 code objects (.hsaco) + manifest.json.

Route: `mojo build --emit object --target-accelerator gfx1201 <file>.mojo` leaves one
complete AMDGPU ELF code object (EM_AMDGPU, ET_DYN, with the AMDGPU metadata note) per
instantiated kernel embedded in the host object. Every kernel must be instantiated by a host
function in the file (enqueue_function[kernel]) or Mojo does not compile it. This script
carves the embedded ELFs out, and writes for each kernel
  <outdir>/<short>.hsaco and an entry in <outdir>/manifest.json: symbol (hipModuleGetFunction
name), kernarg_size, explicit args (offset/size/kind) and hidden args.
The ELFs are valid for hipModuleLoadData as-is (no assembling/linking needed).

usage: mojo2hsaco.py kernels.mojo outdir [--arch gfx1201] [--mojo mojo]
"""
import argparse, json, os, re, struct, subprocess, sys, tempfile


def carve_elfs(blob):
    """Return [bytes] of every AMDGPU ELF (e_machine 224) embedded after the host ELF."""
    out = []
    for m in re.finditer(b"\x7fELF", blob):
        o = m.start()
        if o + 64 > len(blob) or struct.unpack_from("<H", blob, o + 18)[0] != 224:
            continue
        shoff, = struct.unpack_from("<Q", blob, o + 0x28)
        shentsize, shnum = struct.unpack_from("<HH", blob, o + 0x3A)
        end = shoff + shentsize * shnum
        for i in range(shnum):
            sh = o + shoff + i * shentsize
            typ, = struct.unpack_from("<I", blob, sh + 4)
            off, size = struct.unpack_from("<QQ", blob, sh + 0x18)
            if typ != 8:  # SHT_NOBITS occupies no file space
                end = max(end, off + size)
        out.append(blob[o:o + end])
    return out


def msgpack(b, i=0):
    """Minimal msgpack decoder (what the AMDGPU metadata note uses)."""
    t = b[i]; i += 1
    if t <= 0x7F: return t, i
    if t >= 0xE0: return t - 256, i
    if 0xA0 <= t <= 0xBF: n = t & 31; return b[i:i + n].decode(), i + n
    if 0x90 <= t <= 0x9F: return _arr(b, i, t & 15)
    if 0x80 <= t <= 0x8F: return _map(b, i, t & 15)
    if t == 0xC0: return None, i
    if t == 0xC2: return False, i
    if t == 0xC3: return True, i
    for code, fmt, n in ((0xCC, ">B", 1), (0xCD, ">H", 2), (0xCE, ">I", 4), (0xCF, ">Q", 8),
                         (0xD0, ">b", 1), (0xD1, ">h", 2), (0xD2, ">i", 4), (0xD3, ">q", 8)):
        if t == code: return struct.unpack_from(fmt, b, i)[0], i + n
    if t in (0xD9, 0xDA, 0xDB):
        w = {0xD9: 1, 0xDA: 2, 0xDB: 4}[t]
        n = int.from_bytes(b[i:i + w], "big"); i += w
        return b[i:i + n].decode(), i + n
    if t in (0xDC, 0xDD):
        w = 2 if t == 0xDC else 4; n = int.from_bytes(b[i:i + w], "big"); return _arr(b, i + w, n)
    if t in (0xDE, 0xDF):
        w = 2 if t == 0xDE else 4; n = int.from_bytes(b[i:i + w], "big"); return _map(b, i + w, n)
    raise ValueError(f"msgpack type {t:#x}")


def _arr(b, i, n):
    r = []
    for _ in range(n):
        v, i = msgpack(b, i); r.append(v)
    return r, i


def _map(b, i, n):
    r = {}
    for _ in range(n):
        k, i = msgpack(b, i); v, i = msgpack(b, i); r[k] = v
    return r, i


def amdgpu_metadata(elf):
    shoff, = struct.unpack_from("<Q", elf, 0x28)
    shentsize, shnum = struct.unpack_from("<HH", elf, 0x3A)
    for i in range(shnum):
        sh = shoff + i * shentsize
        typ, = struct.unpack_from("<I", elf, sh + 4)
        if typ != 7: continue
        off, size = struct.unpack_from("<QQ", elf, sh + 0x18)
        p = off
        while p < off + size:
            nsz, dsz, nt = struct.unpack_from("<III", elf, p); p += 12
            name = elf[p:p + nsz].rstrip(b"\0"); p += (nsz + 3) & ~3
            desc = elf[p:p + dsz]; p += (dsz + 3) & ~3
            if name == b"AMDGPU" and nt == 32:
                return msgpack(desc)[0]
    raise ValueError("no NT_AMDGPU_METADATA note")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("outdir")
    ap.add_argument("--arch", default="gfx1201"); ap.add_argument("--mojo", default="mojo")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.src))[0]
    defs = re.findall(r"^def\s+(\w+)\s*\(", open(a.src).read(), re.M)
    with tempfile.TemporaryDirectory() as td:
        obj = os.path.join(td, stem + ".o")
        r = subprocess.run([a.mojo, "build", "--emit", "object", "--target-accelerator", a.arch,
                            a.src, "-o", obj], capture_output=True, text=True)
        if r.returncode:
            sys.stderr.write(r.stderr); sys.exit(r.returncode)
        elfs = carve_elfs(open(obj, "rb").read())
    manifest = {"arch": a.arch, "source": os.path.basename(a.src), "kernels": []}
    for elf in elfs:
        md = amdgpu_metadata(elf)
        for k in md["amdhsa.kernels"]:
            sym = k[".name"]
            cands = [d for d in defs if sym.startswith(f"{stem}_{d}_")]
            short = max(cands, key=len) if cands else sym
            args = [{"offset": x[".offset"], "size": x[".size"], "kind": x[".value_kind"]}
                    for x in k[".args"]]
            fname = f"{short}.hsaco"
            open(os.path.join(a.outdir, fname), "wb").write(elf)
            manifest["kernels"].append({
                "name": short, "symbol": sym, "hsaco": fname,
                "kernarg_size": k[".kernarg_segment_size"], "kernarg_align": k[".kernarg_segment_align"],
                "explicit_args": [x for x in args if not x["kind"].startswith("hidden_")],
                "hidden_args": [x for x in args if x["kind"].startswith("hidden_")],
                "group_segment": k[".group_segment_fixed_size"], "private_segment": k[".private_segment_fixed_size"],
                "wavefront_size": k[".wavefront_size"], "max_flat_workgroup_size": k[".max_flat_workgroup_size"],
                "vgpr": k[".vgpr_count"], "sgpr": k[".sgpr_count"]})
    json.dump(manifest, open(os.path.join(a.outdir, "manifest.json"), "w"), indent=1)
    for k in manifest["kernels"]:
        print(k["name"], k["hsaco"], "kernarg", k["kernarg_size"], [(x["offset"], x["size"], x["kind"]) for x in k["explicit_args"]])


main()
