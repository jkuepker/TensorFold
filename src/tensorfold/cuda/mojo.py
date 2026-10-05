"""Mojo GPU kernels as HIP code objects or CUDA cubins: ``mojo build --emit object --target-accelerator <arch>``
embeds one complete AMDGPU ELF per instantiated kernel in the host object; they are carved out (as
``tools/mojo_gate/hipmodule/mojo2hsaco.py`` does) into ``<kernel>.hsaco`` files plus a ``manifest.json`` (symbol,
kernarg layout, registers), which a C++ extension loads with ``hipModuleLoadData``. On NVIDIA the object embeds one
PTX module per kernel instead; ``build_cubin`` carves those (as ``tools/mojo_gate/cumodule/mojo2cubin.py`` does) and
runs ``ptxas`` on each, for ``cuModuleLoadData``. Built at first use and cached by the source's hash, the compiler's
version and the GPU architecture, the way ``build.load`` caches the HIP extensions."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

CARVE_VERSION = 1              # bump when the carving or the manifest changes: cached builds then rebuild
INSTALL = ("pip install mojo --extra-index-url https://whl.modular.com/simple/ (into this venv), or point TF_MOJO at "
           "the mojo binary")


@lru_cache(maxsize=1)
def mojo_binary() -> str:
    """The Mojo compiler: ``TF_MOJO``, else ``mojo`` on PATH, else beside this Python; a clear error without one."""

    named = os.environ.get("TF_MOJO")
    if named:
        if not (os.path.isfile(named) and os.access(named, os.X_OK)):
            raise RuntimeError(f"TF_MOJO={named} is not an executable Mojo compiler")
        return named
    found = shutil.which("mojo") or str(Path(sys.executable).parent / "mojo")
    if not os.access(found, os.X_OK):
        raise RuntimeError(f"a Mojo kernel (TF_ROCM_LANE=mojo, TF_CUDA_PREFILL_GEMM=mojo) needs the Mojo compiler, "
                           f"and none was found: {INSTALL}")
    return found


@lru_cache(maxsize=1)
def mojo_version() -> str:
    out = subprocess.run([mojo_binary(), "--version"], capture_output=True, text=True, check=True)
    return out.stdout.strip()


def _build_root() -> Path:
    from torch.utils import cpp_extension

    return Path(os.environ.get("TORCH_EXTENSIONS_DIR") or cpp_extension.get_default_build_root()) / "mojo"


def build_hsaco(source: Path, arch: str) -> tuple[Path, dict]:
    """``source``'s kernels as .hsaco files for ``arch``: (directory, manifest), built once per source hash."""

    return _cached(Path(source), arch, b"", _build)


def build_cubin(source: Path, arch: str) -> tuple[Path, dict]:
    """``source``'s kernels as .cubin files for ``arch`` (``sm_121``): (directory, manifest), built once per source
    hash, compiler and ptxas."""

    return _cached(Path(source), arch, ptxas_version().encode(), _build_cubin)


def _cached(source: Path, arch: str, extra: bytes, build) -> tuple[Path, dict]:
    text = source.read_bytes()
    parts = [text, mojo_version().encode(), arch.encode(), str(CARVE_VERSION).encode()] + ([extra] if extra else [])
    key = hashlib.sha256(b"\0".join(parts))
    out = _build_root() / f"{source.stem}-{arch}-{key.hexdigest()[:16]}"
    manifest = out / "manifest.json"
    if manifest.is_file():
        return out, json.loads(manifest.read_text())
    from torch.utils.file_baton import FileBaton

    out.parent.mkdir(parents=True, exist_ok=True)
    baton = FileBaton(str(out) + ".lock")
    if baton.try_acquire():
        try:
            if not manifest.is_file():
                print(f"building Mojo kernels {source.name} for {arch} (first use; later starts reuse them)",
                      file=sys.stderr, flush=True)
                build(source, arch, out)
        finally:
            baton.release()
    else:
        baton.wait()
    if not manifest.is_file():
        raise RuntimeError(f"building {source.name} with Mojo failed in another process; see its output")
    return out, json.loads(manifest.read_text())


def _build(source: Path, arch: str, out: Path) -> None:
    with tempfile.TemporaryDirectory(dir=out.parent) as td:
        obj = Path(td) / (source.stem + ".o")
        r = subprocess.run([mojo_binary(), "build", "--emit", "object", "--target-accelerator", arch, str(source),
                            "-o", str(obj)], capture_output=True, text=True, check=False)
        if r.returncode:
            raise RuntimeError(f"mojo build {source.name} failed:\n{r.stderr}")
        stage = Path(td) / "out"
        stage.mkdir()
        manifest = carve(obj.read_bytes(), source, arch, stage)
        if not manifest["kernels"]:
            raise RuntimeError(f"mojo build {source.name}: no AMDGPU kernels found (instantiate them from a host def)")
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=1))
        if out.exists():
            shutil.rmtree(out)
        stage.rename(out)


@lru_cache(maxsize=1)
def ptxas() -> str:
    """CUDA's ptxas: on PATH, else under the toolkit torch builds extensions with."""

    found = shutil.which("ptxas")
    if found:
        return found
    from torch.utils import cpp_extension

    home = cpp_extension.CUDA_HOME or "/usr/local/cuda"
    found = str(Path(home) / "bin" / "ptxas")
    if not os.access(found, os.X_OK):
        raise RuntimeError("the Mojo CUDA kernels need ptxas (the CUDA toolkit); none was found")
    return found


@lru_cache(maxsize=1)
def ptxas_version() -> str:
    out = subprocess.run([ptxas(), "--version"], capture_output=True, text=True, check=True)
    return out.stdout.strip()


def _build_cubin(source: Path, arch: str, out: Path) -> None:
    with tempfile.TemporaryDirectory(dir=out.parent) as td:
        obj = Path(td) / (source.stem + ".o")
        r = subprocess.run([mojo_binary(), "build", "--emit", "object", "--target-accelerator", arch, str(source),
                            "-o", str(obj)], capture_output=True, text=True, check=False)
        if r.returncode:
            raise RuntimeError(f"mojo build {source.name} failed:\n{r.stderr}")
        stage = Path(td) / "out"
        stage.mkdir()
        manifest = carve_cubins(obj.read_bytes(), source, arch, stage)
        if not manifest["kernels"]:
            raise RuntimeError(f"mojo build {source.name}: no PTX kernels found (instantiate them from a host def)")
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=1))
        if out.exists():
            shutil.rmtree(out)
        stage.rename(out)


_PTX_SIZES = {"u8": 1, "s8": 1, "b8": 1, "u16": 2, "s16": 2, "b16": 2, "f16": 2, "u32": 4, "s32": 4, "b32": 4,
              "f32": 4, "u64": 8, "s64": 8, "b64": 8, "f64": 8}


def carve_cubins(blob: bytes, source: Path, arch: str, outdir: Path) -> dict:
    """Write each embedded PTX module and its ptxas cubin to ``outdir`` and return the manifest (mojo2cubin.py's)."""

    stem = source.stem
    defs = re.findall(r"^def\s+(\w+)\s*[\[(]", source.read_text(), re.MULTILINE)
    manifest: dict = {"arch": arch, "source": source.name, "kernels": []}
    for m in re.finditer(rb"\.version [0-9.]+\n\.target sm_\w+\n", blob):
        end = blob.find(b"\0", m.start())
        ptx = blob[m.start():end if end >= 0 else len(blob)].decode()
        target = re.search(r"\.target (\w+)", ptx).group(1)
        for sym, params in ptx_entries(ptx):
            cands = [d for d in defs if sym.startswith(f"{stem}_{d}_")]
            short = max(cands, key=len) if cands else sym
            if any(e["name"] == short for e in manifest["kernels"]):
                raise RuntimeError(f"{source.name}: two kernels named {short} (one top-level def per instantiation)")
            (outdir / f"{short}.ptx").write_text(ptx)
            r = subprocess.run([ptxas(), f"-arch={target}", str(outdir / f"{short}.ptx"), "-o",
                                str(outdir / f"{short}.cubin")], capture_output=True, text=True, check=False)
            if r.returncode:
                raise RuntimeError(f"ptxas {short} ({source.name}) failed:\n{r.stderr}")
            manifest["kernels"].append({"name": short, "symbol": sym, "ptx": f"{short}.ptx", "cubin": f"{short}.cubin",
                                        "target": target, "params": params})
    return manifest


def ptx_entries(ptx: str) -> list[tuple[str, list[dict]]]:
    """[(symbol, [param])] for each ``.entry`` of a PTX module: index, type, offset, size, pointer or not."""

    res = []
    for m in re.finditer(r"\.entry\s+(\w+)\s*\((.*?)\)\s*\n\s*(?:\.\w+[^\n{]*\n\s*)*\{", ptx, re.S):
        params, off = [], 0
        for i, line in enumerate(x.strip().rstrip(",") for x in m.group(2).split("\n") if x.strip()):
            pm = re.match(r"\.param\s+(.*?)\s+(\w+)(?:\[(\d+)\])?$", line)
            if not pm:
                raise ValueError(f"cannot parse PTX param: {line!r}")
            attrs, _, count = pm.groups()
            ty = re.search(r"\.(u8|s8|b8|u16|s16|b16|f16|u32|s32|b32|f32|u64|s64|b64|f64)\b", attrs).group(1)
            al = re.search(r"\.align\s+(\d+)", attrs)
            elem = _PTX_SIZES[ty]
            size = elem * int(count) if count else elem
            align = max(int(al.group(1)) if al and count else 1, elem)
            off = (off + align - 1) // align * align
            params.append({"index": i, "type": ty, "offset": off, "size": size, "ptr": ".ptr" in attrs})
            off += size
        res.append((m.group(1), params))
    return res


def carve(blob: bytes, source: Path, arch: str, outdir: Path) -> dict:
    """Write each embedded kernel's code object to ``outdir`` and return the manifest (mojo2hsaco.py's)."""

    stem = source.stem
    defs = re.findall(r"^def\s+(\w+)\s*[\[(]", source.read_text(), re.MULTILINE)
    manifest: dict = {"arch": arch, "source": source.name, "kernels": []}
    for elf in carve_elfs(blob):
        md = amdgpu_metadata(elf)
        for k in md["amdhsa.kernels"]:
            sym = k[".name"]
            cands = [d for d in defs if sym.startswith(f"{stem}_{d}_")]
            short = max(cands, key=len) if cands else sym
            if any(e["name"] == short for e in manifest["kernels"]):
                raise RuntimeError(f"{source.name}: two kernels named {short} (one top-level def per instantiation)")
            args = [{"offset": x[".offset"], "size": x[".size"], "kind": x[".value_kind"]} for x in k[".args"]]
            fname = f"{short}.hsaco"
            (outdir / fname).write_bytes(elf)
            manifest["kernels"].append({
                "name": short, "symbol": sym, "hsaco": fname,
                "kernarg_size": k[".kernarg_segment_size"], "kernarg_align": k[".kernarg_segment_align"],
                "explicit_args": [x for x in args if not x["kind"].startswith("hidden_")],
                "hidden_args": [x for x in args if x["kind"].startswith("hidden_")],
                "group_segment": k[".group_segment_fixed_size"], "private_segment": k[".private_segment_fixed_size"],
                "wavefront_size": k[".wavefront_size"], "max_flat_workgroup_size": k[".max_flat_workgroup_size"],
                "vgpr": k[".vgpr_count"], "sgpr": k[".sgpr_count"]})
    return manifest


def carve_elfs(blob: bytes) -> list[bytes]:
    """Every AMDGPU ELF (e_machine 224) embedded in ``blob``."""

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
            if typ != 8:                                # SHT_NOBITS occupies no file space
                end = max(end, off + size)
        out.append(blob[o:o + end])
    return out


def amdgpu_metadata(elf: bytes) -> dict:
    shoff, = struct.unpack_from("<Q", elf, 0x28)
    shentsize, shnum = struct.unpack_from("<HH", elf, 0x3A)
    for i in range(shnum):
        sh = shoff + i * shentsize
        typ, = struct.unpack_from("<I", elf, sh + 4)
        if typ != 7:                                    # SHT_NOTE
            continue
        off, size = struct.unpack_from("<QQ", elf, sh + 0x18)
        p = off
        while p < off + size:
            nsz, dsz, nt = struct.unpack_from("<III", elf, p)
            p += 12
            name = elf[p:p + nsz].rstrip(b"\0")
            p += (nsz + 3) & ~3
            desc = elf[p:p + dsz]
            p += (dsz + 3) & ~3
            if name == b"AMDGPU" and nt == 32:
                return _msgpack(desc)[0]
    raise ValueError("no NT_AMDGPU_METADATA note")


def _msgpack(b: bytes, i: int = 0):
    """A minimal msgpack decoder (what the AMDGPU metadata note uses)."""

    t = b[i]
    i += 1
    if t <= 0x7F:
        return t, i
    if t >= 0xE0:
        return t - 256, i
    if 0xA0 <= t <= 0xBF:
        n = t & 31
        return b[i:i + n].decode(), i + n
    if 0x90 <= t <= 0x9F:
        return _array(b, i, t & 15)
    if 0x80 <= t <= 0x8F:
        return _map(b, i, t & 15)
    if t == 0xC0:
        return None, i
    if t in (0xC2, 0xC3):
        return t == 0xC3, i
    for code, fmt, n in ((0xCC, ">B", 1), (0xCD, ">H", 2), (0xCE, ">I", 4), (0xCF, ">Q", 8),
                         (0xD0, ">b", 1), (0xD1, ">h", 2), (0xD2, ">i", 4), (0xD3, ">q", 8)):
        if t == code:
            return struct.unpack_from(fmt, b, i)[0], i + n
    if t in (0xD9, 0xDA, 0xDB):
        w = {0xD9: 1, 0xDA: 2, 0xDB: 4}[t]
        n = int.from_bytes(b[i:i + w], "big")
        i += w
        return b[i:i + n].decode(), i + n
    if t in (0xDC, 0xDD):
        w = 2 if t == 0xDC else 4
        return _array(b, i + w, int.from_bytes(b[i:i + w], "big"))
    if t in (0xDE, 0xDF):
        w = 2 if t == 0xDE else 4
        return _map(b, i + w, int.from_bytes(b[i:i + w], "big"))
    raise ValueError(f"msgpack type {t:#x}")


def _array(b: bytes, i: int, n: int):
    r = []
    for _ in range(n):
        v, i = _msgpack(b, i)
        r.append(v)
    return r, i


def _map(b: bytes, i: int, n: int):
    r = {}
    for _ in range(n):
        k, i = _msgpack(b, i)
        v, i = _msgpack(b, i)
        r[k] = v
    return r, i
