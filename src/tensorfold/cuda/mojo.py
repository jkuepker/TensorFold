"""Mojo GPU kernels as HIP code objects: ``mojo build --emit object --target-accelerator <arch>`` embeds one complete
AMDGPU ELF per instantiated kernel in the host object; they are carved out (as ``tools/mojo_gate/hipmodule/
mojo2hsaco.py`` does) into ``<kernel>.hsaco`` files plus a ``manifest.json`` (symbol, kernarg layout, registers),
which a C++ extension loads with ``hipModuleLoadData``. Built at first use and cached by the source's hash, the
compiler's version and the GPU architecture, the way ``build.load`` caches the HIP extensions."""

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
import threading
from functools import lru_cache
from pathlib import Path

# bump when the carving, the manifest or the ``mojo build`` flags change: cached builds then rebuild. The cache key
# also holds ``mojo --version`` (its commit too); the ``max`` package the kernels import ships version-locked with it
CARVE_VERSION = 1
# Mojo releases the kernels are built and tested with, [first, past): a release outside may rename the kernels'
# symbols or change their arguments
SUPPORTED = ((1, 1, 0), (1, 2, 0))
INSTALL = ('pip install "mojo>=1.1,<1.2" "max[all]>=26.6,<26.7" --extra-index-url https://whl.modular.com/simple/ '
           "(into this venv), or point TF_MOJO at that mojo binary")


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
        raise RuntimeError(f"a TF_ROCM_*=mojo setting needs the Mojo compiler, and none was found: {INSTALL}")
    return found


@lru_cache(maxsize=1)
def mojo_version() -> str:
    """``mojo --version``'s line, e.g. ``Mojo 1.1.0 (8189361e)``; a clear error when the compiler does not run."""

    binary = mojo_binary()
    try:
        out = subprocess.run([binary, "--version"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", None) or str(exc)
        raise RuntimeError(f"{binary} --version failed: {detail.strip()}") from None
    return out.stdout.strip()


def _range() -> str:
    lo, hi = (".".join(map(str, v)) for v in SUPPORTED)
    return f">= {lo}, < {hi}"


def check_version() -> tuple[int, int, int]:
    """The compiler's version as (major, minor, patch), refused with the supported range when outside it."""

    text = mojo_version()
    found = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", text)
    if found is None:
        raise RuntimeError(f"cannot read a version from {mojo_binary()} --version: {text!r}")
    version = (int(found[1]), int(found[2]), int(found[3] or 0))
    if not SUPPORTED[0] <= version < SUPPORTED[1]:
        raise RuntimeError(f"{text} is not supported: TensorFold's Mojo kernels need Mojo {_range()}; {INSTALL}")
    return version


def kernels(manifest: dict, names: list[str] | tuple[str, ...]) -> list[dict]:
    """``names``' manifest entries in that order; a kernel missing from the build is named with the compiler's version
    (a release that renames symbols shows here)."""

    have = {e["name"]: e for e in manifest["kernels"]}
    missing = [n for n in names if n not in have]
    if missing:
        raise RuntimeError(f"Mojo kernel(s) {', '.join(missing)} missing from {manifest['source']} built with "
                           f"{mojo_version()}: unsupported compiler? TensorFold's Mojo kernels need Mojo {_range()}")
    return [have[n] for n in names]


def arg_sizes(entry: dict) -> list[int]:
    """A manifest entry's explicit argument sizes in bytes, in order (the launchers check them against their own)."""

    return [a["size"] for a in entry["explicit_args"]]


def _build_root() -> Path:
    from torch.utils import cpp_extension

    return Path(os.environ.get("TORCH_EXTENSIONS_DIR") or cpp_extension.get_default_build_root()) / "mojo"


def cache_dir(source: Path, arch: str) -> Path:
    """Where ``source``'s build for ``arch`` lives: keyed by its text, the compiler's version, arch, CARVE_VERSION."""

    source = Path(source)
    key = hashlib.sha256(b"\0".join([source.read_bytes(), mojo_version().encode(), arch.encode(),
                                      str(CARVE_VERSION).encode()]))
    return _build_root() / f"{source.stem}-{arch}-{key.hexdigest()[:16]}"


def build_hsaco(source: Path, arch: str) -> tuple[Path, dict]:
    """``source``'s kernels as .hsaco files for ``arch``: (directory, manifest), built once per source hash."""

    source = Path(source)
    check_version()
    out = cache_dir(source, arch)
    manifest = out / "manifest.json"
    if manifest.is_file():
        return out, json.loads(manifest.read_text())
    from torch.utils.file_baton import FileBaton

    from .build import HINT, LOCK_WAIT_SECONDS, _say, _still_waiting

    out.parent.mkdir(parents=True, exist_ok=True)
    lock = str(out) + ".lock"
    baton = FileBaton(lock)
    if baton.try_acquire():
        try:
            if not manifest.is_file():
                _say(f"building Mojo kernels {source.name} for {arch} (first use; later starts reuse them)")
                _build(source, arch, out)
        finally:
            baton.release()
    else:
        timer = None
        try:
            seen = os.stat(lock)
        except OSError:                                 # released between the two calls: nothing to wait on
            seen = None
        if seen is not None:
            _say(f"Mojo kernels {source.name} wait on the build lock {lock}; {HINT}")
            timer = threading.Timer(LOCK_WAIT_SECONDS, _still_waiting, (lock, (seen.st_ino, seen.st_mtime_ns)))
            timer.daemon = True
            timer.start()
        try:
            baton.wait()
        finally:
            if timer is not None:
                timer.cancel()
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
