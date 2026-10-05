"""cuda/mojo.py without a GPU or a Mojo compiler: a fake ``TF_MOJO`` script stands in for ``mojo``, so the build cache,
its lock, the version pin and the code-object carving run on any host."""

import json
import stat
import struct
import threading
import time
from pathlib import Path

import pytest

from tensorfold.cuda import build, mojo

pytestmark = pytest.mark.torch

SOURCE = """def wmma_mt1(x: Int32):
    pass


def wmma_mt2(x: Int32):
    pass
"""


def _fake_mojo(tmp_path: Path, version: str = "Mojo 1.1.0 (8189361e)", fails: bool = False) -> Path:
    """A ``mojo`` that prints ``version`` and, for ``build ... -o OUT``, logs the call and writes OUT."""

    log = tmp_path / "calls.log"
    script = tmp_path / "mojo"
    answer = "echo 'mojo: broken install' >&2; exit 3" if fails else f"echo '{version}'"
    script.write_text(f"""#!/bin/sh
if [ "$1" = "--version" ]; then
    {answer}
    exit 0
fi
echo "$@" >> "{log}"
while [ $# -gt 1 ]; do
    if [ "$1" = "-o" ]; then printf 'object' > "$2"; fi
    shift
done
""")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """A fake compiler, an empty extensions dir and a carve that writes two kernels."""

    monkeypatch.setenv("TF_MOJO", str(_fake_mojo(tmp_path)))
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path / "ext"))

    def carve(blob, source, arch, outdir):
        assert blob == b"object"
        kernels = []
        for name in ("wmma_mt1", "wmma_mt2"):
            (outdir / f"{name}.hsaco").write_bytes(name.encode())
            kernels.append({"name": name, "symbol": f"qmm_{name}_x", "hsaco": f"{name}.hsaco",
                            "explicit_args": [{"offset": 0, "size": 8, "kind": "global_buffer"}]})
        return {"arch": arch, "source": source.name, "kernels": kernels}

    monkeypatch.setattr(mojo, "carve", carve)
    for f in (mojo.mojo_binary, mojo.mojo_version):
        f.cache_clear()
    source = tmp_path / "qmm.mojo"
    source.write_text(SOURCE)
    yield source, tmp_path / "calls.log"
    for f in (mojo.mojo_binary, mojo.mojo_version):
        f.cache_clear()


def _builds(log: Path) -> int:
    return len(log.read_text().splitlines()) if log.exists() else 0


def test_a_build_is_cached_by_its_source(fake, capsys):
    source, log = fake
    out, manifest = mojo.build_hsaco(source, "gfx1201")
    assert _builds(log) == 1 and "building Mojo kernels qmm.mojo for gfx1201" in capsys.readouterr().out
    assert [e["name"] for e in manifest["kernels"]] == ["wmma_mt1", "wmma_mt2"]
    assert (out / "wmma_mt2.hsaco").read_bytes() == b"wmma_mt2" and out == mojo.cache_dir(source, "gfx1201")
    assert mojo.build_hsaco(source, "gfx1201") == (out, manifest) and _builds(log) == 1     # a cache hit
    assert "building" not in capsys.readouterr().out
    mojo.build_hsaco(source, "gfx1100")                                                      # another arch
    source.write_text(SOURCE + "\n# changed\n")
    again, _ = mojo.build_hsaco(source, "gfx1201")                                            # another source
    assert _builds(log) == 3 and again != out
    assert "--target-accelerator gfx1201" in log.read_text().splitlines()[0]


def test_a_start_waits_on_another_starts_build_and_says_so(fake, monkeypatch, capsys):
    source, log = fake
    monkeypatch.setattr(build, "LOCK_WAIT_SECONDS", 0.2)
    out = mojo.cache_dir(source, "gfx1201")
    out.parent.mkdir(parents=True)
    lock = Path(str(out) + ".lock")
    lock.write_text("")                                     # another start holds the build
    got = {}
    waiter = threading.Thread(target=lambda: got.update(r=mojo.build_hsaco(source, "gfx1201")))
    waiter.start()
    time.sleep(0.6)
    assert waiter.is_alive()
    out.mkdir()                                             # the other start finishes
    (out / "manifest.json").write_text(json.dumps({"source": "qmm.mojo", "kernels": []}))
    lock.unlink()
    waiter.join(5)
    text = capsys.readouterr().out
    assert got["r"][0] == out and _builds(log) == 0
    assert f"wait on the build lock {lock}" in text and build.HINT in text and "still waiting after 0.2 s" in text


def test_a_build_that_died_elsewhere_is_named(fake):
    source, _ = fake
    out = mojo.cache_dir(source, "gfx1201")
    out.parent.mkdir(parents=True)
    lock = Path(str(out) + ".lock")
    lock.write_text("")
    threading.Timer(0.3, lock.unlink).start()               # its lock goes, no manifest
    with pytest.raises(RuntimeError, match="failed in another process"):
        mojo.build_hsaco(source, "gfx1201")


@pytest.mark.parametrize("version, ok", [("Mojo 1.1.0 (8189361e)", True), ("Mojo 1.1.3 (abc)", True),
                                         ("Mojo 1.2.0 (abc)", False), ("Mojo 1.0.9 (abc)", False),
                                         ("Mojo 26.6.0", False), ("mojo, unversioned", False)])
def test_the_compiler_version_is_pinned(fake, tmp_path, monkeypatch, version, ok):
    source, log = fake
    monkeypatch.setenv("TF_MOJO", str(_fake_mojo(tmp_path, version)))
    if ok:
        mojo.build_hsaco(source, "gfx1201")
        assert _builds(log) == 1
        return
    with pytest.raises(RuntimeError, match="not supported|cannot read a version"):
        mojo.build_hsaco(source, "gfx1201")
    assert _builds(log) == 0


def test_a_compiler_that_does_not_run_says_why(fake, tmp_path, monkeypatch):
    source, _ = fake
    monkeypatch.setenv("TF_MOJO", str(_fake_mojo(tmp_path, fails=True)))
    with pytest.raises(RuntimeError, match="--version failed: mojo: broken install"):
        mojo.build_hsaco(source, "gfx1201")


def test_a_missing_kernel_is_named_with_the_compiler(fake):
    source, _ = fake
    _, manifest = mojo.build_hsaco(source, "gfx1201")
    assert [e["symbol"] for e in mojo.kernels(manifest, ("wmma_mt2", "wmma_mt1"))] == ["qmm_wmma_mt2_x",
                                                                                        "qmm_wmma_mt1_x"]
    with pytest.raises(RuntimeError, match=r"reduce_kernel missing from qmm.mojo built with Mojo 1.1.0"):
        mojo.kernels(manifest, ("wmma_mt1", "reduce_kernel"))
    assert mojo.arg_sizes(manifest["kernels"][0]) == [8]


# ------------------------------------------------------------------------------------------------- carving
def _pack(x) -> bytes:
    """The msgpack subset the AMDGPU metadata note uses."""

    if isinstance(x, bool):
        return b"\xc3" if x else b"\xc2"
    if isinstance(x, int):
        return bytes([x]) if 0 <= x <= 0x7F else b"\xce" + struct.pack(">I", x)
    if isinstance(x, str):
        b = x.encode()
        return (bytes([0xA0 | len(b)]) if len(b) < 32 else b"\xd9" + bytes([len(b)])) + b
    if isinstance(x, list):
        return bytes([0x90 | len(x)]) + b"".join(_pack(v) for v in x)
    return bytes([0x80 | len(x)]) + b"".join(_pack(k) + _pack(v) for k, v in x.items())


def _elf(machine: int, kernels: list[dict]) -> bytes:
    """A minimal ELF: the header, one NT_AMDGPU_METADATA note, a null and a note section header."""

    desc = _pack({"amdhsa.kernels": kernels})
    name = b"AMDGPU\0\0"
    note = struct.pack("<III", 7, len(desc), 32) + name + desc + b"\0" * (-len(desc) % 4)
    shoff = 64 + len(note)
    head = bytearray(64)
    head[:4] = b"\x7fELF"
    struct.pack_into("<H", head, 18, machine)
    struct.pack_into("<Q", head, 0x28, shoff)
    struct.pack_into("<HH", head, 0x3A, 64, 2)
    null = bytes(64)
    sh = bytearray(64)
    struct.pack_into("<I", sh, 4, 7)                          # SHT_NOTE
    struct.pack_into("<QQ", sh, 0x18, 64, len(note))
    return bytes(head) + note + null + bytes(sh)


def _kernel(symbol: str) -> dict:
    return {".name": symbol, ".kernarg_segment_size": 272, ".kernarg_segment_align": 8,
            ".args": [{".offset": 0, ".size": 8, ".value_kind": "global_buffer"},
                      {".offset": 8, ".size": 4, ".value_kind": "by_value"},
                      {".offset": 16, ".size": 8, ".value_kind": "hidden_block_count_x"}],
            ".group_segment_fixed_size": 1024, ".private_segment_fixed_size": 0, ".wavefront_size": 32,
            ".max_flat_workgroup_size": 256, ".vgpr_count": 96, ".sgpr_count": 40}


def test_carving_finds_each_amdgpu_kernel_in_the_host_object(tmp_path):
    source = tmp_path / "qmm_rocm.mojo"
    source.write_text(SOURCE + "\ndef wmma(x: Int32):\n    pass\n")
    one = _elf(224, [_kernel("qmm_rocm_wmma_mt1_6a1f")])
    two = _elf(224, [_kernel("qmm_rocm_wmma_mt2_b2c3")])
    host = _elf(62, [_kernel("host_code")])                   # x86-64: not a GPU code object
    blob = b"junk" + host + b"\0" * 13 + one + b"pad" + two + b"tail"
    out = tmp_path / "out"
    out.mkdir()
    manifest = mojo.carve(blob, source, "gfx1201", out)
    names = [(e["name"], e["symbol"]) for e in manifest["kernels"]]
    assert names == [("wmma_mt1", "qmm_rocm_wmma_mt1_6a1f"), ("wmma_mt2", "qmm_rocm_wmma_mt2_b2c3")]
    first = manifest["kernels"][0]
    assert (out / "wmma_mt1.hsaco").read_bytes() == one and (out / "wmma_mt2.hsaco").read_bytes() == two
    assert [a["size"] for a in first["explicit_args"]] == [8, 4] and len(first["hidden_args"]) == 1
    assert (first["vgpr"], first["group_segment"], first["kernarg_size"]) == (96, 1024, 272)
    with pytest.raises(RuntimeError, match="two kernels named wmma_mt1"):
        mojo.carve(one + one, source, "gfx1201", out)
