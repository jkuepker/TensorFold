#!/bin/sh
# usage (inside the mojo-p4 container, cwd /work/gate/spark; fleetq slot held): ./run_all.sh 2>&1 | tee results.txt
set -e
. /work/venv/bin/activate
mojo --version
python -c "import max, torch; print('max', max.__version__ if hasattr(max, '__version__') else '?', 'torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
echo "== check 2: vecadd"; mojo vecadd.mojo 2>&1 | grep -E "^(device|mismatches|PASS|FAIL)"
echo "== check 3: launch path"; (cd ../cumodule && ./run.sh 2>&1 | grep -v -E "warning|~|\^|note: |Pointer|\[i\]|^ *$")
echo "== check 4: intrinsics"
for f in intr fp8 fp4; do mojo build --emit shared-lib --target-accelerator sm_121 $f.mojo -o lib$f.so 2>&1 | grep -E "error" || true; done
python intr.py; python fp8.py; python fp4.py
for f in intr fp8 fp4; do ./sass.sh $f.mojo 2>&1 | grep -v -E "warning|deprecated"; done
echo "== check 5: bandwidth"
mojo build --emit shared-lib --target-accelerator sm_121 bw.mojo -o libbw.so 2>&1 | grep -E "error" || true
python bw.py
