#!/bin/sh
# usage (in container): ./sass.sh intr.mojo  -> per-kernel PTX + cubin + SASS under sass/, then grep the instructions of interest
set -e
. /work/venv/bin/activate
rm -rf sass; python ../cumodule/mojo2cubin.py "$1" sass > /dev/null
for c in sass/*.cubin; do cuobjdump -sass "$c" > "${c%.cubin}.sass"; done
echo "--- PTX"; grep -H -o -E "mma\.sync[a-z0-9_.]*|ldmatrix[a-z0-9_.]*|cp\.async[a-z0-9_.]*|mma\.[a-z0-9_.:]*block_scale[a-z0-9_.:]*" sass/*.ptx | sort | uniq -c
echo "--- SASS"; grep -H -o -E "\b(HMMA|QMMA|OMMA|LDSM|LDGSTS|LDGDEPBAR|DEPBAR)[A-Z0-9_.]*" sass/*.sass | sort | uniq -c
