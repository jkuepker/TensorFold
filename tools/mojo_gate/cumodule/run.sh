#!/bin/sh
# usage (inside the mojo-p4 container, cwd /work/gate/cumodule; fleetq slot held):
set -e
. /work/venv/bin/activate
python mojo2cubin.py kernels.mojo out
python bench.py out
