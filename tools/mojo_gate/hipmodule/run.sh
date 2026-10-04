#!/bin/sh
# usage (inside the mojo container, cwd /work/mojo_gate/hipmodule; fleetq slot held, card B):
#   ../run.sh ./run.sh      (the gate's run.sh sets PATH/LD_LIBRARY_PATH)
set -e
python mojo2hsaco.py kernels.mojo out
python bench.py out
