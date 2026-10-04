#!/bin/sh
# usage: run.sh <cmd...>  (inside container, cwd /work/mojo_gate)
export LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib
export PATH=/work/venv/bin:$PATH
exec "$@"
