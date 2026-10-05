#!/bin/sh
# Phase 4: environment on spark2 (GB10, sm_121, aarch64). Image pinned in README.
#   docker run -d --name mojo-p4 --gpus all --ipc=host --network host -v ~/mojo-p4:/work \
#     nvcr.io/nvidia/pytorch:26.07-py3 sleep infinity
set -e
pip install -q uv
uv venv --system-site-packages --python /usr/bin/python3 /work/venv
uv pip install --python /work/venv/bin/python "max[all]" mojo --extra-index-url https://whl.modular.com/simple/
