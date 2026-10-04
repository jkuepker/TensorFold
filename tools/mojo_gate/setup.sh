#!/bin/sh
# Reproduce the environment inside the container (image pinned in README).
#   docker run -d --name mojo-gate ... -e ROCR_VISIBLE_DEVICES=1 -v ~/mojo-gate:/work --entrypoint sleep <image> infinity
set -e
/opt/vllm/bin/pip install -q uv
uv venv --system-site-packages --python /opt/vllm/bin/python /work/venv
uv pip install --python /work/venv/bin/python "max[all]" mojo --extra-index-url https://whl.modular.com/simple/
# torch lives in the image's /opt/vllm env; expose it to the venv without letting pip touch it
echo /opt/vllm/lib/python3.12/site-packages > /work/venv/lib/python3.12/site-packages/zz_torch.pth
