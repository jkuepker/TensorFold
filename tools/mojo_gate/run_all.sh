#!/bin/sh
# Run every gate check inside the container: cd /work/mojo_gate && ./run.sh ./run_all.sh
set -x
python -c "import torch,max;p=torch.cuda.get_device_properties(0);print(torch.__version__,torch.cuda.is_available(),torch.cuda.device_count(),p.pci_bus_id,p.gcnArchName)"
mojo --version
mojo run vecadd.mojo
python pathA.py
mojo build --emit shared-lib pathB.mojo -o libgate.so && python pathB.py
mojo build --emit shared-lib intr.mojo -o libintr.so && python intr.py
./isa_dump.sh
mojo build --emit shared-lib bw.mojo -o libbw.so && python bw.py
