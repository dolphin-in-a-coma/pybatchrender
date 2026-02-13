# PyBatchRender Environment Setup (Twin's instructions)

This guide sets up a reliable local environment for **PyBatchRender** on Ubuntu/Linux.

## 1) Clone and enter repo

```bash
git clone https://github.com/dolphin-in-a-coma/pybatchrender.git
cd pybatchrender
```

## 2) Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

## 3) Install PyBatchRender (editable)

```bash
pip install -e .
```

## 4) Install CPU-only PyTorch (recommended on non-GPU VPS)

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
```

> Why: avoids downloading huge CUDA wheels that often fail on small servers.

## 5) Install minimal runtime libs for headless rendering (Ubuntu)

```bash
sudo apt-get update
sudo apt-get install -y libgl1 libegl1
```

## 6) Quick sanity checks

```bash
python - <<'PY'
import torch, pybatchrender
print('torch:', torch.__version__)
print('cuda_available:', torch.cuda.is_available())
print('pybatchrender import: OK')
PY
```

## 7) Run CartPole test and save images

```bash
python examples/scripts/cartpole_benchmark.py \
  --num-scenes 16 \
  --steps 20 \
  --save-every 10 \
  --save-num 8 \
  --save-dir ./outputs
```

Expected result: PNG images appear in `./outputs`.

---

## Optional: log everything to a file

```bash
python examples/scripts/cartpole_benchmark.py --num-scenes 16 --steps 20 2>&1 | tee -a logs/cartpole-test.log
```

## Optional: GPU/CUDA setup

Only do this if machine has an NVIDIA GPU + drivers. Otherwise stick to CPU-only setup.

### Puhti (CSC) recommended setup (no sudo)

On Puhti, **prefer modules** (fast, supported) instead of trying to install system packages.

In SLURM jobs and non-interactive shells you should initialize the module environment explicitly:

```bash
source /appl/profile/zz-csc-env.sh
module purge
module load pytorch/2.7
```

This provides a working Python + CUDA-enabled PyTorch stack on V100 nodes.

#### Minimal sanity checks

```bash
python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
PY

nvidia-smi
```

#### Panda3D headless renderer check (EGL)

If you need to confirm that offscreen rendering is actually using the GPU, you can install Panda3D in a venv and print the driver strings:

```bash
python -m venv venv-panda3d
source venv-panda3d/bin/activate
pip install -U pip
pip install panda3d==1.10.14

python - <<'PY'
import panda3d
from panda3d.core import loadPrcFileData

loadPrcFileData('', 'window-type offscreen')
loadPrcFileData('', 'audio-library-name null')
loadPrcFileData('', 'load-display pandagl')

from direct.showbase.ShowBase import ShowBase
base = ShowBase(windowType='offscreen')

pipe = base.pipe
print('Pipe:', pipe.getType().getName() if pipe else None)

gsg = base.win.getGsg()
print('Renderer:', gsg.getDriverRenderer())
print('Vendor:', gsg.getDriverVendor())
print('Version:', gsg.getDriverVersion())

base.destroy()
PY
```

Expected on a V100 node: `Pipe: eglGraphicsPipe` and `Renderer: Tesla V100-...`.

### Puhti + Apptainer note

If you run inside Apptainer containers on Puhti, use `--nv` to expose GPU driver libraries into the container:

```bash
apptainer exec --nv myimage.sif python -c "import torch; print(torch.cuda.is_available())"
```
