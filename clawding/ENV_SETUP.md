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
