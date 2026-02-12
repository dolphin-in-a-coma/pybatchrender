# PyBatchRender

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dolphin-in-a-coma/pybatchrender/blob/main/examples/notebooks/cartpole_benchmark.ipynb)

Fast batch 3D rendering for TorchRL environments using Panda3D. Try the **CartPole demo** in Colab to get started.

### Installation

```bash
pip install git+https://github.com/dolphin-in-a-coma/pybatchrender.git
```

From source:
```bash
pip install -e .
```

Optional extras:

```bash
pip install "pybatchrender[images]"  # For saving frames
pip install "pybatchrender[cuda]"    # For CUDA interop (Linux/Windows with NVIDIA)
```

### Quickstart

With pre-made environments:
```python
import pybatchrender as pbr

env = pbr.envs.make("CartPole-v0", num_scenes=256)
td = env.reset()

for step in range(100):
    td["action"] = env.action_spec.rand()
    td = env.step(td)
    td = td["next"]
```

Available built-ins include:
- `CartPole-v0`
- `PacMan-v0`
- `PingPong-v0` (Pong-inspired)
- `Breakout-v0`
- `SpaceInvaders-v0`
- `Asteroids-v0`
- `Freeway-v0`

Example for Atari-inspired envs:
```python
import pybatchrender as pbr

env = pbr.envs.make("Breakout-v0", num_scenes=128, render=True, offscreen=True)
td = env.reset()
for _ in range(50):
    td["action"] = env.action_spec.rand()
    td = env.step(td)["next"]
```

Render simple scene:
```python
from pybatchrender import PBRConfig, PBRRenderer

cfg = PBRConfig(num_scenes=4, tile_resolution=(64, 64), offscreen=True)
renderer = PBRRenderer(cfg)

renderer.add_camera()
renderer.add_light()
renderer.add_node('models/box', instances_per_scene=1, shared_across_scenes=True)
renderer.setup_environment()

img_batch = renderer.step(return_pixels=True)
print(img_batch.shape)  # torch.Size([4, 3, 64, 64])
```

### License
MIT
