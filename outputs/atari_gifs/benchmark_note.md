# Quick benchmark note (2026-02-12)

Host: llvmpipe (CPU raster), offscreen Panda3D, `num_scenes=32`, `render=True`.

Measured over 80 timed steps after 10 warmup steps:

- PacMan-v0: 22.22 steps/s (~711.11 frames/s)
- PingPong-v0: 200.33 steps/s (~6410.54 frames/s)
- Breakout-v0: 101.49 steps/s (~3247.72 frames/s)
- SpaceInvaders-v0: 85.26 steps/s (~2728.46 frames/s)
- Asteroids-v0: 195.91 steps/s (~6268.96 frames/s)
- Freeway-v0: 141.42 steps/s (~4525.58 frames/s)

Note: absolute numbers depend on software rasterizer / host, but smoke-level throughput remains strong after the 2D sprite migration.
