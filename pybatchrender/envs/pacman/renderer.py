"""Renderer for Pac-Man-like 2D environment."""
import torch

from ...config import PBRConfig
from ...renderer.renderer import PBRRenderer


class PacManRenderer(PBRRenderer):
    def __init__(self, cfg: PBRConfig | dict | None = None):
        super().__init__(cfg)

        self.cell = 1.0
        self.base_z = 0.2

        # Simple 2D maze map with walls as squares
        self.map_rows = [
            "###########",
            "#.........#",
            "#.###.###.#",
            "#.#.....#.#",
            "#.#.###.#.#",
            "#...#.#...#",
            "###.#.#.###",
            "#...#.#...#",
            "#.###.###.#",
            "#.........#",
            "###########",
        ]
        self.H = len(self.map_rows)
        self.W = len(self.map_rows[0])

        # Precompute world-space coordinates
        self._wall_coords: list[tuple[float, float]] = []
        self._walkable_coords: list[tuple[float, float]] = []
        for y, row in enumerate(self.map_rows):
            for x, c in enumerate(row):
                wx, wy = self._to_world_xy(x, y)
                if c == "#":
                    self._wall_coords.append((wx, wy))
                else:
                    self._walkable_coords.append((wx, wy))

        # Place collectibles in all walkable cells by convention.
        # Actual alive/dead state is controlled by env step payload.
        self.collectible_coords = list(self._walkable_coords)

        # Nodes
        num_scenes = int(self.cfg.num_scenes)
        n_walls = len(self._wall_coords)
        n_collect = len(self.collectible_coords)

        self.walls = self.add_node(
            "models/box",
            model_pivot_relative_point=(0.5, 0.5, 0.5),
            model_scale=(self.cell * 0.95, self.cell * 0.95, 0.2),
            instances_per_scene=n_walls,
            shared_across_scenes=True,
        )
        self.pacman = self.add_node(
            "models/smiley",
            model_scale=(0.55, 0.55, 0.12),
            model_scale_units="absolute",
            instances_per_scene=1,
            shared_across_scenes=False,
        )
        self.ghost = self.add_node(
            "models/smiley",
            model_scale=(0.55, 0.55, 0.12),
            model_scale_units="absolute",
            instances_per_scene=1,
            shared_across_scenes=False,
        )
        self.pellets = self.add_node(
            "models/smiley",
            model_scale=(0.16, 0.16, 0.05),
            model_scale_units="absolute",
            instances_per_scene=n_collect,
            shared_across_scenes=False,
        )
        self.cherries = self.add_node(
            "models/smiley",
            model_scale=(0.24, 0.24, 0.07),
            model_scale_units="absolute",
            instances_per_scene=n_collect,
            shared_across_scenes=False,
        )
        self.power_pills = self.add_node(
            "models/smiley",
            model_scale=(0.3, 0.3, 0.08),
            model_scale_units="absolute",
            instances_per_scene=n_collect,
            shared_across_scenes=False,
        )

        # Static walls (shared across scenes)
        wall_pos = torch.zeros((1, n_walls, 3), dtype=torch.float32)
        for i, (x, y) in enumerate(self._wall_coords):
            wall_pos[0, i] = torch.tensor([x, y, 0.0])
        self.walls.set_positions(wall_pos)
        self.walls.set_colors(torch.tensor([[[0.12, 0.2, 0.95, 1.0]]], dtype=torch.float32).repeat(1, n_walls, 1))

        # Colors
        self.pacman.set_colors(torch.tensor([[[1.0, 0.95, 0.1, 1.0]]], dtype=torch.float32).repeat(num_scenes, 1, 1))
        self.ghost.set_colors(torch.tensor([[[1.0, 0.2, 0.25, 1.0]]], dtype=torch.float32).repeat(num_scenes, 1, 1))
        self.pellets.set_colors(torch.tensor([[[1.0, 0.9, 0.65, 1.0]]], dtype=torch.float32).repeat(num_scenes, n_collect, 1))
        self.cherries.set_colors(torch.tensor([[[0.95, 0.0, 0.1, 1.0]]], dtype=torch.float32).repeat(num_scenes, n_collect, 1))
        self.power_pills.set_colors(torch.tensor([[[0.35, 1.0, 1.0, 1.0]]], dtype=torch.float32).repeat(num_scenes, n_collect, 1))

        # Camera top-down-ish
        self.add_camera(fov_y_deg=40.0)
        center = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        self._pbr_cam.set_positions(torch.tensor([0.0, -1.0, 14.0], dtype=torch.float32))
        self._pbr_cam.look_at(center)
        self.add_light(ambient=(0.35, 0.35, 0.4), dir_dir=(0.2, -0.3, -1.0), dir_col=(1.0, 1.0, 1.0), strength=1.0)

        self._collect_world = torch.zeros((n_collect, 3), dtype=torch.float32)
        for i, (x, y) in enumerate(self.collectible_coords):
            self._collect_world[i] = torch.tensor([x, y, self.base_z])

        self.setup_environment()

    def _to_world_xy(self, x: int, y: int) -> tuple[float, float]:
        cx = x - (self.W - 1) / 2.0
        cy = (self.H - 1) / 2.0 - y
        return cx * self.cell, cy * self.cell

    def _step(self, state_batch=None):
        if state_batch is None:
            return

        # Expect dictionary payload from env.render_pixels
        pac_xy = torch.as_tensor(state_batch["pac_xy"], dtype=torch.float32).detach().cpu()
        ghost_xy = torch.as_tensor(state_batch["ghost_xy"], dtype=torch.float32).detach().cpu()
        pellet_alive = torch.as_tensor(state_batch["pellet_alive"], dtype=torch.bool).detach().cpu()
        cherry_alive = torch.as_tensor(state_batch["cherry_alive"], dtype=torch.bool).detach().cpu()
        power_alive = torch.as_tensor(state_batch["power_alive"], dtype=torch.bool).detach().cpu()
        powered = torch.as_tensor(state_batch["powered"], dtype=torch.bool).detach().cpu().reshape(-1, 1, 1)

        B = int(self.cfg.num_scenes)
        pac_pos = torch.zeros((B, 1, 3), dtype=torch.float32)
        ghost_pos = torch.zeros((B, 1, 3), dtype=torch.float32)
        pac_pos[:, 0, :2] = pac_xy
        ghost_pos[:, 0, :2] = ghost_xy
        pac_pos[:, 0, 2] = self.base_z
        ghost_pos[:, 0, 2] = self.base_z
        self.pacman.set_positions(pac_pos)
        self.ghost.set_positions(ghost_pos)

        # Ghost color switches when Pac-Man is powered
        ghost_normal = torch.tensor([1.0, 0.2, 0.25, 1.0], dtype=torch.float32).reshape(1, 1, 4)
        ghost_fright = torch.tensor([0.2, 0.5, 1.0, 1.0], dtype=torch.float32).reshape(1, 1, 4)
        ghost_col = torch.where(powered, ghost_fright, ghost_normal).repeat(B, 1, 1)
        self.ghost.set_colors(ghost_col)

        # Collectible visibility by moving eaten ones below scene
        collect_pos = self._collect_world.unsqueeze(0).repeat(B, 1, 1)
        hidden_z = torch.full((B, collect_pos.shape[1], 1), -50.0, dtype=torch.float32)

        pellet_pos = collect_pos.clone()
        pellet_pos[:, :, 2:3] = torch.where(pellet_alive.unsqueeze(-1), pellet_pos[:, :, 2:3], hidden_z)
        cherry_pos = collect_pos.clone()
        cherry_pos[:, :, 2:3] = torch.where(cherry_alive.unsqueeze(-1), cherry_pos[:, :, 2:3], hidden_z)
        power_pos = collect_pos.clone()
        power_pos[:, :, 2:3] = torch.where(power_alive.unsqueeze(-1), power_pos[:, :, 2:3], hidden_z)

        self.pellets.set_positions(pellet_pos)
        self.cherries.set_positions(cherry_pos)
        self.power_pills.set_positions(power_pos)
