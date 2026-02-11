"""Shared map/layout helpers for PacMan-v0."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PacmanLayout:
    width: int
    height: int
    walls: set[tuple[int, int]]

    def is_wall(self, x: int, y: int) -> bool:
        return (x, y) in self.walls

    def is_walkable(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height and (x, y) not in self.walls

    @property
    def walkable(self) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for y in range(self.height):
            for x in range(self.width):
                if self.is_walkable(x, y):
                    out.append((x, y))
        return out


def _rect(x0: int, y0: int, x1: int, y1: int) -> set[tuple[int, int]]:
    pts: set[tuple[int, int]] = set()
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            pts.add((x, y))
    return pts


def _classic_walls_28x31() -> set[tuple[int, int]]:
    """Classic Pac-Man-inspired 28x31 maze walls."""
    rows = [
        "############################",
        "#............##............#",
        "#.####.#####.##.#####.####.#",
        "#o####.#####.##.#####.####o#",
        "#.####.#####.##.#####.####.#",
        "#..........................#",
        "#.####.##.########.##.####.#",
        "#.####.##.########.##.####.#",
        "#......##....##....##......#",
        "######.##### ## #####.######",
        "######.##### ## #####.######",
        "######.##          ##.######",
        "######.## ###--### ##.######",
        "######.## #      # ##.######",
        "      .   #      #   .      ",
        "######.## #      # ##.######",
        "######.## ######## ##.######",
        "######.##          ##.######",
        "######.## ######## ##.######",
        "######.## ######## ##.######",
        "#............##............#",
        "#.####.#####.##.#####.####.#",
        "#.####.#####.##.#####.####.#",
        "#o..##................##..o#",
        "###.##.##.########.##.##.###",
        "###.##.##.########.##.##.###",
        "#......##....##....##......#",
        "#.##########.##.##########.#",
        "#.##########.##.##########.#",
        "#..........................#",
        "############################",
    ]
    walls: set[tuple[int, int]] = set()
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch == '#':
                walls.add((x, y))
    return walls


def _default_walls_21x21() -> set[tuple[int, int]]:
    w, h = 21, 21
    walls: set[tuple[int, int]] = set()

    walls |= _rect(0, 0, w - 1, 0)
    walls |= _rect(0, h - 1, w - 1, h - 1)
    walls |= _rect(0, 0, 0, h - 1)
    walls |= _rect(w - 1, 0, w - 1, h - 1)

    walls |= _rect(3, 2, 5, 4)
    walls |= _rect(8, 2, 10, 4)
    walls |= _rect(12, 2, 14, 4)
    walls |= _rect(16, 2, 18, 4)

    walls |= _rect(2, 6, 6, 7)
    walls |= _rect(8, 6, 12, 7)
    walls |= _rect(14, 6, 18, 7)

    walls |= _rect(7, 9, 13, 9)
    walls |= _rect(7, 13, 13, 13)
    walls |= _rect(7, 9, 7, 13)
    walls |= _rect(13, 9, 13, 13)

    walls |= _rect(2, 15, 6, 16)
    walls |= _rect(8, 15, 12, 16)
    walls |= _rect(14, 15, 18, 16)

    walls |= _rect(2, 18, 4, 19)
    walls |= _rect(6, 18, 8, 19)
    walls |= _rect(12, 18, 14, 19)
    walls |= _rect(16, 18, 18, 19)

    walls |= _rect(6, 2, 6, 8)
    walls |= _rect(11, 2, 11, 8)
    walls |= _rect(15, 2, 15, 8)
    walls |= _rect(6, 12, 6, 19)
    walls |= _rect(11, 12, 11, 19)
    walls |= _rect(15, 12, 15, 19)

    for p in [(10, 2), (10, 3), (10, 4)]:
        walls.discard(p)

    return walls


def nearest_walkable(layout: PacmanLayout, target: tuple[float, float]) -> tuple[int, int]:
    tx, ty = int(round(target[0])), int(round(target[1]))
    if layout.is_walkable(tx, ty):
        return (tx, ty)
    best = None
    best_d = 10**9
    for x, y in layout.walkable:
        d = abs(x - tx) + abs(y - ty)
        if d < best_d:
            best_d = d
            best = (x, y)
    return best if best is not None else (1, 1)


def _matrix_points(mat: list[list[int]] | None) -> tuple[set[tuple[int, int]], int, int]:
    if not mat:
        return set(), 0, 0
    h = len(mat)
    w = len(mat[0]) if h else 0
    pts: set[tuple[int, int]] = set()
    for y in range(h):
        for x in range(w):
            if int(mat[y][x]) != 0:
                pts.add((x, y))
    return pts, w, h


def _border_walls(width: int, height: int) -> set[tuple[int, int]]:
    walls: set[tuple[int, int]] = set()
    for x in range(width):
        walls.add((x, 0))
        walls.add((x, height - 1))
    for y in range(height):
        walls.add((0, y))
        walls.add((width - 1, y))
    return walls


def build_spec(cfg) -> dict:
    """Build fully-resolved layout + item spec from config (with defaults)."""
    width = int(getattr(cfg, "map_width", 21))
    height = int(getattr(cfg, "map_height", 21))

    wall_points, wm_w, wm_h = _matrix_points(getattr(cfg, "wall_matrix", None))
    if wall_points:
        width, height = wm_w, wm_h
        walls = set(wall_points)
    else:
        if width == 28 and height == 31:
            walls = _classic_walls_28x31()
        elif width == 21 and height == 21:
            walls = _default_walls_21x21()
        else:
            walls = _border_walls(width, height)

    border_points, bm_w, bm_h = _matrix_points(getattr(cfg, "border_matrix", None))
    if border_points:
        if wm_w == 0 and wm_h == 0 and (bm_w > 0 and bm_h > 0):
            width, height = bm_w, bm_h
        walls |= border_points

    layout = PacmanLayout(width=width, height=height, walls=walls)
    walkable_sorted = sorted(layout.walkable, key=lambda p: (p[1], p[0]))

    pac_default = nearest_walkable(layout, (5, 14))
    pac_start_cfg = getattr(cfg, "pacman_start", None)
    pac_start_cell = nearest_walkable(layout, pac_start_cfg if pac_start_cfg else pac_default)
    pac_start = (float(pac_start_cell[0]), float(pac_start_cell[1]))

    num_ghosts = int(getattr(cfg, "num_ghosts", 4))
    ghost_default_targets = [(9, 10), (10, 10), (11, 10), (10, 9)]
    ghost_starts_cfg = list(getattr(cfg, "ghost_starts", []) or [])
    ghost_starts: list[tuple[float, float]] = []
    for i in range(num_ghosts):
        if i < len(ghost_starts_cfg):
            tgt = ghost_starts_cfg[i]
        else:
            tgt = ghost_default_targets[i % len(ghost_default_targets)]
        g = nearest_walkable(layout, tgt)
        ghost_starts.append((float(g[0]), float(g[1])))

    blocked = {(int(round(pac_start[0])), int(round(pac_start[1])))} | {
        (int(round(g[0])), int(round(g[1]))) for g in ghost_starts
    }

    pellet_points, _, _ = _matrix_points(getattr(cfg, "pellet_matrix", None))
    if pellet_points:
        pellet_cells = [p for p in sorted(pellet_points, key=lambda q: (q[1], q[0])) if layout.is_walkable(*p)]
    else:
        n = int(getattr(cfg, "default_pellets", 144))
        pellet_cells = [p for p in walkable_sorted if p not in blocked][:n]

    power_points, _, _ = _matrix_points(getattr(cfg, "power_pellet_matrix", None))
    if power_points:
        power_cells = [p for p in sorted(power_points, key=lambda q: (q[1], q[0])) if layout.is_walkable(*p)]
    else:
        preferred_power = [(1, 3), (width - 2, 3), (1, height - 4), (width - 2, height - 4)]
        n = int(getattr(cfg, "default_power_pellets", 4))
        power_cells = []
        for p in preferred_power:
            power_cells.append(nearest_walkable(layout, p))
            if len(power_cells) >= n:
                break
        while len(power_cells) < n:
            power_cells.append(walkable_sorted[len(power_cells) % len(walkable_sorted)])

    cherry_points, _, _ = _matrix_points(getattr(cfg, "cherry_matrix", None))
    if cherry_points:
        cherry_cells = [p for p in sorted(cherry_points, key=lambda q: (q[1], q[0])) if layout.is_walkable(*p)]
    else:
        preferred_cherry = [(width // 2, 5), (width // 2, height - 6)]
        n = int(getattr(cfg, "default_cherries", 2))
        cherry_cells = []
        for p in preferred_cherry:
            cherry_cells.append(nearest_walkable(layout, p))
            if len(cherry_cells) >= n:
                break
        while len(cherry_cells) < n:
            cherry_cells.append(walkable_sorted[(3 + len(cherry_cells)) % len(walkable_sorted)])

    # Remove overlaps among item classes with precedence: power > cherry > pellet
    power_set = set(power_cells)
    cherry_cells = [c for c in cherry_cells if c not in power_set]
    cherry_set = set(cherry_cells)
    pellet_cells = [p for p in pellet_cells if p not in power_set and p not in cherry_set]

    # Backfill to requested defaults when using generated placement
    if not getattr(cfg, "pellet_matrix", None):
        need = int(getattr(cfg, "default_pellets", 144))
        blocked_items = power_set | cherry_set
        for p in walkable_sorted:
            if len(pellet_cells) >= need:
                break
            if p not in blocked_items and p not in pellet_cells and p not in blocked:
                pellet_cells.append(p)

    if not getattr(cfg, "power_pellet_matrix", None):
        need = int(getattr(cfg, "default_power_pellets", 4))
        for p in walkable_sorted:
            if len(power_cells) >= need:
                break
            if p not in power_set and p not in blocked:
                power_cells.append(p)
                power_set.add(p)

    if not getattr(cfg, "cherry_matrix", None):
        need = int(getattr(cfg, "default_cherries", 2))
        for p in walkable_sorted:
            if len(cherry_cells) >= need:
                break
            if p not in power_set and p not in cherry_set and p not in blocked:
                cherry_cells.append(p)
                cherry_set.add(p)

    return {
        "layout": layout,
        "pac_start": pac_start,
        "ghost_starts": ghost_starts,
        "pellet_cells": pellet_cells,
        "power_cells": power_cells,
        "cherry_cells": cherry_cells,
    }
