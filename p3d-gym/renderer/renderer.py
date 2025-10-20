from turtle import update
from typing import Literal
import numpy as np

from direct.showbase.ShowBase import ShowBase
from panda3d.core import loadPrcFileData
from direct.task import Task
from direct.showbase.ShowBaseGlobal import globalClock


from .node import P3DNode
from .camera import P3DCam
from .light import P3DLight
from ..config import P3DConfig


class P3DRenderer(ShowBase):
    """ShowBase subclass that embeds scene helpers for multi-view rendering.

    This merges the responsibilities of `P3DScene` into a parent renderer so
    apps can simply inherit from `P3DRenderer` and call helper methods directly.
    """
    def __init__(self, cfg: P3DConfig | dict | None = None, **cfg_overrides):
        # Process config and apply PRC before initializing ShowBase
        self.cfg = P3DConfig.from_config(cfg, **cfg_overrides)
        prc_data = self.cfg.build_prc()
        loadPrcFileData('', prc_data)

        super().__init__()
        # Initial registries used by node/cam/light helpers
        self._p3d_nodes: list[P3DNode] = []
        self._p3d_cam: P3DCam | None = None
        self._p3d_light: P3DLight | None = None
        self.num_scenes = int(self.cfg.num_scenes)

        if self.cfg.report_fps:
            self._fps_last_print_time = 0.0 # TODO: move to state
            self.taskMgr.add(self.report_fps, 'report_fps')

        if self.cfg.manual_camera_control:
            self.enable_camera_control()

    # --- Node helpers (mirroring P3DScene) ---
    def add_node(self,
            model_path: str | None,
            instances_per_scene: int,
            shared_across_scenes: bool = False,
            parent: 'P3DNode | None' = None,
            name: str | None = None,
            model_pivot_relative_point: tuple[float, float, float] | None = None,
            model_scale: tuple[float, float, float] = (10.0, 1.0, 1.0),
            positions: 'np.ndarray | None' = None,
            hprs: 'np.ndarray | None' = None,
            scales: 'np.ndarray | None' = None,
            colors: 'np.ndarray | None' = None,
            backend: Literal[ "loop", "instanced"] = "instanced",) -> P3DNode:
        node = P3DNode(self, model_path=model_path, num_scenes=self.num_scenes, instances_per_scene=int(instances_per_scene), shared_across_scenes=shared_across_scenes, parent=parent, name=name,
                       model_pivot_relative_point=model_pivot_relative_point, model_scale=model_scale, positions=positions, hprs=hprs, scales=scales, colors=colors, backend=backend)
        return node

    # --- Camera helpers ---
    @property
    def cam_manager(self) -> P3DCam | None:
        return getattr(self, '_p3d_cam', None)

    def add_camera(self) -> P3DCam:
        self._p3d_cam = P3DCam(self, num_scenes=self.cfg.num_scenes, cols=self.cfg.tiles[0], rows=self.cfg.tiles[1])
        return self._p3d_cam

    def _set_tiles_auto(self) -> None:
        if self._p3d_cam is None:
            self.add_camera()
        self._p3d_cam._set_tiles()

    def start(self) -> None:
        # TODO: to remove
        if self._p3d_cam is None:
            self.add_camera()
        self._p3d_cam.start_tasks(self.taskMgr)

    # --- Lighting helpers ---
    def add_light(self,
                  ambient: tuple[float, float, float] = (0.2, 0.2, 0.25),
                  dir_dir: tuple[float, float, float] = (0.4, -0.6, -0.7),
                  dir_col: tuple[float, float, float] = (1.0, 1.0, 1.0),
                  strength: float = 1.0) -> P3DLight:
        self._p3d_light = P3DLight(self, ambient=ambient, dir_dir=dir_dir, dir_col=dir_col, strength=strength)
        return self._p3d_light

    def set_key(self, key, value):
        self.keys[key] = value


    def enable_camera_control(self):
        if self.cfg.offscreen:
            print('WARNING: Manual camera control is not possible in offscreen mode, disabling.')
            self.cfg.manual_camera_control = False
            return

        self.taskMgr.add(self.update_camera, 'camera_control')

        self.keys = {
            "w": False, "s": False, "a": False, "d": False
        }
        self.accept("w", self.set_key, ["w", True])
        self.accept("s", self.set_key, ["s", True])
        self.accept("a", self.set_key, ["a", True])
        self.accept("d", self.set_key, ["d", True])
        self.accept("w-up", self.set_key, ["w", False])
        self.accept("s-up", self.set_key, ["s", False])
        self.accept("a-up", self.set_key, ["a", False])
        self.accept("d-up", self.set_key, ["d", False])


    def report_fps(self, task: Task) -> None:
        now = globalClock.getRealTime()
        if now - self._fps_last_print_time >= self.cfg.report_fps_interval:
            self._fps_last_print_time = now
            fps = globalClock.getAverageFrameRate() * self.cfg.num_scenes
            print(f"FPS: {float(fps):.1f}")

        return task.cont

    def update_camera(self, task):
        dt = self.clock.get_dt()
        speed = 25

        position_k3 = self._p3d_cam.get_eye()
        if self.keys["w"]:
            position_k3[:, 1] += speed * dt
        if self.keys["s"]:
            position_k3[:,1] -= speed * dt
        if self.keys["a"]:
            position_k3[:,0] -= speed * dt
        if self.keys["d"]:
            position_k3[:,0] += speed * dt
        self._p3d_cam.set_eye(position_k3)

        if self.mouseWatcherNode.hasMouse():
            md = self.win.getPointer(0)
            x = md.getX()
            y = md.getY()
            
            fwd_k3 = self._p3d_cam.get_forward()
            fwd_x, fwd_y, fwd_z = fwd_k3[:,0], fwd_k3[:,1], fwd_k3[:,2]

            self.win.movePointer(0, self.win.getXSize() // 2, self.win.getYSize() // 2)

        return task.cont
