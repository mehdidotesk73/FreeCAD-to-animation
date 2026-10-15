"""Frame rendering (offscreen VTK through PyVista).

FreeCAD produces the geometry and this module draws it. Colours come from
the group config, and each group is drawn solid / translucent / hidden
according to the per-frame body ``modes`` computed by the planner.

FreeCAD's own 3D view needs a display server, so frames are rendered with
VTK offscreen (OSMesa / EGL) instead. The geometry is exactly FreeCAD's
tessellation.
"""
from __future__ import annotations

import math
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
import pyvista as pv  # noqa: E402

pv.OFF_SCREEN = True

from .camera import view_direction  # noqa: E402
from .geometry import Geometry, Mesh  # noqa: E402


@dataclass(frozen=True)
class SceneSpec:
    body_group: Mapping[str, str]
    colors: Mapping[str, str]
    style: Mapping[str, object]


@dataclass(frozen=True)
class FrameRequest:
    geometry_key: str
    camera: Mapping
    output: str
    active_group: Optional[str] = None
    caption: Optional[str] = None
    reveal: Mapping[str, float] = field(default_factory=dict)
    modes: Mapping[str, str] = field(default_factory=dict)  # body -> solid/translucent/hidden
    ghost: float = 0.0  # 0..1: fade every body towards ghost_opacity (sketch steps)
    # (model folder holding the sketch, sketch id, opacity): fading profile sketches
    overlays: Sequence[Tuple[str, str, float]] = ()


def _polydata(mesh: Mesh) -> pv.PolyData:
    v, f = mesh
    faces = np.hstack([np.full((len(f), 1), 3, dtype=np.int64), f.astype(np.int64)]).ravel()
    return pv.PolyData(np.asarray(v, dtype=np.float64), faces)


def _polylines(lines: Sequence[np.ndarray], fraction: float = 1.0) -> Optional[pv.PolyData]:
    """Polylines, optionally truncated to ``fraction`` of their total length."""
    lines = [np.asarray(l, dtype=np.float64) for l in lines if len(l) >= 2]
    if not lines:
        return None
    if fraction < 1.0:
        lengths = [float(np.linalg.norm(np.diff(l, axis=0), axis=1).sum()) for l in lines]
        budget = max(fraction, 0.0) * sum(lengths)
        out = []
        for l, L in zip(lines, lengths):
            if budget <= 0:
                break
            if L <= budget:
                out.append(l)
                budget -= L
                continue
            seg = np.linalg.norm(np.diff(l, axis=0), axis=1)
            cum = np.concatenate([[0], np.cumsum(seg)])
            j = int(np.searchsorted(cum, budget))
            a = (budget - cum[j - 1]) / max(seg[j - 1], 1e-12)
            out.append(np.vstack([l[:j], l[j - 1] + a * (l[j] - l[j - 1])]))
            budget = 0
        lines = out
        if not lines:
            return None
    pts = np.vstack(lines)
    cells, off = [], 0
    for l in lines:
        cells.append(np.concatenate([[len(l)], np.arange(off, off + len(l))]))
        off += len(l)
    return pv.PolyData(pts, lines=np.concatenate(cells).astype(np.int64))


def apply_camera(plotter: pv.Plotter, view: Mapping) -> None:
    cam = plotter.camera
    cam.position = tuple(view["Position"])
    cam.focal_point = tuple(view["FocalPoint"])
    cam.up = tuple(view["ViewUp"])
    va = math.radians(float(view.get("ViewAngle", 30.0)))
    zoom = float(view.get("Zoom", 1.0))
    cam.view_angle = math.degrees(2 * math.atan(math.tan(va / 2) / max(zoom, 1e-6)))
    # Without this flag PyVista resets to its default camera on the first render.
    plotter.camera_set = True
    plotter.reset_camera_clipping_range()


class Renderer:
    """Holds one offscreen render window; each ``render`` call starts from an empty scene."""

    def __init__(self, scene: SceneSpec):
        self.scene = scene
        st = scene.style
        self.width, self.height = int(st["width"]), int(st["height"])
        self.plotter = pv.Plotter(off_screen=True, window_size=(self.width, self.height),
                                  lighting="light_kit")
        self.plotter.set_background(st.get("background", "#ffffff"))
        aa = st.get("anti_aliasing")
        if aa:
            # Depth peeling sorts overlapping translucent bodies correctly, but on
            # OSMesa/EGL it only writes their alpha into transparent PNGs when
            # anti-aliasing is on; without it translucent bodies vanish. So peel
            # only together with anti-aliasing.
            self.plotter.enable_depth_peeling(number_of_peels=6)
            self.plotter.enable_anti_aliasing(aa)
        self._mesh_cache: "OrderedDict[Tuple[str, str], Tuple[pv.PolyData, Optional[pv.PolyData]]]" = OrderedDict()

    def close(self):
        self.plotter.close()

    def _body_mesh(self, key: str, name: str, mesh: Mesh):
        ck = (key, name)
        if ck in self._mesh_cache:
            self._mesh_cache.move_to_end(ck)
            return self._mesh_cache[ck]
        pd = _polydata(mesh).clean()
        surf = pd.compute_normals(split_vertices=True, feature_angle=30.0, auto_orient_normals=False)
        edges = None
        if self.scene.style.get("feature_edges", True):
            edges = pd.extract_feature_edges(feature_angle=30.0, boundary_edges=True,
                                             non_manifold_edges=False, manifold_edges=False)
            if edges.n_points == 0:
                edges = None
        self._mesh_cache[ck] = (surf, edges)
        while len(self._mesh_cache) > 48:
            self._mesh_cache.popitem(last=False)
        return surf, edges

    @staticmethod
    def _line_nudge(view: Mapping) -> Tuple[float, float, float]:
        """Sketch lines often lie exactly on a face or edge (a pad's rim); move
        them 0.2% of the camera distance towards the camera so they stay on top."""
        return tuple(view_direction(view) * 0.002 * float(view["Distance"]))

    def render(self, req: FrameRequest, geom: Geometry,
               overlay_geoms: Optional[Mapping[str, Geometry]] = None) -> None:
        """Draw one frame. ``overlay_geoms`` maps an overlay's model folder to its geometry."""
        st = self.scene.style
        p = self.plotter
        p.clear_actors()
        for name, mesh in geom.bodies.items():
            group = self.scene.body_group.get(name)
            mode = req.modes.get(name, "solid") if group else "hidden"
            if mode == "hidden":
                continue
            color = self.scene.colors.get(group, "#aaaaaa")
            opacity = float(st["translucent_opacity"]) if mode == "translucent" else 1.0
            if req.ghost > 0:
                target = min(opacity, float(st["ghost_opacity"]))
                opacity += (target - opacity) * min(float(req.ghost), 1.0)
            surf, edges = self._body_mesh(geom.key, name, mesh)
            p.add_mesh(surf, color=color, opacity=opacity, smooth_shading=True,
                       specular=0.25, specular_power=20, diffuse=0.85, ambient=0.15)
            if edges is not None:
                p.add_mesh(edges, color=st["edge_color"], line_width=1.0,
                           opacity=0.55 if opacity == 1.0 else opacity * 0.6, lighting=False)

        for sk, lines in geom.sketches.items():
            body = geom.info.get("sketches", {}).get(sk, {}).get("body")
            group = self.scene.body_group.get(body)
            if body and req.modes.get(body, "solid") == "hidden":
                continue
            pd = _polylines(lines, float(req.reveal.get(sk, 1.0)))
            if pd is not None:
                a = p.add_mesh(pd, color=st["sketch_color"], line_width=float(st["sketch_width"]),
                               lighting=False)
                a.position = self._line_nudge(req.camera)

        for model, sk, alpha in req.overlays:
            og = (overlay_geoms or {}).get(model)
            lines = og.sketches.get(sk) if og is not None else None
            pd = _polylines(lines) if lines else None
            if pd is not None and alpha > 0:
                a = p.add_mesh(pd, color=st["sketch_color"], line_width=float(st["sketch_width"]),
                               opacity=float(alpha), lighting=False)
                a.position = self._line_nudge(req.camera)

        if req.caption and st.get("captions", True):
            p.add_text(req.caption, position="upper_left", font_size=11, color="#333333",
                       name="caption")
        apply_camera(p, req.camera)
        os.makedirs(os.path.dirname(os.path.abspath(req.output)), exist_ok=True)
        p.screenshot(req.output, transparent_background=bool(st.get("transparent", True)))


# --------------------------------------------------------------------------
# Visibility probe (used by the camera planner)
# --------------------------------------------------------------------------
class VisibilityProbe:
    """Counts how many pixels of a target are visible from a viewpoint.

    One render per viewpoint: occluders are drawn black and the target red,
    both unlit, and the surviving red pixels are counted. Faces the target
    shares with an occluder (for example the walls of a fresh cut) must count
    as visible, so the target is moved towards the camera by a small
    world-space tolerance (0.4% of the camera distance) before drawing.

    Usage per step: ``setup(occluders, target)`` once, then ``visible(view)``
    for every candidate direction (1 render each), and ``unoccluded(view)``
    only where the target's unobstructed size is needed (1 extra render).
    """

    TOLERANCE = 0.004

    def __init__(self, size: Sequence[int] = (240, 180)):
        self.size = (int(size[0]), int(size[1]))
        self.p = pv.Plotter(off_screen=True, window_size=self.size, lighting="none")
        self.p.set_background("white")
        self.p.show(auto_close=False)  # screenshots require an initial render
        self._target = None
        self._occluders: List = []

    def close(self):
        self.p.close()

    def setup(self, occluders: Sequence[pv.PolyData], target: pv.PolyData,
              target_is_lines: bool = False) -> None:
        """Load one step's scene. No render happens here."""
        p = self.p
        p.clear_actors()
        self._occluders = [p.add_mesh(o, color="black", lighting=False, render=False,
                                      reset_camera=False) for o in occluders]
        self._target = p.add_mesh(target, color="red", lighting=False, render=False,
                                  reset_camera=False, line_width=4.0 if target_is_lines else 1.0)

    def _count_red(self, view: Mapping) -> int:
        # Nudge the target towards the camera (see class docstring).
        nudge = view_direction(view) * self.TOLERANCE * float(view["Distance"])
        self._target.position = tuple(nudge)
        apply_camera(self.p, view)
        self.p.render()       # the single render ...
        img = self.p.image    # ... read back without re-rendering (screenshot() may return
                              # a stale image here)
        return int(((img[..., 0] > 180) & (img[..., 1] < 80) & (img[..., 2] < 80)).sum())

    def visible(self, view: Mapping) -> int:
        """Target pixels visible past the occluders (1 render)."""
        return self._count_red(view)

    def unoccluded(self, view: Mapping) -> int:
        """Target pixels with the occluders hidden (1 render)."""
        for a in self._occluders:
            a.visibility = False
        try:
            return self._count_red(view)
        finally:
            for a in self._occluders:
                a.visibility = True

    def measure(self, occluders: Sequence[pv.PolyData], target: pv.PolyData, view: Mapping,
                target_is_lines: bool = False) -> Tuple[int, int]:
        """(visible, unoccluded) target pixels for one view (2 renders)."""
        self.setup(occluders, target, target_is_lines)
        return self.visible(view), self.unoccluded(view)


def as_polydata(mesh: Mesh) -> pv.PolyData:
    return _polydata(mesh)


def as_lines(lines: Sequence[np.ndarray]) -> Optional[pv.PolyData]:
    return _polylines(lines)
