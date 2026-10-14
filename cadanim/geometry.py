"""Layer 2 - stateless geometric compiler.

``GeometryCompiler.compile_many`` turns tree states into STL models by
running FreeCAD in subprocesses. Every state is built from a freshly opened
copy of the source document and then closed (see ``cadanim/fc/worker.py``).
The same state therefore always gives the same geometry, which makes it safe
to cache results under the state's geometry hash (``tree.geometry_hash``).
Render-only keys never cause a rebuild.

Model store layout (one folder per state; every major frame and every growth
increment gets one)::

    <models>/<key>/info.json      bodies (bbox, volume, stl), sketches, focus, errors
    <models>/<key>/<Body>.stl     one binary STL per body, global coordinates
    <models>/<key>/_focus.stl     changed material of the step (planning only)

Cameras are applied by the viewer to these STLs. Re-framing or re-rendering
therefore never needs FreeCAD.

``tree_to_model`` is the single-frame entry point from the spec: compile
(or load from cache) and render one PNG.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import fcbridge
from . import tree as T
from .fc import COMPILER_VERSION
from .stl import read_stl

Mesh = Tuple[np.ndarray, np.ndarray]  # (vertices Nx3 float32, faces Mx3 int32)


@dataclass(frozen=True)
class Geometry:
    key: str
    bodies: Dict[str, Mesh] = field(repr=False)
    sketches: Dict[str, List[np.ndarray]] = field(repr=False)
    focus_mesh: Optional[Mesh] = field(repr=False, default=None)
    info: Dict = field(default_factory=dict, repr=False)

    @property
    def ok(self) -> bool:
        return not self.info.get("errors") and not self.info.get("failed")

    @property
    def focus(self) -> Dict:
        return self.info.get("focus") or {}

    def body_bbox(self, names: Optional[Sequence[str]] = None) -> Optional[List[float]]:
        from .camera import union_bbox
        b = self.info.get("bodies", {})
        return union_bbox(v["bbox"] for n, v in b.items() if names is None or n in names)


EMPTY = Geometry(key="empty", bodies={}, sketches={}, info={"bodies": {}, "sketches": {}, "errors": []})


class CompileError(RuntimeError):
    pass


class GeometryCompiler:
    def __init__(self, source: str, cache_dir: str, workers: int = 4, deflection: float = 0.1,
                 batch_size: int = 12):
        self.source = os.path.abspath(source)
        self.cache_dir = os.path.abspath(cache_dir)
        self.workers = max(1, workers)
        self.deflection = deflection
        self.batch_size = batch_size
        os.makedirs(self.cache_dir, exist_ok=True)

    # -- cache layout ----------------------------------------------------
    def key(self, state: dict, focus: bool = False) -> str:
        h = T.geometry_hash(state) + f"_d{self.deflection:g}_v{COMPILER_VERSION}"
        return h + ("_focus" if focus else "")

    def _base(self, key: str) -> str:
        return os.path.join(self.cache_dir, key)

    def model_dir(self, key: str) -> Optional[str]:
        return None if key == "empty" else self._base(key)

    def cached_key(self, state: dict, focus: bool = False) -> Optional[str]:
        """Existing cache key satisfying the request (a focus build satisfies a plain one)."""
        if not T.feature_ids(state):
            return "empty"
        cands = [self.key(state, True)] if focus else [self.key(state), self.key(state, True)]
        for k in cands:
            if os.path.exists(os.path.join(self._base(k), "info.json")):
                return k
        return None

    # -- compile ----------------------------------------------------------
    def compile_many(self, states: Sequence[dict], focus: bool = False,
                     progress=None) -> List[str]:
        """Ensure every state is compiled; return their cache keys (same order)."""
        keys: List[Optional[str]] = [self.cached_key(s, focus) for s in states]
        todo: Dict[str, dict] = {}
        for s, k in zip(states, keys):
            if k is None:
                todo.setdefault(self.key(s, focus), s)
        if todo:
            items = []
            for k, s in todo.items():
                items.append({"state": s, "out": self._base(k), "focus": focus})
            # Interleave so each worker gets a similar mix of cheap/expensive states.
            nb = max(self.workers, -(-len(items) // self.batch_size))
            batches = [items[i::nb] for i in range(nb) if items[i::nb]]
            jobs = [{"action": "compile", "source": self.source, "deflection": self.deflection,
                     "items": b} for b in batches]
            if progress:
                progress(f"compiling {len(items)} model state(s) in FreeCAD "
                         f"({len(jobs)} job(s), {self.workers} worker(s))")
            results = fcbridge.run_jobs_parallel(jobs, self.workers)
            for r in results:
                for item in r["results"]:
                    if not item["ok"]:
                        # Record the failure so callers can skip the state.
                        os.makedirs(item["out"], exist_ok=True)
                        with open(os.path.join(item["out"], "info.json"), "w") as fh:
                            json.dump({"failed": item["error"], "errors": [], "bodies": {},
                                       "sketches": {}}, fh)
        return [k or self.cached_key(s, focus) for s, k in zip(states, keys)]

    def compile(self, state: dict, focus: bool = False) -> Geometry:
        return self.load(self.compile_many([state], focus)[0])

    def load(self, key: str) -> Geometry:
        if key == "empty":
            return EMPTY
        return _load_geometry(self._base(key), key)


def load_model_dir(path: str, key: Optional[str] = None) -> Geometry:
    """Load a compiled state folder (info.json + STLs). No FreeCAD involved."""
    return _load_geometry(os.path.abspath(path), key or os.path.basename(os.path.normpath(path)))


@lru_cache(maxsize=64)
def _load_geometry(base: str, key: str) -> Geometry:
    with open(os.path.join(base, "info.json")) as fh:
        info = json.load(fh)
    bodies: Dict[str, Mesh] = {}
    for name, b in info.get("bodies", {}).items():
        path = os.path.join(base, b.get("stl", f"{name}.stl"))
        if os.path.exists(path):
            bodies[name] = read_stl(path)
    sketches = {name: [np.asarray(l, dtype=np.float32) for l in sk.get("polylines", [])]
                for name, sk in info.get("sketches", {}).items()}
    focus_mesh = None
    fstl = (info.get("focus") or {}).get("stl")
    if fstl and os.path.exists(os.path.join(base, fstl)):
        focus_mesh = read_stl(os.path.join(base, fstl))
    return Geometry(key=key, bodies=bodies, sketches=sketches, focus_mesh=focus_mesh, info=info)


# --------------------------------------------------------------------------
# spec entry point
# --------------------------------------------------------------------------
def tree_to_model(tree_state: dict, camera_view: dict, output_path: str, *,
                  compiler: GeometryCompiler, renderer=None, active_group: Optional[str] = None,
                  caption: Optional[str] = None) -> Geometry:
    """Compile ``tree_state`` (cached by hash), render it from ``camera_view`` to a PNG.

    ``renderer`` is a ``cadanim.render.Renderer``. The FreeCAD side is
    stateless (a fresh document per state). The renderer only holds an
    offscreen window that is cleared for every frame.
    """
    from .render import FrameRequest
    geom = compiler.compile(tree_state)
    if renderer is None:
        raise ValueError("tree_to_model needs a renderer (cadanim.render.Renderer)")
    reveal = {fid: e["render_reveal"] for fid, e in tree_state.items()
              if fid != T.META_KEY and "render_reveal" in e}
    renderer.render(FrameRequest(geometry_key=geom.key, camera=camera_view, output=output_path,
                                 active_group=active_group, caption=caption, reveal=reveal),
                    geom)
    return geom
