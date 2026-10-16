"""Layer 3 (orchestration) and Layer 4 (frame sequencing + manifest).

Phase 1  parse:   FCStd -> feature tree -> ordered by group/dependencies ->
                  action queue of visual steps.
Phase 2  plan:    for every step: choose its view (visibility-scored
                  candidates), add the inter-step camera transition, add the
                  intra-step growth states and the static hold frames. The
                  result is a list of FrameSpec (state + camera + metadata).
Phase 3  deliver: compile every unique state (parallel, cached), render every
                  frame (parallel), write manifest.json (+ optional MP4).

The full tree is never mutated. Every state is derived with the pure helpers
in ``cadanim.tree`` / ``cadanim.intra``, and ``run`` asserts that the tree's
fingerprint is unchanged at the end.
"""
from __future__ import annotations

import heapq
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import camera as C
from . import groups as G
from . import tree as T
from .geometry import Geometry, GeometryCompiler
from .camplan import plan_sequence, shots
from .intra import GrowthPlan, plan_growth

FRAME_STATIC = "static"
FRAME_INTER = "inter_step"
FRAME_INTRA = "intra_step"


_T0 = time.time()


def _log(msg: str) -> None:
    print(f"[cadanim {time.time() - _T0:6.0f}s] {msg}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# Phase 1: ordering and the action queue
# --------------------------------------------------------------------------
def order_tree(full_tree: Mapping[str, Any], cfg: G.ProjectConfig) -> T.Tree:
    """Reorder features body by body: dependencies first, then group order.

    Bodies stay contiguous and keep their internal feature order. Among
    bodies whose dependencies are satisfied, the one in the earliest group
    (then earliest in the document) is built next.
    """
    bodies_meta = T.meta(full_tree).get("bodies", {})
    b2g = cfg.body_to_group()
    gidx = {g.name: i for i, g in enumerate(cfg.groups)}
    doc_order = {b: i for i, b in enumerate(bodies_meta)}
    deps = {b: set(d for d in m.get("deps", []) if d in bodies_meta) for b, m in bodies_meta.items()}
    rank = lambda b: (gidx.get(b2g.get(b), len(gidx)), doc_order[b])
    ready = [(rank(b), b) for b in bodies_meta if not deps[b]]
    heapq.heapify(ready)
    done: List[str] = []
    while ready:
        _, b = heapq.heappop(ready)
        done.append(b)
        for other in bodies_meta:
            if b in deps[other]:
                deps[other].discard(b)
                if not deps[other] and other not in done and all(other != x for _, x in ready):
                    heapq.heappush(ready, (rank(other), other))
    if len(done) != len(bodies_meta):
        raise ValueError("cyclic dependencies between bodies")
    by_body: Dict[str, List[str]] = {b: [] for b in done}
    for fid in T.feature_ids(full_tree):
        by_body.setdefault(full_tree[fid]["body"], []).append(fid)
    order = [fid for b in done for fid in by_body[b]]
    return T.reorder(full_tree, order)


def action_queue(ordered: Mapping[str, Any], cfg: G.ProjectConfig,
                 only_groups: Optional[Sequence[str]] = None) -> List[str]:
    b2g = cfg.body_to_group()
    out = []
    for fid in T.feature_ids(ordered):
        e = ordered[fid]
        g = b2g.get(e["body"])
        if g is None or g == G.HIDDEN_GROUP or not T.is_visual(e):
            continue
        if only_groups and g not in only_groups:
            continue
        out.append(fid)
    return out


# --------------------------------------------------------------------------
# data records
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class FrameSpec:
    state: dict = field(repr=False)
    camera: dict = field(repr=False)
    frame_type: str
    step_id: Optional[str]
    step_index: Optional[int]
    group: Optional[str]
    modes: Mapping[str, str]          # body -> solid / translucent / hidden
    caption: str = ""
    params: Optional[Mapping[str, Any]] = None
    ghost: float = 0.0                # 0..1: how far every body is faded to see-through
    # Sketch lines drawn on top of the state: (geometry key holding the sketch,
    # sketch id, opacity). Used to fade a sketch out once its feature grows.
    overlays: Tuple[Tuple[str, str, float], ...] = ()


@dataclass
class StepRecord:
    step_index: int
    step_id: str
    label: str
    kind: str
    role: str
    body: str
    body_label: str
    group: str
    animation: str
    view: dict
    ghost: bool = False
    shot: int = -1          # shared-camera shot id (-1 for sketch views)
    first_frame: int = -1
    last_frame: int = -1


@dataclass
class Timeline:
    ordered_tree: dict
    frames: List[FrameSpec]
    steps: List[StepRecord]


# --------------------------------------------------------------------------
# Phase 2: view planning
# --------------------------------------------------------------------------
def elevation_preference(direction, preferred_deg: float = 30.0) -> float:
    """1 at the preferred elevation, 0.5 level, 0 straight down, -1 from below."""
    e = math.degrees(math.asin(max(-1.0, min(1.0, float(direction[2])))))
    if e >= preferred_deg:
        return 1.0 - (e - preferred_deg) / (90.0 - preferred_deg)
    return max(-1.0, 1.0 - (preferred_deg - e) / (2 * preferred_deg))


class ViewPlanner:
    """Measures candidate views per step and turns planned directions into views.

    ``measure`` scores every candidate direction of a step by how many pixels
    of its target (sketch wires, added/removed material, or the changed region
    of a pattern) are visible past the solid context. One probe render per
    candidate. The choice across the whole sequence is made by
    ``cadanim.camplan.plan_sequence``; ``view_for`` frames the chosen
    direction on a box (a step's own region, or its shot's shared region).
    """

    def __init__(self, cfg: G.ProjectConfig):
        from .render import VisibilityProbe
        self.cfg = cfg
        cam = cfg.camera
        self.view_angle = float(cam["view_angle"])
        self.aspect = cfg.render["width"] / cfg.render["height"]
        self.probe = VisibilityProbe(cam["probe_size"])

    def close(self):
        self.probe.close()

    def region(self, focus_bbox, context_bbox) -> List[float]:
        """Framing box for a step.

        With ``context_weight`` = 1 the whole active body plus the feature is
        framed. Features smaller than ``small_feature_ratio`` of that region
        pull the framing towards them (never below ``min_context``), but never
        smaller than the feature itself (scaled by ``focus_scale``).
        """
        cam = self.cfg.camera
        union = C.union_bbox([context_bbox, focus_bbox])
        fc, fr = C.bbox_sphere(focus_bbox)
        uc, ur = C.bbox_sphere(union)
        w = min(max(float(cam["context_weight"]), 0.0), 1.0)
        ratio = fr / max(ur, 1e-9)
        small = float(cam["small_feature_ratio"])
        if ratio < small:
            w *= max(float(cam["min_context"]), ratio / small)
        if w >= 1.0:
            return list(union)
        center = fc + w * (uc - fc)
        radius = max(fr * float(cam["focus_scale"]), ur * w,
                     float(np.linalg.norm(center - fc)) + fr, 1.0)
        return C.sphere_box(center, radius)

    def step_override(self, step_id: str, entry: Mapping[str, Any]) -> Optional[dict]:
        """``camera.step_views`` entry for a step (by feature id, label or "Body/Label")."""
        table = self.cfg.camera.get("step_views") or {}
        for k in (step_id, entry.get("label"), f"{entry.get('body_label')}/{entry.get('label')}"):
            if k in table:
                return table[k]
        return None

    @staticmethod
    def override_direction(ov: Mapping[str, Any]) -> Optional[np.ndarray]:
        if "direction" in ov:
            return C.normalize(ov["direction"])
        if "azimuth" in ov or "elevation" in ov:
            return C.spherical(float(ov.get("azimuth", 45.0)), float(ov.get("elevation", 30.0)))
        return None

    def view_for(self, direction, region, override: Optional[Mapping[str, Any]] = None,
                 outline: bool = True) -> dict:
        """Frame ``region`` (a box) looking along ``direction``; apply override
        zoom / distance_scale / focal_offset if given.

        ``outline``: fit the box's projected outline (3D shots). Otherwise fit
        its bounding sphere (sketch views): looking straight at a tall part,
        its outline is narrow and outline framing would zoom in far more than
        the shots around it."""
        ov = override or {}
        margin = float(self.cfg.camera["margin"])
        if outline:
            view = C.fit_box(direction, region, self.view_angle, self.aspect, margin=margin)
        else:
            center, radius = C.bbox_sphere(region)
            view = C.make_view(center, direction,
                               C.fit_distance(radius, self.view_angle, self.aspect, margin=margin),
                               view_angle=self.view_angle)
        if ov:
            focal = np.asarray(view["FocalPoint"]) + np.asarray(ov.get("focal_offset", [0, 0, 0]), float)
            view = C.complete_view({**view, "FocalPoint": list(focal),
                                    "Distance": view["Distance"] * float(ov.get("distance_scale", 1.0)),
                                    "Zoom": float(ov.get("zoom", 1.0))})
        return view

    def measure(self, step_id: str, entry: Mapping[str, Any], after: Geometry,
                solid_bodies: Sequence[str], active_body_bbox):
        """Candidate directions of one step with their visibility and quality."""
        from .camplan import StepCandidates
        from .render import as_lines, as_polydata
        cam = self.cfg.camera
        focus = after.focus
        role = entry.get("role")
        target, is_lines = None, False
        if role == "sketch" and after.sketches.get(step_id):
            target, is_lines = as_lines(after.sketches[step_id]), True
        elif after.focus_mesh is not None and len(after.focus_mesh[1]):
            target = as_polydata(after.focus_mesh)
        elif entry.get("body") in after.bodies:
            target = as_polydata(after.bodies[entry["body"]])
        focus_bbox = focus.get("bbox") or active_body_bbox or after.body_bbox() or [-1, -1, -1, 1, 1, 1]
        # Context: the active body, or for the first sketch of a new body the
        # solid geometry it is being placed against. A body much smaller than
        # what is already built is framed with some of that around it.
        assembly = after.body_bbox(solid_bodies)
        context_bbox = active_body_bbox or assembly
        if active_body_bbox and assembly:
            bc, br = C.bbox_sphere(active_body_bbox)
            _, ar = C.bbox_sphere(assembly)
            floor = float(cam.get("min_body_fraction", 0.5)) * ar
            if br < floor:
                context_bbox = C.union_bbox([active_body_bbox, C.sphere_box(bc, floor)])
        region = self.region(focus_bbox, context_bbox)

        normal = C.normalize(focus["normal"]) if role == "sketch" and focus.get("normal") else None
        # Sketches are always seen perpendicular to their plane; only the side is chosen.
        dirs = [normal, -normal] if normal is not None else C.candidate_directions()
        override = self.step_override(step_id, entry) or {}
        fixed, keep_prev = None, False
        od = self.override_direction(override)
        if od is not None:
            dirs = list(dirs) + [od]
            fixed = len(dirs) - 1
        elif override.get("keep_previous") and role != "sketch":
            keep_prev = True

        if target is not None:
            self.probe.setup([as_polydata(after.bodies[b]) for b in solid_bodies if b in after.bodies],
                             target, is_lines)
            # Same scale for every direction (bounding sphere of the region):
            # outline framing would zoom in on narrow outlines and inflate
            # their pixel counts, making directions incomparable.
            center, radius = C.bbox_sphere(region)
            dist = C.fit_distance(radius, self.view_angle, self.aspect,
                                  margin=float(self.cfg.camera["margin"]))
            vis = [float(self.probe.visible(C.make_view(center, d, dist, view_angle=self.view_angle)))
                   for d in dirs]
        else:
            vis = [1.0] * len(dirs)
        best = max(vis) or 1.0
        # sqrt: once most of the target is visible, a nicer angle matters more.
        quality = [math.sqrt(v / best) + float(cam["elevation_bias"]) * elevation_preference(d)
                   for d, v in zip(dirs, vis)]
        return StepCandidates(step=step_id, role=role,
                              dirs=tuple(tuple(float(x) for x in d) for d in dirs),
                              vis=tuple(vis), quality=tuple(quality), region=tuple(region),
                              fixed=fixed, keep_previous=keep_prev, body=str(entry.get("body", "")))

    def plan(self, step_id: str, entry: Mapping[str, Any], after: Geometry,
             solid_bodies: Sequence[str], active_body_bbox, prev_view: Optional[dict] = None):
        """Single-step view (the ``frame`` command): best candidate on its own."""
        from .camplan import plan_sequence
        c = self.measure(step_id, entry, after, solid_bodies, active_body_bbox)
        j = plan_sequence([c], self.cfg.camera.get("sequence"))[0]
        return (self.view_for(c.dirs[j], c.region, self.step_override(step_id, entry)),
                bool((self.step_override(step_id, entry) or {}).get("ghost", entry.get("role") == "sketch")))


# --------------------------------------------------------------------------
# Phase 2: timeline construction
# --------------------------------------------------------------------------
def _caption(i: int, n: int, entry: Mapping[str, Any], group: str, extra: str = "") -> str:
    s = f"Step {i}/{n}  {group} › {entry.get('body_label', entry.get('body'))} › {entry.get('label')}"
    return s + (f"  ({extra})" if extra else "")


def plan_timeline(full_tree: Mapping[str, Any], cfg: G.ProjectConfig, compiler: GeometryCompiler,
                  only_groups: Optional[Sequence[str]] = None,
                  log: Callable[[str], None] = _log) -> Timeline:
    tm = cfg.timing
    ordered = order_tree(full_tree, cfg)
    ids = T.feature_ids(ordered)
    queue = action_queue(ordered, cfg, only_groups)
    if not queue:
        raise ValueError("nothing to animate (empty action queue)")
    b2g = cfg.body_to_group()
    n = len(queue)
    log(f"action queue: {n} visual steps across {len({b2g[ordered[s]['body']] for s in queue})} group(s)")

    def before(step):
        i = ids.index(step)
        return T.state_up_to(ordered, ids[i - 1]) if i > 0 else T.empty_state(ordered)

    # Compile all finished-step states (with focus data) and their baselines in one batch.
    afters = [T.state_up_to(ordered, s) for s in queue]
    baselines = [before(s) for s in queue]
    compiler.compile_many(afters, focus=True, progress=log)
    compiler.compile_many(baselines, progress=log)

    # Where each sketch's lines live: the model folder of its own finished step.
    sketch_keys = {s: compiler.cached_key(afters[k], focus=True)
                   for k, s in enumerate(queue) if ordered[s]["role"] == "sketch"}
    fade_n = int(tm.get("sketch_fade_frames", 0))

    def ramp(a: float, b: float, count: int) -> List[float]:
        """``count`` values easing from a (exclusive) to b (inclusive)."""
        return [a + (b - a) * C.ease((i + 1) / count, tm["easing"]) for i in range(count)]

    planner = ViewPlanner(cfg)
    frames: List[FrameSpec] = []
    steps: List[StepRecord] = []
    prev_view: Optional[dict] = None
    prev_ghost = 0.0
    try:
        # --- camera plan for the whole sequence (task 02) ---------------------
        cands = []
        for k, step in enumerate(queue):
            entry = ordered[step]
            modes = G.body_modes(cfg, ordered, b2g[entry["body"]])
            after_geom = compiler.compile(afters[k], focus=True)
            if after_geom.info.get("errors") or after_geom.info.get("failed"):
                log(f"warning: step {step} has FreeCAD errors: "
                    f"{after_geom.info.get('errors') or after_geom.info.get('failed')}")
            cands.append(planner.measure(step, entry, after_geom,
                                         [b for b in after_geom.bodies if modes.get(b) == "solid"],
                                         after_geom.body_bbox([entry["body"]])))
        choice = plan_sequence(cands, cfg.camera.get("sequence"))
        shot_ids = shots(cands, choice)
        shot_region: Dict[int, List[float]] = {}
        for c, sid in zip(cands, shot_ids):
            if sid >= 0:
                shot_region[sid] = C.union_bbox([shot_region.get(sid), list(c.region)])
        planned_views = []
        for k, (c, j, sid) in enumerate(zip(cands, choice, shot_ids)):
            override = planner.step_override(queue[k], ordered[queue[k]])
            region = shot_region[sid] if sid >= 0 else list(c.region)
            planned_views.append(planner.view_for(c.dirs[j], region, override, outline=sid >= 0))
        log(f"camera plan: {len(set(s for s in shot_ids if s >= 0))} shot(s) for "
            f"{sum(1 for c in cands if not c.is_sketch)} 3D step(s), "
            f"{sum(1 for c in cands if c.is_sketch)} sketch view(s)")

        for k, step in enumerate(queue):
            entry = ordered[step]
            group = b2g[entry["body"]]
            modes = G.body_modes(cfg, ordered, group)
            after_geom = compiler.compile(afters[k], focus=True)
            view = planned_views[k]
            override = planner.step_override(step, entry) or {}
            ghost = bool(override.get("ghost", entry["role"] == "sketch"))
            growth: Optional[GrowthPlan] = plan_growth(
                afters[k], step,
                tm["sketch_frames"] if entry["role"] == "sketch" else tm["growth_frames"],
                after_geom.focus, tm["easing"])
            rec = StepRecord(step_index=k + 1, step_id=step, label=entry.get("label", step),
                             kind=entry["kind"], role=entry["role"], body=entry["body"],
                             body_label=entry.get("body_label", entry["body"]), group=group,
                             animation=growth.description if growth else "appear", view=view,
                             ghost=ghost, shot=shot_ids[k])
            cap = _caption(k + 1, n, entry, group, rec.animation if growth else "")
            base_kw = dict(step_id=step, step_index=k + 1, group=group, modes=modes)
            g_level = 1.0 if ghost else 0.0

            if k == 0:  # Tree_State_0 / intro: the canvas before anything is built
                for _ in range(max(1, int(tm["intro_frames"]))):
                    frames.append(FrameSpec(state=baselines[0], camera=view, frame_type=FRAME_STATIC,
                                            step_id=None, step_index=0, group=group, modes=modes,
                                            caption=""))
            rec.first_frame = len(frames)
            # Inter-step: camera move over the historical (pre-step) state. The
            # see-through amount eases from the previous step's level to this one's
            # (into a sketch: everything fades to translucent while the camera turns
            # perpendicular to it; out of a sketch: it fades back).
            if prev_view is not None and not C.views_equal(prev_view, view):
                cams = C.interpolate_camera(prev_view, view, int(tm["transition_frames"]) + 1,
                                            tm["easing"])[1:]
                for cv, gv in zip(cams, ramp(prev_ghost, g_level, len(cams))):
                    frames.append(FrameSpec(state=baselines[k], camera=cv, frame_type=FRAME_INTER,
                                            caption=cap, ghost=gv, **base_kw))
            elif prev_ghost != g_level:  # same view: fade in place
                count = max(1, int(tm["transition_frames"]) // 2)
                for gv in ramp(prev_ghost, g_level, count):
                    frames.append(FrameSpec(state=baselines[k], camera=view, frame_type=FRAME_INTER,
                                            caption=cap, ghost=gv, **base_kw))

            # Frames showing the step itself: growth (or appear), then hold.
            body_frames = []
            if growth:
                for st, vals in zip(growth.states, growth.values):
                    body_frames.append(dict(state=st, frame_type=FRAME_INTRA, params=vals))
            else:
                body_frames += [dict(state=afters[k], frame_type=FRAME_STATIC)] * int(tm["appear_frames"])
            body_frames += [dict(state=afters[k], frame_type=FRAME_STATIC)] * int(tm["hold_frames"])
            # The profile sketch has served its purpose: fade it out over the first
            # frames of the feature instead of letting it vanish.
            prof = entry.get("profile")
            fading = prof in sketch_keys and fade_n > 0 and sketch_keys[prof]
            for i, bf in enumerate(body_frames):
                overlays = ()
                if fading and i < fade_n:
                    overlays = ((sketch_keys[prof], prof, round(1.0 - (i + 1) / (fade_n + 1), 4)),)
                frames.append(FrameSpec(camera=view, caption=cap, ghost=g_level, overlays=overlays,
                                        **bf, **base_kw))
            rec.last_frame = len(frames) - 1
            steps.append(rec)
            prev_view = view
            prev_ghost = g_level
            log(f"  [{k + 1:>3}/{n}] {entry.get('body_label')}/{entry.get('label'):<16} "
                f"{rec.animation:<28} {'sketch view' if rec.shot < 0 else f'shot {rec.shot}':<10} "
                f"dir={np.round(C.view_direction(view), 2)}"
                f"{'  (see-through)' if ghost else ''}")

        # Finale: turntable of the finished model (everything solid).
        nf = int(tm.get("finale_frames", 0))
        if nf > 0:
            final_state = afters[-1]
            fg = compiler.compile(final_state)
            visible = [b for b in fg.bodies if b2g.get(b) != G.HIDDEN_GROUP]
            bbox = fg.body_bbox(visible)
            if bbox:
                center, radius = C.bbox_sphere(bbox)
                start = C.make_view(center, C.spherical(-60, 28), C.fit_distance(
                    radius, planner.view_angle, planner.aspect, margin=1.05),
                    view_angle=planner.view_angle)
                all_solid = G.body_modes(cfg, ordered, None)
                kw = dict(step_id="finale", step_index=n + 1, group=None, modes=all_solid,
                          caption="Finished model")
                cams = C.interpolate_camera(prev_view, start, int(tm["transition_frames"]) + 1,
                                            tm["easing"])[1:]
                for cv, gv in zip(cams, ramp(prev_ghost, 0.0, len(cams))):
                    frames.append(FrameSpec(state=final_state, camera=cv, frame_type=FRAME_INTER,
                                            ghost=gv, **kw))
                for cv in C.orbit(start, nf):
                    frames.append(FrameSpec(state=final_state, camera=cv, frame_type=FRAME_INTER, **kw))
    finally:
        planner.close()
    return Timeline(ordered_tree=ordered, frames=frames, steps=steps)


# --------------------------------------------------------------------------
# Phase 3: compile, render, manifest
# --------------------------------------------------------------------------
def _render_chunk(args):
    """Worker process: render requests from model folders (STLs). No FreeCAD."""
    scene_d, requests = args
    from .geometry import EMPTY, load_model_dir
    from .render import FrameRequest, Renderer, SceneSpec
    r = Renderer(SceneSpec(**scene_d))
    try:
        for req in requests:
            req = dict(req)
            model = req.pop("model_dir")
            geom = load_model_dir(model, req["geometry_key"]) if model else EMPTY
            overlays = {m: load_model_dir(m) for m, _, _ in req.get("overlays", ())}
            r.render(FrameRequest(**req), geom, overlays)
    finally:
        r.close()
    return len(requests)


def render_requests(requests: List[dict], scene_d: dict, workers: int,
                    log: Callable[[str], None] = _log) -> None:
    if not requests:
        return
    nchunks = max(1, min(workers, len(requests)))
    size = -(-len(requests) // nchunks)
    chunks = [(scene_d, requests[i:i + size]) for i in range(0, len(requests), size)]
    if len(chunks) == 1:
        _render_chunk(chunks[0])
    else:
        import multiprocessing as mp
        with ProcessPoolExecutor(max_workers=len(chunks), mp_context=mp.get_context("spawn")) as ex:
            list(ex.map(_render_chunk, chunks))


def _scene_dict(cfg: G.ProjectConfig) -> dict:
    return {"body_group": cfg.body_to_group(), "colors": cfg.colors(), "style": dict(cfg.render)}


def _clear_frames(frames_dir: str) -> None:
    os.makedirs(frames_dir, exist_ok=True)
    for old in os.listdir(frames_dir):
        if old.startswith("frame_") and old.endswith(".png"):
            os.remove(os.path.join(frames_dir, old))


def render_timeline(tl: Timeline, cfg: G.ProjectConfig, compiler: GeometryCompiler, outdir: str,
                    workers: int = 4, log: Callable[[str], None] = _log) -> Dict[str, Any]:
    t0 = time.time()
    keys = compiler.compile_many([f.state for f in tl.frames], progress=log)

    # Drop intra-step frames whose state failed to build (FreeCAD rejected an
    # intermediate parameter value); the timeline stays continuous.
    kept: List[tuple] = []
    dropped = 0
    for f, k in zip(tl.frames, keys):
        if f.frame_type == FRAME_INTRA and not compiler.load(k).ok:
            dropped += 1
            continue
        kept.append((f, k))
    if dropped:
        log(f"dropped {dropped} intra-step frame(s) whose intermediate state FreeCAD could not build")

    manifest = write_manifest(tl, cfg, kept, outdir, compiler)
    _clear_frames(os.path.join(outdir, "frames"))
    log(f"rendering {len(kept)} frame(s) at {cfg.render['width']}x{cfg.render['height']} "
        f"with {workers} worker(s)")
    render_requests(_requests_from_manifest(manifest, outdir), _scene_dict(cfg), workers, log)
    log(f"compiled + rendered in {time.time() - t0:.0f}s")
    return manifest


def _requests_from_manifest(manifest: Mapping[str, Any], outdir: str,
                            indices: Optional[Sequence[int]] = None) -> List[dict]:
    models = os.path.join(outdir, manifest["models_dir"])
    frames = manifest["frames"] if indices is None else [manifest["frames"][i] for i in indices]
    return [{"geometry_key": fr["geometry_key"],
             "model_dir": os.path.join(models, fr["geometry_key"]) if fr["model"] else None,
             "camera": fr["camera"],
             "output": os.path.join(outdir, fr["file"]),
             "active_group": fr["group"], "caption": fr["caption"], "reveal": fr["reveal"],
             "modes": fr["context_modes"], "ghost": fr.get("ghost", 0.0),
             "overlays": [(os.path.join(outdir, o["model"]), o["sketch"], o["opacity"])
                          for o in fr.get("overlays", [])]}
            for fr in frames]


def rerender(outdir: str, workers: int = 4, indices: Optional[Sequence[int]] = None,
             log: Callable[[str], None] = _log) -> Dict[str, Any]:
    """Re-draw frames from ``manifest.json`` + the stored STL models, without FreeCAD.

    Edit a frame's ``camera`` (or ``config.json``'s render/colour settings)
    and rerender. Cameras are viewer-side only.
    """
    with open(os.path.join(outdir, "manifest.json")) as fh:
        manifest = json.load(fh)
    with open(os.path.join(outdir, "config.json")) as fh:
        cfg_d = json.load(fh)
    tree = T.load(os.path.join(outdir, manifest["tree_file"]))
    cfg = G.from_dict(cfg_d, tree, base_dir=outdir)
    if indices is None:
        _clear_frames(os.path.join(outdir, "frames"))
    reqs = _requests_from_manifest(manifest, outdir, indices)
    for r in reqs:
        r["camera"] = C.complete_view(r["camera"])  # accept hand-edited views
    t0 = time.time()
    render_requests(reqs, _scene_dict(cfg), workers, log)
    log(f"re-rendered {len(reqs)} frame(s) in {time.time() - t0:.0f}s (no FreeCAD)")
    return manifest


def review_sheet(outdir: str, cols: int = 6, tile: Tuple[int, int] = (480, 270),
                 log: Callable[[str], None] = _log) -> str:
    """``<out>/steps_sheet.png``: one tile per step (its last frame) plus the finale.

    Rendered directly from the stored models and the manifest's cameras, so it
    works right after ``--plan-only`` too: the quick way to review a camera
    plan without rendering the animation.
    """
    import tempfile
    from PIL import Image, ImageDraw
    from .geometry import EMPTY, load_model_dir
    from .render import FrameRequest, Renderer, SceneSpec
    with open(os.path.join(outdir, "manifest.json")) as fh:
        manifest = json.load(fh)
    with open(os.path.join(outdir, "config.json")) as fh:
        cfg_d = json.load(fh)
    cfg = G.from_dict(cfg_d, T.load(os.path.join(outdir, manifest["tree_file"])), base_dir=outdir)
    picks = [(s, s["last_frame"]) for s in manifest["steps"] if s.get("last_frame") is not None]
    if manifest["frames"] and manifest["frames"][-1]["step_id"] == "finale":
        picks.append(({"step_index": "end", "label": "finished", "shot": None}, len(manifest["frames"]) - 1))
    scene = _scene_dict(cfg)
    scene["style"] = dict(scene["style"], width=tile[0], height=tile[1], transparent=False)
    r = Renderer(SceneSpec(**scene))
    rows = -(-len(picks) // cols)
    sheet = Image.new("RGB", (tile[0] * cols, tile[1] * rows), "white")
    draw = ImageDraw.Draw(sheet)
    with tempfile.TemporaryDirectory() as tmp:
        for n, (step, idx) in enumerate(picks):
            req = _requests_from_manifest(manifest, outdir, [idx])[0]
            model = req.pop("model_dir")
            req["output"] = os.path.join(tmp, f"{n}.png")
            shot = step.get("shot")
            req["caption"] = (f"{step['step_index']} {step.get('body_label', '')}/{step['label']}"
                              f"  [{'sketch' if shot == -1 else 'end' if shot is None else f'shot {shot}'}]")
            geom = load_model_dir(model, req["geometry_key"]) if model else EMPTY
            r.render(FrameRequest(**req), geom, {m: load_model_dir(m) for m, _, _ in req["overlays"]})
            x, y = (n % cols) * tile[0], (n // cols) * tile[1]
            sheet.paste(Image.open(req["output"]).convert("RGB"), (x, y))
            draw.rectangle([x, y, x + tile[0] - 1, y + tile[1] - 1], outline="#bbbbbb")
    r.close()
    path = os.path.join(outdir, "steps_sheet.png")
    sheet.save(path)
    log(f"review sheet: {path}")
    return path


def write_manifest(tl: Timeline, cfg: G.ProjectConfig, kept: List[tuple], outdir: str,
                   compiler: GeometryCompiler) -> Dict[str, Any]:
    fps = float(cfg.timing["fps"])
    full = tl.ordered_tree
    models_rel = os.path.relpath(compiler.cache_dir, os.path.abspath(outdir))
    frames = []
    for i, (f, key) in enumerate(kept):
        g = compiler.load(key) if key else None
        model = None if (not key or key == "empty") else f"{models_rel}/{key}"
        stl = {b: f"{model}/{v.get('stl', b + '.stl')}" for b, v in (g.info.get("bodies", {}) if g else {}).items()} if model else {}
        frames.append({
            "index": i,
            "file": f"frames/frame_{i:05d}.png",
            "timestamp": round(i / fps, 4),
            "frame_type": f.frame_type,
            "step_id": f.step_id,
            "step_index": f.step_index,
            "group": f.group,
            "params": dict(f.params) if f.params else None,
            "caption": f.caption,
            "camera": f.camera,
            "geometry_key": key or "empty",
            "model": model,
            "stl": stl,
            "reveal": {fid: e["render_reveal"] for fid, e in f.state.items()
                       if fid != T.META_KEY and "render_reveal" in e},
            "ghost": round(float(f.ghost), 4),
            "overlays": [{"model": f"{models_rel}/{k}", "sketch": sk, "opacity": a}
                         for k, sk, a in f.overlays],
            "context_modes": dict(f.modes),
            "state": {"upto": T.last_feature(f.state), "overrides": T.diff_overrides(full, f.state)},
        })
    steps = []
    for s in tl.steps:
        d = asdict(s)
        own = [fr["index"] for fr in frames if fr["step_id"] == s.step_id]
        d["first_frame"], d["last_frame"] = (own[0], own[-1]) if own else (None, None)
        d["start_time"] = round(own[0] / fps, 4) if own else None
        steps.append(d)
    manifest = {
        "schema": "cadanim.manifest/1",
        "source": T.meta(full).get("source"),
        "source_sha256": T.meta(full).get("source_sha256"),
        "fps": fps,
        "resolution": [cfg.render["width"], cfg.render["height"]],
        "frame_count": len(frames),
        "duration": round(len(frames) / fps, 3),
        "frame_types": {t: sum(1 for f in frames if f["frame_type"] == t)
                        for t in (FRAME_STATIC, FRAME_INTER, FRAME_INTRA)},
        "groups": [{"name": g.name, "color": g.color, "export": g.export, "bodies": list(g.bodies)}
                   for g in cfg.groups],
        "tree_file": "tree.json",
        "models_dir": models_rel,
        "steps": steps,
        "frames": frames,
    }
    T.save(full, os.path.join(outdir, "tree.json"))
    with open(os.path.join(outdir, "config.json"), "w") as fh:
        json.dump(cfg.to_dict(), fh, indent=1)
    with open(os.path.join(outdir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    return manifest


def run(full_tree: Mapping[str, Any], cfg: G.ProjectConfig, outdir: str, *, cache_dir: str,
        workers: int = 4, only_groups: Optional[Sequence[str]] = None,
        plan_only: bool = False, log: Callable[[str], None] = _log) -> Dict[str, Any]:
    fp = T.fingerprint(full_tree)
    os.makedirs(outdir, exist_ok=True)
    compiler = GeometryCompiler(cfg.source, cache_dir, workers=workers,
                                deflection=float(cfg.render["deflection"]))
    tl = plan_timeline(full_tree, cfg, compiler, only_groups, log)
    log(f"timeline: {len(tl.frames)} frames "
        f"({len(tl.frames) / float(cfg.timing['fps']):.1f}s at {cfg.timing['fps']} fps)")
    if plan_only:
        kept = [(f, compiler.cached_key(f.state) or "") for f in tl.frames]
        manifest = write_manifest(tl, cfg, kept, outdir, compiler)
    else:
        manifest = render_timeline(tl, cfg, compiler, outdir, workers, log)
    if T.fingerprint(full_tree) != fp:
        raise AssertionError("the source tree was mutated during the pipeline run")
    review_sheet(outdir, log=log)
    return manifest
