"""Part groups: colour, export behaviour and context visibility.

A *group* is a set of bodies that belongs together physically. Typical uses:

* ``"export": "union"``: the bodies are separately modelled modules that are
  printed as one solid. Their meshes are written into a single STL; no
  boolean fuse is needed because slicers merge touching shells.
* ``"export": "separate"``: one STL per body.

Every body of a group shares the group colour. While a step of group A is
being animated, every *other* group B is shown according to the context
policy:

* ``"solid"``: fully visible (the default for the specific *bodies* that A
  binds to through SubShapeBinders, i.e. binding/mating surfaces)
* ``"translucent"``: ghosted, visible for as long as it exists
* ``"hidden"``: not drawn

Project config file (``*.cadanim.json``)::

    {
      "source": "BirdFeeder.FCStd",
      "groups": [
        {"name": "Feeder", "color": "#4f7cac", "export": "union",
         "bodies": ["Shaft", "Base", "ShaftFastner", "Platform"]},
        {"name": "Roof", "color": "#e0a03a", "export": "separate", "bodies": ["Top"]}
      ],
      "context": {"default": "translucent", "binding": "solid",
                  "overrides": {"Roof": {"Feeder": "translucent"}}},
      "timing": {...}, "render": {...}, "camera": {...}
    }

Bodies may be referenced by FreeCAD Name (``Body003``) or Label (``Top``).
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from . import tree as T

PALETTE = ["#4f7cac", "#e0a03a", "#5cb85c", "#c0504d", "#8e6bbf", "#2aa198", "#d17fa8", "#7f8c8d"]
MODES = ("solid", "translucent", "hidden")
HIDDEN_GROUP = "__ungrouped__"

DEFAULT_TIMING = {
    "fps": 24,
    "sketch_frames": 14,       # sketch drawn progressively
    "growth_frames": 24,       # feature growth (intra-step)
    "appear_frames": 6,        # features without a growth animation
    "hold_frames": 8,          # static frames after each step
    "sketch_fade_frames": 12,  # a profile sketch fades out over this many frames of its feature
    "transition_frames": 18,   # camera move between steps (inter-step)
    "intro_frames": 6,         # blank canvas at the start
    "finale_frames": 96,       # turntable of the finished model (0 = off)
    "easing": "ease_in_out",
}
DEFAULT_RENDER = {
    "width": 1280,
    "height": 720,
    "transparent": True,       # alpha PNG frames; the video gets a background
    "background": "#ffffff",
    "translucent_opacity": 0.18,
    "ghost_opacity": 0.35,     # every body while a sketch step is shown (faded in/out)
    "feature_edges": True,
    "edge_color": "#222222",
    "sketch_color": "#111111",
    "sketch_width": 3.0,
    "captions": True,
    "deflection": 0.1,         # tessellation tolerance (mm)
    "anti_aliasing": "ssaa",
}
DEFAULT_CAMERA = {
    "view_angle": 24.0,        # narrower = less perspective distortion
    "margin": 1.12,            # breathing room around the framed outline
    "context_weight": 1.0,     # 1 = whole active body in frame; lower zooms towards the feature
    "focus_scale": 1.6,        # framing radius relative to the feature radius
    "small_feature_ratio": 0.25,  # features smaller than this (vs body) get zoomed in on ...
    "min_context": 0.65,       # ... but at least this much context weight is kept
    "min_body_fraction": 0.5,  # a small body is framed at >= this share of the built assembly
    "elevation_bias": 0.35,    # preference for looking from above
    "probe_size": [240, 180],
    # Sequence planning (cadanim/camplan.py): costs traded off across all steps.
    "sequence": {
        "quality": 3.0,         # per step: losing visibility / preferred elevation
        "fixed_move": 1.0,      # per camera move (moves into/out of sketches always count)
        "move": 1.0,            # per 90 deg of rotation
        "home": 1.0,            # per 90 deg between 3D views separated by sketches
        "min_visibility": 0.6,  # 3D steps: never below this share of the best visibility
    },
    # Manual per-step views, applied by the viewer (no FreeCAD rerun). Keyed by
    # feature id, label or "BodyLabel/FeatureLabel". Each value may use
    # direction [x,y,z] | azimuth/elevation (deg) | keep_previous, plus
    # zoom, distance_scale, focal_offset [x,y,z] and ghost (bool).
    "step_views": {},
}


@dataclass(frozen=True)
class Group:
    name: str
    color: str
    bodies: Tuple[str, ...]
    export: str = "union"


@dataclass(frozen=True)
class ProjectConfig:
    source: str
    groups: Tuple[Group, ...]
    context_default: str = "translucent"
    context_binding: str = "solid"
    context_overrides: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    timing: Mapping[str, Any] = field(default_factory=lambda: dict(DEFAULT_TIMING))
    render: Mapping[str, Any] = field(default_factory=lambda: dict(DEFAULT_RENDER))
    camera: Mapping[str, Any] = field(default_factory=lambda: dict(DEFAULT_CAMERA))

    # ------------------------------------------------------------------
    def group_names(self) -> List[str]:
        return [g.name for g in self.groups]

    def group(self, name: str) -> Group:
        for g in self.groups:
            if g.name == name:
                return g
        raise KeyError(name)

    def body_to_group(self) -> Dict[str, str]:
        return {b: g.name for g in self.groups for b in g.bodies}

    def colors(self) -> Dict[str, str]:
        return {g.name: g.color for g in self.groups}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "groups": [{"name": g.name, "color": g.color, "export": g.export,
                        "bodies": list(g.bodies)} for g in self.groups if g.name != HIDDEN_GROUP],
            "context": {"default": self.context_default, "binding": self.context_binding,
                        "overrides": {k: dict(v) for k, v in self.context_overrides.items()}},
            "timing": dict(self.timing),
            "render": dict(self.render),
            "camera": dict(self.camera),
        }


# --------------------------------------------------------------------------
# loading / validation
# --------------------------------------------------------------------------
def _resolve_body(ref: str, bodies_meta: Mapping[str, Any]) -> str:
    if ref in bodies_meta:
        return ref
    matches = [n for n, m in bodies_meta.items() if m.get("label") == ref]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        known = ", ".join(f"{n} ({m.get('label')})" for n, m in bodies_meta.items())
        raise ValueError(f"body {ref!r} not found (known: {known})")
    raise ValueError(f"body label {ref!r} is ambiguous: {matches}")


def from_dict(data: Mapping[str, Any], tree: Mapping[str, Any], base_dir: str = ".") -> ProjectConfig:
    bodies_meta = T.meta(tree).get("bodies", {})
    groups: List[Group] = []
    used: Dict[str, str] = {}
    for i, g in enumerate(data.get("groups", [])):
        name = g["name"]
        if name in [x.name for x in groups]:
            raise ValueError(f"duplicate group name {name!r}")
        bodies = []
        for ref in g.get("bodies", []):
            b = _resolve_body(ref, bodies_meta)
            if b in used:
                raise ValueError(f"body {ref!r} is in both {used[b]!r} and {name!r}")
            used[b] = name
            bodies.append(b)
        export = g.get("export", "union")
        if export not in ("union", "separate"):
            raise ValueError(f"group {name!r}: export must be 'union' or 'separate'")
        groups.append(Group(name, g.get("color", PALETTE[i % len(PALETTE)]), tuple(bodies), export))

    # Bodies that are not assigned: hidden helper bodies are still built (other
    # bodies may bind to them) but never drawn or exported.
    leftover = tuple(b for b in bodies_meta if b not in used)
    if leftover:
        groups.append(Group(HIDDEN_GROUP, "#999999", leftover, "separate"))

    ctx = data.get("context", {})
    for key in ("default", "binding"):
        if ctx.get(key, "solid") not in MODES:
            raise ValueError(f"context.{key} must be one of {MODES}")
    for a, row in ctx.get("overrides", {}).items():
        for b, mode in row.items():
            if mode not in MODES:
                raise ValueError(f"context override {a}->{b}: {mode!r} not in {MODES}")

    src = data.get("source") or T.meta(tree).get("source", "")
    if src and not os.path.isabs(src):
        src = os.path.normpath(os.path.join(base_dir, src))
    return ProjectConfig(
        source=src,
        groups=tuple(groups),
        context_default=ctx.get("default", "translucent"),
        context_binding=ctx.get("binding", "solid"),
        context_overrides=copy.deepcopy(ctx.get("overrides", {})),
        timing={**DEFAULT_TIMING, **data.get("timing", {})},
        render={**DEFAULT_RENDER, **data.get("render", {})},
        camera={**DEFAULT_CAMERA, **data.get("camera", {})},
    )


def load_config(path: str, tree: Mapping[str, Any]) -> ProjectConfig:
    with open(path) as fh:
        data = json.load(fh)
    return from_dict(data, tree, base_dir=os.path.dirname(os.path.abspath(path)))


def default_config(tree: Mapping[str, Any], source: Optional[str] = None) -> ProjectConfig:
    """One group per visible body: the starting point when nothing is known."""
    bodies = T.meta(tree).get("bodies", {})
    data = {"source": source or T.meta(tree).get("source"),
            "groups": [{"name": m.get("label", n), "bodies": [n], "export": "separate",
                        "color": PALETTE[i % len(PALETTE)]}
                       for i, (n, m) in enumerate((n, m) for n, m in bodies.items() if m.get("visible", True))]}
    return from_dict(data, tree)


# --------------------------------------------------------------------------
# context policy
# --------------------------------------------------------------------------
def bound_bodies(tree: Mapping[str, Any], cfg: ProjectConfig, group: str) -> set:
    """Bodies of *other* groups that ``group`` binds to through SubShapeBinders."""
    b2g = cfg.body_to_group()
    out = set()
    for fid in T.feature_ids(tree):
        e = tree[fid]
        if e.get("role") == "binder" and b2g.get(e.get("body")) == group:
            out.update(b for b in e.get("binds", []) if b2g.get(b) not in (None, group))
    return out


def body_modes(cfg: ProjectConfig, tree: Mapping[str, Any], active: Optional[str]) -> Dict[str, str]:
    """Display mode of every body while a step of group ``active`` is shown.

    Precedence: hidden helper bodies > active group (solid) > explicit
    override for (active, other group) > binding surfaces (bodies the active
    group binds to) > default.
    """
    bound = bound_bodies(tree, cfg, active) if active else set()
    out: Dict[str, str] = {}
    for g in cfg.groups:
        for b in g.bodies:
            if g.name == HIDDEN_GROUP:
                out[b] = "hidden"
            elif active is None or g.name == active:
                out[b] = "solid"
            elif g.name in cfg.context_overrides.get(active, {}):
                out[b] = cfg.context_overrides[active][g.name]
            elif b in bound:
                out[b] = cfg.context_binding
            else:
                out[b] = cfg.context_default
    return out


def suggest_report(tree: Mapping[str, Any], inspect: Mapping[str, Any]) -> str:
    """Human-readable facts that help decide on a grouping."""
    bodies = T.meta(tree).get("bodies", {})
    lab = lambda n: f"{bodies.get(n, {}).get('label', n)} ({n})"
    lines = ["Bodies:"]
    for b in inspect.get("bodies", []):
        bb = b["bbox"]
        size = " x ".join(f"{bb[i + 3] - bb[i]:.1f}" for i in range(3)) if bb else "?"
        lines.append(f"  - {lab(b['name'])}: {size} mm, volume {b['volume']:.0f} mm3, "
                     f"{b['solids']} solid(s){'' if b['visible'] else ', HIDDEN'}")
    if inspect.get("binders"):
        lines.append("Binder links (body built on another body's geometry):")
        for x in inspect["binders"]:
            lines.append(f"  - {lab(x['body'])} <- {lab(x['source'])} via {x['binder']}")
    if inspect.get("contacts"):
        lines.append("Touching / overlapping bodies (fusing them gives one solid):")
        for c in inspect["contacts"]:
            ov = c.get("overlap_volume")
            kind = f"overlap {ov:.1f} mm3" if ov and ov > 1e-6 else "touching"
            lines.append(f"  - {lab(c['a'])} & {lab(c['b'])}: {kind}")
    lines.append("Decide which bodies print as one solid (export: union) and which print "
                 "separately; bodies in the same group share a colour.")
    return "\n".join(lines)
