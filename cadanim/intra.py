"""Object-space iteration: transient tree states that show a feature growing.

``create_intra_step_iterator`` is the single-parameter form from the spec.
``create_intra_step_iterator_fn`` is the general higher-order form: it takes a
function ``t -> {property: value}`` so that features needing coupled
parameters (polar patterns change Occurrences *and* Angle) or a mode switch
(ThroughAll pockets are animated as Length pockets) use the same machinery.

``plan_growth`` maps a feature to its animation behaviour:

    Pad / Pocket (Length, TwoLengths)    -> Length (+Length2) 0 -> L
    Pad / Pocket (ThroughAll, UpTo*)     -> anim_sweep 0 -> 1: the finished feature's
                                            material revealed in growing slabs from the
                                            sketch plane (compiler side)
    Revolution / Groove (Angle, TwoAngles) -> Angle (+Angle2) 0 -> A
    PolarPattern                         -> all copies, Angle 0 -> final (Spacing mode: Offset)
    LinearPattern                        -> all copies, Length 0 -> final (Spacing mode: Offset)
    Fillet                               -> Radius 0 -> R
    Chamfer                              -> Size 0 -> S
    Draft                                -> Angle 0 -> A
    Sketch                               -> render_reveal 0 -> 1 (drawn progressively)

All functions are pure: the base tree is deep-copied for every state.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional

from . import tree as T
from .camera import ease

Overrides = Dict[str, Any]


def create_intra_step_iterator(base_tree: dict, target_step_id: str, param_key: str,
                               start_val: float, end_val: float, steps_count: int) -> List[dict]:
    """Uniformly spaced states with ``tree[target_step_id][param_key]`` varying.

    ``base_tree`` must already contain the entry for ``target_step_id`` (with
    its final definition). It is never modified.
    """
    if target_step_id not in base_tree:
        raise KeyError(f"{target_step_id!r} is not part of the base tree")
    step_size = (end_val - start_val) / max(steps_count - 1, 1)
    states = []
    for i in range(steps_count):
        state_copy = copy.deepcopy(base_tree)
        state_copy[target_step_id][param_key] = start_val + i * step_size
        states.append(state_copy)
    return states


def create_intra_step_iterator_fn(base_tree: dict, target_step_id: str,
                                  overrides_at: Callable[[float], Mapping[str, Any]],
                                  steps_count: int, easing: str = "linear") -> List[dict]:
    """General form: ``overrides_at(t)`` gives the property overrides at t in [0, 1].

    Returns ``steps_count`` standalone states with t sampled uniformly over
    (0, 1]. The final sample is always t = 1, so the last state equals the
    finished feature.
    """
    if target_step_id not in base_tree:
        raise KeyError(f"{target_step_id!r} is not part of the base tree")
    states = []
    n = max(steps_count, 1)
    for i in range(n):
        t = ease((i + 1) / n, easing)
        state_copy = copy.deepcopy(base_tree)
        state_copy[target_step_id].update(copy.deepcopy(dict(overrides_at(t))))
        states.append(state_copy)
    return states


@dataclass(frozen=True)
class GrowthPlan:
    step_id: str
    description: str                 # e.g. "Length 0 -> 120 mm"
    states: List[dict] = field(repr=False)
    values: List[Overrides] = field(repr=False)  # overrides per state (for the manifest/captions)


# --------------------------------------------------------------------------
# feature behaviours
# --------------------------------------------------------------------------
MIN_FRACTION = 0.02  # FreeCAD rejects zero-length pads / zero-angle revolutions


def _scaled(final: float, t: float) -> float:
    return final * max(t, MIN_FRACTION)


def _sketch(entry, focus, frames):
    return "drawn", (lambda t: {"render_reveal": round(t, 4)}), frames


def _sweep(entry, focus, frames):
    typ = entry.get("Type", "Length")
    L1, L2 = float(entry.get("Length", 0) or 0), float(entry.get("Length2", 0) or 0)
    if typ == "Length" and L1 > 0:
        return f"Length 0 → {L1:g} mm", (lambda t: {"Length": _scaled(L1, t)}), frames
    if typ == "TwoLengths" and (L1 > 0 or L2 > 0):
        return (f"Lengths 0 → {L1:g}/{L2:g} mm",
                (lambda t: {"Length": _scaled(L1, t), "Length2": L2 * t}), frames)
    if entry.get("role") in ("subtractive", "additive") and entry.get("profile"):
        # ThroughAll / UpTo*: a Length feature cannot reproduce these in general
        # (both sides of the sketch plane, air gaps before the material starts),
        # so the compiler reveals the finished feature's material in growing
        # slabs around the sketch plane instead. t = 1 is the exact feature.
        def at(t):
            return {} if t >= 1.0 else {"anim_sweep": round(max(t, MIN_FRACTION), 4)}
        depth = (focus or {}).get("sweep_depth")
        size = f" {depth:.3g} mm" if depth else ""
        return f"{typ}: cut grows to full depth{size}", at, frames
    return None


def _revolve(entry, focus, frames):
    typ = entry.get("Type", "Angle")
    A1, A2 = float(entry.get("Angle", 0) or 0), float(entry.get("Angle2", 0) or 0)
    if typ == "Angle" and A1 > 0:
        return f"Angle 0 → {A1:g}°", (lambda t: {"Angle": _scaled(A1, t)}), frames
    if typ == "TwoAngles" and (A1 > 0 or A2 > 0):
        return (f"Angles 0 → {A1:g}/{A2:g}°",
                (lambda t: {"Angle": _scaled(A1, t), "Angle2": A2 * t}), frames)
    return None


def _polar(entry, focus, frames):
    """All copies from the first frame; the pattern angle opens from 0 to final.

    Extent mode spaces copies Angle/(n-1) apart, except a full 360 deg circle,
    which uses 360/n. Sweeping to 360*(n-1)/n therefore ends on exactly the
    final layout without a jump; the last frame restores the original Angle.
    Spacing mode sweeps the per-copy Offset instead.
    """
    n = int(entry.get("Occurrences", 1) or 1)
    if n < 2:
        return None
    angle = float(entry.get("Angle", 360.0) or 360.0)
    if entry.get("Mode") == "Spacing":
        offset = float(entry.get("Offset", 0) or 0)
        if offset <= 0:
            return None
        return (f"{n} copies, spacing 0 \u2192 {offset:g}\u00b0",
                lambda t: {"Occurrences": n, "Offset": _scaled(offset, t)}, frames)
    end = angle * (n - 1) / n if math.isclose(angle, 360.0, abs_tol=1e-6) else angle

    def at(t):
        if t >= 1.0:
            return {"Occurrences": n, "Angle": angle}
        return {"Occurrences": n, "Angle": _scaled(end, t)}
    return f"{n} copies, angle 0 \u2192 {angle:g}\u00b0", at, frames


def _linear(entry, focus, frames):
    """All copies from the first frame; the pattern length (or spacing) grows from 0."""
    n = int(entry.get("Occurrences", 1) or 1)
    if n < 2:
        return None
    if entry.get("Mode") == "Spacing":
        offset = float(entry.get("Offset", 0) or 0)
        if offset <= 0:
            return None
        return (f"{n} copies, spacing 0 \u2192 {offset:g} mm",
                lambda t: {"Occurrences": n, "Offset": _scaled(offset, t)}, frames)
    length = float(entry.get("Length", 0) or 0)
    if length <= 0:
        return None
    return (f"{n} copies, length 0 \u2192 {length:g} mm",
            lambda t: {"Occurrences": n, "Length": _scaled(length, t)}, frames)


def _dressup(prop, unit):
    def fn(entry, focus, frames):
        v = float(entry.get(prop, 0) or 0)
        if v <= 0:
            return None
        return f"{prop} 0 → {v:g}{unit}", (lambda t: {prop: _scaled(v, t)}), frames
    return fn


ANIMATORS: Dict[str, Callable] = {
    "Sketcher::SketchObject": _sketch,
    "PartDesign::Pad": _sweep,
    "PartDesign::Pocket": _sweep,
    "PartDesign::Revolution": _revolve,
    "PartDesign::Groove": _revolve,
    "PartDesign::PolarPattern": _polar,
    "PartDesign::LinearPattern": _linear,
    "PartDesign::Fillet": _dressup("Radius", " mm"),
    "PartDesign::Chamfer": _dressup("Size", " mm"),
    "PartDesign::Draft": _dressup("Angle", "°"),
}


def plan_growth(step_state: dict, step_id: str, frames: int, focus: Optional[dict] = None,
                easing: str = "ease_in_out") -> Optional[GrowthPlan]:
    """Growth animation for ``step_id`` or ``None`` when the feature just appears.

    ``step_state`` is the state that includes the finished step.
    """
    entry = step_state[step_id]
    fn = ANIMATORS.get(entry.get("kind"))
    if fn is None or frames <= 0:
        return None
    spec = fn(entry, focus, frames)
    if spec is None:
        return None
    desc, overrides_at, n = spec
    states = create_intra_step_iterator_fn(step_state, step_id, overrides_at, n, easing)
    values = [{k: v for k, v in s[step_id].items()
               if k in overrides_at(1.0) or k.startswith(T.ANIM_PREFIX)} for s in states]
    return GrowthPlan(step_id=step_id, description=desc, states=states, values=values)
