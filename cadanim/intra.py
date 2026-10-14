"""Object-space iteration: transient tree states that show a feature growing.

``create_intra_step_iterator`` is the single-parameter form from the spec.
``create_intra_step_iterator_fn`` is the general higher-order form: it takes a
function ``t -> {property: value}`` so that features needing coupled
parameters (polar patterns change Occurrences *and* Angle) or a mode switch
(ThroughAll pockets are animated as Length pockets) use the same machinery.

``plan_growth`` maps a feature to its animation behaviour:

    Pad / Pocket (Length, TwoLengths)    -> Length (+Length2) 0 -> L
    Pad / Pocket (ThroughAll, UpTo*)     -> temporary Type=Length 0 -> swept depth,
                                            final frame restores the original type
    Revolution / Groove (Angle, TwoAngles) -> Angle (+Angle2) 0 -> A
    PolarPattern / LinearPattern         -> instances appear one at a time
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
    depth = (focus or {}).get("sweep_depth")
    if depth:
        total = 2 * depth if entry.get("Midplane") else depth

        def at(t):
            if t >= 1.0:
                return {"Type": typ, "Length": L1}  # exact original definition
            return {"Type": "Length", "Length": _scaled(total, t)}
        return f"{typ}: depth 0 → {total:.3g} mm", at, frames
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
    n = int(entry.get("Occurrences", 1) or 1)
    if n < 2:
        return None
    angle = float(entry.get("Angle", 360.0) or 360.0)
    spacing_mode = entry.get("Mode") == "Spacing"
    full_circle = math.isclose(angle, 360.0, abs_tol=1e-6)

    def at(t):
        k = max(1, min(n, 1 + int(round(t * (n - 1)))))
        if k == n:
            return {"Occurrences": n, "Angle": angle}
        if spacing_mode:
            return {"Occurrences": k}
        # Keep each instance at its final position while adding instances.
        step = angle / n if full_circle else angle / (n - 1)
        return {"Occurrences": k, "Angle": step * (k - 1) if k > 1 else step}
    return f"{n} instances", at, max(frames, n)


def _linear(entry, focus, frames):
    n = int(entry.get("Occurrences", 1) or 1)
    if n < 2:
        return None
    length = float(entry.get("Length", 0) or 0)
    spacing_mode = entry.get("Mode") == "Spacing"

    def at(t):
        k = max(1, min(n, 1 + int(round(t * (n - 1)))))
        if k == n:
            return {"Occurrences": n, "Length": length}
        if spacing_mode:
            return {"Occurrences": k}
        step = length / (n - 1)
        return {"Occurrences": k, "Length": step * (k - 1) if k > 1 else step}
    return f"{n} instances", at, max(frames, n)


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
    # Patterns are discrete: sample linearly so every instance gets equal time.
    is_discrete = entry.get("role") == "pattern"
    states = create_intra_step_iterator_fn(step_state, step_id, overrides_at, n,
                                           "linear" if is_discrete else easing)
    values = [{k: s[step_id][k] for k in overrides_at(1.0)} for s in states]
    return GrowthPlan(step_id=step_id, description=desc, states=states, values=values)
