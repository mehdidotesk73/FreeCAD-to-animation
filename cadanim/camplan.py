"""Sequence-level camera planning (pure: no FreeCAD, no rendering).

Choosing each step's best view on its own makes the camera wander. This
module plans the whole sequence at once:

* Every step has candidate directions with a measured visibility of what
  the step changes (``StepCandidates``). Feature steps get the standard
  candidate directions. Sketch steps get exactly two: perpendicular to the
  sketch plane, from either side.
* ``plan_sequence`` picks one candidate per step by dynamic programming,
  minimising
      quality   * (best quality - chosen quality)            per step
    + fixed_move * (number of camera moves)
    + move      * (rotation of each move / 90 deg)
    + home      * (angle between consecutive 3D views that are separated by
                   sketches / 90 deg)   ("come back to the same 3D view")
  Candidates showing less than ``min_visibility`` of the step's best
  visibility are not allowed for feature steps. A move into or out of a
  sketch always counts as a move (its framing differs even if the direction
  matches), so there is no reward for doing a 3D step from a sketch's flat
  view.
* ``shots`` groups consecutive feature steps that use the same direction,
  sketches in between included, into shots. The pipeline gives every step
  of a shot one shared framing: within a shot the camera does not move at
  all, and after a sketch it returns to exactly the same view. Shots end at
  body boundaries.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

DEFAULT_WEIGHTS = {"quality": 3.0, "fixed_move": 1.0, "move": 1.0, "home": 1.0,
                   "min_visibility": 0.6}
_SAME_DEG = 1.0
_MAX_SKETCH_RUN = 10  # exhaustive side choice up to 2^10 combinations


@dataclass(frozen=True)
class StepCandidates:
    step: str
    role: str                        # "sketch" or a feature role
    dirs: Tuple[Tuple[float, float, float], ...]
    vis: Tuple[float, ...]           # visible target pixels per direction
    quality: Tuple[float, ...]       # higher is better (visibility + preferred elevation)
    region: Tuple[float, ...] = ()   # framing box of the step itself
    body: str = ""                   # shots never span bodies (framing scale differs)
    fixed: Optional[int] = None      # forced option (camera.step_views override)
    keep_previous: bool = False      # reuse the previous 3D step's direction

    @property
    def is_sketch(self) -> bool:
        return self.role == "sketch"


def angle_deg(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _options(s: StepCandidates, w: Mapping[str, float]) -> List[Tuple[int, float]]:
    """Allowed option indices with their quality loss."""
    if s.fixed is not None:
        return [(s.fixed, 0.0)]
    qbest = max(s.quality)
    best_vis = max(s.vis) or 1.0
    opts = [(j, qbest - q) for j, (v, q) in enumerate(zip(s.vis, s.quality))
            if s.is_sketch or v / best_vis >= float(w["min_visibility"])]
    return opts or [(int(np.argmax(s.quality)), 0.0)]


def _move(rot_deg: float, forced: bool, w: Mapping[str, float]) -> float:
    moved = forced or rot_deg > _SAME_DEG
    return (float(w["fixed_move"]) if moved else 0.0) + float(w["move"]) * rot_deg / 90.0


def _sketch_run(prev_dir, run: Sequence[StepCandidates], next_dir, w) -> Tuple[float, Tuple[int, ...]]:
    """Cheapest side choice for a run of sketches between two 3D views."""
    opt_lists = [_options(s, w) for s in run]
    best: Tuple[float, Tuple[int, ...]] = (math.inf, ())
    combos = itertools.product(*opt_lists) if len(run) <= _MAX_SKETCH_RUN else \
        [tuple(min(o, key=lambda x: x[1]) for o in opt_lists)]
    for combo in combos:
        dirs = [np.asarray(s.dirs[j]) for s, (j, _) in zip(run, combo)]
        path = ([prev_dir] if prev_dir is not None else []) + dirs + \
               ([next_dir] if next_dir is not None else [])
        cost = sum(float(w["quality"]) * q for _, q in combo)
        cost += sum(_move(angle_deg(a, b), True, w) for a, b in zip(path, path[1:]))
        if cost < best[0]:
            best = (cost, tuple(j for j, _ in combo))
    return best


def plan_sequence(steps: Sequence[StepCandidates],
                  weights: Optional[Mapping[str, float]] = None) -> List[int]:
    """Chosen option index for every step (see module docstring)."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    n = len(steps)
    groups: List[Tuple[List[int], int]] = []   # (sketches before, feature index)
    pending: List[int] = []
    for i, s in enumerate(steps):
        if s.is_sketch:
            pending.append(i)
        else:
            groups.append((pending, i))
            pending = []
    trailing = pending
    choice: List[Optional[int]] = [None] * n

    if not groups:  # sketches only
        _, sides = _sketch_run(None, [steps[i] for i in trailing], None, w)
        return list(sides)

    # cost[g][j] = best total cost with feature of group g using option j
    cost: List[Dict[int, float]] = []
    back: List[Dict[int, Tuple[Optional[int], Tuple[int, ...]]]] = []
    for g, (sk, fi) in enumerate(groups):
        f = steps[fi]
        cost.append({}); back.append({})
        run = [steps[i] for i in sk]
        for j, qloss in _options(f, w):
            d = np.asarray(f.dirs[j])
            base = float(w["quality"]) * qloss
            if g == 0:
                c, sides = _sketch_run(None, run, d, w)
                cost[g][j] = base + c
                back[g][j] = (None, sides)
                continue
            pf = steps[groups[g - 1][1]]
            best = (math.inf, None, ())
            for pj, pc in cost[g - 1].items():
                pd = np.asarray(pf.dirs[pj])
                if f.keep_previous and angle_deg(pd, d) > _SAME_DEG:
                    continue
                if run:
                    c, sides = _sketch_run(pd, run, d, w)
                    c += float(w["home"]) * angle_deg(pd, d) / 90.0
                else:
                    c, sides = _move(angle_deg(pd, d), False, w), ()
                if pc + c < best[0]:
                    best = (pc + c, pj, sides)
            if best[1] is not None:
                cost[g][j] = base + best[0]
                back[g][j] = (best[1], best[2])
        if not cost[g]:  # keep_previous impossible (no matching direction): plan freely
            relaxed = list(steps)
            relaxed[fi] = replace(f, keep_previous=False)
            return plan_sequence(relaxed, w)

    # trailing sketches after the last feature
    last = len(groups) - 1
    final = {}
    for j, c in cost[last].items():
        if trailing:
            tc, tsides = _sketch_run(np.asarray(steps[groups[last][1]].dirs[j]),
                                     [steps[i] for i in trailing], None, w)
        else:
            tc, tsides = 0.0, ()
        final[j] = (c + tc, tsides)
    j = min(final, key=lambda k: final[k][0])
    for i, side in zip(trailing, final[j][1]):
        choice[i] = side
    for g in range(last, -1, -1):
        sk, fi = groups[g]
        choice[fi] = j
        pj, sides = back[g][j]
        for i, side in zip(sk, sides):
            choice[i] = side
        j = pj
    return [int(c) for c in choice]


def shots(steps: Sequence[StepCandidates], choice: Sequence[int]) -> List[int]:
    """Shot id per step: consecutive feature steps of the same body with the
    same direction share a shot (sketches in between do not break it). A new
    body starts a new shot: the direction may stay, but the framing follows
    the new body's size. Sketch steps get -1."""
    out: List[int] = []
    shot, prev, prev_body = -1, None, None
    for s, j in zip(steps, choice):
        if s.is_sketch:
            out.append(-1)
            continue
        d = np.asarray(s.dirs[j])
        if prev is None or angle_deg(prev, d) > _SAME_DEG or s.body != prev_body:
            shot += 1
        prev, prev_body = d, s.body
        out.append(shot)
    return out
