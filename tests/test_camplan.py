"""Sequence camera planning on synthetic candidates (no FreeCAD, no rendering)."""
import math

import numpy as np
import pytest

from cadanim import camera as C
from cadanim.camplan import StepCandidates, angle_deg, plan_sequence, shots

A = (0.0, 0.57, 0.82)           # three-quarter view from the front
B = (0.58, 0.58, -0.58)         # from below, the side the pattern needs
SIDE = (0.0, 1.0, 0.0)          # perpendicular to an XZ sketch
W = {"quality": 3.0, "fixed_move": 1.0, "move": 1.0, "home": 1.0, "min_visibility": 0.6}


def feat(name, vis_by_dir, **kw):
    dirs = tuple(vis_by_dir)
    vis = tuple(vis_by_dir.values())
    best = max(vis)
    quality = tuple(math.sqrt(v / best) for v in vis)
    return StepCandidates(step=name, role="additive", dirs=dirs, vis=vis, quality=quality, **kw)


def sketch(name, normal):
    n = tuple(normal)
    m = tuple(-x for x in normal)
    return StepCandidates(step=name, role="sketch", dirs=(n, m), vis=(1.0, 1.0), quality=(1.0, 1.0))


def test_feature_shares_the_next_steps_view():
    # Arch sketch -> pocket (slightly better from A, fine from B) -> pattern (only B works).
    steps = [sketch("arch", SIDE),
             feat("pocket", {A: 100, B: 90}),
             feat("pattern", {A: 20, B: 100})]
    choice = plan_sequence(steps, W)
    dirs = [steps[i].dirs[j] for i, j in enumerate(choice)]
    assert dirs[1] == B and dirs[2] == B          # pocket done from the pattern's view
    assert shots(steps, choice) == [-1, 0, 0]     # one shot: the camera does not move


def test_returns_to_home_view_after_sketch():
    # Revolve from A; a sketch; pocket would be marginally better from another view.
    other = (0.58, -0.58, 0.58)
    steps = [feat("revolve", {A: 100, other: 70}),
             sketch("pocket sketch", (0, 0, 1)),
             feat("pocket", {A: 92, other: 100})]
    choice = plan_sequence(steps, W)
    assert steps[2].dirs[choice[2]] == A
    assert shots(steps, choice) == [0, -1, 0]     # same shot across the sketch


def test_min_visibility_is_a_hard_limit():
    steps = [feat("f1", {A: 100, B: 10}), feat("f2", {A: 50, B: 100})]
    choice = plan_sequence(steps, W)
    assert steps[1].dirs[choice[1]] == B          # 50% < 60%: A is not allowed for f2


def test_sketch_side_follows_the_camera():
    steps = [feat("f", {A: 100}), sketch("s", (0, -1, 0)), feat("g", {A: 100})]
    choice = plan_sequence(steps, W)
    assert steps[1].dirs[choice[1]] == (0, 1, 0)  # the side facing the 3D view


def test_fixed_override_and_keep_previous():
    steps = [feat("f", {A: 100, B: 100}), feat("g", {A: 10, B: 100}, keep_previous=True)]
    choice = plan_sequence(steps, W)
    assert steps[0].dirs[choice[0]] == steps[1].dirs[choice[1]] == B
    fixed = [feat("f", {A: 100, B: 10}, fixed=1)]
    assert plan_sequence(fixed, W) == [1]


def test_sketches_only():
    assert plan_sequence([sketch("s", (0, 0, 1))], W) in ([0], [1])


def test_fit_box_contains_and_fills_the_outline():
    box = [-100, -100, 0, 100, 100, 20]          # wide and flat, like the base
    view_angle, aspect, margin = 24.0, 16 / 9, 1.0
    v = C.fit_box(C.normalize([0, 0.57, 0.82]), box, view_angle, aspect, margin)
    r = C.quat_to_matrix(v["Rotation"]); eye = np.asarray(v["Position"])
    tan_v = math.tan(math.radians(view_angle) / 2); tan_h = tan_v * aspect
    worst = 0.0
    for x in (box[0], box[3]):
        for y in (box[1], box[4]):
            for z in (box[2], box[5]):
                p = (np.array([x, y, z]) - eye) @ r      # camera frame
                depth = -p[2]
                worst = max(worst, abs(p[0]) / (depth * tan_h), abs(p[1]) / (depth * tan_v))
    assert worst <= 1.0 + 1e-6        # every corner inside the frame
    assert worst > 0.8                # and the outline (not a sphere) fills it
    sphere_d = C.fit_distance(C.bbox_sphere(box)[1], view_angle, aspect, margin=1.0)
    assert v["Distance"] < sphere_d   # closer than sphere framing for flat parts


def test_shots_end_at_body_boundaries():
    a = feat("a", {A: 100}, body="Fastener")
    b = feat("b", {A: 100}, body="Platform")
    c = feat("c", {A: 100}, body="Platform")
    assert shots([a, b, c], [0, 0, 0]) == [0, 1, 1]
