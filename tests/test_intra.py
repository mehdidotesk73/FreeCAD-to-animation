import math

import pytest

from cadanim import tree as T
from cadanim.intra import (create_intra_step_iterator, create_intra_step_iterator_fn,
                           plan_growth)


def test_spec_iterator_values_and_immutability(sample_tree):
    base = T.state_up_to(sample_tree, "PadA")
    fp = T.fingerprint(base)
    states = create_intra_step_iterator(base, "PadA", "Length", 0.1, 10.0, 5)
    assert [round(s["PadA"]["Length"], 6) for s in states] == [0.1, 2.575, 5.05, 7.525, 10.0]
    assert T.fingerprint(base) == fp
    states[0]["PadA"]["Length"] = -1
    assert states[1]["PadA"]["Length"] != -1  # every state is an independent deep copy


def test_spec_iterator_single_step(sample_tree):
    base = T.state_up_to(sample_tree, "PadA")
    states = create_intra_step_iterator(base, "PadA", "Length", 3.0, 9.0, 1)
    assert len(states) == 1 and states[0]["PadA"]["Length"] == 3.0


def test_iterator_requires_step_in_base(sample_tree):
    with pytest.raises(KeyError):
        create_intra_step_iterator(T.state_up_to(sample_tree, "SkA"), "PadA", "Length", 0, 1, 2)


def test_general_iterator_ends_at_final_value(sample_tree):
    base = T.state_up_to(sample_tree, "PadA")
    states = create_intra_step_iterator_fn(base, "PadA", lambda t: {"Length": 120.0 * t}, 4)
    assert states[-1]["PadA"]["Length"] == pytest.approx(120.0)
    assert states[0]["PadA"]["Length"] == pytest.approx(30.0)


def test_pad_growth(sample_tree):
    plan = plan_growth(T.state_up_to(sample_tree, "PadA"), "PadA", 10)
    lengths = [s["PadA"]["Length"] for s in plan.states]
    assert len(lengths) == 10
    assert lengths == sorted(lengths)
    assert lengths[0] > 0 and lengths[-1] == pytest.approx(120.0)


def test_through_all_pocket_uses_sweep_depth_and_restores_type(sample_tree):
    st = T.state_up_to(sample_tree, "PocketB")
    plan = plan_growth(st, "PocketB", 6, focus={"sweep_depth": 40.0})
    assert all(s["PocketB"]["Type"] == "Length" for s in plan.states[:-1])
    assert plan.states[-1]["PocketB"]["Type"] == "ThroughAll"
    assert plan.states[-1]["PocketB"]["Length"] == 5.0  # original definition, exactly
    assert plan.states[-2]["PocketB"]["Length"] < 40.0


def test_through_all_without_depth_appears(sample_tree):
    assert plan_growth(T.state_up_to(sample_tree, "PocketB"), "PocketB", 6) is None


def test_polar_pattern_keeps_instance_positions(sample_tree):
    st = T.state_up_to(sample_tree, "PolarB")
    plan = plan_growth(st, "PolarB", 8)
    seen = {}
    for s in plan.states:
        e = s["PolarB"]
        seen[e["Occurrences"]] = e["Angle"]
    assert sorted(seen) == [1, 2, 3, 4]
    # 4 instances over a full circle sit at 90 degree spacing at every stage.
    assert seen[2] == pytest.approx(90.0)
    assert seen[3] == pytest.approx(180.0)
    assert seen[4] == pytest.approx(360.0)


def test_sketch_reveal_is_render_only(sample_tree):
    st = T.state_up_to(sample_tree, "SkA")
    plan = plan_growth(st, "SkA", 5)
    reveals = [s["SkA"]["render_reveal"] for s in plan.states]
    assert reveals[-1] == 1.0 and reveals == sorted(reveals)
    assert len({T.geometry_hash(s) for s in plan.states}) == 1  # no rebuilds needed


def test_revolution_growth(sample_tree):
    plan = plan_growth(T.state_up_to(sample_tree, "RevC"), "RevC", 12)
    angles = [s["RevC"]["Angle"] for s in plan.states]
    assert angles[0] > 0 and angles[-1] == pytest.approx(360.0)


def test_binder_has_no_growth(sample_tree):
    assert plan_growth(T.state_up_to(sample_tree, "BindB"), "BindB", 10) is None
