"""End-to-end checks against the example model. Skipped when FreeCAD is absent."""
import os

import pytest

from cadanim import fcbridge
from cadanim import tree as T

pytestmark = pytest.mark.skipif(not fcbridge.available(), reason="freecadcmd not installed")

MODEL = os.path.join(os.path.dirname(__file__), "..", "examples", "birdfeeder", "BirdFeeder.FCStd")


@pytest.fixture(scope="module")
def tree():
    return fcbridge.run_job({"action": "extract", "source": MODEL})["tree"]


def test_extract(tree):
    bodies = T.meta(tree)["bodies"]
    assert set(bodies) == {"Body", "Body001", "Body002", "Body003", "Body004"}
    assert bodies["Body003"]["deps"] == ["Body"]
    assert tree["Pad"]["Length"] == pytest.approx(120.0)


def test_compile_is_deterministic_and_cached(tree, tmp_path):
    from cadanim.geometry import GeometryCompiler
    comp = GeometryCompiler(MODEL, str(tmp_path), workers=1)
    st = T.with_overrides(T.state_up_to(tree, "Pad"), "Pad", {"Length": 60.0})
    g1 = comp.compile(st, focus=True)
    assert g1.ok
    zmin = g1.info["bodies"]["Body001"]["bbox"][2]
    assert zmin == pytest.approx(-60.0, abs=1e-6)  # the override took effect
    assert g1.focus["role"] == "additive"
    assert comp.cached_key(st) == g1.key          # a focus build also serves plain requests
    g2 = comp.compile(st)
    assert g2.key == g1.key


# "Pocket" sits on a sketch whose feature has a rotated placement: its
# AddSubShape must be moved into body coordinates before it is clipped.
@pytest.mark.parametrize("step", ["Pocket002", "Pocket"])
def test_through_all_pocket_focus_has_sweep_depth(tree, tmp_path, step):
    from cadanim.geometry import GeometryCompiler
    comp = GeometryCompiler(MODEL, str(tmp_path), workers=1)
    g = comp.compile(T.state_up_to(tree, step), focus=True)
    assert g.focus.get("bbox") is not None
    assert g.focus.get("sweep_depth", 0) > 0


# Pruning removes everything a state does not need before recompute. It must
# never change the geometry: compare against full-document builds.
PRUNE_CASES = [("Pad", {"Length": 60.0}), ("Pocket004", None), ("Pocket", None),
               ("Pad001", None), ("Groove001", {"Angle": 9.0}), ("Fillet", None)]


@pytest.mark.parametrize("step,overrides", PRUNE_CASES)
def test_pruned_build_matches_full_build(tree, tmp_path, step, overrides):
    from cadanim.geometry import GeometryCompiler
    st = T.state_up_to(tree, step)
    if overrides:
        st = T.with_overrides(st, step, overrides)
    full = GeometryCompiler(MODEL, str(tmp_path / "full"), workers=1, prune=False).compile(st)
    pruned = GeometryCompiler(MODEL, str(tmp_path / "pruned"), workers=1).compile(st)
    assert full.ok and pruned.ok
    assert pruned.info["pruned_objects"] > 0 or step == "Fillet"
    assert set(pruned.bodies) == set(full.bodies)
    for name, b in full.info["bodies"].items():
        p = pruned.info["bodies"][name]
        assert p["volume"] == pytest.approx(b["volume"], rel=1e-9)
        assert p["bbox"] == pytest.approx(b["bbox"], abs=1e-6)
        assert len(pruned.bodies[name][1]) == len(full.bodies[name][1])  # same tessellation
    assert set(pruned.sketches) == set(full.sketches)


# ThroughAll pockets are animated by revealing the finished cut's material in
# growing slabs (anim_sweep), not by switching to a Length pocket, which for
# the arch slot (both walls, hollow bore in between) removed nothing for most
# of the animation.
@pytest.mark.parametrize("step,body", [("Pocket004", "Body001"), ("Pocket003", "Body")])
def test_through_all_sweep_removes_material_progressively(tree, tmp_path, step, body):
    from cadanim.geometry import GeometryCompiler
    comp = GeometryCompiler(MODEL, str(tmp_path), workers=4)
    before = T.state_up_to(tree, tree_prev(tree, step))
    full = T.state_up_to(tree, step)
    states = [before] + [T.with_overrides(full, step, {"anim_sweep": t}) for t in (0.25, 0.5, 0.75)] + [full]
    keys = comp.compile_many(states)
    vols = [comp.load(k).info["bodies"][body]["volume"] for k in keys]
    removed = [vols[0] - v for v in vols[1:]]
    assert removed[0] > 0                                            # cutting from the first quarter
    assert all(a < b for a, b in zip(removed, removed[1:]))          # strictly growing
    assert removed[-2] < removed[-1]                                 # sweep stops short of the end
    assert comp.load(keys[-2]).info.get("anim_sweep") == 0.75


def tree_prev(tree, step):
    ids = T.feature_ids(tree)
    return ids[ids.index(step) - 1]
