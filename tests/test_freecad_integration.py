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
