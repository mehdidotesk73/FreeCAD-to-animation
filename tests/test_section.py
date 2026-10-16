"""Section cuts in the viewer (no FreeCAD)."""
import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from cadanim.section import cap_color, cap_for, cut_lines, cut_surface, normalize_section


def test_box_cut_keeps_half_and_caps_exact_area():
    box = pv.Box(bounds=(-10, 10, -5, 5, 0, 30)).triangulate().clean()
    sec = {"normal": [0, 1, 0], "origin": [0, 0, 0]}
    kept, cap = cut_surface(box, box, sec)
    assert kept.bounds[3] == pytest.approx(0.0)      # +y half removed
    assert kept.bounds[2] == pytest.approx(-5.0)
    assert cap.area == pytest.approx(20 * 30)        # x-z cross-section


def test_hollow_tube_cap_is_two_wall_strips_not_a_disc():
    outer = pv.Cylinder(radius=25, height=40, resolution=96).triangulate()
    inner = pv.Cylinder(radius=22, height=42, resolution=96).triangulate()
    tube = outer.boolean_difference(inner).clean()
    cap = cap_for(tube, (0, 1, 0), (0, 0, 0))
    assert cap.area == pytest.approx(2 * 3 * 40, rel=0.01)


def test_plane_missing_the_body_gives_no_cap():
    box = pv.Box(bounds=(-1, 1, -1, 1, -1, 1)).triangulate()
    assert cap_for(box, (0, 1, 0), (0, 5, 0)) is None


def test_normal_flipped_towards_camera():
    sec = normalize_section({"normal": [0, -2, 0], "origin": [1, 2, 3]}, view_dir=[0, 1, 0])
    assert sec["normal"] == [0, 1, 0] and sec["origin"] == [1, 2, 3]


def test_lines_on_removed_side_are_dropped():
    line = pv.Line((0, -5, 0), (0, 5, 0))
    kept = cut_lines(line, {"normal": [0, 1, 0], "origin": [0, 0, 0]})
    assert kept.bounds[3] <= 1e-9
    assert cut_lines(pv.Line((0, 1, 0), (0, 5, 0)), {"normal": [0, 1, 0], "origin": [0, 0, 0]}) is None


def test_cap_colour_is_slightly_darker():
    assert cap_color("#4f7cac", 0.85) == "#436992"


def test_renderer_draws_section(tmp_path):
    from PIL import Image
    from cadanim import camera as C
    from cadanim.geometry import Geometry
    from cadanim.groups import DEFAULT_RENDER
    from cadanim.render import FrameRequest, Renderer, SceneSpec
    box = pv.Box(bounds=(-10, 10, -10, 10, 0, 30)).triangulate()
    g = Geometry(key="t", bodies={"B": (np.asarray(box.points, np.float32),
                                        box.faces.reshape(-1, 4)[:, 1:].astype(np.int32))},
                 sketches={}, info={"bodies": {}})
    r = Renderer(SceneSpec(body_group={"B": "G"}, colors={"G": "#4f7cac"},
                           style=dict(DEFAULT_RENDER, width=160, height=90, captions=False)))
    view = C.make_view([0, 0, 15], [0.3, 1, 0.4], 120)
    out = []
    for sec in (None, normalize_section({"normal": [0, 1, 0], "origin": [0, 0, 0]}, C.view_direction(view))):
        path = str(tmp_path / f"{bool(sec)}.png")
        r.render(FrameRequest(geometry_key="t", camera=view, output=path, modes={"B": "solid"},
                              section=sec), g)
        out.append(np.asarray(Image.open(path))[..., 3] > 0)
    r.close()
    assert out[0].sum() > 0 and out[1].sum() > 0
    assert out[1].sum() < out[0].sum()               # the front half is gone
