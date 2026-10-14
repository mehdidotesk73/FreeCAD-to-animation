import math

import numpy as np
import pytest

from cadanim import camera as C


def test_quaternion_matrix_round_trip():
    for d in C.candidate_directions():
        q = C.look_rotation(d)
        m = C.quat_to_matrix(q)
        assert np.allclose(m.T @ m, np.eye(3), atol=1e-9)
        assert np.allclose(m[:, 2], C.normalize(d), atol=1e-9)
        assert np.allclose(C.quat_from_matrix(m), q, atol=1e-9) or \
            np.allclose(C.quat_from_matrix(m), -np.asarray(q), atol=1e-9)


def test_make_view_derives_position_and_up():
    v = C.make_view([1, 2, 3], [1, 0, 0], 100)
    assert np.allclose(v["Position"], [101, 2, 3])
    assert np.allclose(v["ViewUp"], [0, 0, 1])


def test_top_view_has_valid_up():
    v = C.make_view([0, 0, 0], [0, 0, 1], 50)
    assert np.allclose(v["Position"], [0, 0, 50])
    assert abs(np.dot(v["ViewUp"], [0, 0, 1])) < 1e-9


def test_interpolate_endpoints_and_count():
    a = C.make_view([0, 0, 0], [1, 0, 0], 100)
    b = C.make_view([10, 0, 0], [0, 1, 0], 400, zoom=2.0)
    seq = C.interpolate_camera(a, b, 7)
    assert len(seq) == 7
    assert C.views_equal(seq[0], a) and C.views_equal(seq[-1], b)
    mid = seq[3]
    assert C.angle_between_views(a, mid) == pytest.approx(45.0, abs=1e-6)
    assert mid["Distance"] == pytest.approx(200.0)       # geometric distance interpolation
    assert mid["Zoom"] == pytest.approx(1.5)
    assert np.allclose(mid["FocalPoint"], [5, 0, 0])


def test_interpolation_orbits_instead_of_cutting_through():
    a = C.make_view([0, 0, 0], [1, 0, 0], 100)
    b = C.make_view([0, 0, 0], [-1, 0.001, 0], 100)
    for v in C.interpolate_camera(a, b, 9):
        assert np.linalg.norm(v["Position"]) == pytest.approx(100.0)


def test_slerp_shortest_path():
    q0 = C.look_rotation([1, 0, 0])
    q1 = list(-np.asarray(C.look_rotation([0, 1, 0])))  # same rotation, opposite sign
    mid = C.slerp(q0, q1, 0.5)
    m = C.quat_to_matrix(mid)
    assert np.allclose(m[:, 2], C.normalize([1, 1, 0]), atol=1e-9)


def test_views_equal_tolerances():
    a = C.make_view([0, 0, 0], [1, 1, 1], 100)
    assert C.views_equal(a, dict(a))
    assert not C.views_equal(a, C.make_view([0, 0, 0], [1, 1, 1.2], 100))
    assert not C.views_equal(a, C.make_view([0, 0, 0], [1, 1, 1], 120))


def test_fit_distance_contains_sphere():
    r, va, aspect = 50.0, 30.0, 16 / 9
    d = C.fit_distance(r, va, aspect, margin=1.0)
    assert math.asin(r / d) == pytest.approx(math.radians(va) / 2)
    d_portrait = C.fit_distance(r, va, 0.5, margin=1.0)
    assert d_portrait > d  # narrow frames need more distance


def test_orbit_full_turn():
    v = C.make_view([0, 0, 0], [1, 0, 0.5], 100)
    seq = C.orbit(v, 4)
    assert len(seq) == 4
    assert np.allclose(seq[1]["Position"][:2], [0, 100 / math.sqrt(1.25)], atol=1e-6)
    for s in seq:
        assert s["Position"][2] == pytest.approx(v["Position"][2])


def test_union_bbox():
    assert C.union_bbox([None, [0, 0, 0, 1, 1, 1], [-1, 0, 0, 0, 2, 0]]) == [-1, 0, 0, 1, 2, 1]
    assert C.union_bbox([None]) is None
