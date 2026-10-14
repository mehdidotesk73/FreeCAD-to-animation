import numpy as np

from cadanim.stl import read_stl, write_stl


def test_round_trip_merges_shared_vertices(tmp_path):
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    f = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    path = str(tmp_path / "t.stl")
    assert write_stl(path, [(v, f)]) == 4
    rv, rf = read_stl(path)
    assert len(rv) == 4 and rf.shape == (4, 3)
    assert np.allclose(sorted(map(tuple, rv)), sorted(map(tuple, v)))


def test_several_meshes_in_one_file(tmp_path):
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    f = np.array([[0, 1, 2]])
    path = str(tmp_path / "u.stl")
    assert write_stl(path, [(v, f), (v + 5, f)]) == 2
    rv, rf = read_stl(path)
    assert len(rv) == 6 and len(rf) == 2


def test_empty(tmp_path):
    path = str(tmp_path / "e.stl")
    assert write_stl(path, []) == 0
    rv, rf = read_stl(path)
    assert rv.shape == (0, 3) and rf.shape == (0, 3)
