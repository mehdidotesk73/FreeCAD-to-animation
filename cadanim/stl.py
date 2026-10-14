"""Binary STL read/write (numpy only, usable inside FreeCAD's Python too).

STL is the interchange format between the FreeCAD compiler and the viewer:
every compiled state is a folder of one STL per body. Several bodies written
into one STL need no boolean fuse; slicers treat touching or overlapping
shells as one solid.
"""
from __future__ import annotations

import os
from typing import Iterable, Tuple

import numpy as np

_DTYPE = np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])


def write_stl(path: str, meshes: Iterable[Tuple[np.ndarray, np.ndarray]], header: str = "cadanim") -> int:
    """Write one or more (vertices Nx3, faces Mx3) meshes into one binary STL.

    Written atomically. Returns the number of triangles.
    """
    tris = []
    for v, f in meshes:
        v = np.asarray(v, dtype=np.float64).reshape(-1, 3)
        f = np.asarray(f, dtype=np.int64).reshape(-1, 3)
        if len(f):
            tris.append(v[f])
    t = np.concatenate(tris) if tris else np.zeros((0, 3, 3))
    data = np.zeros(len(t), dtype=_DTYPE)
    if len(t):
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        norm = np.linalg.norm(n, axis=1, keepdims=True)
        data["normal"] = np.divide(n, norm, out=np.zeros_like(n), where=norm > 0)
        data["v"] = t
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(header.encode("ascii", "replace")[:80].ljust(80, b" "))
        fh.write(np.uint32(len(t)).tobytes())
        fh.write(data.tobytes())
    os.replace(tmp, path)
    return len(t)


def read_stl(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Binary STL -> (vertices, faces) with exactly shared corners merged."""
    with open(path, "rb") as fh:
        fh.seek(80)
        n = int(np.frombuffer(fh.read(4), dtype="<u4")[0])
        data = np.frombuffer(fh.read(n * _DTYPE.itemsize), dtype=_DTYPE, count=n)
    pts = data["v"].reshape(-1, 3).astype(np.float32)
    if not len(pts):
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)
    verts, inverse = np.unique(pts, axis=0, return_inverse=True)
    return verts, inverse.reshape(-1, 3).astype(np.int32)
