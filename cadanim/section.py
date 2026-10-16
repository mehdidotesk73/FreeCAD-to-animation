"""Section views: cut every body with a plane in the viewer (VTK only).

A section is ``{"normal": [x, y, z], "origin": [x, y, z]}``. Everything on
the side the normal points to is removed. The cut face (cap) is built from the
cut outline itself: the mesh is sliced along the plane into closed loops, and
``vtkContourTriangulator`` fills them, holes included (a hollow tube gives a
ring, not a disc). Selecting existing faces by orientation would also pick up
real model faces that happen to be parallel to the plane.

Pure geometry, no FreeCAD: changing a cut never rebuilds the model.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
import pyvista as pv


def normalize_section(section: Mapping, view_dir: Optional[Sequence[float]] = None) -> dict:
    """Unit normal, and (when ``view_dir`` is given) the normal flipped so the
    removed half is the one facing the camera."""
    n = np.asarray(section["normal"], float)
    n = n / np.linalg.norm(n)
    if view_dir is not None and float(np.dot(n, view_dir)) < 0:
        n = -n
    return {"normal": [float(x) for x in n], "origin": [float(x) for x in section["origin"]]}


def cap_color(hex_color: str, factor: float = 0.85) -> str:
    """Slightly darker shade of ``hex_color`` for cut faces."""
    c = hex_color.lstrip("#")
    return "#" + "".join(f"{int(int(c[i:i + 2], 16) * factor):02x}" for i in (0, 2, 4))


def cut_surface(display: pv.PolyData, closed: pv.PolyData, section: Mapping
                ) -> Tuple[pv.PolyData, Optional[pv.PolyData]]:
    """(kept part of ``display``, cap) for one body.

    ``display`` is the shaded surface (normals may have split vertices);
    ``closed`` is the same body as a connected mesh, used for the cut outline
    so the loops close.
    """
    n, o = section["normal"], section["origin"]
    kept = display.clip(normal=n, origin=o, invert=True)
    return kept, cap_for(closed, n, o)


def cap_for(closed: pv.PolyData, normal: Sequence[float], origin: Sequence[float]
            ) -> Optional[pv.PolyData]:
    import vtk
    loops = closed.slice(normal=normal, origin=origin)
    if loops.n_points == 0:
        return None
    tri = vtk.vtkContourTriangulator()
    tri.SetInputData(loops)
    tri.TriangulationErrorDisplayOff()
    tri.Update()
    cap = pv.wrap(tri.GetOutput())
    if cap.n_cells == 0:
        return None
    cap.point_data["Normals"] = np.tile(np.asarray(normal, float), (cap.n_points, 1))
    return cap


def cut_lines(lines: Optional[pv.PolyData], section: Mapping) -> Optional[pv.PolyData]:
    """Sketch lines on the removed side are removed too."""
    if lines is None or lines.n_points == 0:
        return lines
    out = lines.clip(normal=section["normal"], origin=section["origin"], invert=True)
    return out if out.n_points else None
