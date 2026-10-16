"""Camera-space math: view dicts, quaternions, interpolation and framing.

A *camera view* is a JSON-serialisable dict:

    {
      "FocalPoint": [x, y, z],      # point the camera looks at
      "Rotation":   [w, x, y, z],   # unit quaternion, camera-to-world
      "Distance":   d,              # focal point -> eye distance (mm)
      "Zoom":       1.0,            # multiplies the field of view (2.0 = 2x tighter)
      "ViewAngle":  30.0,           # base vertical field of view (degrees)
      "Position":   [x, y, z],      # derived: FocalPoint + R @ (0, 0, Distance)
      "ViewUp":     [x, y, z],      # derived: R @ (0, 1, 0)
    }

Camera-local axes follow the OpenGL convention: the camera looks down its
local -Z axis and +Y is up. ``Position`` and ``ViewUp`` are always derived from
the other fields so a view can never be inconsistent.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

Vec = Sequence[float]
WORLD_UP = np.array([0.0, 0.0, 1.0])  # FreeCAD is Z-up


# --------------------------------------------------------------------------
# vectors & quaternions
# --------------------------------------------------------------------------
def normalize(v: Vec) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    n = np.linalg.norm(a)
    if n < 1e-12:
        raise ValueError("cannot normalize a zero vector")
    return a / n


def quat_from_matrix(m: np.ndarray) -> np.ndarray:
    """Rotation matrix (columns = local axes in world) -> quaternion [w,x,y,z]."""
    m = np.asarray(m, dtype=float)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    q = q / np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def quat_to_matrix(q: Vec) -> np.ndarray:
    w, x, y, z = normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def slerp(q0: Vec, q1: Vec, t: float) -> List[float]:
    """Spherical linear interpolation along the shortest arc."""
    a, b = normalize(q0), normalize(q1)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b, dot = -b, -dot
    if dot > 0.9995:
        out = a + t * (b - a)
        return list(out / np.linalg.norm(out))
    theta0 = math.acos(dot)
    theta = theta0 * t
    s0 = math.sin(theta0 - theta) / math.sin(theta0)
    s1 = math.sin(theta) / math.sin(theta0)
    return list(s0 * a + s1 * b)


def lerp(a, b, t: float):
    if isinstance(a, (int, float)):
        return float(a + (b - a) * t)
    return [float(x + (y - x) * t) for x, y in zip(a, b)]


def ease(t: float, kind: str = "linear") -> float:
    t = min(max(t, 0.0), 1.0)
    if kind == "linear":
        return t
    if kind in ("ease_in_out", "smooth"):
        return t * t * (3 - 2 * t)
    if kind == "smoother":
        return t * t * t * (t * (6 * t - 15) + 10)
    raise ValueError(f"unknown easing {kind!r}")


# --------------------------------------------------------------------------
# view construction
# --------------------------------------------------------------------------
def look_rotation(to_camera: Vec, up_hint: Vec = WORLD_UP) -> List[float]:
    """Quaternion for a camera whose eye lies along ``to_camera`` from the focus."""
    back = normalize(to_camera)
    up_hint = np.asarray(up_hint, dtype=float)
    if abs(float(np.dot(back, normalize(up_hint)))) > 0.999:
        up_hint = np.array([0.0, 1.0, 0.0]) if abs(back[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    right = normalize(np.cross(up_hint, back))
    up = np.cross(back, right)
    return list(quat_from_matrix(np.column_stack([right, up, back])))


def complete_view(view: Dict) -> Dict:
    """Return a copy of ``view`` with derived Position/ViewUp filled in."""
    out = {
        "FocalPoint": [float(x) for x in view["FocalPoint"]],
        "Rotation": [float(x) for x in normalize(view["Rotation"])],
        "Distance": float(view["Distance"]),
        "Zoom": float(view.get("Zoom", 1.0)),
        "ViewAngle": float(view.get("ViewAngle", 30.0)),
    }
    r = quat_to_matrix(out["Rotation"])
    out["Position"] = list(np.asarray(out["FocalPoint"]) + r[:, 2] * out["Distance"])
    out["ViewUp"] = list(r[:, 1])
    return out


def make_view(focal: Vec, to_camera: Vec, distance: float, *, up_hint: Vec = WORLD_UP,
              view_angle: float = 30.0, zoom: float = 1.0) -> Dict:
    return complete_view({"FocalPoint": list(focal), "Rotation": look_rotation(to_camera, up_hint),
                          "Distance": distance, "Zoom": zoom, "ViewAngle": view_angle})


def view_direction(view: Dict) -> np.ndarray:
    """Unit vector from the focal point towards the eye."""
    return quat_to_matrix(view["Rotation"])[:, 2]


def effective_fov(view_angle: float, zoom: float, aspect: float) -> float:
    """Smallest of the vertical/horizontal field of view, radians."""
    v = math.radians(view_angle) / max(zoom, 1e-6)
    h = 2 * math.atan(math.tan(v / 2) * aspect)
    return min(v, h)


def fit_distance(radius: float, view_angle: float, aspect: float, zoom: float = 1.0,
                 margin: float = 1.1) -> float:
    """Eye distance at which a sphere of ``radius`` fills the view (with margin)."""
    fov = effective_fov(view_angle, zoom, aspect)
    return max(radius, 1e-6) * margin / math.sin(fov / 2)


def bbox_sphere(bbox: Sequence[float]) -> Tuple[np.ndarray, float]:
    lo, hi = np.asarray(bbox[:3], float), np.asarray(bbox[3:], float)
    return (lo + hi) / 2, float(np.linalg.norm(hi - lo) / 2)


def union_bbox(boxes: Iterable[Sequence[float] | None]) -> List[float] | None:
    boxes = [b for b in boxes if b]
    if not boxes:
        return None
    a = np.asarray(boxes, float)
    return list(np.concatenate([a[:, :3].min(0), a[:, 3:].max(0)]))


def candidate_directions(extra: Iterable[Vec] = ()) -> List[np.ndarray]:
    """Standard viewpoints: 8 isometric octants, 3 elevated 'three-quarter' rings and 6 axes."""
    dirs = []
    for elev in (35.264, -35.264):
        for az in (45, 135, 225, 315):
            dirs.append(spherical(az, elev))
    for elev in (20.0, 55.0):
        for az in (0, 90, 180, 270, 22.5, 112.5, 202.5, 292.5):
            dirs.append(spherical(az, elev))
    for axis in np.eye(3):
        dirs.append(axis.copy())
        dirs.append(-axis)
    for e in extra:
        dirs.append(normalize(e))
    return dirs


def spherical(az_deg: float, elev_deg: float) -> np.ndarray:
    """Unit vector at azimuth/elevation (degrees), Z up."""
    az, el = math.radians(az_deg), math.radians(elev_deg)
    return np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])


def rotate_about(v: Vec, axis: Vec, angle_rad: float) -> np.ndarray:
    v, k = np.asarray(v, float), normalize(axis)
    return (v * math.cos(angle_rad) + np.cross(k, v) * math.sin(angle_rad)
            + k * np.dot(k, v) * (1 - math.cos(angle_rad)))


# --------------------------------------------------------------------------
# Inter-step interpolation (task 02)
# --------------------------------------------------------------------------
def interpolate_camera(view_start: Dict, view_end: Dict, frames_count: int,
                       easing: str = "linear") -> List[Dict]:
    """Camera path from ``view_start`` to ``view_end`` (both inclusive).

    * Rotation: SLERP (shortest arc).
    * FocalPoint and Zoom: LERP.
    * Distance: geometric interpolation, so zooming feels uniform.
    * Position is *derived* (FocalPoint + R @ [0,0,Distance]). The camera
      therefore orbits around the focus instead of cutting a straight line
      through the model, which a plain Position LERP would do.
    """
    seq = []
    d0, d1 = float(view_start["Distance"]), float(view_end["Distance"])
    for i in range(frames_count):
        t = ease(i / max(frames_count - 1, 1), easing)
        seq.append(complete_view({
            "FocalPoint": lerp(view_start["FocalPoint"], view_end["FocalPoint"], t),
            "Rotation": slerp(view_start["Rotation"], view_end["Rotation"], t),
            "Distance": d0 * (d1 / d0) ** t if d0 > 0 and d1 > 0 else lerp(d0, d1, t),
            "Zoom": lerp(float(view_start.get("Zoom", 1.0)), float(view_end.get("Zoom", 1.0)), t),
            "ViewAngle": lerp(float(view_start.get("ViewAngle", 30.0)),
                              float(view_end.get("ViewAngle", 30.0)), t),
        }))
    return seq


def angle_between_views(a: Dict, b: Dict) -> float:
    """Rotation angle (degrees) between two camera orientations."""
    qa, qb = normalize(a["Rotation"]), normalize(b["Rotation"])
    d = min(1.0, abs(float(np.dot(qa, qb))))
    return math.degrees(2 * math.acos(d))


def views_equal(a: Dict | None, b: Dict | None, angle_tol: float = 0.5,
                rel_tol: float = 0.01) -> bool:
    if a is None or b is None:
        return a is b
    if angle_between_views(a, b) > angle_tol:
        return False
    scale = max(float(a["Distance"]), float(b["Distance"]), 1e-9)
    if np.linalg.norm(np.subtract(a["FocalPoint"], b["FocalPoint"])) > rel_tol * scale:
        return False
    if abs(float(a["Distance"]) - float(b["Distance"])) > rel_tol * scale:
        return False
    return abs(float(a.get("Zoom", 1)) - float(b.get("Zoom", 1))) <= rel_tol


def orbit(view: Dict, frames_count: int, degrees: float = 360.0,
          axis: Vec = WORLD_UP) -> List[Dict]:
    """Turntable path around ``axis`` through the focal point (end exclusive)."""
    r0 = quat_to_matrix(view["Rotation"])
    out = []
    for i in range(frames_count):
        ang = math.radians(degrees) * i / max(frames_count, 1)
        cols = [rotate_about(r0[:, j], axis, ang) for j in range(3)]
        out.append(complete_view({**view, "Rotation": list(quat_from_matrix(np.column_stack(cols)))}))
    return out


def fit_box(direction: Vec, box: Sequence[float], view_angle: float, aspect: float,
            margin: float = 1.1, zoom: float = 1.0, up_hint: Vec = WORLD_UP) -> Dict:
    """View along ``direction`` (towards the eye) that frames ``box``'s outline.

    Unlike ``fit_distance`` (which fits the bounding sphere), this projects the
    8 box corners into the camera frame, centres the focal point on their
    projected extent and moves the eye just far enough that every corner is
    inside the perspective frustum (with ``margin``). Wide, flat parts fill
    the frame instead of sitting small inside a sphere-sized margin.
    """
    rot = look_rotation(direction, up_hint)
    r = quat_to_matrix(rot)
    lo, hi = np.asarray(box[:3], float), np.asarray(box[3:], float)
    c = (lo + hi) / 2
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                        for z in (lo[2], hi[2])]) - c
    local = corners @ r                      # columns: right, up, towards camera
    ox = (local[:, 0].max() + local[:, 0].min()) / 2
    oy = (local[:, 1].max() + local[:, 1].min()) / 2
    v = math.radians(view_angle) / max(zoom, 1e-6)
    tan_v = math.tan(v / 2)
    tan_h = tan_v * aspect
    need = local[:, 2] + margin * np.maximum(np.abs(local[:, 0] - ox) / tan_h,
                                             np.abs(local[:, 1] - oy) / tan_v)
    focal = c + r[:, 0] * ox + r[:, 1] * oy
    dist = max(float(need.max()), 1e-3)
    return complete_view({"FocalPoint": list(focal), "Rotation": rot, "Distance": dist,
                          "Zoom": zoom, "ViewAngle": view_angle})


def sphere_box(center: Vec, radius: float) -> List[float]:
    """Axis-aligned cube whose half-diagonal is ``radius``."""
    h = float(radius) / math.sqrt(3.0)
    c = np.asarray(center, float)
    return list(np.concatenate([c - h, c + h]))
