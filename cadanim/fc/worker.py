"""Entry point executed *inside* FreeCAD's Python (``freecadcmd``).

The pipeline never imports FreeCAD itself. It writes a job JSON file and runs

    freecadcmd -c "<bootstrap that runs this module>"

with ``CADANIM_JOB`` / ``CADANIM_RESULT`` pointing to the job and result
files (see ``cadanim/fcbridge.py``). Supported actions:

* ``extract``: FCStd -> feature tree (Layer 1 data)
* ``compile``: list of tree states -> one folder per state holding an STL
  per body (plus info.json with sketches and focus data) (Layer 2). Each
  state is built in a freshly opened document that is closed afterwards, so
  there is no state between items. Cameras never reach FreeCAD; the viewer
  applies them to these STLs.
* ``export``: final tree + groups -> STL files (union = the group's bodies
  written into one STL, no boolean fuse; separate = one STL per body)
* ``inspect``: body overview, binder links and body contacts, used to propose
  groupings
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback

import FreeCAD  # noqa: F401  (provided by freecadcmd)
import Part

from cadanim import tree as T
from cadanim.stl import write_stl

SOLID_ROLES = ("additive", "subtractive", "pattern", "dressup")

# FreeCAD properties captured into the tree (when present on a feature).
CAPTURED_PROPS = (
    "Type", "Length", "Length2", "Angle", "Angle2", "Occurrences", "Midplane",
    "Reversed", "Offset", "Radius", "Size", "Size2", "Mode", "UseAllEdges",
    "Refine", "Depth", "Diameter",
)

SUBTRACTIVE_KINDS = {
    "PartDesign::Pocket", "PartDesign::Groove", "PartDesign::Hole",
    "PartDesign::SubtractivePipe", "PartDesign::SubtractiveLoft",
    "PartDesign::SubtractiveHelix",
}
PATTERN_KINDS = {
    "PartDesign::PolarPattern", "PartDesign::LinearPattern",
    "PartDesign::Mirrored", "PartDesign::MultiTransform", "PartDesign::Scaled",
}
DRESSUP_KINDS = {
    "PartDesign::Fillet", "PartDesign::Chamfer", "PartDesign::Draft",
    "PartDesign::Thickness",
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _value(v):
    if hasattr(v, "Value"):          # Quantity
        return float(v.Value)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float, str)):
        return v
    return None


def _role(obj):
    tid = obj.TypeId
    if tid == "Sketcher::SketchObject":
        return "sketch"
    if tid in ("PartDesign::SubShapeBinder", "PartDesign::ShapeBinder"):
        return "binder"
    if tid.startswith("PartDesign::") and tid in ("PartDesign::Plane", "PartDesign::Line",
                                                   "PartDesign::Point", "PartDesign::CoordinateSystem"):
        return "datum"
    if tid in SUBTRACTIVE_KINDS or tid.startswith("PartDesign::Subtractive"):
        return "subtractive"
    if tid in PATTERN_KINDS:
        return "pattern"
    if tid in DRESSUP_KINDS:
        return "dressup"
    if obj.isDerivedFrom("PartDesign::Feature"):
        return "additive"
    return "other"


def _is_solid_feature(obj):
    return obj.isDerivedFrom("PartDesign::Feature") and _role(obj) in SOLID_ROLES


def _bodies(doc):
    return [o for o in doc.Objects if o.TypeId == "PartDesign::Body"]


def _body_features(body):
    """Features of a body up to (and including) its Tip, in build order."""
    feats = [o for o in body.Group if not o.TypeId.startswith("App::")]
    tip = body.Tip
    if tip is not None and tip in feats:
        tip_idx = feats.index(tip)
        # Keep sketches/binders listed after the tip only if a feature before
        # the tip uses them (rare); otherwise everything after the tip is
        # inactive design history.
        feats = feats[: tip_idx + 1]
    return feats


def _binder_sources(obj):
    out = []
    for link in getattr(obj, "Support", []) or []:
        try:
            src = link[0]
        except Exception:
            continue
        b = src if src.TypeId == "PartDesign::Body" else _owner_body(src)
        if b is not None:
            out.append(b.Name)
    return sorted(set(out))


def _owner_body(obj):
    for p in obj.InList:
        if p.TypeId == "PartDesign::Body":
            return p
    return None


def _profile_of(obj):
    prof = getattr(obj, "Profile", None)
    if prof:
        try:
            return prof[0].Name
        except Exception:
            return None
    return None


def _bbox_list(bb):
    if bb is None or not bb.isValid():
        return None
    return [bb.XMin, bb.YMin, bb.ZMin, bb.XMax, bb.YMax, bb.ZMax]


def _tess(shape, deflection):
    pts, tris = shape.tessellate(deflection)
    v = [[p.x, p.y, p.z] for p in pts]
    f = [list(t) for t in tris]
    return v, f


def _sketch_polylines(sketch, deflection):
    shp = sketch.Shape.copy()
    shp.Placement = sketch.getGlobalPlacement()
    lines = []
    for e in shp.Edges:
        try:
            pts = e.discretize(Deflection=deflection)
        except Exception:
            continue
        if len(pts) >= 2:
            lines.append([[p.x, p.y, p.z] for p in pts])
    return lines


# --------------------------------------------------------------------------
# extract
# --------------------------------------------------------------------------
def action_extract(job):
    import hashlib
    src = job["source"]
    sha = hashlib.sha256(open(src, "rb").read()).hexdigest()
    doc = FreeCAD.openDocument(src, True)
    try:
        bodies = _bodies(doc)
        body_meta = {}
        body_feats = {}
        for b in bodies:
            feats = _body_features(b)
            deps = set()
            for f in feats:
                if _role(f) == "binder":
                    deps.update(x for x in _binder_sources(f) if x != b.Name)
            body_meta[b.Name] = {
                "label": b.Label,
                "visible": bool(b.Visibility),
                "deps": sorted(deps),
                "features": [f.Name for f in feats],
                "bbox": _bbox_list(b.Shape.BoundBox) if not b.Shape.isNull() else None,
                "volume": float(b.Shape.Volume) if not b.Shape.isNull() else 0.0,
            }
            body_feats[b.Name] = feats

        # Topological body order (binder dependencies), document order tiebreak.
        order, seen, temp = [], set(), set()
        names = [b.Name for b in bodies]

        def visit(n):
            if n in seen:
                return
            if n in temp:
                raise RuntimeError(f"cyclic body dependency through {n}")
            temp.add(n)
            for d in body_meta[n]["deps"]:
                if d in body_meta:
                    visit(d)
            temp.discard(n)
            seen.add(n)
            order.append(n)

        for n in names:
            visit(n)

        tree = {T.META_KEY: {
            "schema": T.SCHEMA,
            "source": os.path.abspath(src),
            "source_sha256": sha,
            "program_version": FreeCAD.Version()[:3],
            "bodies": {n: body_meta[n] for n in order},
        }}
        for bname in order:
            for f in body_feats[bname]:
                role = _role(f)
                entry = {
                    "kind": f.TypeId,
                    "label": f.Label,
                    "role": role,
                    "body": bname,
                    "body_label": body_meta[bname]["label"],
                }
                prof = _profile_of(f)
                if prof:
                    entry["profile"] = prof
                if role == "binder":
                    entry["binds"] = _binder_sources(f)
                if role != "sketch":
                    for p in CAPTURED_PROPS:
                        if p in f.PropertiesList:
                            v = _value(getattr(f, p))
                            if v is not None:
                                entry[p] = v
                if role == "pattern":
                    try:
                        entry["originals"] = [o.Name for o in f.Originals]
                    except Exception:
                        pass
                tree[f.Name] = entry
        return {"tree": tree}
    finally:
        FreeCAD.closeDocument(doc.Name)


# --------------------------------------------------------------------------
# compile (Layer 2 core)
# --------------------------------------------------------------------------
def _apply_state(doc, state):
    """Configure ``doc`` to represent ``state``. Returns the included ids."""
    included = [fid for fid in T.feature_ids(state)]
    inc = set(included)

    # Property overrides (only where they differ from the document).
    for fid in included:
        obj = doc.getObject(fid)
        if obj is None:
            raise RuntimeError(f"feature {fid} missing from source document")
        for key, val in T.properties(state[fid]).items():
            if key not in obj.PropertiesList:
                continue
            cur = _value(getattr(obj, key))
            if isinstance(val, float) and isinstance(cur, (int, float)) and not isinstance(cur, bool):
                if math.isclose(float(cur), val, rel_tol=0, abs_tol=1e-9):
                    continue
            elif cur == val:
                continue
            setattr(obj, key, val)  # enumerations accept their string value

    # Body tips: last included solid feature per body.
    active_bodies = {}
    for b in _bodies(doc):
        feats = [f for f in b.Group if f.Name in inc and _is_solid_feature(f)]
        if feats:
            if b.Tip != feats[-1]:
                b.Tip = feats[-1]
            active_bodies[b.Name] = b
    return included, active_bodies


def _focus_info(doc, state, step_id, active_bodies, deflection):
    """Extra data about the step being presented, used for camera planning."""
    obj = doc.getObject(step_id)
    entry = state[step_id]
    role = entry.get("role")
    info = {"step": step_id, "role": role}
    tool = None
    body = _owner_body(obj)
    body_gp = body.getGlobalPlacement() if body is not None else FreeCAD.Placement()

    if role == "sketch":
        shp = obj.Shape.copy()
        shp.Placement = obj.getGlobalPlacement()
        info["bbox"] = _bbox_list(shp.BoundBox)
        n = obj.getGlobalPlacement().Rotation.multVec(FreeCAD.Vector(0, 0, 1))
        info["normal"] = [n.x, n.y, n.z]
        return info, None

    base = getattr(obj, "BaseFeature", None)
    base_shape = base.Shape.copy() if base is not None and not base.Shape.isNull() else None
    result = obj.Shape.copy() if not obj.Shape.isNull() else None

    try:
        if role in ("additive", "subtractive") and "AddSubShape" in obj.PropertiesList \
                and not obj.AddSubShape.isNull():
            tool = obj.AddSubShape.copy()
            # AddSubShape lives in the feature's own frame; bring it to body coordinates.
            tool.Placement = obj.Placement.multiply(tool.Placement)
            if role == "subtractive" and base_shape is not None:
                # Only the material actually removed (clips ThroughAll tools).
                tool = tool.common(base_shape)
        elif result is not None:
            # Patterns / dress-ups: symmetric difference with the base.
            pieces = []
            if base_shape is not None:
                added = result.cut(base_shape)
                removed = base_shape.cut(result)
                for s in (added, removed):
                    if s is not None and not s.isNull() and s.Volume > 1e-6:
                        pieces.append(s)
            else:
                pieces.append(result)
            if pieces:
                tool = pieces[0] if len(pieces) == 1 else Part.makeCompound(pieces)
    except Exception as exc:  # booleans can fail on odd geometry
        info["focus_error"] = str(exc)
        tool = None

    if tool is not None and not tool.isNull():
        # Feature shapes are body-local; move them to global coordinates.
        tool = tool.copy()
        tool.Placement = body_gp.multiply(tool.Placement)
        info["bbox"] = _bbox_list(tool.BoundBox)
    elif body is not None and body.Name in active_bodies:
        info["bbox"] = _bbox_list(active_bodies[body.Name].Shape.BoundBox)

    # Profile normal + sweep depth (used to animate ThroughAll / UpTo* types).
    prof = entry.get("profile")
    if prof:
        sk = doc.getObject(prof)
        if sk is not None and sk.TypeId == "Sketcher::SketchObject":
            pl = sk.getGlobalPlacement()
            n = pl.Rotation.multVec(FreeCAD.Vector(0, 0, 1))
            info["normal"] = [n.x, n.y, n.z]
            if tool is not None and not tool.isNull():
                o = pl.Base
                depths = [abs((v.Point - o).dot(n)) for v in tool.Vertexes]
                if depths:
                    info["sweep_depth"] = max(depths)
    return info, tool


def _prune(doc, included):
    """Delete every object the state does not need, so recompute only touches the state.

    Kept: the included features, everything they depend on according to
    FreeCAD's own dependency graph (``OutListRecursive``: profiles, base
    features, attachment faces, binder sources, expression targets such as
    spreadsheets), and the owning bodies with their origins. Everything else
    (later features, bodies not started yet) is removed from this in-memory
    copy only; the file on disk is never saved. Returns the number removed.
    """
    keep = set()
    for fid in included:
        obj = doc.getObject(fid)
        keep.add(obj.Name)
        keep.update(o.Name for o in obj.OutListRecursive)
        body = _owner_body(obj)
        if body is not None:
            keep.add(body.Name)
            keep.add(body.Origin.Name)
            keep.update(o.Name for o in body.Origin.OutListRecursive)
    doomed = {o.Name for o in doc.Objects if o.Name not in keep}
    removed = 0
    while doomed:
        # Remove objects nothing else still left in the document depends on.
        ready = [n for n in doomed
                 if doc.getObject(n) is None
                 or not any(u.Name in doomed and u.Name != n for u in doc.getObject(n).InList)]
        if not ready:  # dependency cycle among doomed objects: remove in reverse order
            ready = [o.Name for o in reversed(doc.Objects) if o.Name in doomed][:1]
        for n in ready:
            doomed.discard(n)
            if doc.getObject(n) is not None:
                doc.removeObject(n)
                removed += 1
    return removed


def compile_state(source, state, deflection, focus, prune=True):
    t0 = time.time()
    timing = {}
    doc = FreeCAD.openDocument(source, True)
    timing["open"] = time.time() - t0
    try:
        t = time.time()
        included, active = _apply_state(doc, state)
        timing["apply"] = time.time() - t
        t = time.time()
        removed = _prune(doc, included) if prune else 0
        timing["prune"] = time.time() - t
        t = time.time()
        doc.recompute()
        timing["recompute"] = time.time() - t
        t = time.time()
        errors = []
        for fid in included:
            o = doc.getObject(fid)
            st = list(o.State)
            if "Invalid" in st or "Error" in st:
                errors.append({"feature": fid, "state": st})

        out = {"bodies": {}, "sketches": {}, "errors": errors}
        meshes = {}  # STL file stem -> (vertices, faces)
        for name, b in active.items():
            shp = b.Shape
            if shp.isNull():
                continue
            shp = shp.copy()
            shp.Placement = b.getGlobalPlacement()
            meshes[name] = _tess(shp, deflection)
            out["bodies"][name] = {"bbox": _bbox_list(shp.BoundBox), "volume": float(shp.Volume),
                                   "stl": f"{name}.stl"}

        # Sketches that are included but not yet consumed by an included feature.
        consumed = set()
        for fid in included:
            p = state[fid].get("profile")
            if p:
                consumed.add(p)
        for fid in included:
            if state[fid].get("role") == "sketch" and fid not in consumed:
                sk = doc.getObject(fid)
                lines = _sketch_polylines(sk, max(deflection * 0.5, 0.02))
                out["sketches"][fid] = {"body": state[fid].get("body"),
                                        "polylines": [[[round(c, 5) for c in p] for p in ln]
                                                      for ln in lines]}

        timing["tessellate"] = time.time() - t
        t = time.time()
        if focus:
            step = T.last_visual_feature(state)
            if step is not None:
                info, tool = _focus_info(doc, state, step, active, deflection)
                if tool is not None and not tool.isNull():
                    v, f = _tess(tool, max(deflection * 3, 0.3))
                    if f:
                        meshes["_focus"] = (v, f)
                        info["stl"] = "_focus.stl"
                out["focus"] = info
        timing["focus"] = time.time() - t
        out["pruned_objects"] = removed
        out["timing"] = {k: round(v, 4) for k, v in timing.items()}
        out["seconds"] = round(time.time() - t0, 3)
        return out, meshes
    finally:
        FreeCAD.closeDocument(doc.Name)


def action_compile(job):
    """Write ``<out>/<Body>.stl`` (+ ``_focus.stl``) and ``<out>/info.json`` per state."""
    results = []
    for item in job["items"]:
        out_dir = item["out"]
        try:
            info, meshes = compile_state(job["source"], item["state"], job.get("deflection", 0.1),
                                         item.get("focus", False), job.get("prune", True))
            t = time.time()
            os.makedirs(out_dir, exist_ok=True)
            for stem, mesh in meshes.items():
                write_stl(os.path.join(out_dir, stem + ".stl"), [mesh], header=f"cadanim {stem}")
            tmp = os.path.join(out_dir, "info.json.tmp")
            info["timing"]["write"] = round(time.time() - t, 4)
            with open(tmp, "w") as fh:
                json.dump(info, fh)
            os.replace(tmp, os.path.join(out_dir, "info.json"))  # written last = complete
            results.append({"out": out_dir, "ok": True, "errors": info["errors"]})
        except Exception as exc:
            results.append({"out": out_dir, "ok": False, "error": f"{exc}",
                            "trace": traceback.format_exc()})
    return {"results": results}


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------
def action_export(job):
    """Group STLs. A union group is its bodies' meshes in one file (no fuse)."""
    doc = FreeCAD.openDocument(job["source"], True)
    written = []
    try:
        _apply_state(doc, job["state"])
        doc.recompute()
        os.makedirs(job["outdir"], exist_ok=True)
        tol = job.get("deflection", 0.05)
        safe = lambda n: "".join(c if c.isalnum() or c in "-_" else "_" for c in n)

        def write(items, fname):
            path = os.path.join(job["outdir"], fname)
            n = write_stl(path, [_tess(s, tol) for _, s in items], header=f"cadanim {fname}")
            written.append({"file": path, "facets": n, "bodies": [b.Label for b, _ in items],
                            "volume": float(sum(s.Volume for _, s in items))})

        for g in job["groups"]:
            shapes = []
            for bname in g["bodies"]:
                b = doc.getObject(bname)
                if b is None or b.Shape.isNull():
                    continue
                s = b.Shape.copy()
                s.Placement = b.getGlobalPlacement()
                shapes.append((b, s))
            if not shapes:
                continue
            if g.get("export", "union") == "union":
                write(shapes, f"{safe(g['name'])}.stl")
            else:
                for b, s in shapes:
                    name = f"{safe(g['name'])}__{safe(b.Label)}.stl" if len(shapes) > 1 else f"{safe(g['name'])}.stl"
                    write([(b, s)], name)
        return {"written": written}
    finally:
        FreeCAD.closeDocument(doc.Name)


# --------------------------------------------------------------------------
# inspect
# --------------------------------------------------------------------------
def action_inspect(job):
    doc = FreeCAD.openDocument(job["source"], True)
    try:
        bodies = [b for b in _bodies(doc) if not b.Shape.isNull()]
        info = {"bodies": [], "binders": [], "contacts": []}
        for b in bodies:
            info["bodies"].append({
                "name": b.Name, "label": b.Label, "visible": bool(b.Visibility),
                "bbox": _bbox_list(b.Shape.BoundBox), "volume": float(b.Shape.Volume),
                "solids": len(b.Shape.Solids),
            })
            for f in b.Group:
                if _role(f) == "binder":
                    for src in _binder_sources(f):
                        if src != b.Name:
                            info["binders"].append({"body": b.Name, "binder": f.Name, "source": src})
        for i, a in enumerate(bodies):
            for b in bodies[i + 1:]:
                sa, sb = a.Shape, b.Shape
                if not sa.BoundBox.intersect(sb.BoundBox):
                    continue
                try:
                    dist = sa.distToShape(sb)[0]
                except Exception:
                    continue
                if dist > 1e-3:
                    continue
                try:
                    common = sa.common(sb).Volume
                except Exception:
                    common = None
                info["contacts"].append({"a": a.Name, "b": b.Name, "distance": dist,
                                         "overlap_volume": common})
        return info
    finally:
        FreeCAD.closeDocument(doc.Name)


ACTIONS = {"extract": action_extract, "compile": action_compile,
           "export": action_export, "inspect": action_inspect}


def main():
    job_path = os.environ["CADANIM_JOB"]
    result_path = os.environ["CADANIM_RESULT"]
    with open(job_path) as fh:
        job = json.load(fh)
    try:
        result = {"ok": True, **ACTIONS[job["action"]](job)}
    except Exception as exc:
        result = {"ok": False, "error": str(exc), "trace": traceback.format_exc()}
    with open(result_path, "w") as fh:
        json.dump(result, fh)


if __name__ == "__main__":
    main()
