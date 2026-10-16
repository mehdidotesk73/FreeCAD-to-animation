"""Layer 1 - the immutable feature-tree data model.

A *tree* is a plain JSON-serialisable dict whose insertion order is the build
order of the model:

    {
      "__meta__": {...source document, bodies, schema...},
      "Sketch001": {"kind": "Sketcher::SketchObject", "body": "Body001", ...},
      "Pad":       {"kind": "PartDesign::Pad", "role": "additive",
                    "Type": "Length", "Length": 120.0, ...},
      ...
    }

Conventions
-----------
* Every key other than ``__meta__`` is a feature id (the FreeCAD object Name).
* Inside a feature entry, keys starting with an upper-case letter are FreeCAD
  property values. Changing one of them changes the geometry the compiler
  builds (``state[step]["Length"] = 42.0``).
* Lower-case keys are metadata (kind, role, body, deps, ...).
* Keys starting with ``anim_`` are synthetic animation parameters the
  compiler applies on top of the FreeCAD model (``anim_sweep`` reveals the
  material of a ThroughAll/UpTo* feature progressively). They change geometry.
* Keys starting with ``render_`` are presentation-only hints (for example
  ``render_reveal`` for drawing a sketch progressively). They are excluded
  from the geometry hash, so they never trigger a FreeCAD rebuild.

A *state* is a tree that contains only a prefix of the features (everything
built up to a given step), possibly with some property overrides.

Every helper here is pure: inputs are never mutated and new dicts are always
returned. This module only uses the standard library because it is imported
both by the pipeline and inside FreeCAD's bundled Python.
"""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Dict, Iterable, List, Mapping

META_KEY = "__meta__"
SCHEMA = "cadanim.tree/1"
RENDER_PREFIX = "render_"
# Synthetic animation parameters that change geometry but are not FreeCAD
# properties (e.g. ``anim_sweep``: reveal a ThroughAll cut progressively).
# Unlike ``render_*`` keys they are part of the geometry hash.
ANIM_PREFIX = "anim_"

# Feature roles assigned during extraction.
VISUAL_ROLES = ("sketch", "additive", "subtractive", "pattern", "dressup")

Tree = Dict[str, Any]


# --------------------------------------------------------------------------
# Inspection
# --------------------------------------------------------------------------
def feature_ids(tree: Mapping[str, Any]) -> List[str]:
    """Feature ids in build order."""
    return [k for k in tree if k != META_KEY]


def meta(tree: Mapping[str, Any]) -> Dict[str, Any]:
    return tree.get(META_KEY, {})


def is_property_key(key: str) -> bool:
    return key[:1].isupper()


def properties(entry: Mapping[str, Any]) -> Dict[str, Any]:
    """FreeCAD property values stored in a feature entry."""
    return {k: v for k, v in entry.items() if is_property_key(k)}


def is_visual(entry: Mapping[str, Any]) -> bool:
    return entry.get("role") in VISUAL_ROLES


def last_feature(tree: Mapping[str, Any]) -> str | None:
    ids = feature_ids(tree)
    return ids[-1] if ids else None


def last_visual_feature(tree: Mapping[str, Any]) -> str | None:
    for fid in reversed(feature_ids(tree)):
        if is_visual(tree[fid]):
            return fid
    return None


# --------------------------------------------------------------------------
# Hashing / canonical form
# --------------------------------------------------------------------------
def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=False, separators=(",", ":"), default=float)


def _strip_render(tree: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in tree.items():
        if k == META_KEY:
            # Only the source identity affects geometry.
            m = v or {}
            out[k] = {"source_sha256": m.get("source_sha256"), "schema": m.get("schema")}
        else:
            out[k] = {pk: pv for pk, pv in v.items() if not pk.startswith(RENDER_PREFIX)}
    return out


def geometry_hash(tree: Mapping[str, Any]) -> str:
    """Hash of everything that influences geometry. Feature order matters."""
    payload = canonical_json(_strip_render(tree))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def fingerprint(tree: Mapping[str, Any]) -> str:
    """Hash of the entire tree, render hints included (for immutability checks)."""
    return hashlib.sha256(canonical_json(tree).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Pure transformations
# --------------------------------------------------------------------------
def empty_state(full_tree: Mapping[str, Any]) -> Tree:
    """Tree_State_0: the blank environment (metadata only)."""
    return {META_KEY: copy.deepcopy(meta(full_tree))}


def state_up_to(full_tree: Mapping[str, Any], step_id: str | None) -> Tree:
    """New state containing every feature up to and including ``step_id``."""
    out = empty_state(full_tree)
    if step_id is None:
        return out
    if step_id not in full_tree:
        raise KeyError(f"unknown step {step_id!r}")
    for fid in feature_ids(full_tree):
        out[fid] = copy.deepcopy(full_tree[fid])
        if fid == step_id:
            break
    return out


def with_overrides(tree: Mapping[str, Any], step_id: str, overrides: Mapping[str, Any]) -> Tree:
    """Deep copy of ``tree`` with ``overrides`` merged into ``tree[step_id]``."""
    if step_id not in tree:
        raise KeyError(f"step {step_id!r} not present in state")
    out = copy.deepcopy(dict(tree))
    out[step_id].update(copy.deepcopy(dict(overrides)))
    return out


def reorder(tree: Mapping[str, Any], order: Iterable[str]) -> Tree:
    """New tree with features in ``order`` (which must be a permutation)."""
    order = list(order)
    ids = feature_ids(tree)
    if sorted(order) != sorted(ids):
        missing = set(ids) ^ set(order)
        raise ValueError(f"order is not a permutation of the tree features: {sorted(missing)}")
    out: Tree = {META_KEY: copy.deepcopy(meta(tree))}
    for fid in order:
        out[fid] = copy.deepcopy(tree[fid])
    return out


def diff_overrides(full_tree: Mapping[str, Any], state: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Property/render values in ``state`` that differ from ``full_tree``.

    Together with the last feature id this fully reproduces a state, which is
    what the manifest stores per frame.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for fid in feature_ids(state):
        base = full_tree.get(fid, {})
        delta = {k: v for k, v in state[fid].items()
                 if (is_property_key(k) or k.startswith((RENDER_PREFIX, ANIM_PREFIX)))
                 and base.get(k) != v}
        if delta:
            out[fid] = delta
    return out


def body_of(tree: Mapping[str, Any], fid: str) -> str | None:
    return tree[fid].get("body")


def load(path: str) -> Tree:
    with open(path, "r", encoding="utf-8") as fh:
        tree = json.load(fh)
    if meta(tree).get("schema") != SCHEMA:
        raise ValueError(f"{path}: not a {SCHEMA} tree")
    return tree


def save(tree: Mapping[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(tree, fh, indent=1)
