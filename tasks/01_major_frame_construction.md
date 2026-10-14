# Task 01: Major frame construction (Layers 1 and 2)

**Focus:** compiling structural trees into CAD geometry and frames,
deterministically.

## Layer 1: the feature tree (`cadanim/tree.py`)

The source model is serialised once by `worker.action_extract`
(`python -m cadanim extract MODEL -o tree.json`):

```json
{
  "__meta__": {"schema": "cadanim.tree/1", "source": "...FCStd", "source_sha256": "...",
               "bodies": {"Body001": {"label": "Shaft", "deps": [], "features": [...], "bbox": [...]}}},
  "Sketch001": {"kind": "Sketcher::SketchObject", "role": "sketch", "body": "Body001", ...},
  "Pad":       {"kind": "PartDesign::Pad", "role": "additive", "body": "Body001",
                "profile": "Sketch001", "Type": "Length", "Length": 120.0, "Reversed": true, ...},
  "Binder005": {"kind": "PartDesign::SubShapeBinder", "role": "binder", "binds": ["Body001"], ...}
}
```

* Insertion order = build order. Bodies are topologically sorted by binder
  dependencies at extraction. `pipeline.order_tree` re-sorts them by group
  priority without breaking dependencies.
* Only features up to each body's Tip are included (later ones are inactive
  history).
* `role` is one of `sketch`, `additive`, `subtractive`, `pattern`,
  `dressup`, `binder`, `datum`, `other`. Only the first five are *visual*
  steps (`tree.VISUAL_ROLES`).
* Captured properties: `worker.CAPTURED_PROPS`. Add one there when a new
  feature behaviour needs it.

All helpers are pure (`state_up_to`, `with_overrides`, `reorder`,
`diff_overrides`, `geometry_hash`, `fingerprint`). Tests:
`tests/test_tree.py`.

## Layer 2: the stateless geometric compiler

### FreeCAD side: `cadanim/fc/worker.py :: compile_state`

For every state:

1. `FreeCAD.openDocument(source)`: a fresh document every time.
2. `_apply_state`: write every upper-case property that differs from the
   document. Then set each body's `Tip` to its last included solid feature.
   Bodies without included solid features produce nothing.
3. `doc.recompute()`, then record any included feature that is
   `Invalid`/`Error` (`info.errors`).
4. Write one binary STL per active body in global coordinates
   (`shape.tessellate(deflection)`). Record the polylines of every included
   sketch not yet consumed by an included feature (as `profile`) in
   `info.json`.
5. With `focus=True`, also describe the last visual step for camera
   planning: its bbox, its profile normal, the *changed material* mesh
   (additive tool, clipped removed material, or the symmetric difference
   for patterns/dress-ups) and `sweep_depth` (used to animate ThroughAll /
   UpTo* sweeps).
6. `closeDocument`. Nothing survives to the next state.

Output: the **model store** folder `<models>/<key>/` with `<Body>.stl`,
`_focus.stl` (planning only) and `info.json` (written last, which marks the
folder complete). The key is `geometry_hash(state)` + deflection +
`COMPILER_VERSION` (+ `_focus`). FreeCAD never sees a camera.

### Pipeline side: `cadanim/geometry.py`

* `load_model_dir(path)` reads a state folder (STLs + info) without
  FreeCAD. The renderer and `rerender` use only this.
* `GeometryCompiler.compile_many(states, focus)` de-duplicates by key,
  splits the misses over `workers` parallel `freecadcmd` processes and
  returns keys. Failed states are recorded (`info.failed`), and the
  sequencer drops failing intra-step frames.
* `tree_to_model(tree_state, camera_view, output_path, *, compiler, renderer, ...)`
  is the spec's single-frame entry point: compile (cache), then render one PNG.

### Rendering: `cadanim/render.py`

The viewer draws the stored STLs with VTK offscreen and applies the frame's
camera:

* colour per **group**, display mode per **body** (`solid`, `translucent`,
  `hidden`) from `groups.body_modes`
* `ghost_active`: the active group's bodies are drawn semi-transparent while
  an occluded sketch is presented
* feature edges, sketch polylines (`render_reveal` truncates them by arc
  length), optional caption, transparent RGBA PNG

## Requirements checklist

- [x] Parsing (tree) is fully separate from rendering loops.
- [x] Same tree prefix gives the same geometry (fresh document per state,
      hash-keyed cache, integration test `test_compile_is_deterministic_and_cached`).
- [x] Property overrides map to `setattr` on the FreeCAD objects (the
      tree-to-FreeCAD mapper).
- [ ] Rebuilding from a document-free tree (re-creating sketches from JSON)
      is out of scope. The source `.FCStd` remains the geometry authority and
      the tree selects and overrides into it.

## Extending

* New feature kind: make sure `_role` classifies it and its parameters are in
  `CAPTURED_PROPS`. If its changed material needs special handling, extend
  `_focus_info`. Bump `COMPILER_VERSION` when the outputs change.
