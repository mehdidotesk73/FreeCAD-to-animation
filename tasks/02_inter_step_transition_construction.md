# Task 02: Inter-step transition construction (camera space)

**Focus:** choosing a view per step and moving the camera smoothly between
steps.

## Camera view schema (`cadanim/camera.py`)

```python
{"FocalPoint": [x, y, z], "Rotation": [w, x, y, z], "Distance": d,
 "Zoom": 1.0, "ViewAngle": 24.0,           # inputs
 "Position": [...], "ViewUp": [...]}        # derived by complete_view()
```

The camera looks down its local -Z and +Y is up (OpenGL convention).
FreeCAD world is Z-up.

## `interpolate_camera(view_start, view_end, frames_count, easing="linear")`

* Rotation: **SLERP** along the shortest arc.
* FocalPoint, Zoom, ViewAngle: **LERP**.
* Distance: **geometric** interpolation (`d0 * (d1/d0)**t`), which gives a
  uniform-feeling zoom.
* Position is **derived** from the above rather than LERPed. A straight
  Position LERP cuts through the model when the camera swings around it, so
  the camera orbits the moving focus instead. The inputs of the spec's
  signature are kept; only Position's derivation differs.
* Endpoints are inclusive. The sequencer drops the first frame (it equals the
  previous step's view).
* `views_equal(a, b)` (angle < 0.5 deg, focal/distance within 1%) produces
  the spec's "zero-frame transition" for identical views.

## Choosing View_k: `ViewPlanner.plan` (`cadanim/pipeline.py`)

1. **Target** = what the step changes: the sketch polylines, or the
   additive/removed material mesh, or the changed region of a
   pattern/dress-up (from the compiler's focus data).
2. **Framing sphere** (`_framing`): the union of the active body's bbox and
   the target. With `camera.context_weight` = 1 the whole body is in view.
   Targets smaller than `small_feature_ratio` of the body pull the camera in
   (never below `min_context`). For the first sketch of a new body, the
   existing solid geometry is the context.
   `fit_distance` fits the sphere into the narrower of the two FOVs.
3. **Candidates**: 30 standard directions (isometric octants, two elevated
   rings, six axes). A sketch is always viewed **perpendicular to its
   plane**; only the side (normal or opposite) is chosen.
4. **Score** (`VisibilityProbe`, 240x180, unlit): visible target pixels
   past the *solid* context, measured with **exactly one render per
   candidate**. The step's scene is loaded once (`setup`); per candidate only
   the camera moves. Occluders are black and the target red, and red pixels
   are counted. The target is nudged 0.4% of the camera distance towards the
   camera so faces it shares with an occluder (cut walls) count as visible.
   Read images with
   `p.render()` + `p.image`; `p.screenshot()` can return a stale image there.
   Each score is
   `sqrt(visible/best) + elevation_bias * elevation_preference(d)
   + continuity_bias * max(0, d . prev_d)`. Sketch scores are multiplied by
   `|d . normal|` (face-on).
5. **Hysteresis** (features only): keep the previous direction if it scores
   at least `hysteresis` x best. If the direction is kept and the framing is
   within 25% / ±33%, keep the whole previous view (no transition).

## Sketch steps

A sketch step runs in this order:

1. **Turn and fade in.** The camera turns perpendicular to the sketch, and
   during the same frames every body fades to `render.ghost_opacity` (frame
   field `ghost` ramps 0 -> 1).
2. **Draw.** The sketch is drawn progressively (`render_reveal`), then held.
3. **Turn and fade out.** The next step's transition turns the camera to
   that step's view while `ghost` ramps back to 0. If two views are equal,
   the fade happens in place over `transition_frames / 2` frames.
4. **Sketch fade-out.** When the feature that uses the sketch as its profile
   starts growing, the sketch lines fade out over `timing.sketch_fade_frames`
   frames instead of vanishing. These are frame `overlays`: lines taken from
   the sketch step's own model folder and drawn with decreasing opacity.

All of these knobs live in the project config under `"camera"` (defaults in
`groups.DEFAULT_CAMERA`).

## Manual adjustments (viewer-side, no FreeCAD)

Cameras only exist in the viewer. Every state's geometry is already in the
model store, so re-framing never recompiles:

* `camera.step_views` in the config, keyed by feature id, label or
  `"BodyLabel/FeatureLabel"`:
  ```json
  "step_views": {"Pad": {"azimuth": 30, "elevation": 25, "zoom": 1.3},
                 "Sketch012": {"direction": [0, -1, 0.3], "ghost": true},
                 "Groove001": {"keep_previous": true}}
  ```
  Optional keys: `distance_scale` and `focal_offset`. Rerun `animate`;
  transitions and holds are re-planned around the new view.
* For frame-level edits, change `frames[i].camera` in `manifest.json` and run
  `python -m cadanim rerender OUT --frames i-j`. Derived fields are
  recomputed from `FocalPoint` / `Rotation` / `Distance`.

## Finale

After the last step the camera moves to a three-quarter view fitted to the
whole model and orbits 360 deg (`camera.orbit`, `timing.finale_frames`), with
every group solid.

## Tests

`tests/test_camera.py`: quaternion round trips, endpoint equality, 45-deg
mid-SLERP, orbit radius preservation (no cut-through), fit distance,
`views_equal`.
