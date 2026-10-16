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

## Choosing the views: plan the whole sequence, not each step

Picking each step's best view on its own makes the camera wander. Views are
therefore planned for the whole sequence at once.

### 1. Measure (`ViewPlanner.measure`, `cadanim/pipeline.py`)

For every step:

* **Target** = what the step changes: the sketch polylines, or the
  additive/removed material mesh, or the changed region of a
  pattern/dress-up (from the compiler's focus data).
* **Region** (`ViewPlanner.region`, a box): the union of the active body's
  bbox and the target. Targets smaller than `small_feature_ratio` pull the
  region towards them (never below `min_context`). A body smaller than
  `min_body_fraction` of the already-built assembly is framed with some of
  the assembly around it. For the first sketch of a new body, the existing
  solid geometry is the context.
* **Candidates**: 30 standard directions for 3D steps. A sketch has exactly
  two: perpendicular to its plane, from either side.
* **Visibility** (`VisibilityProbe`, 240x180, unlit, **one render per
  candidate**): target pixels visible past the *solid* context. Occluders
  are black and the target red; the target is nudged 0.4% of the camera
  distance towards the camera so faces it shares with an occluder (cut
  walls) count. Every candidate uses the same scale (the region's bounding
  sphere), so pixel counts are comparable between directions. Read images
  with `p.render()` + `p.image`; `p.screenshot()` can return a stale image.
* **Quality** = `sqrt(visible/best) + elevation_bias * elevation_preference(d)`.

### 2. Plan (`cadanim/camplan.py`, pure and unit-tested)

`plan_sequence` chooses one candidate per step by dynamic programming,
minimising (weights in `camera.sequence`):

    quality    x (best quality - chosen quality)        per step
    fixed_move x number of camera moves
    move       x rotation / 90 deg                      per move
    home       x angle between 3D views separated by sketches / 90 deg

3D candidates below `min_visibility` (60%) of the step's best are not
allowed. A move into or out of a sketch always counts as a move (the framing
changes even when the direction does not).

The effect: a pocket that reads fine from the next pattern's angle is shown
from that angle, so the camera goes sketch -> that view once, and the pattern
follows with no move. A sequence like revolve / sketch / pocket / pattern
comes back to the revolve's view after the sketch.

### 3. Shots and framing

`shots` groups consecutive 3D steps **of the same body** on the same
direction (sketches in between included) into a shot. All steps of a shot
share one framing: the union of their regions. Within a shot the camera does
not move at all, and after a sketch it returns to exactly the same view.

3D views fit the region box's **projected outline** (`camera.fit_box`), so
wide, flat parts fill the frame. Sketch views fit the bounding sphere, which
keeps a steady scale when looking straight at a tall part.

### Review sheet

Every run writes `<out>/steps_sheet.png`: one tile per step (its last
frame), labelled with its shot. `--plan-only` plus the sheet reviews a camera
plan in about a minute without rendering the animation; `cadanim sheet OUT`
rebuilds it.

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
