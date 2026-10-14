# CLAUDE.md: orchestrator instructions

This repository turns a FreeCAD parametric model (`.FCStd`) into an
animation. The animation shows the model being built step by step: sketches
are drawn, features grow, and the camera moves to the best view of each
step. The output is PNG frames, a JSON manifest and optionally an MP4.

Read this file first. The three task files in `tasks/` cover the layers in depth:

| Task file | Layer | Code |
|---|---|---|
| `tasks/01_major_frame_construction.md` | Layer 1 data + Layer 2 geometry | `cadanim/tree.py`, `cadanim/fc/worker.py`, `cadanim/geometry.py`, `cadanim/render.py` |
| `tasks/02_inter_step_transition_construction.md` | camera / inter-step | `cadanim/camera.py`, `ViewPlanner` in `cadanim/pipeline.py` |
| `tasks/03_intra_step_transition_construction.md` | feature growth / intra-step | `cadanim/intra.py` |

Orchestration (Layer 3) and the frame sequencer and manifest (Layer 4) live in
`cadanim/pipeline.py`.

## Environment

* FreeCAD is **not** importable from the system Python. Geometry work runs in
  `freecadcmd` subprocesses (`cadanim/fcbridge.py` -> `cadanim/fc/worker.py`).
  Install it with `scripts/install_freecad.sh` (FreeCAD 1.1.x AppImage,
  extracted to `/opt/freecad`), or point `CADANIM_FREECADCMD` at an existing
  `freecadcmd`.
* Rendering uses PyVista/VTK offscreen. A headless machine needs
  `libosmesa6` and/or `libegl1` (also installed by the script).
* Python deps: `pip install -r requirements.txt`.
* Tests: `python -m pytest -q tests`. FreeCAD integration tests skip
  themselves when `freecadcmd` is missing.

## Workflow for a new model (follow this order)

1. **Study the model**: `python -m cadanim inspect MODEL.FCStd`. This lists
   bodies, sizes, binder links (which body is built on another body's
   faces) and touching or overlapping bodies.
2. **Decide the grouping with the user.** Groups decide colours, STL export
   and what the context looks like during the animation:
   * Bodies in the **same group share a colour**. Different groups get
     different colours.
   * `"export": "union"`: separately modelled modules that are printed as
     **one solid**, fused into a single STL.
   * `"export": "separate"`: one STL per body.
   * Propose a grouping from the inspect report (touching bodies that print
     together, parts that print separately such as lids or roofs), then
     **ask the user to confirm or correct it**. Do not guess silently. Some
     bodies touch but are still printed separately.
   * Bodies left out of every group are built (other bodies may bind to them)
     but never drawn or exported.
3. **Write the project config** `<model>.cadanim.json` next to the model
   (`inspect --init-config` writes a one-group-per-body starting point). See
   `examples/birdfeeder/birdfeeder.cadanim.json`. Its `context` block decides
   how other groups look while a group is animated:
   * `binding` (default `solid`): bodies the active group is bound to via
     SubShapeBinders, i.e. mating/binding surfaces, stay fully visible.
   * `default` (default `translucent`): every other existing body is ghosted
     but visible for as long as it exists.
   * `overrides`: `{"ActiveGroup": {"OtherGroup": "solid|translucent|hidden"}}`.
4. **Export STLs**: `python -m cadanim export CONFIG [-o DIR]`.
5. **Animate**:
   * quick look: `python -m cadanim animate CONFIG --preview --video`
   * final: `python -m cadanim animate CONFIG --video`
   * one group only: `--group Roof` (repeatable). Earlier groups are
     still built and shown per the context policy.
   * plan only, no rendering: `--plan-only` (writes the manifest).
   * a single frame for debugging: `python -m cadanim frame CONFIG --step Pad -o pad.png`
6. **Review** the output before handing it over: open a contact sheet of the
   last frame of every step (`manifest.json` -> `steps[].last_frame`) and
   watch the MP4. Tune `camera` / `timing` / `render` in the config rather
   than the code when a view or pacing looks off.

## Pipeline contract (Layer 3 coordination)

```
FCStd --extract--> full tree (immutable)
      --order_tree(groups)--> ordered tree --action_queue--> [step ids]
for each step k:
    view_k            = ViewPlanner.plan(...)                     # task 02
    inter-step frames = interpolate_camera(view_{k-1}, view_k)    # task 02 (skipped if equal)
                        over the *historical* state (everything before step k)
    intra-step frames = plan_growth(state_k, step) under view_k   # task 03
    hold frames       = finished state_k under view_k
baseline_{k+1} = state_k
finale: turntable of the finished model
compile every unique state (parallel, cached) -> render every frame -> manifest.json
```

### Rules (do not break)

* **Immutability.** Never mutate a tree or state in place. Use
  `tree.state_up_to`, `tree.with_overrides`, `intra.create_intra_step_iterator*`
  (all deep-copy). `pipeline.run` asserts that the source tree fingerprint is
  unchanged at the end, so keep that assertion.
* **Tree keys.** Upper-case keys of a feature entry are FreeCAD properties
  (they change geometry). Lower-case keys are metadata. `render_*` keys are
  presentation-only and excluded from the geometry hash.
* **Determinism.** `compile(state)` must depend only on the state (and the
  source file). The geometry cache is keyed by `geometry_hash(state)`,
  the tessellation deflection and `cadanim.fc.COMPILER_VERSION`. **Bump
  `COMPILER_VERSION`** whenever the worker's output for a state changes.
* **Statelessness.** Every state is built in a freshly opened FreeCAD
  document that is closed afterwards. Never reuse a document between states.
* **Typed interfaces.** Camera views always carry `FocalPoint`, `Rotation`
  (quaternion w,x,y,z), `Distance`, `Zoom`, `ViewAngle`, plus derived
  `Position`/`ViewUp`. Build them with `camera.make_view` / `complete_view`,
  never by hand.
* **Separation.** FreeCAD-side code (`cadanim/fc/`) imports only FreeCAD,
  the standard library and `cadanim.tree`. Pipeline code never imports
  FreeCAD.

## Output layout (Layer 4)

```
<out>/frames/frame_00000.png ...   RGBA PNGs (transparent background)
<out>/manifest.json                frames[] and steps[] index (see below)
<out>/tree.json                    the ordered full tree (reference for state deltas)
<out>/config.json                  the effective config
<out>/animation.mp4                with --video (frames composited on render.background)
```

Each `manifest.frames[i]` holds `file`, `timestamp`, `frame_type`
(`static` | `inter_step` | `intra_step`), `step_id`, `step_index`, `group`,
`params` (the growing parameter values), `camera`, `geometry_key`,
`context_modes` (body -> mode) and `state = {"upto": <last feature id>,
"overrides": {feature: {prop: value}}}`. That pair reproduces the exact
state from `tree.json`. `manifest.steps[]` gives each step's frame range,
start time, view, ghosting and animation description.

## Known limits

* Only `PartDesign::Body` content is animated (sketches, additive /
  subtractive features, patterns, dress-ups, binders). Part workbench objects,
  assemblies and App::Links are not yet handled.
* Features without a growth mapping simply appear (with hold frames). Add new
  behaviours to `intra.ANIMATORS` (see task 03).
* Colours come from the groups, not from FreeCAD's view providers.
