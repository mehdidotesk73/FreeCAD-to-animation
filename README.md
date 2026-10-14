# FreeCAD-to-animation

Turns a FreeCAD PartDesign model into a build-up animation. Each sketch is
drawn, each pad, pocket, revolution or pattern grows, and the camera moves to
the clearest view of every step. The output is transparent PNG frames, a JSON
manifest indexing every frame, and optionally an MP4. The same project config
exports print-ready STLs per part group.

```
FCStd ──extract──▶ feature tree ──order/queue──▶ steps ──plan──▶ frames (state + camera)
                    (immutable)                    │
                                                   ├─ view per step (visibility-scored)
                                                   ├─ inter-step camera moves (SLERP/LERP)
                                                   └─ intra-step feature growth
frames ──FreeCAD: STL per body per state (parallel, cached)──▶ viewer applies camera (VTK offscreen) ──▶ PNGs + manifest.json (+ MP4)
```

FreeCAD only produces geometry: an STL per body for every finished step and
every growth increment (`<out>/models/`). The camera lives entirely in the
viewer, so camera tweaks (`camera.step_views` in the config, or edited
cameras in `manifest.json` + `python -m cadanim rerender OUT`) never rerun
FreeCAD.

## Quick start

```bash
scripts/install_freecad.sh            # FreeCAD 1.1 (headless) + offscreen GL + ffmpeg
pip install -r requirements.txt

python -m cadanim inspect examples/birdfeeder/BirdFeeder.FCStd       # bodies, binders, contacts
python -m cadanim export  examples/birdfeeder/birdfeeder.cadanim.json   # group STLs
python -m cadanim animate examples/birdfeeder/birdfeeder.cadanim.json --preview --video
```

## Groups

Bodies are organised in **groups** (in `<model>.cadanim.json`):

```json
{
  "source": "BirdFeeder.FCStd",
  "groups": [
    {"name": "Feeder", "color": "#4f7cac", "export": "union",
     "bodies": ["Shaft", "Base", "ShaftFastner", "Platform"]},
    {"name": "Roof", "color": "#e0a03a", "export": "separate", "bodies": ["Top"]}
  ],
  "context": {"default": "translucent", "binding": "solid", "overrides": {}}
}
```

* One colour per group.
* `union`: modules modelled separately but printed as one solid. Their meshes
  go into one STL with no boolean fuse; slicers merge touching shells.
  `separate`: one STL per body.
* While a group is being animated, other bodies are shown according to
  `context`. Bodies the group is bound to (SubShapeBinders, i.e. mating
  surfaces) stay `solid`, other existing bodies are `translucent`, and
  `overrides` can set `solid` / `translucent` / `hidden` per group pair.
* `animate --group Roof` renders frames only for that group, with the
  others in context.

Optional `timing`, `render` and `camera` blocks override the defaults in
`cadanim/groups.py` (fps, frame counts per phase, resolution, opacity, view
heuristics...).

## Layout

| Path | Role |
|---|---|
| `cadanim/tree.py` | Layer 1: immutable feature tree and pure state helpers |
| `cadanim/fc/worker.py` | runs inside FreeCAD: extract, stateless compile, export, inspect |
| `cadanim/fcbridge.py` | launches `freecadcmd` jobs (in parallel) |
| `cadanim/geometry.py` | Layer 2: hash-keyed STL model store, `tree_to_model` |
| `cadanim/stl.py` | binary STL read/write (shared by FreeCAD side and viewer) |
| `cadanim/render.py` | offscreen frame renderer and visibility probe |
| `cadanim/camera.py` | view schema, quaternions, `interpolate_camera`, framing |
| `cadanim/intra.py` | `create_intra_step_iterator`, feature-growth mapping |
| `cadanim/groups.py` | groups, colours, context policy, defaults |
| `cadanim/pipeline.py` | Layers 3 and 4: queue, view planning, timeline, rendering, manifest |
| `CLAUDE.md`, `tasks/*.md` | agent instructions per layer |

Tests: `python -m pytest -q tests` (FreeCAD integration tests run when
`freecadcmd` is available).
