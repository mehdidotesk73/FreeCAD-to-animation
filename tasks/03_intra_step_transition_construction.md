# Task 03: Intra-step transition construction (object space)

**Focus:** transient tree states that show a feature growing under a fixed
view.

## Iterators (`cadanim/intra.py`)

```python
create_intra_step_iterator(base_tree, target_step_id, param_key,
                           start_val, end_val, steps_count) -> List[dict]
```

The spec's form: uniform values from `start_val` to `end_val` (inclusive),
one deep copy per state. `base_tree` must already contain the step (with its
final definition), and only `param_key` changes.

```python
create_intra_step_iterator_fn(base_tree, target_step_id,
                              overrides_at: Callable[[float], dict],
                              steps_count, easing="linear") -> List[dict]
```

The general higher-order form. `overrides_at(t)` returns the property
overrides at `t` in (0, 1]. The last sample is always `t = 1`, so the final
state is the finished feature. Some features need it: polar patterns change
`Occurrences` and `Angle` together, and ThroughAll pockets temporarily become
Length pockets.

## Feature behaviours (`intra.ANIMATORS`)

| Feature | Animation | Notes |
|---|---|---|
| Sketch | `render_reveal` 0 -> 1 | drawn by arc length; render-only, so no FreeCAD rebuilds |
| Pad / Pocket, `Length` | `Length` 2% -> L | FreeCAD rejects zero length |
| Pad / Pocket, `TwoLengths` | `Length`, `Length2` scaled together | |
| Pad / Pocket, `ThroughAll` / `UpTo*` | temporary `Type=Length`, depth 0 -> `sweep_depth` | last frame restores the original Type exactly; `Midplane` doubles the depth |
| Revolution / Groove, `Angle` | `Angle` 2% -> A | `TwoAngles` scales both |
| PolarPattern | all copies from the first frame, `Angle` 2% -> final | `Occurrences` fixed at its final value. Extent mode spaces copies A/(n-1), except a full 360 deg circle (A/n), so a full circle sweeps to 360(n-1)/n (identical layout, no jump) and the last frame restores 360. `Mode=Spacing` sweeps `Offset` instead |
| LinearPattern | all copies from the first frame, `Length` 2% -> final | `Mode=Spacing` sweeps `Offset` instead |
| Fillet / Chamfer / Draft | `Radius` / `Size` / `Angle` 2% -> value | |
| others (Mirrored, MultiTransform, Thickness, Pipe, Loft, ...) | appear | `timing.appear_frames` static frames |

`plan_growth(step_state, step_id, frames, focus, easing)` returns a
`GrowthPlan(description, states, values)`, or `None` when the feature just
appears. All behaviours use the configured easing.

Intermediate states that FreeCAD cannot build (for example a fillet radius
that is too large mid-way) are recorded as failed by the compiler and
dropped by the sequencer. The step still ends on the exact finished feature.

## Adding a behaviour

```python
def _my_feature(entry, focus, frames):
    v = float(entry.get("SomeProp", 0))
    if v <= 0:
        return None                       # -> appears instead
    return f"SomeProp 0 -> {v:g}", (lambda t: {"SomeProp": _scaled(v, t)}), frames

ANIMATORS["PartDesign::MyFeature"] = _my_feature
```

Make sure the property is captured (`fc/worker.py: CAPTURED_PROPS`) and add a
test in `tests/test_intra.py`.

## Tests

`tests/test_intra.py`: spec iterator values and independence of copies,
monotonic pad growth, ThroughAll type restoration, polar instance positions,
render-only sketch reveal (one geometry hash for all frames), binders produce
no growth.
