"""Command line interface: ``python -m cadanim <command> ...``.

    inspect  MODEL.FCStd [--init-config CFG]   bodies, binder links, contacts (grouping aid)
    extract  MODEL.FCStd -o tree.json          dump the feature tree (Layer 1 data)
    export   CFG [-o DIR]                      group STLs (union / separate)
    frame    CFG --step ID -o out.png          one frame via tree_to_model
    animate  CFG [-o DIR] [--group G] [--preview] [--plan-only] [--video]
    rerender OUTDIR [--frames A-B] [--video]   redraw frames from manifest + STLs (no FreeCAD)
    video    OUTDIR                            encode frames/ into an MP4

FreeCAD only produces geometry: one STL per body for every compiled state
(stored under <out>/models for animate). Cameras are applied by the viewer,
so camera tweaks (config camera.step_views, or editing manifest.json and
running rerender) never rerun FreeCAD when the geometry is already stored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

from . import fcbridge
from . import groups as G
from . import tree as T

DEFAULT_CACHE = os.path.join(os.getcwd(), ".cadanim-cache")


def _extract(model: str, cache_dir: str | None = None) -> dict:
    """Feature tree of ``model``; cached by file hash so unchanged models skip FreeCAD."""
    from .fc import COMPILER_VERSION
    path = None
    if cache_dir:
        sha = hashlib.sha256(open(model, "rb").read()).hexdigest()[:20]
        path = os.path.join(cache_dir, f"tree_{sha}_v{COMPILER_VERSION}.json")
        if os.path.exists(path):
            tree = T.load(path)
            tree[T.META_KEY]["source"] = os.path.abspath(model)
            return tree
    tree = fcbridge.run_job({"action": "extract", "source": model})["tree"]
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        T.save(tree, path)
    return tree


def _load_project(cfg_path: str, cache_dir: str | None = None):
    with open(cfg_path) as fh:
        data = json.load(fh)
    base = os.path.dirname(os.path.abspath(cfg_path))
    src = data.get("source")
    if not src:
        raise SystemExit(f"{cfg_path}: 'source' (path to the .FCStd) is required")
    model = src if os.path.isabs(src) else os.path.join(base, src)
    tree = _extract(model, cache_dir)
    return tree, G.from_dict(data, tree, base_dir=base)


def cmd_inspect(a):
    tree = _extract(a.model)
    info = fcbridge.run_job({"action": "inspect", "source": a.model})
    print(G.suggest_report(tree, info))
    if a.init_config:
        cfg = G.default_config(tree, source=os.path.relpath(os.path.abspath(a.model),
                                                            os.path.dirname(os.path.abspath(a.init_config))))
        with open(a.init_config, "w") as fh:
            json.dump(cfg.to_dict(), fh, indent=2)
        print(f"\nwrote {a.init_config} (one group per body; edit the grouping before animating)")


def cmd_extract(a):
    tree = _extract(a.model)
    T.save(tree, a.output)
    print(f"wrote {a.output}: {len(T.feature_ids(tree))} features in "
          f"{len(T.meta(tree)['bodies'])} bodies")


def cmd_export(a):
    tree, cfg = _load_project(a.config, a.cache or DEFAULT_CACHE)
    out = a.output or os.path.join(os.path.dirname(os.path.abspath(a.config)), "stl")
    groups = [{"name": g.name, "bodies": list(g.bodies), "export": g.export}
              for g in cfg.groups if g.name != G.HIDDEN_GROUP]
    res = fcbridge.run_job({"action": "export", "source": cfg.source, "state": tree,
                            "groups": groups, "outdir": out})
    for w in res["written"]:
        print(f"{w['file']}: {w['facets']} facets, bodies {', '.join(w['bodies'])}, "
              f"volume {w['volume']:.0f} mm3 (sum of bodies)")


def _apply_preview(cfg: G.ProjectConfig) -> G.ProjectConfig:
    d = cfg.to_dict()
    d["render"].update(width=640, height=360)
    for k in ("sketch_frames", "growth_frames", "transition_frames", "hold_frames", "finale_frames",
              "sketch_fade_frames"):
        d["timing"][k] = max(1, int(d["timing"][k]) // 2)
    d["timing"]["fps"] = max(1, int(d["timing"]["fps"]) // 2)
    return d


def cmd_animate(a):
    from . import pipeline
    out = a.output or os.path.join(os.path.dirname(os.path.abspath(a.config)), "animation")
    models = a.cache or os.path.join(out, "models")
    tree, cfg = _load_project(a.config, models)
    if a.preview:
        cfg = G.from_dict(_apply_preview(cfg), tree)
    if a.group:
        unknown = set(a.group) - set(cfg.group_names())
        if unknown:
            raise SystemExit(f"unknown group(s): {sorted(unknown)}; known: {cfg.group_names()}")
    manifest = pipeline.run(tree, cfg, out, cache_dir=models, workers=a.workers,
                            only_groups=a.group, plan_only=a.plan_only)
    print(f"{manifest['frame_count']} frames ({manifest['duration']}s) -> {out}/manifest.json")
    if a.video and not a.plan_only:
        print(encode_video(out, manifest["fps"], cfg.render.get("background", "#ffffff")))


def cmd_frame(a):
    from .camera import complete_view
    from .geometry import GeometryCompiler, tree_to_model
    from .pipeline import ViewPlanner, order_tree
    from .render import Renderer, SceneSpec
    cache = a.cache or DEFAULT_CACHE
    tree, cfg = _load_project(a.config, cache)
    ordered = order_tree(tree, cfg)
    state = T.state_up_to(ordered, a.step)
    comp = GeometryCompiler(cfg.source, cache, workers=1, deflection=float(cfg.render["deflection"]))
    geom = comp.compile(state, focus=True)
    b2g = cfg.body_to_group()
    group = b2g.get(ordered[a.step]["body"])
    modes = G.body_modes(cfg, ordered, group)
    vp = ViewPlanner(cfg)
    try:
        view, ghost = vp.plan(a.step, ordered[a.step], geom,
                          [b for b in geom.bodies if modes.get(b) == "solid"],
                          geom.body_bbox([ordered[a.step]["body"]]), None)
    finally:
        vp.close()
    r = Renderer(SceneSpec(body_group=b2g, colors=cfg.colors(), style=dict(cfg.render)))
    try:
        from .render import FrameRequest
        r.render(FrameRequest(geometry_key=geom.key, camera=complete_view(view), output=a.output,
                              active_group=group, caption=a.step, modes=modes,
                              ghost=1.0 if ghost else 0.0), geom)
    finally:
        r.close()
    print(f"wrote {a.output}")


def encode_video(outdir: str, fps: float, background: str = "#ffffff") -> str:
    """Composite the alpha frames over ``background`` and encode an H.264 MP4."""
    frames = os.path.join(outdir, "frames", "frame_%05d.png")
    out = os.path.join(outdir, "animation.mp4")
    bg = background.lstrip("#")
    cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", frames,
           "-filter_complex",
           f"color=c=0x{bg}:s=16x16[bg];[bg][0:v]scale2ref[bg2][fg];[bg2][fg]overlay=shortest=1:format=auto,setsar=1,format=yuv420p",
           "-c:v", "libx264", "-crf", "20", "-movflags", "+faststart", out]
    subprocess.run(cmd, check=True)
    return out


def cmd_rerender(a):
    from . import pipeline
    idx = None
    if a.frames:
        lo, _, hi = a.frames.partition("-")
        idx = list(range(int(lo), int(hi or lo) + 1))
    m = pipeline.rerender(a.outdir, workers=a.workers, indices=idx)
    if a.video:
        with open(os.path.join(a.outdir, "config.json")) as fh:
            bg = json.load(fh)["render"].get("background", "#ffffff")
        print(encode_video(a.outdir, m["fps"], bg))


def cmd_video(a):
    with open(os.path.join(a.outdir, "manifest.json")) as fh:
        m = json.load(fh)
    print(encode_video(a.outdir, m["fps"], a.background))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cadanim", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=None,
                    help="model store (STL per body per state); default <out>/models for "
                         "animate, ./.cadanim-cache otherwise")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inspect"); p.add_argument("model"); p.add_argument("--init-config")
    p.set_defaults(fn=cmd_inspect)
    p = sub.add_parser("extract"); p.add_argument("model"); p.add_argument("-o", "--output", required=True)
    p.set_defaults(fn=cmd_extract)
    p = sub.add_parser("export"); p.add_argument("config"); p.add_argument("-o", "--output")
    p.set_defaults(fn=cmd_export)
    p = sub.add_parser("frame"); p.add_argument("config"); p.add_argument("--step", required=True)
    p.add_argument("-o", "--output", required=True); p.set_defaults(fn=cmd_frame)
    p = sub.add_parser("animate"); p.add_argument("config"); p.add_argument("-o", "--output")
    p.add_argument("--group", action="append", help="only animate these group(s); repeatable")
    p.add_argument("--preview", action="store_true", help="half resolution / half frame counts")
    p.add_argument("--plan-only", action="store_true", help="plan + manifest, no rendering")
    p.add_argument("--video", action="store_true", help="also encode animation.mp4")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    p.set_defaults(fn=cmd_animate)
    p = sub.add_parser("rerender"); p.add_argument("outdir")
    p.add_argument("--frames", help="frame index or range A-B (default: all)")
    p.add_argument("--video", action="store_true")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    p.set_defaults(fn=cmd_rerender)
    p = sub.add_parser("video"); p.add_argument("outdir"); p.add_argument("--background", default="#ffffff")
    p.set_defaults(fn=cmd_video)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
