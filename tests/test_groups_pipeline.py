import pytest

from cadanim import groups as G
from cadanim import tree as T
from cadanim.pipeline import action_queue, order_tree


def _cfg(tree, groups, **ctx):
    return G.from_dict({"source": "/tmp/x.FCStd", "groups": groups, "context": ctx}, tree)


def test_bodies_resolved_by_label_or_name(sample_tree):
    cfg = _cfg(sample_tree, [{"name": "Main", "bodies": ["Shaft", "b"]},
                             {"name": "Lid", "bodies": ["Lid"]}])
    assert cfg.group("Main").bodies == ("a", "b")
    assert cfg.body_to_group()["c"] == "Lid"


def test_unknown_or_duplicate_body_rejected(sample_tree):
    with pytest.raises(ValueError):
        _cfg(sample_tree, [{"name": "X", "bodies": ["Nope"]}])
    with pytest.raises(ValueError):
        _cfg(sample_tree, [{"name": "X", "bodies": ["a"]}, {"name": "Y", "bodies": ["a"]}])


def test_order_follows_groups_but_respects_dependencies(sample_tree):
    # Group order puts 'b' (Base) before the Lid; 'b' needs 'a' first.
    cfg = _cfg(sample_tree, [{"name": "Main", "bodies": ["a", "b"]}, {"name": "Lid", "bodies": ["c"]}])
    order = T.feature_ids(order_tree(sample_tree, cfg))
    assert order == ["SkA", "PadA", "BindB", "SkB", "PocketB", "PolarB", "SkC", "RevC"]
    # If the Lid group comes first it is built first.
    cfg2 = _cfg(sample_tree, [{"name": "Lid", "bodies": ["c"]}, {"name": "Main", "bodies": ["a", "b"]}])
    assert T.feature_ids(order_tree(sample_tree, cfg2))[:2] == ["SkC", "RevC"]


def test_dependency_beats_group_priority(sample_tree):
    # 'b' is in the first group but depends on 'a' from the second group.
    cfg = _cfg(sample_tree, [{"name": "B", "bodies": ["b"]}, {"name": "A", "bodies": ["a", "c"]}])
    order = T.feature_ids(order_tree(sample_tree, cfg))
    assert order.index("PadA") < order.index("BindB")


def test_action_queue_skips_binders_and_ungrouped(sample_tree):
    cfg = _cfg(sample_tree, [{"name": "Main", "bodies": ["a", "b"]}])  # 'c' left ungrouped
    ordered = order_tree(sample_tree, cfg)
    q = action_queue(ordered, cfg)
    assert "BindB" not in q and "SkC" not in q and "RevC" not in q
    assert q == ["SkA", "PadA", "SkB", "PocketB", "PolarB"]
    assert action_queue(ordered, cfg, only_groups=["Main"]) == q


def test_body_modes_binding_surfaces(sample_tree):
    cfg = _cfg(sample_tree, [{"name": "Shaft", "bodies": ["a"]}, {"name": "Base", "bodies": ["b"]},
                             {"name": "Lid", "bodies": ["c"]}])
    modes = G.body_modes(cfg, sample_tree, "Base")
    assert modes == {"a": "solid", "b": "solid", "c": "translucent"}  # Base binds to the Shaft
    assert G.body_modes(cfg, sample_tree, "Shaft") == {"a": "solid", "b": "translucent",
                                                       "c": "translucent"}
    assert set(G.body_modes(cfg, sample_tree, None).values()) == {"solid"}


def test_context_overrides_win(sample_tree):
    cfg = _cfg(sample_tree, [{"name": "Shaft", "bodies": ["a"]}, {"name": "Base", "bodies": ["b"]},
                             {"name": "Lid", "bodies": ["c"]}],
               default="hidden", binding="solid",
               overrides={"Base": {"Shaft": "translucent", "Lid": "solid"}})
    assert G.body_modes(cfg, sample_tree, "Base") == {"a": "translucent", "b": "solid", "c": "solid"}
    assert G.body_modes(cfg, sample_tree, "Lid")["a"] == "hidden"


def test_ungrouped_bodies_are_hidden(sample_tree):
    cfg = _cfg(sample_tree, [{"name": "Main", "bodies": ["a", "b"]}])
    assert G.body_modes(cfg, sample_tree, "Main")["c"] == "hidden"
    assert G.HIDDEN_GROUP not in [g["name"] for g in cfg.to_dict()["groups"]]


def test_step_view_override_lookup(sample_tree):
    import numpy as np
    from cadanim import camera as C
    from cadanim.pipeline import ViewPlanner
    cfg = G.from_dict({"source": "/tmp/x.FCStd", "groups": [{"name": "All", "bodies": ["a", "b", "c"]}],
                       "camera": {"step_views": {"Shaft/Pad": {"azimuth": 0, "elevation": 0, "zoom": 2}}}},
                      sample_tree)
    vp = ViewPlanner.__new__(ViewPlanner)  # lookup and view math only, no probe
    vp.cfg = cfg
    vp.view_angle, vp.aspect = 24.0, 16 / 9
    entry = dict(sample_tree["PadA"], label="Pad", body_label="Shaft")
    ov = vp.step_override("PadA", entry)
    assert ov == {"azimuth": 0, "elevation": 0, "zoom": 2}
    d = vp.override_direction(ov)
    assert np.allclose(d, [1, 0, 0])
    view = vp.view_for(d, [-1, -1, -1, 1, 1, 1], ov)
    assert np.allclose(C.view_direction(view), [1, 0, 0]) and view["Zoom"] == 2
