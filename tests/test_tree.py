import pytest

from cadanim import tree as T


def test_state_up_to_is_prefix_and_independent(sample_tree):
    fp = T.fingerprint(sample_tree)
    st = T.state_up_to(sample_tree, "SkC")
    assert T.feature_ids(st) == ["SkA", "PadA", "SkC"]
    st["PadA"]["Length"] = 1.0
    assert sample_tree["PadA"]["Length"] == 120.0
    assert T.fingerprint(sample_tree) == fp


def test_empty_state_has_only_meta(sample_tree):
    st = T.empty_state(sample_tree)
    assert T.feature_ids(st) == []
    assert st[T.META_KEY]["source_sha256"] == "abc"


def test_unknown_step_raises(sample_tree):
    with pytest.raises(KeyError):
        T.state_up_to(sample_tree, "Nope")


def test_geometry_hash_ignores_render_hints(sample_tree):
    st = T.state_up_to(sample_tree, "SkA")
    h = T.geometry_hash(st)
    st2 = T.with_overrides(st, "SkA", {"render_reveal": 0.5})
    assert T.geometry_hash(st2) == h
    st3 = T.with_overrides(T.state_up_to(sample_tree, "PadA"), "PadA", {"Length": 10.0})
    assert T.geometry_hash(st3) != T.geometry_hash(T.state_up_to(sample_tree, "PadA"))


def test_geometry_hash_depends_on_order(sample_tree):
    a = T.reorder(sample_tree, ["SkC", "RevC", "SkA", "PadA", "BindB", "SkB", "PocketB", "PolarB"])
    assert T.geometry_hash(a) != T.geometry_hash(sample_tree)


def test_reorder_requires_permutation(sample_tree):
    with pytest.raises(ValueError):
        T.reorder(sample_tree, ["SkA"])


def test_diff_overrides_round_trip(sample_tree):
    st = T.with_overrides(T.state_up_to(sample_tree, "PadA"), "PadA", {"Length": 30.0})
    assert T.diff_overrides(sample_tree, st) == {"PadA": {"Length": 30.0}}


def test_last_visual_feature_skips_binders(sample_tree):
    st = T.state_up_to(sample_tree, "BindB")
    assert T.last_feature(st) == "BindB"
    assert T.last_visual_feature(st) == "RevC"
