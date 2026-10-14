import copy
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cadanim import tree as T  # noqa: E402


def _feat(kind, role, body, **props):
    return {"kind": kind, "label": props.pop("label", kind.split("::")[-1]), "role": role,
            "body": body, "body_label": body.title(), **props}


@pytest.fixture
def sample_tree():
    """Three bodies: 'b' binds to 'a', 'c' is independent (document order a, c, b)."""
    tree = {
        T.META_KEY: {
            "schema": T.SCHEMA, "source": "/tmp/x.FCStd", "source_sha256": "abc",
            "bodies": {
                "a": {"label": "Shaft", "visible": True, "deps": []},
                "c": {"label": "Lid", "visible": True, "deps": []},
                "b": {"label": "Base", "visible": True, "deps": ["a"]},
            },
        },
        "SkA": _feat("Sketcher::SketchObject", "sketch", "a"),
        "PadA": _feat("PartDesign::Pad", "additive", "a", Type="Length", Length=120.0, profile="SkA"),
        "SkC": _feat("Sketcher::SketchObject", "sketch", "c"),
        "RevC": _feat("PartDesign::Revolution", "additive", "c", Type="Angle", Angle=360.0, profile="SkC"),
        "BindB": _feat("PartDesign::SubShapeBinder", "binder", "b", binds=["a"]),
        "SkB": _feat("Sketcher::SketchObject", "sketch", "b"),
        "PocketB": _feat("PartDesign::Pocket", "subtractive", "b", Type="ThroughAll", Length=5.0,
                         profile="SkB"),
        "PolarB": _feat("PartDesign::PolarPattern", "pattern", "b", Angle=360.0, Occurrences=4,
                        Mode="Extent"),
    }
    return copy.deepcopy(tree)
