"""Scene values: instance / compose / move, reuse without mutation, nested rigid transforms and
deterministic instance paths (orb/scenes.py)."""
import math

import pytest

from orb.lang import OrbError, Program
from orb.scenes import Scene, flatten

LIB = """
part "bench" { mass = 20 kg  visual "box" { size = [1.2, 0.6, 0.9] m } }
part "crate" { mass = 2 kg   visual "box" { size = [0.4, 0.4, 0.4] m } }
site "room"  { floor = [6, 4] m  zone "dock" { at = [2, 1, 0] m } }
item "crate-01" { of = part.crate  serial = "C1"  stage = "new" }
"""


def run(src, name="out"):
    p = Program()
    p.add_source(LIB + src, "t.orb")
    return p.resolve_global(name, None)


def leaves(scene, prefix="demo"):
    return {l.path: l for l in flatten(scene, prefix)}


def close(a, b):
    return all(math.isclose(x, y, abs_tol=1e-9) for x, y in zip(a, b))


def test_instance_and_compose():
    s = run("""
fn cell(bench: part, crate: part) -> scene {
  return compose(instance(bench, id = "bench"), instance(crate, id = "crate", at = [0, 0, 0.9] m))
}
let out = cell(part.bench, part.crate)
""")
    assert isinstance(s, Scene) and s.orb_get("ids", None) == ["bench", "crate"]
    ls = leaves(s)
    assert list(ls) == ["demo/bench", "demo/crate"]
    assert close(ls["demo/crate"].pos, (0, 0, 0.9)) and ls["demo/crate"].node.kind == "part"


def test_reuse_twice_rotated_does_not_mutate():
    s = run("""
fn cell() -> scene { return compose(instance(part.bench), instance(part.crate, at = [1, 0, 0] m)) }
let c = cell()
let out = compose(instance(c, id = "a"), instance(c, id = "b", at = [10, 0, 0] m, yaw = 90 deg))
""")
    ls = leaves(s)
    assert list(ls) == ["demo/a/bench", "demo/a/crate", "demo/b/bench", "demo/b/crate"]
    assert close(ls["demo/a/crate"].pos, (1, 0, 0))
    assert close(ls["demo/b/crate"].pos, (10, 1, 0))  # turned 90 deg about the instance origin
    assert close(ls["demo/b/crate"].quat, (math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)))
    # the reused scene is untouched
    c = run("fn cell() -> scene { return compose(instance(part.bench), instance(part.crate, at = [1, 0, 0] m)) }\nlet out = cell()")
    assert close(c.node("crate").pos, (1, 0, 0))


def test_nested_transforms_compose():
    s = run("""
let inner = instance(part.crate, at = [1, 0, 0] m)
let mid   = instance(inner, id = "mid", at = [0, 2, 0] m, yaw = 90 deg)
let out   = instance(mid, id = "outer", at = [5, 0, 0] m, yaw = 90 deg)
""")
    (leaf,) = leaves(s).values()
    assert leaf.path == "demo/outer/mid/crate"
    # crate (1,0,0) -> mid yaw 90: (0,1,0)+(0,2,0) = (0,3,0) -> outer yaw 90: (-3,0,0)+(5,0,0)
    assert close(leaf.pos, (2, 0, 0))
    assert close(leaf.quat, (0, 0, 0, 1))  # 180 deg about z


def test_move_keeps_paths():
    s = run("let out = move(compose(instance(part.bench), instance(part.crate)), at = [1, 2, 0] m)")
    ls = leaves(s, "")
    assert list(ls) == ["bench", "crate"] and close(ls["crate"].pos, (1, 2, 0))


def test_item_lends_definition_and_attributes():
    s = run('let out = instance(item["crate-01"], id = "c", stage = "used")')
    (leaf,) = leaves(s).values()
    assert leaf.node.target.label == "crate"
    assert leaf.attrs == {"serial": "C1", "stage": "used"}


def test_site_and_scene_scope_inherit():
    s = run("""
let room = instance(site.room, at = [0, 0, 0] m)
let out = compose(room, instance(compose(instance(part.bench)), id = "g", only = ["mujoco"]))
""")
    ls = leaves(s)
    assert ls["demo/room"].node.kind == "site"
    assert ls["demo/g/bench"].attrs == {"only": ["mujoco"]}


def test_scene_access():
    s = run("let s = compose(instance(part.bench, id = \"b1\"), instance(part.bench, id = \"b2\"))\n"
            "let out = [s.ids, len(s), s.b2.ids, s[\"b1\"].count]")
    assert s == [["b1", "b2"], 2, ["b2"], 1]


def test_grid_by_comprehension():
    s = run("let out = compose([for i in range(3) : instance(part.crate, id = \"c${i}\", at = [i * 0.5, 0, 0] m)])")
    assert [round(l.pos[0], 6) for l in flatten(s)] == [0, 0.5, 1.0]


@pytest.mark.parametrize("src,msg", [
    ("let out = compose(instance(part.bench), instance(part.bench))", "two instances named 'bench'"),
    ('let out = instance(part.bench, id = "a/b")', "without '.', '/'"),
    ("let out = instance(1)", "places a part, assembly, item, site or scene"),
    ("let out = instance(part.bench, at = [1, 2, 3])", "at takes lengths"),
    ("let out = instance(part.bench, yaw = 90 deg, rpy = [0, 0, 0] deg)", "give one rotation"),
    ("let out = instance(instance(part.bench), id = \"g\", stage = \"x\")", "instance of a scene takes"),
    ("let out = compose(1)", "compose\\(\\) takes scenes"),
    ("let out = move(instance(part.bench), scale = 2)", "no parameter 'scale'"),
])
def test_errors(src, msg):
    with pytest.raises(OrbError, match=msg):
        run(src)


def test_typed_scene_param():
    with pytest.raises(OrbError, match="argument 's' of fn f must be a scene, got number"):
        run("fn f(s: scene) { return s }\nlet out = f(3)")


@pytest.mark.parametrize("kw,msg", [
    ("position = [1, 0, 0] m", "instance\\(\\) has no 'position' \\(use at\\)"),
    ("positon = [1, 0, 0] m", "\\(use at\\)"),
    ("rotation = 90 deg", "\\(use yaw, rpy or quat\\)"),
    ("jionts = {}", "did you mean 'joints'"),
])
def test_instance_catches_misnamed_pose(kw, msg):
    with pytest.raises(OrbError, match=msg):
        run(f'part "p" {{ mass = 1 kg }}\nlet out = instance(part.p, {kw})')


def test_instance_passes_item_attributes():
    s = run('part "p" { mass = 1 kg }\nlet out = instance(part.p, serial = "A1", note = "x", body = "dynamic")')
    assert s.nodes[0].attrs == {"serial": "A1", "note": "x", "body": "dynamic"}
