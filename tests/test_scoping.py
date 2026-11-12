"""Seam scoping (contract v0.3, section 1): `for_seam`, `only`/`skip`, natives, addendum files."""
import copy
import json
from pathlib import Path

import pytest

from orb.lang import OrbError, load
from orb.world import compile_world, for_seam, in_seam

ROOT = Path(__file__).parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "motion_ir.json"


@pytest.fixture
def ir():
    return json.loads(FIXTURE.read_text())


def test_in_seam():
    assert in_seam({}, "isaac") and in_seam({"only": ["isaac"]}, "isaac")
    assert not in_seam({"only": ["minecraft"]}, "isaac")
    assert not in_seam({"skip": ["isaac", "step"]}, "isaac") and in_seam({"skip": ["step"]}, "isaac")


def test_fixture_minecraft_only_horse(ir):
    for seam in ("isaac", "mujoco", "glb", "usd", "step"):
        s = for_seam(ir, seam)
        assert "horse-01" not in s["items"] and "horse" not in s["parts"], seam
        assert s["seam"] == seam
    s = for_seam(ir, "minecraft")
    assert s["items"]["horse-01"]["native"] == {"minecraft": {"entity": "minecraft:horse", "nbt": "{Tame:1b}"}}
    assert s["parts"]["crate"]["native"] == {"minecraft": {"block": "minecraft:barrel"}}
    assert "native" not in s["items"]["truck-01"]


def test_natives_reduced_to_the_seam(ir):
    s = for_seam(ir, "isaac")
    assert s["items"]["truck-01"]["native"] == {"isaac": {"usd": "assets/isaac/truck.usda", "prim": "/Truck",
                                                          "replace": "visual"}}
    assert "native" not in s["parts"]["crate"]
    s = for_seam(ir, "usd")  # usd has its own natives (it reads the isaac keys), not isaac's
    assert "native" not in s["items"]["truck-01"]


def test_for_seam_is_a_copy_and_idempotent(ir):
    before = copy.deepcopy(ir)
    s = for_seam(ir, "mujoco")
    assert ir == before
    assert for_seam(s, "mujoco") == s


def test_removed_item_takes_its_timeline_references(ir):
    ir["items"]["carrier-01"]["skip"] = ["mujoco"]
    s = for_seam(ir, "mujoco")
    tl = s["timelines"]["demo"]
    assert "carrier-01" not in tl["tracks"] and "carrier-01" not in tl["joints"]
    assert "carrier-01" not in tl["bodies"]
    assert [r["to"] for r in tl["attach"]] == ["truck-01"]
    assert [x["name"] for x in s["shots"]] == ["overview"]  # the chase shot followed the carrier
    s2 = for_seam(ir, "isaac")
    assert len(s2["timelines"]["demo"]["attach"]) == 2 and len(s2["shots"]) == 2


def test_removed_part_removes_its_items(ir):
    ir["parts"]["tag"]["only"] = ["isaac"]
    s = for_seam(ir, "glb")
    assert "tag-01" not in s["items"] and "tag-01" not in s["timelines"]["demo"]["visible"]
    assert "tag-01" in for_seam(ir, "isaac")["items"]


def test_removed_link_takes_subtree_joints_and_attach(ir):
    links = ir["assemblies"]["carrier"]["links"]
    links[1]["skip"] = ["step"]  # the boom (and the gripper under it)
    s = for_seam(ir, "step")
    asm = s["assemblies"]["carrier"]
    assert [l["name"] for l in asm["links"]] == ["base"] and asm["mass"] == pytest.approx(150)
    tl = s["timelines"]["demo"]
    assert "carrier-01" not in tl["joints"]  # lift was the boom's joint
    assert [r["to"] for r in tl["attach"]] == ["truck-01"]  # attached to the gripper link: gone


def test_geoms_fixtures_zones_scoped(ir):
    ir["sites"]["yard"]["fixtures"]["wall"]["only"] = ["isaac"]
    ir["sites"]["yard"]["fixtures"]["shelf"]["geoms"][1]["skip"] = ["minecraft"]
    ir["sites"]["yard"]["zones"]["dock"]["skip"] = ["glb"]
    s = for_seam(ir, "minecraft")
    assert "wall" not in s["sites"]["yard"]["fixtures"] and len(s["sites"]["yard"]["fixtures"]["shelf"]["geoms"]) == 2
    s = for_seam(ir, "glb")
    assert "dock" not in s["sites"]["yard"]["zones"]
    assert s["timelines"]["demo"]["phases"][1]["station"] is None  # pointed at the removed zone


def test_scoped_phases_events_cameras_shots(ir):
    tl = ir["timelines"]["demo"]
    tl["phases"][1]["only"] = ["isaac"]
    tl["events"][0]["skip"] = ["mujoco"]
    ir["cameras"]["overview"]["only"] = ["isaac"]
    s = for_seam(ir, "mujoco")
    ph = s["timelines"]["demo"]["phases"]
    assert [p["name"] for p in ph] == ["arrive", "carry", "shelve", "return"] and ph[0]["t1"] == 24.0
    assert [e["name"] for e in s["timelines"]["demo"]["events"]][0] == "crate-picked"
    assert s["cameras"] == {} and [x["name"] for x in s["shots"]] == ["chase"]


def test_timeline_scope_map(ir):
    ir["timelines"]["demo"]["scope"] = {"tracks": {"truck-01": {"only": ["isaac"]}}}
    s = for_seam(ir, "mujoco")
    assert "truck-01" not in s["timelines"]["demo"]["tracks"] and "scope" not in s["timelines"]["demo"]
    assert "truck-01" in for_seam(ir, "isaac")["timelines"]["demo"]["tracks"]


def test_whole_timeline_scoped(ir):
    ir["timelines"]["demo"]["only"] = ["isaac"]
    s = for_seam(ir, "mujoco")
    assert s["timelines"] == {} and s["shots"] == []


# --------------------------------------------------------------------------- OrbLang: only / skip / native

PROJ = '''
site "s" {
  floor = [10, 10] m
  zone "z" { at = [1, 1, 0] m  size = [2, 2] m  skip = ["glb"] }
  fixture "lamp" { at = [0, 0, 2] m  only = ["isaac"]  collision "box" { size = [0.2, 0.2, 0.2] m } }
  fixture "wall" {
    at = [0, 5, 0] m
    kind = "wall"
    collision "box" { size = [10, 0.2, 3] m  at = [0, 0, 1.5] m }
    visual "box" { size = [10, 0.21, 3] m  at = [0, 0, 1.5] m  only = "isaac" }
    native "minecraft" { block = "minecraft:bricks" }
  }
  native "isaac" { usd = "assets/site.usda" }
}
part "p" {
  mass = 1 kg  inertia = [0.01, 0.01, 0.01, 0, 0, 0] kg*m2
  collision "box" { size = [0.2, 0.2, 0.2] m }
  native "glb" { glb = "assets/p.glb"  at = [0, 0, 10] cm  rpy = [0, 0, 90] deg  scale = 2 }
}
assembly "a" {
  link "base" {
    mass = 1 kg  inertia = [0.01, 0.01, 0.01, 0, 0, 0] kg*m2
    link "extra" { at = [0, 0, 1] m  skip = ["mujoco"]  mass = 1 kg  inertia = [0.01, 0.01, 0.01, 0, 0, 0] kg*m2
      native "mujoco" { mjcf = "assets/a.xml"  body = "extra" } }
  }
  native "isaac" { usd = "assets/a.usda"  replace = "all" }
}
item "a-01" { of = assembly.a  location = site.s  at = [2, 0, 0] m }
item "p-01" { of = part.p  location = site.s  skip = ["step", "minecraft"] }
'''


def project(tmp_path, files, assets=("assets/site.usda", "assets/p.glb", "assets/a.xml", "assets/a.usda")):
    d = tmp_path / "proj"
    for name, text in files.items():
        (d / name).parent.mkdir(parents=True, exist_ok=True)
        (d / name).write_text(text)
    for a in assets:
        (d / a).parent.mkdir(parents=True, exist_ok=True)
        (d / a).write_text("x")
    return d


def build(d):
    return compile_world(load(d), d)


def test_only_skip_and_natives_land_in_the_ir(tmp_path):
    ir = build(project(tmp_path, {"main.orb": PROJ}))
    s = ir["sites"]["s"]
    assert s["zones"]["z"]["skip"] == ["glb"] and s["fixtures"]["lamp"]["only"] == ["isaac"]
    assert "only" not in s["fixtures"]["lamp"].get("attrs", {})
    wall = s["fixtures"]["wall"]
    assert wall["geoms"][1]["only"] == ["isaac"]  # a single string is a list of one
    assert wall["native"] == {"minecraft": {"block": "minecraft:bricks"}} and wall["attrs"] == {"kind": "wall"}
    assert s["native"] == {"isaac": {"usd": "assets/site.usda"}}
    assert ir["parts"]["p"]["native"]["glb"] == {"glb": "assets/p.glb", "at": [0, 0, pytest.approx(0.1)],
                                                 "rpy": [0, 0, pytest.approx(1.5707963)], "scale": 2}
    link = ir["assemblies"]["a"]["links"][1]
    assert link["skip"] == ["mujoco"] and link["native"] == {"mujoco": {"mjcf": "assets/a.xml", "body": "extra"}}
    assert ir["assemblies"]["a"]["native"] == {"isaac": {"usd": "assets/a.usda", "replace": "all"}}
    assert ir["items"]["p-01"]["skip"] == ["step", "minecraft"]


def test_for_seam_on_a_compiled_project(tmp_path):
    ir = build(project(tmp_path, {"main.orb": PROJ}))
    mj = for_seam(ir, "mujoco")
    assert [l["name"] for l in mj["assemblies"]["a"]["links"]] == ["base"]
    assert "lamp" not in mj["sites"]["s"]["fixtures"] and len(mj["sites"]["s"]["fixtures"]["wall"]["geoms"]) == 1
    assert "native" not in mj["sites"]["s"] and "native" not in mj["assemblies"]["a"]
    isaac = for_seam(ir, "isaac")
    assert "lamp" in isaac["sites"]["s"]["fixtures"] and isaac["assemblies"]["a"]["native"]["isaac"]["replace"] == "all"
    assert "z" not in for_seam(ir, "glb")["sites"]["s"]["zones"]
    assert "p-01" not in for_seam(ir, "minecraft")["items"] and "p-01" in for_seam(ir, "glb")["items"]


def test_scope_errors(tmp_path):
    def err(src):
        with pytest.raises(OrbError) as e:
            build(project(tmp_path, {"main.orb": src}))
        return e.value.msg

    assert "not both" in err('part "x" { only = ["isaac"]  skip = ["glb"] }')
    assert "list of seam names" in err('part "x" { only = [] }')
    assert "list of seam names" in err('part "x" { only = [1] }')
    assert "cannot go on a key" in err('item "i" { of = part.x }\npart "x" { mass = 1 kg }\n'
                                       'timeline "t" { duration = 2 s  track "i" { key { t = 0 s  only = ["isaac"] } } }')
    assert "two natives for isaac" in err('part "x" { native "isaac" { prim = "/A" }  native "isaac" { prim = "/B" } }')
    assert "replace is one of" in err('part "x" { native "isaac" { replace = "some" } }')
    assert "file not found" in err('part "x" { native "isaac" { usd = "nope.usda" } }')
    assert "needs the seam" in err('part "x" { native { usd = "x" } }')


def test_unknown_seam_name_warns(tmp_path):
    ir = build(project(tmp_path, {"main.orb": 'part "x" { only = ["isac"] }\nseam "emu" {}\npart "y" { only = ["emu"] }'}))
    assert any("'isac'" in w for w in ir["warnings"]) and not any("'emu'" in w for w in ir["warnings"])


def test_top_level_native_with_target(tmp_path):
    src = PROJ + '''
native "isaac" { target = item["a-01"]  usd = "assets/a.usda"  prim = "/A" }
native "minecraft" { target = site.s.fixture.lamp  entity = "minecraft:glow_squid" }
native "glb" { target = assembly.a.link.base.link.extra  glb = "assets/p.glb" }
'''
    ir = build(project(tmp_path, {"main.orb": src}))
    assert ir["items"]["a-01"]["native"] == {"isaac": {"usd": "assets/a.usda", "prim": "/A"}}
    assert ir["sites"]["s"]["fixtures"]["lamp"]["native"] == {"minecraft": {"entity": "minecraft:glow_squid"}}
    assert set(ir["assemblies"]["a"]["links"][1]["native"]) == {"mujoco", "glb"}
    with pytest.raises(OrbError, match="already has a native for isaac"):
        build(project(tmp_path, {"main.orb": PROJ + 'native "isaac" { target = assembly.a  usd = "assets/a.usda" }'}))
    with pytest.raises(OrbError, match="needs `target"):
        build(project(tmp_path, {"main.orb": PROJ + 'native "isaac" { usd = "assets/a.usda" }'}))
    with pytest.raises(OrbError, match="target is a"):
        build(project(tmp_path, {"main.orb": PROJ + 'native "isaac" { target = site.s.zone.z }'}))


def test_addendum_files(tmp_path):
    files = {"main.orb": PROJ,
             "trucks.isaac.orb": 'native "isaac" { target = item["p-01"]  usd = "assets/a.usda" }\n'
                                 'fixture_rig "lights" { intensity = 3 }\n'
                                 'item "lamp-01" { of = part.p  location = site.s }',
             "fun.minecraft.orb": 'item "horse-01" { of = part.p  location = site.s  '
                                  'native "minecraft" { entity = "minecraft:horse" } }',
             "notes.v2.orb": 'part "q" { mass = 1 kg  inertia = [0.01, 0.01, 0.01, 0, 0, 0] kg*m2 }'}
    ir = build(project(tmp_path, files))
    assert ir["items"]["p-01"]["native"] == {"isaac": {"usd": "assets/a.usda"}}
    assert ir["items"]["lamp-01"]["only"] == ["isaac"] and ir["other"]["fixture_rig"]["lights"]["only"] == ["isaac"]
    assert ir["items"]["horse-01"]["only"] == ["minecraft"]
    assert "only" not in ir["parts"]["q"]  # v2 is not a seam
    assert "horse-01" not in for_seam(ir, "isaac")["items"] and "lamp-01" not in for_seam(ir, "minecraft")["items"]


def test_addendum_errors(tmp_path):
    with pytest.raises(OrbError, match="already .drop `only`"):
        build(project(tmp_path, {"main.orb": PROJ, "x.isaac.orb": 'part "z" { only = ["glb"] }'}))
    with pytest.raises(OrbError, match="an addendum for isaac"):
        build(project(tmp_path, {"main.orb": PROJ,
                                 "y.isaac.orb": 'native "mujoco" { target = item["p-01"]  body = "x" }'}))


def test_declared_seam_addendum(tmp_path):
    ir = build(project(tmp_path, {"main.orb": PROJ + 'seam "emu" { command = "x" }',
                                  "site.emu.orb": 'item "e-01" { of = part.p }'}))
    assert ir["items"]["e-01"]["only"] == ["emu"]


# --------------------------------------------------------------------------- include


LIB_PART = '''part "{name}" {{
  mass = 1 kg  inertia = [0.01, 0.01, 0.01, 0, 0, 0] kg*m2
  visual "mesh" {{ file = "meshes/{name}.obj" }}
  native "isaac" {{ usd = "assets/{name}.usda" }}
}}
'''


def test_include_a_library(tmp_path):
    lib = tmp_path / "assets-lib"
    (lib / "mfg" / "meshes").mkdir(parents=True)
    (lib / "mfg" / "assets").mkdir()
    (lib / "mfg" / "cart.orb").write_text(LIB_PART.format(name="cart"))
    (lib / "mfg" / "meshes" / "cart.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    (lib / "mfg" / "assets" / "cart.usda").write_text("#usda 1.0\n")
    (lib / ".orb" / "sources" / "x").mkdir(parents=True)
    (lib / ".orb" / "sources" / "x" / "junk.orb").write_text("this is not orb")  # hidden: not read
    (lib / "build").mkdir()
    (lib / "build" / "junk.orb").write_text("nor this")
    (lib / "fun.minecraft.orb").write_text('part "hay" { mass = 1 kg  inertia = [0.01, 0.01, 0.01, 0, 0, 0] kg*m2 }')
    d = tmp_path / "proj"
    d.mkdir()
    (d / "main.orb").write_text('include "../assets-lib"\ninclude "../assets-lib"\n'
                                'item "c-01" { of = part.cart }\n')
    ir = build(d)
    assert ir["includes"] == [str(lib)]
    g = ir["parts"]["cart"]["geoms"][0]
    assert g["file"] == "../assets-lib/mfg/meshes/cart.obj"
    assert ir["parts"]["cart"]["native"]["isaac"]["usd"] == "../assets-lib/mfg/assets/cart.usda"
    assert ir["parts"]["hay"]["only"] == ["minecraft"]  # addenda work inside libraries too
    # one file, and a library that includes the project back (read once)
    (lib / "mfg" / "back.orb").write_text('include "../../proj/main.orb"\n')
    ir = build(d)
    assert set(ir["parts"]) == {"cart", "hay"}


def test_include_errors(tmp_path):
    d = tmp_path / "proj"
    d.mkdir()
    (d / "main.orb").write_text('include "nowhere"')
    with pytest.raises(OrbError, match="no such file or directory"):
        load(d)
    (tmp_path / "empty").mkdir()
    (d / "main.orb").write_text('include "../empty"')
    with pytest.raises(OrbError, match="no .orb files"):
        load(d)
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "a.orb").write_text('part "dup" { }')
    (d / "main.orb").write_text('include "../lib"\npart "dup" { }')
    with pytest.raises(OrbError) as e:
        load(d)
    assert "duplicate part 'dup'" in e.value.msg and "a.orb" in e.value.msg and "main.orb" in str(e.value)
    (d / "main.orb").write_text('part "x" { include = 1 }\ninclude x')
    with pytest.raises(OrbError):
        load(d)
