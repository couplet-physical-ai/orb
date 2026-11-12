"""glb seam playback of v0.3 timelines (contract sections 1 and 5): one glTF animation per timeline
sampled from Motion.bake (translation/rotation per moving item, visibility as STEP scale keys,
articulated joints as per-link channels), nominal poses that agree with the evaluator, the rest
pose at t = 0, seam scoping (`only`/`skip`) and `native "glb"` assets. Read back with pygltflib,
an independent reader."""
import copy
import json
import math
import struct
from pathlib import Path

import pytest

from orb.motion import Motion
from orb.seams import glb

pygltflib = pytest.importorskip("pygltflib")
pytest.importorskip("numpy")

ROOT = Path(__file__).parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "motion_ir.json"
FPS = 30.0


def fixture_ir():
    if not FIXTURE.exists():
        pytest.skip("tests/fixtures/motion_ir.json (core) not there yet")
    ir = json.loads(FIXTURE.read_text())
    ir["root"] = str((ROOT / ir["root"]).resolve()) if not Path(ir["root"]).is_absolute() else ir["root"]
    return ir


def emit(ir, out, cfg=None):
    files = glb.emit(ir, out, cfg or {})
    assert files == [out / "scene.glb"]
    return pygltflib.GLTF2().load(str(files[0]))


def floats(g, idx):
    """An accessor's values as Python floats, grouped by its type."""
    acc = g.accessors[idx]
    view = g.bufferViews[acc.bufferView]
    data = g.binary_blob()
    width = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}[acc.type]
    assert acc.componentType == pygltflib.FLOAT
    off = (view.byteOffset or 0) + (acc.byteOffset or 0)
    vals = struct.unpack_from(f"<{acc.count * width}f", data, off)
    return [vals[i:i + width] if width > 1 else vals[i] for i in range(0, len(vals), width)]


def channel_map(g, anim):
    """{(node name, path): (times, values, interpolation)}"""
    out = {}
    for ch in anim.channels:
        s = anim.samplers[ch.sampler]
        out[(g.nodes[ch.target.node].name, ch.target.path)] = (floats(g, s.input), floats(g, s.output), s.interpolation)
    return out


def sample(times, values, t, rotation=False):
    """glTF LINEAR sampling (slerp for rotations), holding the ends."""
    if t <= times[0]:
        return list(values[0])
    if t >= times[-1]:
        return list(values[-1])
    k = max(i for i, x in enumerate(times) if x <= t)
    t0, t1 = times[k], times[k + 1]
    u = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
    a, b = values[k], values[k + 1]
    if not rotation:
        return [x + (y - x) * u for x, y in zip(a, b)]
    dot = sum(x * y for x, y in zip(a, b))
    if dot < 0:
        b, dot = [-y for y in b], -dot
    if dot > 0.9995:
        q = [x + (y - x) * u for x, y in zip(a, b)]
    else:
        th = math.acos(min(1.0, dot))
        q = [(math.sin((1 - u) * th) * x + math.sin(u * th) * y) / math.sin(th) for x, y in zip(a, b)]
    n = math.sqrt(sum(x * x for x in q))
    return [x / n for x in q]


def quat_close(a_xyzw, b_wxyz, tol=2e-3):
    """Same rotation (q and -q are the same)."""
    b = [b_wxyz[1], b_wxyz[2], b_wxyz[3], b_wxyz[0]]
    return abs(abs(sum(x * y for x, y in zip(a_xyzw, b))) - 1.0) < tol


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    ir = fixture_ir()
    g = emit(copy.deepcopy(ir), tmp_path_factory.mktemp("glb") / "glb")
    return ir, g


def test_glb_seam_declares_playback():
    assert glb.MOTION == "playback"


def test_one_animation_per_timeline(tmp_path):
    ir = fixture_ir()
    second = copy.deepcopy(ir["timelines"]["demo"])
    second["duration"] = 30.0
    second["label"] = "second"
    ir["timelines"]["again"] = second
    g = emit(ir, tmp_path / "glb")
    assert [a.name for a in g.animations] == ["demo", "again"]
    for a in g.animations:
        meta = a.extras["orb"]
        assert meta["timeline"] == a.name and meta["poses"] == "nominal (no physics)"
        assert meta["fps"] == FPS
        ends = [max(floats(g, s.input)) for s in a.samplers]
        assert ends == pytest.approx([meta["duration"]] * len(ends))  # every clip lasts exactly its timeline
    assert g.asset.extras["orb"]["motion"]["rest"] == "demo"


def test_channel_targets(built):
    ir, g = built
    (anim,) = g.animations
    ch = channel_map(g, anim)
    # whole-body motion on the item nodes; joints on their links; visibility as scale
    assert ("truck-01", "translation") in ch
    assert ("carrier-01", "translation") in ch and ("carrier-01", "rotation") in ch
    assert ("crate-01", "translation") in ch  # carried by the truck, then the carrier, then released
    assert ("tag-01", "scale") in ch
    assert ("carrier-01/boom", "translation") in ch  # prismatic lift
    assert ("carrier-01/gripper", "translation") in ch  # rides the boom
    assert ("truck-01/door", "translation") in ch or ("truck-01/door", "rotation") in ch
    # nothing else animates: no site, fixture, geom or untouched link channels
    animated = {n for n, _ in ch}
    m = Motion(ir, "demo")
    assert animated - set(m.items()) <= {"carrier-01/boom", "carrier-01/gripper", "truck-01/door"}
    assert not any("/visual_" in n or "/collision_" in n for n in animated)
    # a static channel is never written (the base link of an articulated item stays put)
    assert ("carrier-01/base", "translation") not in ch and ("carrier-01/base", "rotation") not in ch


def test_key_counts_match_the_bake(built):
    ir, g = built
    (anim,) = g.animations
    ch = channel_map(g, anim)
    m = Motion(ir, "demo")
    bake = m.bake(FPS)
    for item in ("truck-01", "carrier-01", "crate-01"):
        times, values, interp = ch[(item, "translation")]
        samples = bake[item]
        expected = len(samples) + (samples[0][0] > 0) + (samples[-1][0] < m.duration)
        assert len(times) == len(values) == expected
        assert interp == "LINEAR"
        assert times[0] == 0.0 and times[-1] == pytest.approx(m.duration)
        assert all(b > a for a, b in zip(times, times[1:])), "glTF key times must strictly increase"
    for (node, path), (times, values, _) in ch.items():
        assert len(times) == len(values) >= 2, (node, path)
    # joints: one key per bake_joints sample (plus holds)
    jt = m.bake_joints(FPS)["carrier-01"]
    times, _, _ = ch[("carrier-01/boom", "translation")]
    assert len(times) == len(jt) + (jt[0][0] > 0) + (jt[-1][0] < m.duration)


def test_poses_agree_with_the_evaluator(built):
    ir, g = built
    (anim,) = g.animations
    ch = channel_map(g, anim)
    m = Motion(ir, "demo")
    for item in ("truck-01", "carrier-01", "crate-01"):
        for t in (0.0, 5.3, 11.0, 22.0, 23.7, 30.0, 45.5, 46.0, 52.25, 59.9):
            pos, quat = m.pose(item, t)
            got = sample(*ch[(item, "translation")][:2], t)
            assert max(abs(a - b) for a, b in zip(got, pos)) < 2e-3, (item, t, got, pos)
            if (item, "rotation") in ch:
                assert quat_close(sample(*ch[(item, "rotation")][:2], t, rotation=True), quat), (item, t)
    # the crate rides the carrier's gripper while attached (nominal poses: welded exactly)
    for t in (25.0, 35.0, 44.0):
        c, _ = m.pose("crate-01", t)
        got = sample(*ch[("crate-01", "translation")][:2], t)
        assert max(abs(a - b) for a, b in zip(got, c)) < 2e-3
    # released at 46 s: the nominal pose holds where it was let go
    a = sample(*ch[("crate-01", "translation")][:2], 47.0)
    b = sample(*ch[("crate-01", "translation")][:2], 59.0)
    assert max(abs(x - y) for x, y in zip(a, b)) < 1e-4


def test_visibility_is_step_scale(built):
    ir, g = built
    (anim,) = g.animations
    times, values, interp = channel_map(g, anim)[("tag-01", "scale")]
    assert interp == "STEP"
    assert times[:2] == [0.0, 46.0]
    assert [list(v) for v in values[:2]] == [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    # hidden at rest: a viewer that ignores animations still shows t = 0
    tag = next(n for n in g.nodes if n.name == "tag-01")
    assert tag.scale == [0.0, 0.0, 0.0]


def test_rest_pose_is_t0(built):
    ir, g = built
    m = Motion(ir, "demo")
    nodes = {n.name: n for n in g.nodes}
    for item in ("truck-01", "carrier-01", "crate-01"):
        pos, quat = m.pose(item, 0.0)
        got = nodes[item].translation or [0.0, 0.0, 0.0]
        assert max(abs(a - b) for a, b in zip(got, pos)) < 1e-6, item
        rot = nodes[item].rotation or [0.0, 0.0, 0.0, 1.0]
        assert quat_close(rot, quat, 1e-6), item
    # the lift starts at 0.18 m (its key at t = 0), not at the joint's own initial value
    boom = nodes["carrier-01/boom"]
    assert boom.translation[2] == pytest.approx(1.0 + 0.18, abs=1e-6)


def test_minecraft_only_item_is_absent(built):
    ir, g = built
    assert "horse-01" in ir["items"] and ir["items"]["horse-01"]["only"] == ["minecraft"]
    names = [n.name for n in g.nodes]
    assert not any(n == "horse-01" or n.startswith("horse-01/") for n in names)
    for a in g.animations:
        assert not any(g.nodes[c.target.node].name.startswith("horse-01") for c in a.channels)
    # natives for other seams are ignored: the truck keeps its universal geometry, no native node
    assert not any(n.endswith("/native_glb") for n in names)
    assert any(n.startswith("truck-01/") and g.nodes[i].mesh is not None for i, n in enumerate(names))


def tiny_glb(path: Path):
    """A one-triangle Y-up glTF (what an artist's .glb looks like), written with the seam's own writer."""
    np = pytest.importorskip("numpy")
    G = glb.Gltf()
    pos = G.accessor(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32), "VEC3", bounds=True)
    idx = G.accessor(np.array([0, 1, 2]), "SCALAR", target=glb.ELEMENTS)
    G.add("materials", {"name": "native-red", "pbrMetallicRoughness": {"baseColorFactor": [1, 0, 0, 1]}})
    G.add("meshes", {"name": "tri", "primitives": [{"attributes": {"POSITION": pos}, "indices": idx, "material": 0}]})
    G.add("nodes", {"name": "tri", "mesh": 0})
    path.parent.mkdir(parents=True, exist_ok=True)
    G.write(path)


def test_native_glb_replaces_the_visuals(tmp_path):
    ir = fixture_ir()
    root = tmp_path / "proj"
    tiny_glb(root / "assets" / "glb" / "crate.glb")
    ir["root"] = str(root)
    ir["parts"]["crate"]["native"] = {"glb": {"glb": "assets/glb/crate.glb", "at": [0, 0, 0.2], "scale": 0.5,
                                              "replace": "visual"},
                                      "isaac": {"usd": "assets/isaac/crate.usda"}}
    g = emit(ir, tmp_path / "glb")
    names = [n.name for n in g.nodes]
    nodes = {n.name: n for n in g.nodes}
    wrapper = nodes["crate-01/native_glb"]
    assert names.index("crate-01/native_glb") in nodes["crate-01"].children  # moves with the crate
    assert wrapper.translation == [0.0, 0.0, 0.2] and wrapper.scale == [0.5, 0.5, 0.5]
    # turned from glTF's Y up to the Orb's Z up (+90 deg about X)
    assert wrapper.rotation == pytest.approx([math.sqrt(0.5), 0, 0, math.sqrt(0.5)])
    (tri,) = [nodes[names[c]] for c in wrapper.children]
    assert tri.name == "tri" and g.meshes[tri.mesh].name == "tri"
    assert g.materials[g.meshes[tri.mesh].primitives[0].material].name == "native-red"
    # the universal visuals are not drawn (replace = "visual"); the collision box stays as hidden metadata
    crate_geoms = [n for n in names if n.startswith("crate-01/") and n != "crate-01/native_glb"]
    assert crate_geoms and all(nodes[n].mesh is None for n in crate_geoms)
    # still animated
    assert any(g.nodes[c.target.node].name == "crate-01" for c in g.animations[0].channels)


def test_native_glb_missing_file_is_a_clear_error(tmp_path):
    ir = fixture_ir()
    ir["root"] = str(tmp_path)
    ir["items"]["crate-01"]["native"] = {"glb": {"glb": "assets/missing.glb"}}
    with pytest.raises(Exception, match="missing.glb"):
        glb.emit(ir, tmp_path / "glb", {})


def test_no_timeline_no_animations(tmp_path):
    ir = fixture_ir()
    ir["timelines"] = {}
    ir["shots"] = []
    g = emit(ir, tmp_path / "glb")
    assert not g.animations
    assert "motion" not in g.asset.extras["orb"]
    # crate-01 rests at its own placement (no timeline poses it)
    crate = next(n for n in g.nodes if n.name == "crate-01")
    assert (crate.translation or [0.0, 0.0, 0.0]) == pytest.approx(ir["items"]["crate-01"]["pos"])


def test_seam_timeline_picks_the_rest_pose(tmp_path):
    ir = fixture_ir()
    late = copy.deepcopy(ir["timelines"]["demo"])
    for keys in late["tracks"].values():
        for k in keys:
            k["pos"] = [k["pos"][0], k["pos"][1] + 100.0, k["pos"][2]]
    ir["timelines"] = {"demo": ir["timelines"]["demo"], "shifted": late}
    g = emit(ir, tmp_path / "glb", {"timeline": "shifted"})
    truck = next(n for n in g.nodes if n.name == "truck-01")
    pos, _ = Motion(ir, "shifted").pose("truck-01", 0.0)
    assert truck.translation[1] == pytest.approx(pos[1]) and pos[1] > 50
    assert g.asset.extras["orb"]["motion"]["rest"] == "shifted"
    assert [a.name for a in g.animations] == ["demo", "shifted"]
