"""glTF seam: the whole compiled scene as one binary glTF (build/glb/scene.glb).

For three.js, immersive visualization tools, Blender, Omniverse, Reality Composer, any glTF viewer. The
scene is Y up (glTF's convention): everything hangs under one root node `world` that turns
the Orb's Z-up frame onto Y-up, so every other node keeps its Orb pose. Node names are the
Orb ids (site, zone, fixture, item, `item/link`), and every node carries its provenance in
`extras.orb` (fixture attributes such as `source`, item stage/location, joint state);
`asset.extras.orb` has the capture blocks, the OrbLang sources and the world graph, which is
what `orb import` reads to regenerate the project exactly.

Scan meshes keep their textures (embedded PNG/JPEG) or vertex colours (COLOR_0), robot
meshes their flat colours. Assemblies are posed at their initial joint values. Collision
geoms hidden by the simulator seams (capture boxes in a scanned room, link colliders behind
visuals) become mesh-less nodes whose `extras.orb.geom` still describes the shape.

Motion (contract v0.3, section 4): every timeline becomes one glTF animation, named after it,
sampled from `orb.motion.Motion.bake` at the seam's `fps` (default 30): translation and/or
rotation channels per moving item node (only what changes), visibility as STEP scale keys (0 hidden, 1 shown), and
articulated joints as per-link node channels from `bake_joints`. Poses are nominal (no
physics): a released payload holds where it was let go. Each animation's `extras.orb` carries
the timeline's duration, loop flag, phases and events. Nodes rest at t = 0 of the seam's
timeline (`seam "glb" { timeline = "..." }`, default the first), so a viewer that ignores
animations still shows the start of the scene.

Scoping: the seam sees the IR through `orb.world.for_seam(ir, "glb")` (`only`/`skip`), and a
`native "glb" { glb = "x.glb" at rpy scale replace }` on a site, fixture, item, part,
assembly or link embeds that glTF under its owner's node (turned from glTF's Y up to the
Orb's Z up, then placed by at/rpy/scale). `replace = "visual"` (default) or `"all"` leaves the
universal geometry out of the picture (mesh-less nodes keep its description); `"none"`
overlays it.

Needs numpy (the `export` extra). Seam settings: `show_capture_boxes = true`,
`embed_sources = false` (leave the OrbLang sources and world graph out of the file),
`fps = 30` (animation sampling), `timeline = "<name>"` (the rest pose).
"""
from __future__ import annotations

import base64
import json
import math
import struct
from pathlib import Path

from ..lang import OrbError
from ..world import frames, qmul, rpy_quat
from . import scene
from .meshio import load_obj

PROCESS_FIDELITY = ()
MOTION = "playback"  # timelines play back as glTF animations: nominal poses, no physics
Y_UP = [-math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]  # glTF [x, y, z, w]: -90 deg about X, Z-up -> Y-up
Z_UP = [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]  # the inverse: a Y-up glTF (a native asset) -> Z up
DEFAULT_RGBA = [0.72, 0.72, 0.72, 1.0]
FLOAT, UINT32 = 5126, 5125
ARRAY, ELEMENTS = 34962, 34963
DEFAULT_FPS = 30.0


def _np():
    try:
        import numpy
    except ImportError as e:  # pragma: no cover
        raise OrbError("the glb seam needs NumPy and Pillow: uv sync --extra glb") from e
    return numpy


def srgb_to_linear(c):
    return [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c]


class Gltf:
    def __init__(self):
        self.doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"name": "orb", "nodes": [0]}], "nodes": [],
                    "meshes": [], "accessors": [], "bufferViews": [], "buffers": [], "materials": []}
        self.bin = bytearray()

    def view(self, data: bytes, target=None):
        while len(self.bin) % 4:
            self.bin.append(0)
        bv = {"buffer": 0, "byteOffset": len(self.bin), "byteLength": len(data)}
        if target:
            bv["target"] = target
        self.bin += data
        self.doc["bufferViews"].append(bv)
        return len(self.doc["bufferViews"]) - 1

    def accessor(self, arr, kind, target=ARRAY, bounds=False, floats=False):
        """SCALAR is uint32 (indices) unless `floats` (animation times); everything else float32."""
        np = _np()
        if kind == "SCALAR" and not floats:
            arr = np.ascontiguousarray(arr, dtype=np.uint32).reshape(-1)
            comp, count = UINT32, len(arr)
        else:
            arr = np.ascontiguousarray(arr, dtype=np.float32)
            if kind == "SCALAR":
                arr = arr.reshape(-1)
            comp, count = FLOAT, len(arr)
        acc = {"bufferView": self.view(arr.tobytes(), target), "componentType": comp, "count": count, "type": kind}
        if bounds and count:
            acc["min"] = [float(x) for x in np.atleast_1d(arr.min(axis=0))]
            acc["max"] = [float(x) for x in np.atleast_1d(arr.max(axis=0))]
        self.doc["accessors"].append(acc)
        return len(self.doc["accessors"]) - 1

    def add(self, key, item):
        self.doc.setdefault(key, []).append(item)
        return len(self.doc[key]) - 1

    def write(self, path: Path):
        if self.bin:
            self.doc["buffers"] = [{"byteLength": len(self.bin)}]
        else:
            self.doc.pop("buffers")
        for k in [k for k, v in self.doc.items() if isinstance(v, list) and not v]:
            del self.doc[k]
        js = json.dumps(self.doc, separators=(",", ":"), ensure_ascii=False).encode()
        js += b" " * (-len(js) % 4)
        bn = bytes(self.bin) + b"\0" * (-len(self.bin) % 4)
        total = 12 + 8 + len(js) + (8 + len(bn) if bn else 0)
        with open(path, "wb") as f:
            f.write(struct.pack("<4sII", b"glTF", 2, total))
            f.write(struct.pack("<I4s", len(js), b"JSON") + js)
            if bn:
                f.write(struct.pack("<I4s", len(bn), b"BIN\0") + bn)


def normals(np, pos, tris):
    """Smooth per-vertex normals over welded positions (area weighted)."""
    key, inv = np.unique(np.round(pos, 5), axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    a, b, c = (pos[tris[:, i]] for i in range(3))
    fn = np.cross(b - a, c - a)
    acc = np.zeros((len(key), 3))
    for i in range(3):
        np.add.at(acc, inv[tris[:, i]], fn)
    n = acc[inv]
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.where(ln > 1e-20, n / np.where(ln > 1e-20, ln, 1), [0.0, 0.0, 1.0])
    return n


def primitive_mesh(np, g):
    """Box/sphere/cylinder/capsule -> (positions, normals, triangles) at the geom's true size."""
    shape = g["shape"]
    if shape == "box":
        sx, sy, sz = (s / 2 for s in g["size"])
        P, N, T = [], [], []
        for axis in range(3):
            for sign in (-1, 1):
                u, v = [(1, 2), (2, 0), (0, 1)][axis]
                base = len(P)
                for du, dv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                    p = [0.0, 0.0, 0.0]
                    p[axis], p[u], p[v] = sign, du, dv
                    P.append([p[0] * sx, p[1] * sy, p[2] * sz])
                    n = [0.0, 0.0, 0.0]
                    n[axis] = sign
                    N.append(n)
                T += [[base, base + 1, base + 2], [base, base + 2, base + 3]] if sign > 0 else \
                     [[base, base + 2, base + 1], [base, base + 3, base + 2]]
        return np.array(P), np.array(N), np.array(T)
    seg, rings = 32, 16
    if shape == "sphere":
        r = g["radius"]
        P, T = [], []
        for i in range(rings + 1):
            th = math.pi * i / rings
            for j in range(seg + 1):
                ph = 2 * math.pi * j / seg
                P.append([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])
        for i in range(rings):
            for j in range(seg):
                a, b = i * (seg + 1) + j, (i + 1) * (seg + 1) + j
                T += [[a, b, a + 1], [a + 1, b, b + 1]]
        P = np.array(P)
        return P * r, P, np.array(T)
    r, h = g["radius"], g["length"] / 2
    P, N, T = [], [], []
    if shape == "capsule":  # hemispheres + side, as one smooth surface
        prof = [(math.sin(math.pi / 2 * k / 8), h + r * math.cos(math.pi / 2 * k / 8), math.cos(math.pi / 2 * k / 8))
                for k in range(9)]
        prof += [(math.cos(math.pi / 2 * k / 8), -h - r * math.sin(math.pi / 2 * k / 8), -math.sin(math.pi / 2 * k / 8))
                 for k in range(9)]
        for rad, z, nz in prof:
            nr = math.sqrt(max(0.0, 1 - nz * nz))
            for j in range(seg + 1):
                ph = 2 * math.pi * j / seg
                P.append([rad * r * math.cos(ph), rad * r * math.sin(ph), z])
                N.append([nr * math.cos(ph), nr * math.sin(ph), nz])
        for i in range(len(prof) - 1):
            for j in range(seg):
                a, b = i * (seg + 1) + j, (i + 1) * (seg + 1) + j
                T += [[a, b, a + 1], [a + 1, b, b + 1]]
        return np.array(P), np.array(N), np.array(T)
    for z in (h, -h):  # cylinder side
        for j in range(seg + 1):
            ph = 2 * math.pi * j / seg
            P.append([r * math.cos(ph), r * math.sin(ph), z])
            N.append([math.cos(ph), math.sin(ph), 0.0])
    for j in range(seg):
        T += [[j, seg + 1 + j, j + 1], [j + 1, seg + 1 + j, seg + 2 + j]]
    for z, s in ((h, 1), (-h, -1)):  # caps
        c = len(P)
        P.append([0.0, 0.0, z])
        N.append([0.0, 0.0, s])
        for j in range(seg + 1):
            ph = 2 * math.pi * j / seg
            P.append([r * math.cos(ph), r * math.sin(ph), z])
            N.append([0.0, 0.0, s])
        for j in range(seg):
            T.append([c, c + 1 + j, c + 2 + j] if s > 0 else [c, c + 2 + j, c + 1 + j])
    return np.array(P), np.array(N), np.array(T)


def emit(ir, out: Path, cfg):
    np = _np()
    world = ir  # the unscoped IR: the world graph `orb import` rebuilds the project from
    ir = scene.scoped(ir, "glb")  # orb.world.for_seam(ir, "glb"), once
    root = Path(ir["root"])
    out.mkdir(parents=True, exist_ok=True)
    G = Gltf()
    mats, images, meshes, geo = {}, {}, {}, {}
    rest = rest_motion(ir, cfg)

    def material(rgba, name=None, double=False):
        key = ("c", tuple(round(c, 4) for c in rgba), double)
        if key not in mats:
            m = {"name": name or "c_" + "_".join(str(round(c * 255)) for c in rgba),
                 "pbrMetallicRoughness": {"baseColorFactor": srgb_to_linear(rgba[:3]) + [rgba[3]],
                                          "metallicFactor": 0.0, "roughnessFactor": 0.7}}
            if rgba[3] < 1:
                m["alphaMode"] = "BLEND"
            if double:
                m["doubleSided"] = True
            mats[key] = G.add("materials", m)
        return mats[key]

    def texture_material(path: Path):
        key = ("t", str(path))
        if key not in mats:
            if path not in images:
                data, mime = path.read_bytes(), "image/png"
                if path.suffix.lower() in (".jpg", ".jpeg"):
                    mime = "image/jpeg"
                elif path.suffix.lower() != ".png":
                    import io

                    from PIL import Image
                    buf = io.BytesIO()
                    Image.open(path).convert("RGB").save(buf, "PNG")
                    data = buf.getvalue()
                img = G.add("images", {"name": path.name, "bufferView": G.view(data), "mimeType": mime})
                if "sampler" not in images:
                    images["sampler"] = G.add("samplers", {"magFilter": 9729, "minFilter": 9987, "wrapS": 10497,
                                                           "wrapT": 10497})
                images[path] = G.add("textures", {"sampler": images["sampler"], "source": img})
            mats[key] = G.add("materials", {"name": "tex_" + path.stem, "doubleSided": True,
                                            "pbrMetallicRoughness": {"baseColorTexture": {"index": images[path]},
                                                                     "metallicFactor": 0.0, "roughnessFactor": 0.9}})
        return mats[key]

    def obj_geometry(path: Path):
        """OBJ -> (attributes, [(indices accessor, material slot, object name)], kind): one primitive
        per object/material group, so the `o` groups survive (`extras.orb.object`)."""
        if path in geo:
            return geo[path]
        obj = load_obj(path)
        pos = np.asarray(obj.pos, dtype=np.float64).reshape(-1, 3)
        tv = np.asarray(obj.tri_v, dtype=np.int64).reshape(-1, 3)
        kind = obj.kind
        attrs, prims = {}, []
        if kind == "textured":  # unweld so each (position, uv) corner pair is one glTF vertex
            tt = np.asarray(obj.tri_t, dtype=np.int64).reshape(-1, 3)
            uv = np.vstack([np.asarray(obj.uv, dtype=np.float64).reshape(-1, 2), [[0.0, 0.0]]])
            tt = np.where(tt < 0, len(uv) - 1, tt)
            pairs = np.stack([tv.reshape(-1), tt.reshape(-1)], axis=1)
            uniq, inv = np.unique(pairs, axis=0, return_inverse=True)
            tris = inv.reshape(-1, 3)
            P = pos[uniq[:, 0]]
            T = uv[uniq[:, 1]].copy()
            T[:, 1] = 1.0 - T[:, 1]  # OBJ v points up, glTF v points down
            attrs["TEXCOORD_0"] = G.accessor(T, "VEC2")
        else:
            tris, P = tv, pos
            if kind == "vcolor":
                col = np.asarray(obj.col, dtype=np.float64)
                lin = np.where(col <= 0.04045, col / 12.92, ((col + 0.055) / 1.055) ** 2.4)
                attrs["COLOR_0"] = G.accessor(lin, "VEC3")
        attrs["POSITION"] = G.accessor(P, "VEC3", bounds=True)
        attrs["NORMAL"] = G.accessor(normals(np, P, tris), "VEC3")
        named = any(name is not None for name in obj.objects)
        if named:  # the OBJ's own groups, in file order
            for part in obj.parts():
                sel = tris[np.asarray(part["tris"], dtype=np.int64)]
                prims.append((G.accessor(sel, "SCALAR", ELEMENTS), part["slot"], part["object"]))
        else:
            slots = np.asarray(obj.tri_m) if obj.tri_m is not None else np.zeros(len(tris), dtype=np.int64)
            for m in sorted(set(slots.tolist())):
                sel = tris[slots == m]
                prims.append((G.accessor(sel, "SCALAR", ELEMENTS), m, None))
        geo[path] = (attrs, prims, kind, obj)
        return geo[path]

    def mesh_for(g, name):
        """The glTF mesh index for a geom; None for an OBJ without a face (a glTF mesh needs a
        primitive and an accessor at least one element long)."""
        if g["shape"] == "mesh":
            src = (root / g["file"]).resolve()
            key = ("mesh", str(src), tuple(g.get("color") or ()))
            if key not in meshes and src.suffix.lower() == ".obj" and load_obj(src).ntris == 0:
                meshes[key] = None
            if key not in meshes:
                attrs, prims, kind, obj = obj_geometry(src)
                ps = []
                for acc, slot, oname in prims:
                    tex = obj.texture(slot) if kind == "textured" else None
                    if tex is not None:
                        mi = texture_material(tex)
                    elif kind == "vcolor":
                        mi = material([1.0, 1.0, 1.0, 1.0], "vertex_color", double=True)
                    elif g.get("color"):
                        mi = material(g["color"])
                    elif obj.kd(slot):
                        mi = material(obj.kd(slot) + [1.0])
                    else:
                        mi = material(DEFAULT_RGBA)
                    ps.append({"attributes": dict(attrs), "indices": acc, "material": mi,
                               **({"extras": {"orb": {"object": oname}}} if oname is not None else {})})
                meshes[key] = G.add("meshes", {"name": g["file"], "primitives": ps,
                                               "extras": {"orb": {"file": g["file"], "kind": kind}}})
            return meshes[key]
        dims = {k: g[k] for k in ("size", "radius", "length") if k in g}
        key = ("prim", g["shape"], json.dumps(dims), tuple(g.get("color") or DEFAULT_RGBA))
        if key not in meshes:
            P, N, T = primitive_mesh(np, g)
            prim = {"attributes": {"POSITION": G.accessor(P, "VEC3", bounds=True), "NORMAL": G.accessor(N, "VEC3")},
                    "indices": G.accessor(T, "SCALAR", ELEMENTS), "material": material(g.get("color") or DEFAULT_RGBA)}
            meshes[key] = G.add("meshes", {"name": f"{g['shape']}", "primitives": [prim]})
        return meshes[key]

    def gnode(n, k, g):
        role = g["role"]
        geom = {k2: v for k2, v in g.items() if k2 != "show"}  # the Orb geom as written (provenance)
        if n["type"] == "fixture" and g["shape"] == "box":  # doors/windows drawn proud of their wall
            g = {**g, "size": scene.opening_box_size(g, n["orb"].get("kind"))}
        nd = {"name": f"{n['name']}/{role}_{k}", "extras": {"orb": {"geom": geom, "visible": g["show"]}}}
        if any(abs(x) > 0 for x in g["pos"]):
            nd["translation"] = [float(x) for x in g["pos"]]
        if g["quat"] != scene.IDENTITY:
            w, x, y, z = g["quat"]
            nd["rotation"] = [x, y, z, w]
        if g["shape"] == "mesh" and g.get("scale", [1, 1, 1]) != [1, 1, 1]:
            nd["scale"] = list(g["scale"])
        if g["show"]:
            mi = mesh_for(g, nd["name"])
            if mi is not None:
                nd["mesh"] = mi
        return G.add("nodes", nd)

    item_nodes, link_nodes = {}, {}  # item -> node, (item, link) -> node: what animations target

    def add(n):
        idx = G.add("nodes", {"name": n["name"]})
        nd = G.doc["nodes"][idx]
        if any(abs(x) > 0 for x in n["pos"]):
            nd["translation"] = [float(x) for x in n["pos"]]
        if n["quat"] != scene.IDENTITY:
            w, x, y, z = n["quat"]
            nd["rotation"] = [x, y, z, w]
        nd["extras"] = {"orb": {"type": n["type"], **n["orb"]}}
        if n["type"] == "item":
            item_nodes[n["orb"]["item"]] = idx
            if rest is not None and n["orb"]["item"] in rest.items() and not rest.visible(n["orb"]["item"], 0.0):
                nd["scale"] = [0.0, 0.0, 0.0]  # hidden at the start: the animation's scale keys show it
        elif n["type"] == "link":
            link_nodes[(n["orb"]["item"], n["orb"]["link"])] = idx
        kids = [gnode(n, k, g) for k, g in enumerate(n["geoms"])] + [add(c) for c in n["children"]]
        if native_file(n.get("native")):  # scene.build already left out the universal geoms it replaces
            kids.append(embed_native(G, root, n["native"], n["name"]))
        if kids:
            nd["children"] = kids
        return idx

    G.add("nodes", {"name": "world", "rotation": Y_UP,
                    "extras": {"orb": {"frame": "children are in the Orb frame: Z up, metres; this node turns it Y up"}}})
    tree = scene.build(posed(ir, rest), cfg, "glb")
    G.doc["nodes"][0]["children"] = [add(n) for n in tree]
    fps = float(cfg.get("fps", DEFAULT_FPS))
    if rest is not None:
        from ..motion import Motion

        for name in ir["timelines"]:
            m = rest if name == rest.name else Motion(ir, name)
            anim = animation(G, ir, m, fps, item_nodes, link_nodes)
            if anim["channels"]:  # a glTF animation needs a channel; a timeline nothing moves in has none
                G.add("animations", anim)
    meta = scene.provenance(world, "glb", with_sources=cfg.get("embed_sources", True))
    if rest is not None:
        meta["motion"] = {"playback": "one glTF animation per timeline, nominal poses (no physics)",
                          "rest": rest.name, "fps": fps}
    G.doc["asset"].update(generator=meta["generator"], extras={"orb": meta})
    G.write(out / "scene.glb")
    return [out / "scene.glb"]


# --------------------------------------------------------------------------- motion


def rest_motion(ir, cfg):
    """The Motion whose t = 0 the nodes rest at: the seam's `timeline`, else the first; None without timelines."""
    if not ir.get("timelines"):
        return None
    from ..motion import Motion, timeline_name

    return Motion(ir, cfg.get("timeline") or timeline_name(ir, "glb"))


def posed(ir, m):
    """The IR with every item the timeline touches at its t = 0 pose (and joints), placed even when it
    has no `location` of its own (a payload that only rides on a carrier still needs a node)."""
    if m is None:
        return ir
    items = dict(ir["items"])
    for name in m.items():
        it = items.get(name)
        if it is None:
            continue
        pos, quat = m.pose(name, 0.0)
        it = {**it, "pos": [float(x) for x in pos], "quat": [float(x) for x in quat]}
        if it.get("location") is None:
            it["location"] = f"timeline.{m.name}"
        if it.get("kind") == "assembly":
            js = m.joints(name, 0.0)
            if js:
                it["joints"] = {k: float(v) for k, v in js.items()}
        items[name] = it
    return {**ir, "items": items}


def _continuous(quats):
    """[w, x, y, z] samples -> glTF [x, y, z, w], each in the hemisphere of the one before (no long way round)."""
    out, prev = [], None
    for w, x, y, z in quats:
        q = [float(x), float(y), float(z), float(w)]
        if prev is not None and sum(a * b for a, b in zip(q, prev)) < 0:
            q = [-c for c in q]
        out.append(q)
        prev = q
    return out


def _static(values, tol=1e-6):
    first = values[0]
    return all(max(abs(a - b) for a, b in zip(v, first)) <= tol for v in values)


def animation(G, ir, m, fps, item_nodes, link_nodes):
    """One timeline -> one glTF animation (dict), sampled from m.bake / m.bake_joints."""
    np = _np()
    duration = float(m.duration)
    channels, samplers = [], []

    def track(node, path, times, values, interp="LINEAR"):
        keep = [i for i, t in enumerate(times) if t <= duration + 1e-9] or [0]  # the clip is [0, duration]
        times, values = [min(float(times[i]), duration) for i in keep], [list(values[i]) for i in keep]
        if times[0] > 0:  # hold the first key from t = 0 and the last to the end: the clip is the timeline
            times, values = [0.0, *times], [values[0], *values]
        if times[-1] < duration:
            times, values = [*times, duration], [*values, values[-1]]
        t = np.asarray(times, dtype=np.float32)
        for i in range(1, len(t)):  # glTF wants strictly increasing times; a jump keeps both keys, 1 ulp apart
            if t[i] <= t[i - 1]:
                t[i] = np.nextafter(t[i - 1], np.float32(np.inf))
        kind = {"translation": "VEC3", "scale": "VEC3", "rotation": "VEC4"}[path]
        samplers.append({"input": G.accessor(t, "SCALAR", target=None, bounds=True, floats=True),
                         "output": G.accessor(np.asarray(values, dtype=np.float32), kind, target=None),
                         "interpolation": interp})
        channels.append({"sampler": len(samplers) - 1, "target": {"node": node, "path": path}})

    for name, samples in m.bake(fps).items():  # whole-body poses (world frame; items hang under `world`)
        node = item_nodes.get(name)
        if node is None or not samples:
            continue
        times = [s[0] for s in samples]
        pos = [[float(x) for x in s[1]] for s in samples]
        quat = _continuous([s[2] for s in samples])
        if len(samples) > 1 and not _static(pos):
            track(node, "translation", times, pos)
        if len(samples) > 1 and not _static(quat):
            track(node, "rotation", times, quat)

    tl = ir["timelines"][m.name]
    for name in tl.get("visible") or {}:  # visibility: scale 0 / 1, STEP keys from t = 0
        node = item_nodes.get(name)
        if node is None:
            continue
        keys = []
        for t, v in m.visibility(name):
            v = 1.0 if v else 0.0
            if not keys or keys[-1][1] != v:
                keys.append((float(t), v))
        if len(keys) > 1 or keys[0][1] == 0.0:
            track(node, "scale", [k[0] for k in keys], [[k[1]] * 3 for k in keys], "STEP")

    for name, samples in m.bake_joints(fps).items():  # articulated items: each moving link's own channels
        it = ir["items"].get(name) or {}
        asm = ir["assemblies"].get(it.get("of")) if it.get("kind") == "assembly" else None
        if asm is None or not samples:
            continue
        times = [s[0] for s in samples]
        fk = [frames(asm, {**(it.get("joints") or {}), **values}) for _, values in samples]
        for link in asm["links"]:
            node = link_nodes.get((name, link["name"]))
            if node is None:
                continue
            pos = [[float(x) for x in f[link["name"]][0]] for f in fk]
            quat = _continuous([f[link["name"]][1] for f in fk])
            if len(fk) > 1 and not _static(pos):  # a revolute joint turns its link, a prismatic one slides it
                track(node, "translation", times, pos)
            if len(fk) > 1 and not _static(quat):
                track(node, "rotation", times, quat)

    return {"name": m.name, "channels": channels, "samplers": samplers,
            "extras": {"orb": {"timeline": m.name, "label": m.label, "duration": duration, "loop": m.loop,
                               "physics": m.physics, "fps": fps, "poses": "nominal (no physics)",
                               "phases": m.phases, "events": m.events}}}


# --------------------------------------------------------------------------- natives


def native_file(entry):
    """The glTF a `native "glb"` entry names (its `glb` key), or None."""
    return entry.get("glb") if isinstance(entry, dict) and isinstance(entry.get("glb"), str) else None


def read_gltf(path: Path):
    """(json doc, [buffer bytes]) of a .glb or .gltf (buffers from the BIN chunk, files or data: URIs)."""
    raw = path.read_bytes()
    bin_chunk = None
    if raw[:4] == b"glTF":
        off, doc = 12, None
        while off + 8 <= len(raw):
            n, kind = struct.unpack_from("<I4s", raw, off)
            data = raw[off + 8:off + 8 + n]
            if kind == b"JSON":
                doc = json.loads(data.decode("utf-8"))
            elif kind == b"BIN\0":
                bin_chunk = bytes(data)
            off += 8 + n
        if doc is None:
            raise OrbError(f"native glb {path}: no JSON chunk")
    else:
        doc = json.loads(raw.decode("utf-8-sig"))

    def load(uri):
        if uri.startswith("data:"):
            return base64.b64decode(uri.split(",", 1)[1])
        from urllib.parse import unquote
        return (path.parent / unquote(uri)).read_bytes()

    bufs = [load(b["uri"]) if b.get("uri") else (bin_chunk or b"") for b in doc.get("buffers", [])]
    return doc, bufs


def _remap_textures(v, tex):
    """Every textureInfo ({"index": i, ...} under a *Texture key) inside a material, renumbered."""
    if isinstance(v, dict):
        return {k: ({**x, "index": tex[x["index"]], **({"extensions": _remap_textures(x["extensions"], tex)}
                                                       if "extensions" in x else {})}
                    if k.endswith("Texture") and isinstance(x, dict) and isinstance(x.get("index"), int)
                    else _remap_textures(x, tex)) for k, x in v.items()}
    if isinstance(v, list):
        return [_remap_textures(x, tex) for x in v]
    return v


def embed_native(G, root: Path, entry, owner):
    """Copy a native glTF (its default scene) into G under one node `<owner>/native_glb`; returns that node.
    The asset is Y up like every glTF: the node turns it Z up, then applies at / rpy / scale."""
    path = (root / native_file(entry)).resolve()
    if not path.is_file():
        raise OrbError(f"native \"glb\" of {owner}: {native_file(entry)} not found ({path})")
    doc, bufs = read_gltf(path)
    views = []
    for bv in doc.get("bufferViews", []):
        data = bufs[bv.get("buffer", 0)][bv.get("byteOffset", 0):bv.get("byteOffset", 0) + bv["byteLength"]]
        i = G.view(bytes(data), bv.get("target"))
        if "byteStride" in bv:
            G.doc["bufferViews"][i]["byteStride"] = bv["byteStride"]
        views.append(i)
    acc = []
    for a in doc.get("accessors", []):
        a = json.loads(json.dumps(a))
        if "bufferView" in a:
            a["bufferView"] = views[a["bufferView"]]
        for part in ("indices", "values"):
            if part in a.get("sparse", {}):
                a["sparse"][part]["bufferView"] = views[a["sparse"][part]["bufferView"]]
        acc.append(G.add("accessors", a))
    imgs = []
    for im in doc.get("images", []):
        im = dict(im)
        if "bufferView" in im:
            im["bufferView"] = views[im["bufferView"]]
        elif im.get("uri"):  # an external or data: image goes into the binary chunk too
            uri = im.pop("uri")
            if uri.startswith("data:"):
                im.setdefault("mimeType", uri[5:].split(";", 1)[0])
                data = base64.b64decode(uri.split(",", 1)[1])
            else:
                from urllib.parse import unquote
                data = (path.parent / unquote(uri)).read_bytes()
                im.setdefault("mimeType", "image/jpeg" if uri.lower().endswith((".jpg", ".jpeg")) else "image/png")
            im["bufferView"] = G.view(data)
        imgs.append(G.add("images", im))
    smp = [G.add("samplers", dict(s)) for s in doc.get("samplers", [])]
    tex = []
    for t in doc.get("textures", []):
        t = json.loads(json.dumps(t))
        if "source" in t:
            t["source"] = imgs[t["source"]]
        if "sampler" in t:
            t["sampler"] = smp[t["sampler"]]
        for ext in (t.get("extensions") or {}).values():  # EXT_texture_webp, KHR_texture_basisu, ...
            if isinstance(ext, dict) and isinstance(ext.get("source"), int):
                ext["source"] = imgs[ext["source"]]
        tex.append(G.add("textures", t))
    mat = [G.add("materials", _remap_textures(m, tex)) for m in doc.get("materials", [])]
    msh = []
    for me in doc.get("meshes", []):
        me = json.loads(json.dumps(me))
        for p in me.get("primitives", []):
            p["attributes"] = {k: acc[v] for k, v in p["attributes"].items()}
            if "indices" in p:
                p["indices"] = acc[p["indices"]]
            if "material" in p:
                p["material"] = mat[p["material"]]
            if "targets" in p:
                p["targets"] = [{k: acc[v] for k, v in tg.items()} for tg in p["targets"]]
        msh.append(G.add("meshes", me))
    base = len(G.doc["nodes"])
    skins_base = len(G.doc.get("skins", []))
    for nd in doc.get("nodes", []):
        nd = {k: v for k, v in nd.items() if k != "camera"}  # the asset's own cameras stay out
        if "children" in nd:
            nd["children"] = [base + c for c in nd["children"]]
        if "mesh" in nd:
            nd["mesh"] = msh[nd["mesh"]]
        if "skin" in nd:
            nd["skin"] += skins_base
        G.add("nodes", nd)
    for sk in doc.get("skins", []):
        sk = dict(sk, joints=[base + j for j in sk["joints"]])
        if "skeleton" in sk:
            sk["skeleton"] += base
        if "inverseBindMatrices" in sk:
            sk["inverseBindMatrices"] = acc[sk["inverseBindMatrices"]]
        G.add("skins", sk)
    for key in ("extensionsUsed", "extensionsRequired"):
        for e in doc.get(key, []):
            if e not in G.doc.setdefault(key, []):
                G.doc[key].append(e)
    scenes = doc.get("scenes") or [{"nodes": [i for i in range(len(doc.get("nodes", [])))
                                              if not any(i in (n.get("children") or []) for n in doc["nodes"])]}]
    roots = scenes[doc.get("scene", 0)].get("nodes", [])
    pos = entry.get("pos", entry.get("at")) or [0.0, 0.0, 0.0]
    quat = entry.get("quat") or (rpy_quat(*entry["rpy"]) if entry.get("rpy") else scene.IDENTITY)
    q = qmul(quat, [Z_UP[3], Z_UP[0], Z_UP[1], Z_UP[2]])  # [w, x, y, z]: placed, after turning Y up to Z up
    s = entry.get("scale", 1.0)
    s = [float(s)] * 3 if isinstance(s, (int, float)) else [float(c) for c in s]
    wrapper = {"name": f"{owner}/native_glb", "translation": [float(c) for c in pos],
               "rotation": [q[1], q[2], q[3], q[0]], "children": [base + r for r in roots],
               "extras": {"orb": {"native": {k: v for k, v in entry.items()},
                                  "frame": "a Y-up glTF turned Z up, then at / rpy / scale"}}}
    if s != [1.0, 1.0, 1.0]:
        wrapper["scale"] = s
    return G.add("nodes", wrapper)
