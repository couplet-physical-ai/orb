"""The posed scene the export seams share (glb, usd, minecraft, step).

`build(ir, cfg)` walks the world graph once and returns a tree of nodes in the Orb frame
(Z up, metres): one node per site (its ground slab, zones and fixtures), one per placed item
(assemblies posed at their initial joint values, or `item.joints`, through forward
kinematics), each carrying its geoms and the provenance an exporter writes next to it.
Visibility follows the simulator seams: capture-derived boxes in a scanned site are hidden
unless `show_capture_boxes = true`, link collision geoms hide behind the link's visuals.

Node: {"name", "type", "pos", "quat" (w, x, y, z; relative to the parent node),
       "orb": {...provenance...}, "geoms": [{**geom, "show": bool}], "children": [...],
       "native": {...} (only on a node with a native for the seam asking)}

Seam scoping (v0.3): `scoped(ir, seam)` is `orb.world.for_seam` (blocks with `only`/`skip`
that leave this seam out are gone, every `native` map holds this seam's entry only), done once.
`build(ir, cfg, seam)` honours that seam's natives on items (or their part/assembly), links,
fixtures and sites: the node carries `native`, and its universal geoms go as `replace` says:
"visual" (default) drops the visuals and keeps the collisions hidden (`replaced: True`), "all"
drops every geom, "none" keeps them all (the native is an overlay). `snapshot(ir, t)` poses the
items where the timeline has them at `t` (for the seams that show motion as a static snapshot).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from ..world import frames, hidden_capture_boxes, qmul, qrot, with_mimics

GROUND_RGBA = [0.42, 0.42, 0.41, 1.0]
IDENTITY = [1.0, 0.0, 0.0, 0.0]


def version():
    try:
        from importlib.metadata import version as v

        return v("orb-lang")
    except Exception:  # pragma: no cover - running from a checkout without metadata
        return "0.1.0"


def node(name, type_, pos=None, quat=None, orb=None, geoms=(), children=(), native=None):
    n = {"name": name, "type": type_, "pos": list(pos or [0.0, 0.0, 0.0]), "quat": list(quat or IDENTITY),
         "orb": orb or {}, "geoms": list(geoms), "children": list(children)}
    if native:
        n["native"] = native
    return n


SCOPED = "seam"  # the key an IR scoped for one seam carries (its name), so scoping happens once
REPLACE = ("visual", "all", "none")


def scoped(ir, seam):
    """The IR as `seam` sees it: `orb.world.for_seam(ir, seam)`, once (an IR already scoped for
    this seam comes back as is)."""
    if ir.get(SCOPED) == seam:
        return ir
    from ..world import for_seam

    out = for_seam(ir, seam)
    out[SCOPED] = seam
    return out


def native_of(block, seam):
    """`seam`'s native spec on an IR block (item, part, assembly, link, fixture, site), or None."""
    if not seam or not isinstance(block, dict):
        return None
    spec = (block.get("native") or {}).get(seam)
    return spec if isinstance(spec, dict) else None


def framed_native(spec, pos, quat):
    """A site's native posed in the site's place: the site node stays at the origin (its ground,
    zones and fixtures are already world positions), so its at / turn go into the native's pose."""
    from ..world import rpy_quat

    if not spec or (not any(pos) and list(quat) == IDENTITY):
        return spec
    p = spec.get("pos", spec.get("at")) or [0.0, 0.0, 0.0]
    q = spec.get("quat") or (rpy_quat(*spec["rpy"]) if spec.get("rpy") is not None else
                             rpy_quat(0.0, 0.0, spec["yaw"]) if spec.get("yaw") is not None else IDENTITY)
    out = {k: v for k, v in spec.items() if k not in ("at", "rpy", "yaw")}
    out["pos"] = out["at"] = [a + b for a, b in zip(pos, qrot(list(quat), list(p)))]  # (minecraft reads at)
    out["quat"] = qmul(list(quat), list(q))
    return out


def replace_geoms(geoms, spec):
    """A node's geoms once a native replaces them (`spec["replace"]`, default "visual")."""
    if not spec:
        return geoms
    how = spec.get("replace", "visual")
    if how == "none":
        return geoms
    if how == "all":
        return []
    return [{**g, "show": False, "replaced": True} for g in geoms if g["role"] != "visual"]


def snapshot(ir, t=0.0, timeline=None):
    """A copy of the IR with every item the timeline moves posed where it is at `t` (world pose,
    joint values) and the items it hides at `t` left out: what a seam that cannot play motion
    shows. `timeline` default: the seam's (`seam "x" { timeline }`), else the first. An IR
    without timelines comes back as is."""
    if not ir.get("timelines"):
        return ir
    from ..motion import Motion, timeline_name

    m = Motion(ir, timeline or timeline_name(ir, ir.get(SCOPED)))
    items = dict(ir["items"])
    for name in m.items():
        if name not in items:
            continue
        if not m.visible(name, t):
            del items[name]
            continue
        it = dict(items[name])
        pos, quat = m.pose(name, t)
        it["pos"], it["quat"] = [float(v) for v in pos], [float(v) for v in quat]
        if it.get("kind") == "assembly":
            joints = m.joints(name, t)
            if joints:
                it["joints"] = {**(it.get("joints") or {}), **joints}
        items[name] = it
    return {**ir, "items": items, "snapshot": {"timeline": m.name, "t": float(t)}}


GROUND_DROP = 0.01  # m: the ground under a site that has its own floor fixture sits this much lower
OPENING_KINDS = ("door", "window")
OPENING_PROUD = 0.005  # m: door/window visuals stand this far off each wall face, and in from their edges


def ground_drop(site):
    """How far below the site's origin its ground goes: GROUND_DROP when a `floor` fixture covers
    the floor (its top face would otherwise share z = 0 with the ground and flicker), else 0."""
    return GROUND_DROP if any((fx.get("attrs") or {}).get("kind") == "floor"
                              for fx in (site.get("fixtures") or {}).values()) else 0.0


def opening_box_size(g, kind):
    """A box geom's drawn size: a door or window visual is OPENING_PROUD thicker than its wall on
    both faces (its thin horizontal axis) and OPENING_PROUD in from its other edges, so no face of it
    shares a plane with the wall around it or the scanned wall behind it. Anything else: as is."""
    size = list(g["size"])
    if kind not in OPENING_KINDS or g.get("role") != "visual" or g.get("shape") != "box" or len(size) != 3:
        return size
    thin = 0 if size[0] <= size[1] else 1
    out = [max(s - 2 * OPENING_PROUD, s * 0.5) for s in size]
    out[thin] = size[thin] + 2 * OPENING_PROUD
    return out


def zone_tint(z):
    """Opaque zone tint (as the Isaac seam draws it)."""
    return [0.8 * c + 0.1 for c in (z.get("color") or [0.5, 0.5, 0.5, 1])[:3]] + [1.0]


def build(ir, cfg, seam=None):
    """The posed scene. `seam` (default: the one the IR was scoped for) picks the natives honoured."""
    seam = seam or ir.get(SCOPED)
    out = []
    for sname, site in ir["sites"].items():
        w, d = site["floor"] or [20, 20]
        sq = list(site.get("quat") or IDENTITY)  # a site an output turned: its floor and zones turn with it
        # a placed site's zones and fixtures are named under it ("o/A/bench"): one site placed twice
        # must not give two nodes (STEP solids, GLB nodes) the same name
        under = (lambda k: f"{sname}/{k}") if "/" in sname else (lambda k: k)
        kids = [node(f"{sname}/ground", "ground", orb={"site": sname, "ground": True},
                     geoms=[{"role": "collision", "shape": "box",
                             "pos": [site["pos"][0], site["pos"][1], site["pos"][2] - 0.05 - ground_drop(site)],
                             "quat": sq, "size": [w, d, 0.1], "color": GROUND_RGBA, "show": True}])]
        for zname, z in site["zones"].items():
            orb = {"zone": zname, "site": sname, **{k: v for k, v in z.items() if k not in ("pos", "size", "color")}}
            geoms = []
            if z["size"]:
                geoms.append({"role": "visual", "shape": "box", "pos": [0.0, 0.0, 0.002], "quat": IDENTITY,
                              "size": [z["size"][0], z["size"][1], 0.004], "color": zone_tint(z), "show": True})
            kids.append(node(under(zname), "zone", z["pos"], None if sq == IDENTITY else sq, orb, geoms))
        hidden = hidden_capture_boxes(site, cfg)
        for fname, fx in site["fixtures"].items():
            geoms = [{**g, "show": fname not in hidden or g["role"] == "visual"} for g in fx["geoms"]]
            fnat = native_of(fx, seam)
            kids.append(node(under(fname), "fixture", fx["pos"], fx["quat"],
                             {**fx.get("attrs", {}), "fixture": fname, "site": sname}, replace_geoms(geoms, fnat),
                             native=fnat))
        out.append(node(sname, "site", None, None, {"site": sname, **{k: v for k, v in site.items()
                                                                       if k not in ("zones", "fixtures")}}, (), kids,
                        native=framed_native(native_of(site, seam), site["pos"], sq)))
    for iname, it in ir["items"].items():
        if it.get("location") is None:  # installed inside something else, or not placed
            continue
        orb = {"item": iname, **{k: v for k, v in it.items() if k not in ("pos", "quat")}}
        if it["kind"] == "part":
            part = ir["parts"][it["of"]]
            nat = native_of(it, seam) or native_of(part, seam)
            gs = part["geoms"]
            has_visual = any(g["role"] == "visual" for g in gs)
            geoms = [{**g, "show": g["role"] == "visual" or not has_visual} for g in gs]
            out.append(node(iname, "item", it["pos"], it["quat"], orb, replace_geoms(geoms, nat), native=nat))
            continue
        asm = ir["assemblies"][it["of"]]
        nat = native_of(it, seam) or native_of(asm, seam)
        fk = frames(asm, it.get("joints"))
        values = with_mimics(asm, it.get("joints"))  # mimic joints at their leader's value
        links = []
        for link in asm["links"]:
            pos, quat = fk[link["name"]]
            lnat = native_of(link, seam)
            has_visual = any(g["role"] == "visual" for g in link["geoms"])
            geoms = [{**g, "show": g["role"] == "visual" or not has_visual} for g in link["geoms"]]
            geoms = replace_geoms(replace_geoms(geoms, nat), lnat)
            j = link["joint"]
            lorb = {"item": iname, "link": link["name"], "parent": link["parent"], "part": link["part"],
                    "joint": {k: v for k, v in j.items()}}
            if j["type"] in ("revolute", "prismatic"):
                lorb["joint"]["position"] = values.get(j["name"], j["initial"])
            links.append(node(f"{iname}/{link['name']}", "link", pos, quat, lorb, geoms, native=lnat))
        out.append(node(iname, "item", it["pos"], it["quat"], orb, (), links, native=nat))
    return out


def walk(nodes, pos=None, quat=None):
    """Depth-first (node, world_pos, world_quat) over a node tree."""
    pos, quat = pos or [0.0, 0.0, 0.0], quat or IDENTITY
    for n in nodes:
        p = [a + b for a, b in zip(pos, qrot(quat, n["pos"]))]
        q = qmul(quat, n["quat"])
        yield n, p, q
        yield from walk(n["children"], p, q)


def geom_world(g, pos, quat):
    """A geom's pose in the world, given its node's world pose."""
    return [a + b for a, b in zip(pos, qrot(quat, g["pos"]))], qmul(quat, g["quat"])


def quat_matrix(q):
    w, x, y, z = q
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]


def sources(ir):
    """The project's OrbLang sources (file name -> text), for exports that carry them."""
    root = Path(ir["root"])
    # top-level *.orb files only: `orb pull` keeps a `.orb/` DIRECTORY (sources, lock) beside them
    return {p.name: p.read_text(encoding="utf-8-sig") for p in sorted(root.glob("*.orb"))
            if p.is_file() and not p.name.startswith(".")} if root.is_dir() else {}


def licenses(ir, limit=256 * 1024):
    """The project's mesh licences (meshes/**/LICENSE*, path -> text): exports that carry the
    meshes carry their terms too, and `orb import` puts them back."""
    root = Path(ir["root"])
    out = {}
    dirs = ["meshes"]
    for f in sorted(files_used(ir)):  # an imported project's meshes/ carries its own terms (../x/meshes,
        parts = Path(f).parts         # or deps/x/meshes once an import packaged it)
        if "meshes" in parts:
            dirs.append(Path(*parts[:parts.index("meshes") + 1]).as_posix())
    for d in dict.fromkeys(dirs):
        if Path(d).is_absolute() or not (root / d).is_dir():  # (an absolute path: no place to put it back)
            continue
        for p in sorted((root / d).rglob("*")):
            if p.is_file() and p.name.upper().startswith(("LICENSE", "LICENCE", "COPYING")) \
                    and p.stat().st_size <= limit:
                out[f"{d}/{p.relative_to(root / d).as_posix()}"] = p.read_text(errors="replace")
    return out


def files_outside(ir):
    """Project-relative `file` paths in the IR that leave the project (../robot-cell/meshes/...):
    files of projects it imports."""
    return {f for f in files_used(ir) if f.startswith("../")}


def files_used(ir):
    """Every project-relative `file` path the parts, assemblies and sites name."""
    found = set()

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k == "file" and isinstance(v, str):
                    found.add(v)
                else:
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk({k: v for k, v in ir.items() if k in ("parts", "assemblies", "sites")})
    return found


def dep_roots(ir):
    """{"../../lib/robot-cell": "robot-cell"}: for each file outside the project, the folder of the
    project it belongs to (the nearest folder above it holding .orb files) as the path climbs to
    it, and the name it is packaged under (deps/<name>/) when the export is imported. Names that
    two projects share get a suffix."""
    import os

    root, roots = ir["root"], {}
    for f in sorted(files_outside(ir)):
        if any(f.startswith(k + "/") for k in roots):
            continue
        parts = f.split("/")
        for base in (root, os.path.realpath(root)):
            if os.path.exists(os.path.join(base, f)):
                break
        else:
            continue
        for depth in range(len(parts) - 1, 0, -1):  # the folder holding the file, then its parents
            key = "/".join(parts[:depth])
            if set(key.split("/")) == {".."}:
                break
            if any(n.endswith(".orb") for n in os.listdir(os.path.join(base, key))):
                name, k = os.path.basename(os.path.normpath(os.path.join(base, key))), 2
                while name in roots.values():
                    name, k = f"{name.rstrip('0123456789-')}-{k}", k + 1
                roots[key] = name
                break
    return roots


VIEW_KEYS = ("root", "warnings", SCOPED, "snapshot")  # what a seam's view adds to the IR: not the world


def provenance(ir, seam, with_sources=True):
    """What every export says about where it came from. The embedded world graph is the IR handed
    in, without the seam-view markers (pass the unscoped IR to carry every seam's natives)."""
    meta = {"generator": f"orb {version()} seam {seam}", "ir": ir.get("orb"), "project": Path(ir["root"]).name,
            "frame": "Orb: Z up, metres, quaternions [w, x, y, z]"}
    caps = (ir.get("other") or {}).get("capture") or {}
    if caps:
        meta["capture"] = caps
    if with_sources:
        meta["sources"] = sources(ir)
        deps = dep_roots(ir)
        if deps:
            meta["deps"] = deps
        lic = licenses(ir)
        if lic:
            meta["licenses"] = lic
        meta["world"] = json.dumps({k: v for k, v in ir.items() if k not in VIEW_KEYS},
                                   separators=(",", ":"), sort_keys=True)
    return meta


def deg(rad):
    return rad * 180 / math.pi


INSTANCE_MIN = 4  # a fixture's visual meshes sharing a file this often are instanced (a PointInstancer)


def instance_groups(gs, least=INSTANCE_MIN):
    """{(file, colour or None): [geom index, ...]} of a fixture's visual mesh geoms that share a
    file at least `least` times (the colour is part of the key: a prototype binds one material),
    in first-use order; files used less often are left out (drawn one prim each)."""
    by_file = {}
    for k, g in enumerate(gs):
        if g["role"] == "visual" and g["shape"] == "mesh":
            by_file.setdefault(g["file"], []).append(k)
    out = {}
    for file, ks in by_file.items():
        if len(ks) >= least:
            for k in ks:
                c = gs[k].get("color")
                out.setdefault((file, tuple(c) if c else None), []).append(k)
    return out
