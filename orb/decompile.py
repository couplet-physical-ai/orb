"""World graph (IR) -> OrbLang source: the way back when only the compiled scene survived.

`orb import` uses the original sources when an export carries them (exact, comments and
modules included). Without them, it rebuilds a project from the world graph an export
embeds: `to_orb(world)` writes one canonical `.orb` file that compiles back to the same world
graph (same ids, kinds, poses, dimensions, inertias, joints and provenance; numbers in SI
units, modules expanded, comments gone).
"""
from __future__ import annotations

import json
import re

IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
IDENTITY = [1.0, 0.0, 0.0, 0.0]


def num(x):
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int):
        return str(x)
    s = repr(float(x))
    return "0" if s in ("0.0", "-0.0") else s.removesuffix(".0") if s.endswith(".0") else s


def qstr(s):
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"').replace("${", "\\${").replace("\n", "\\n") + '"'


def value(v):
    """A plain IR value (no units) as an OrbLang literal."""
    if v is None:
        return "null"
    if isinstance(v, (bool, int, float)):
        return num(v)
    if isinstance(v, str):
        return qstr(v)
    if isinstance(v, list):
        return "[" + ", ".join(value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{key(k)} = {value(x)}" for k, x in v.items()) + " }" if v else "{}"
    raise TypeError(f"cannot write {type(v).__name__}")


def key(k):
    return k if IDENT.match(k) else qstr(k)


def vec(xs, unit=""):
    return "[" + ", ".join(num(x) for x in xs) + "]" + (f" {unit}" if unit else "")


def label_ref(type_, label):
    """part.robot_joint, site["room-scan"]"""
    return type_ + (f".{label}" if IDENT.match(label) else f"[{qstr(label)}]")


def path_ref(path, world):
    """An IR reference path (site.office.zone.bench, item.robot-01, ...) -> OrbLang reference."""
    if not isinstance(path, str):
        return value(path)
    if path.startswith("site."):
        for s, site in world["sites"].items():
            if path == f"site.{s}":
                return label_ref("site", s)
            for z in site["zones"]:
                if path == f"site.{s}.zone.{z}":
                    return label_ref("site", s) + ".zone" + label_ref("", z)
    t, _, lab = path.partition(".")
    return label_ref(t, lab)


def loc_base(it, world):
    """The frame an item's `at` (and its timeline keys') is in: its zone's or site's position."""
    loc = it.get("location")
    for s, sd in world["sites"].items():
        if loc == f"site.{s}":
            return sd["pos"]
        for z, zd in sd["zones"].items():
            if loc == f"site.{s}.zone.{z}":
                return zd["pos"]
    return [0.0, 0.0, 0.0]


SHOT_UNITS = {"move_to": "m", "look_to": "m", "offset_to": "m", "look_offset": "m", "fov_to": "rad", "orbit": "rad", "lag": "s", "blend": "s"}


def timeline(w, name, tl, world):
    """A normalized timeline -> canonical OrbLang: every key explicit, world poses back in each
    item's location frame, attach/detach/show/hide/set as events."""
    scope = (tl.get("scope") or {})

    def scoped(kind, label):
        for k, v in ((scope.get(kind) or {}).get(label) or {}).items():
            w(f"{k} = {value(v)}")

    skip = ("label", "duration", "loop", "physics", "tracks", "joints", "attach", "visible", "channels",
            "phases", "events", "bodies", "scope", "blend")
    with w.block(f"timeline {qstr(name)}"):
        w(f"duration = {num(tl['duration'])} s")
        for k in ("loop", "physics"):
            if k in tl:
                w(f"{k} = {value(tl[k])}")
        if tl.get("label") is not None:
            w(f"label = {value(tl['label'])}")
        if isinstance(tl.get("blend"), (int, float)):  # its shots' default blend (they carry it resolved too)
            w(f"blend = {num(tl['blend'])} s")
        attrs(w, tl, skip=skip)
        for item, keys in (tl.get("tracks") or {}).items():
            base = loc_base(world["items"].get(item) or {}, world)
            with w.block(f"track {qstr(item)}"):
                scoped("tracks", item)
                for k in keys:
                    at = vec([a - b for a, b in zip(k["pos"], base)], "m")
                    w(f"key {{ t = {num(k['t'])} s  at = {at}  quat = {vec(k['quat'])}  ease = {qstr(k['ease'])} }}")
        for item, keys in (tl.get("joints") or {}).items():
            asm = world["assemblies"].get((world["items"].get(item) or {}).get("of"), {})
            kinds = {l["joint"]["name"]: l["joint"]["type"] for l in asm.get("links", []) if l.get("joint")}
            with w.block(f"joints {qstr(item)}"):
                scoped("joints", item)
                for k in keys:
                    vals = ", ".join(f"{qstr(j)} = {num(v)} {'m' if kinds.get(j) == 'prismatic' else 'rad'}"
                                     for j, v in k["values"].items())
                    w(f"key {{ t = {num(k['t'])} s  values = {{ {vals} }}  ease = {qstr(k['ease'])} }}")
        att = tl.get("attach") or []
        for i, a in enumerate(att):
            parts = [f"t = {num(a['t0'])} s", f"to = {label_ref('item', a['to'])}"]
            if a.get("link"):
                parts.append(f"link = {qstr(a['link'])}")
            parts += [f"at = {vec(a['pos'], 'm')}", f"quat = {vec(a['quat'])}"]
            parts += [f"{k} = {value(a[k])}" for k in ("only", "skip") if k in a]
            w(f"attach {qstr(a['item'])} {{ " + "  ".join(parts) + " }")
            t1 = a.get("t1")
            handed_on = any(b["item"] == a["item"] and b["t0"] == t1 for b in att[i + 1:])
            if t1 is not None and not handed_on:
                w(f"detach {qstr(a['item'])} {{ t = {num(t1)} s }}")
        for item, flips in (tl.get("visible") or {}).items():
            for t, vis in flips:  # a first `show` at 0 too: it may be all that names an unplaced item
                w(f"{'show' if vis else 'hide'} {qstr(item)} {{ t = {num(t)} s }}")
        for ch, sets in (tl.get("channels") or {}).items():
            for t, v in sets:
                w(f"set {qstr(ch)} {{ t = {num(t)} s  value = {num(v)} }}")
        for p in tl.get("phases") or []:
            parts = [f"t = {num(p['t0'])} s"]
            if p.get("label") is not None:
                parts.append(f"label = {value(p['label'])}")
            if p.get("station"):
                parts.append(f"station = {path_ref(p['station'], world)}")
            parts += [f"{key(k)} = {value(v)}" for k, v in p.items() if k not in ("name", "label", "t0", "t1", "station")]
            w(f"phase {qstr(p['name'])} {{ " + "  ".join(parts) + " }")
        for e in tl.get("events") or []:
            parts = [f"t = {num(e['t'])} s"] + [f"{key(k)} = {value(v)}" for k, v in e.items() if k not in ("name", "t")]
            w(f"event {qstr(e['name'])} {{ " + "  ".join(parts) + " }")


def shot(w, s):
    with w.block(f"shot {qstr(s['name'])}"):
        w(f"timeline = {label_ref('timeline', s['timeline'])}")
        w(f"t = [{num(s['t0'])}, {num(s['t1'])}] s")
        if s.get("camera"):
            w(f"camera = {label_ref('camera', s['camera'])}")
        if s.get("pos") is not None:
            w(f"at = {vec(s['pos'], 'm')}")
        if s.get("look_at") is not None:
            w(f"look_at = {vec(s['look_at'], 'm')}")
        if s.get("follow"):
            w(f"follow = {label_ref('item', s['follow'])}")
        if s.get("offset") is not None:
            w(f"offset = {vec(s['offset'], 'm')}")
        if s.get("relative"):
            w("relative = true")
        w(f"fov = {num(s['fov'])} rad")
        w(f"label = {value(s['label'])}")
        for k, v in s.items():
            if k in ("name", "timeline", "t0", "t1", "camera", "pos", "look_at", "follow", "offset", "relative",
                     "fov", "label"):
                continue
            u = SHOT_UNITS.get(k)
            w(f"{key(k)} = {(vec(v, u) if isinstance(v, list) else f'{num(v)} {u}') if u else value(v)}")


class W:
    def __init__(self):
        self.lines, self.ind = [], 0

    def __call__(self, s=""):
        self.lines.append(("  " * self.ind + s) if s else "")

    def block(self, head):
        self(head + " {")
        self.ind += 1
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.ind -= 1
        self("}")


def geom(w, g):
    head = f'{g["role"]} "{g["shape"]}"'
    parts = []
    if g["shape"] == "box":
        parts.append(f"size = {vec(g['size'], 'm')}")
    elif g["shape"] == "sphere":
        parts.append(f"radius = {num(g['radius'])} m")
    elif g["shape"] in ("cylinder", "capsule"):
        parts.append(f"radius = {num(g['radius'])} m")
        parts.append(f"length = {num(g['length'])} m")
    else:
        parts.append(f"file = {qstr(g['file'])}")
        if g.get("scale", [1, 1, 1]) != [1, 1, 1]:
            parts.append(f"scale = {vec(g['scale'])}")
    if any(g["pos"]):
        parts.append(f"at = {vec(g['pos'], 'm')}")
    if g["quat"] != IDENTITY:
        parts.append(f"quat = {vec(g['quat'])}")
    if g.get("color") is not None:
        parts.append(f"color = {vec(g['color'])}")
    if "friction" in g:
        parts.append(f"friction = {num(g['friction'])}")
    w(f"{head} {{ " + "  ".join(parts) + " }")


def generic_children(w, kids):
    for c in kids or []:
        attrs = {k: v for k, v in c.items() if k not in ("type", "name", "children")}
        with w.block(f"{c['type']} {qstr(c['name'])}" if c.get("name") is not None else c["type"]):
            for k, v in attrs.items():
                w(f"{key(k)} = {value(v)}")
            generic_children(w, c.get("children"))


def attrs(w, d, skip=()):
    for k, v in d.items():
        if k not in skip and k != "children":
            w(f"{key(k)} = {value(v)}")


def inertial(w, i):
    if i:
        w(f"mass = {num(i['mass'])} kg")
        w(f"com = {vec(i['com'], 'm')}")
        w(f"inertia = {vec(i['inertia'], 'kg*m2')}")


JOINT_UNITS = {"revolute": {"range": "rad", "initial": "rad", "effort": "Nm", "velocity": "rad/s"},
               "prismatic": {"range": "m", "initial": "m", "effort": "N", "velocity": "m/s"}}


def joint(w, j, link, is_root, fixed_base):
    default_type = "free" if is_root and not fixed_base else "fixed"
    if j["name"] == f"{link}_joint" and j["type"] == default_type:
        return
    with w.block(f"joint {qstr(j['name'])}"):
        w(f"type = {qstr(j['type'])}")
        if j["type"] not in ("revolute", "prismatic"):
            return
        u = JOINT_UNITS[j["type"]]
        w(f"axis = {vec(j['axis'])}")
        if j.get("range"):
            w(f"range = {vec(j['range'], u['range'])}")
        if j.get("mimic"):  # its initial is derived from the leader's
            w(f"mimic = {qstr(j['mimic']['joint'])}")
            w(f"multiplier = {num(j['mimic']['multiplier'])}")
            w(f"offset = {num(j['mimic']['offset'])} {u['initial']}")
        else:
            w(f"initial = {num(j['initial'])} {u['initial']}")
        for k in ("effort", "velocity"):
            if j.get(k) is not None:
                w(f"{k} = {num(j[k])} {u[k]}")
        for k in ("damping", "armature", "friction"):
            w(f"{k} = {num(j[k])}")
        if j.get("drive"):
            w(f"kp = {num(j['drive']['kp'])}")
            w(f"kd = {num(j['drive']['kd'])}")


def to_orb(world, name="imported"):
    """-> {file name: OrbLang text} for a world graph (the `orb ir` JSON)."""
    w = W()
    w(f"# {name}: rebuilt by `orb import` from the world graph an Orb export carried.")
    w("# Canonical form: SI units, modules expanded, original comments gone. It compiles to the same world graph.")
    for typ, blocks in (world.get("other") or {}).items():
        for lab, b in blocks.items():
            w()
            with w.block(f"{typ} {qstr(lab)}"):
                attrs(w, b)
                generic_children(w, b.get("children"))
    for pname, p in world["parts"].items():
        w()
        with w.block(f"part {qstr(pname)}"):
            attrs(w, p, skip=("inertial", "geoms"))
            inertial(w, p["inertial"])
            for g in p["geoms"]:
                geom(w, g)
    for aname, a in world["assemblies"].items():
        w()
        links = a["links"]
        kids = {}
        for l in links:
            kids.setdefault(l["parent"], []).append(l)
        fixed_base = bool(a.get("fixed_base", False))
        with w.block(f"assembly {qstr(aname)}"):
            attrs(w, a, skip=("root", "links", "mass"))

            def link(l):
                part = world["parts"].get(l["part"]) if l["part"] else None
                with w.block(f"link {qstr(l['name'])}"):
                    if part is not None:
                        w(f"part = {label_ref('part', l['part'])}")
                    if any(l["pos"]):
                        w(f"at = {vec(l['pos'], 'm')}")
                    if l["quat"] != IDENTITY:
                        w(f"quat = {vec(l['quat'])}")
                    if l["inertial"] and (part is None or part["inertial"] != l["inertial"]):
                        inertial(w, l["inertial"])
                    for g in l["geoms"][len(part["geoms"]) if part else 0:]:
                        geom(w, g)
                    joint(w, l["joint"], l["name"], l["parent"] is None, fixed_base)
                    for s in l["sensors"]:
                        w(f"sensor {qstr(s['name'])} {{ type = {qstr(s['type'])}  at = {vec(s['pos'], 'm')}  "
                          f"quat = {vec(s['quat'])} }}")
                    for c in kids.get(l["name"], []):
                        link(c)

            link(kids[None][0])
    for lname, lc in world["lifecycles"].items():
        w()
        with w.block(f"lifecycle {qstr(lname)}"):
            attrs(w, lc, skip=("transitions",))
    for pname, p in world["processes"].items():
        w()
        with w.block(f"process {qstr(pname)}"):
            for k, v in p.items():
                if k in ("assembly", "station"):
                    w(f"{k} = {path_ref(v, world)}")
                elif k == "timescale" or k not in ("steps", "duration", "station_pos", "fidelity"):
                    w(f"{key(k)} = {value(v)}")
            w(f"fidelity = {qstr(p['fidelity'])}")
            for s in p["steps"]:
                rest = {k: v for k, v in s.items() if k not in ("name", "start", "takes")}
                body = [f"takes = {num(s['takes'])} s"] + [f"{key(k)} = {value(v)}" for k, v in rest.items()]
                w(f"step {qstr(s['name'])} {{ " + "  ".join(body) + " }")
    for sname, s in world["sites"].items():
        w()
        with w.block(f"site {qstr(sname)}"):
            attrs(w, s, skip=("pos", "floor", "zones", "fixtures"))
            if any(s["pos"]):
                w(f"at = {vec(s['pos'], 'm')}")
            if s["floor"] is not None:
                w(f"floor = {vec(s['floor'], 'm')}")
            for zname, z in s["zones"].items():
                parts = [f"at = {vec([a - b for a, b in zip(z['pos'], s['pos'])], 'm')}"]
                if z["size"] is not None:
                    parts.append(f"size = {vec(z['size'], 'm')}")
                if z.get("color") is not None:
                    parts.append(f"color = {vec(z['color'])}")
                parts += [f"{key(k)} = {value(v)}" for k, v in z.items() if k not in ("pos", "size", "color")]
                w(f"zone {qstr(zname)} {{ " + "  ".join(parts) + " }")
            for fname, fx in s["fixtures"].items():
                with w.block(f"fixture {qstr(fname)}"):
                    fa = fx.get("attrs", {})
                    attrs(w, fa)
                    w(f"at = {vec([a - b for a, b in zip(fx['pos'], s['pos'])], 'm')}")
                    if fx["quat"] != IDENTITY:
                        w(f"quat = {vec(fx['quat'])}")
                    for g in fx["geoms"]:
                        geom(w, g)
                    generic_children(w, fa.get("children"))
    for iname, it in world["items"].items():
        w()
        with w.block(f"item {qstr(iname)}"):
            w(f"of = {label_ref(it['kind'], it['of'])}")
            loc = it.get("location")
            if loc:
                w(f"location = {path_ref(loc, world)}")
            base = loc_base(it, world)
            rel = [a - b for a, b in zip(it["pos"], base)]
            if any(rel):
                w(f"at = {vec(rel, 'm')}")
            if it["quat"] != IDENTITY:
                w(f"quat = {vec(it['quat'])}")
            for k in ("lifecycle", "made_by", "installed_in"):
                if it.get(k):
                    w(f"{k} = {path_ref(it[k], world)}")
            if it.get("joints"):
                asm = world["assemblies"][it["of"]]
                kinds = {l["joint"]["name"]: l["joint"]["type"] for l in asm["links"]}
                w("joints = { " + ", ".join(f"{key(k)} = {num(v)} {'m' if kinds[k] == 'prismatic' else 'rad'}"
                                            for k, v in it["joints"].items()) + " }")
            for k, v in it.items():
                if k not in ("kind", "of", "location", "pos", "quat", "lifecycle", "made_by", "installed_in", "joints"):
                    w(f"{key(k)} = {value(v)}")
    for cname, c in (world.get("cameras") or {}).items():
        w()
        with w.block(f"camera {qstr(cname)}"):
            w(f"at = {vec(c['pos'], 'm')}")
            w(f"look_at = {vec(c['look_at'], 'm')}")
            w(f"fov = {num(c['fov'])} rad")
            attrs(w, c, skip=("pos", "look_at", "fov"))
    for tname, tl in (world.get("timelines") or {}).items():
        w()
        timeline(w, tname, tl, world)
    for sh in world.get("shots") or []:
        w()
        shot(w, sh)
    for oname, o in (world.get("outputs") or {}).items():
        # the scene an output placed is in the items and sites above (location = output.<name>,
        # `scene` records); the functions that built it are not in a world graph
        w()
        with w.block(f"output {qstr(oname)}"):
            w("value = compose()")
            if o.get("prefix") != oname:
                w(f"prefix = {value(o.get('prefix', ''))}")
            if o.get("label") != oname:
                w(f"label = {value(o.get('label'))}")
    for sname, cfg in world["seams"].items():
        w()
        if cfg:
            with w.block(f"seam {qstr(sname)}"):
                attrs(w, cfg)
                generic_children(w, cfg.get("children"))
        else:
            w(f"seam {qstr(sname)} {{}}")
    return {f"{name}.orb": "\n".join(w.lines) + "\n"}


def same_world(a, b, tol=1e-9):
    """Compare two world graphs (minus root/warnings), floats to `tol`. -> list of differences."""
    diffs = []

    def walk(x, y, path):
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y)):
                if k in ("root", "warnings") and not path:
                    continue
                if k not in x or k not in y:
                    diffs.append(f"{path}/{k}: only in {'second' if k not in x else 'first'}")
                else:
                    walk(x[k], y[k], f"{path}/{k}")
        elif isinstance(x, list) and isinstance(y, list):
            if len(x) != len(y):
                diffs.append(f"{path}: {len(x)} vs {len(y)} entries")
            for i, (p, q) in enumerate(zip(x, y)):
                walk(p, q, f"{path}[{i}]")
        elif isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
            if abs(x - y) > tol * max(1.0, abs(x), abs(y)):
                diffs.append(f"{path}: {x!r} vs {y!r}")
        elif x != y:
            diffs.append(f"{path}: {json.dumps(x)[:80]} vs {json.dumps(y)[:80]}")

    walk(a, b, "")
    return diffs
