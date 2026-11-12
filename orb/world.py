"""Evaluated Orb blocks -> the world graph (IR) that seams consume.

The IR is plain JSON in SI units (m, kg, s, rad). Quaternions are [w, x, y, z].
Known block types (part, assembly, site, item, lifecycle, process, seam) get a
normalized shape; any other top-level block passes through under "other" so a
third-party seam can define its own vocabulary.
"""
from __future__ import annotations

import copy
import math
import os
from pathlib import Path

from .lang import DIMLESS, Block, OrbError, Q, did_you_mean, dim_name, type_name

IR_VERSION = "0.1"

M, L, T, ANG = (1, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0), (0, 0, 0, 0, 0, 1)
INERTIA = (1, 2, 0, 0, 0, 0)
TORQUE, FORCE = (1, 2, -2, 0, 0, 0), (1, 1, -2, 0, 0, 0)
ANGVEL, VEL = (0, 0, -1, 0, 0, 1), (0, 1, -1, 0, 0, 0)
HINT = {L: "mm or m", M: "kg", ANG: "deg or rad", T: "s or min", INERTIA: "kg*m2", TORQUE: "Nm",
        FORCE: "N", ANGVEL: "rad/s", VEL: "m/s"}

COLORS = {
    "black": [0.02, 0.02, 0.02, 1], "white": [0.95, 0.95, 0.95, 1], "gray": [0.67, 0.69, 0.77, 1],
    "metal": [0.9, 0.95, 0.95, 1], "red": [0.8, 0.1, 0.1, 1], "green": [0.1, 0.6, 0.2, 1],
    "blue": [0.1, 0.3, 0.8, 1], "orange": [1.0, 0.45, 0.0, 1], "wood": [0.55, 0.4, 0.25, 1],
    "concrete": [0.62, 0.62, 0.6, 1], "wall": [0.85, 0.85, 0.82, 1],
}
SHAPES = ("box", "sphere", "cylinder", "capsule", "mesh")
JOINTS = ("revolute", "prismatic", "fixed", "free")
POSE = ("at", "rpy", "quat", "yaw")
SCOPE = ("only", "skip")
BODIES = ("static", "kinematic", "dynamic")
EASES = ("linear", "smooth", "step")
REPLACE = ("visual", "all", "none")
# keys of a `native "<seam>"` block that name files (resolved like `file`, stored root-relative)
NATIVE_FILES = ("usd", "mjcf", "glb", "gltf", "schematic", "mesh", "file", "texture", "obj")
NATIVE_ON = ("part", "assembly", "link", "site", "fixture", "item")
# timeline entries; the ones in NO_SCOPE cannot carry only/skip (they are steps of something scoped)
TIMELINE_ENTRIES = ("track", "joints", "attach", "detach", "show", "hide", "set", "phase", "event")
NO_SCOPE = ("key", "path", "detach", "show", "hide", "set", "native")
_MISSING = object()


# --------------------------------------------------------------------------- math


def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw]


def qrot(q, v):
    w, x, y, z = qmul(qmul(q, [0, *v]), [q[0], -q[1], -q[2], -q[3]])
    return [x, y, z]


def rpy_quat(r, p, y):
    """URDF convention: fixed-axis roll(X), pitch(Y), yaw(Z) == Rz*Ry*Rx."""
    cr, sr, cp, sp, cy, sy = (math.cos(r / 2), math.sin(r / 2), math.cos(p / 2), math.sin(p / 2),
                              math.cos(y / 2), math.sin(y / 2))
    return [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy]


def axis_angle(axis, a):
    s = math.sin(a / 2)
    return [math.cos(a / 2), axis[0] * s, axis[1] * s, axis[2] * s]


def x_to(axis):
    """Quaternion rotating +X onto `axis` (USD joints only take X/Y/Z axes)."""
    d = axis[0]
    if d > 1 - 1e-9:
        return [1.0, 0.0, 0.0, 0.0]
    if d < -1 + 1e-9:
        return [0.0, 0.0, 0.0, 1.0]
    w, c = 1 + d, [0.0, -axis[2], axis[1]]  # cross(x, axis)
    n = math.sqrt(w * w + c[1] ** 2 + c[2] ** 2)
    return [w / n, 0.0, c[1] / n, c[2] / n]


def mat_quat(m):
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1) * 2
        return [s / 4, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s]
    i = max(range(3), key=lambda k: m[k][k])
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1 + m[i][i] - m[j][j] - m[k][k]) * 2
    q = [0.0] * 4
    q[0], q[i + 1], q[j + 1], q[k + 1] = (m[k][j] - m[j][k]) / s, s / 4, (m[j][i] + m[i][j]) / s, (m[k][i] + m[i][k]) / s
    return q


def principal(inertia):
    """[ixx,iyy,izz,ixy,ixz,iyz] -> (principal moments, quat of principal axes). Jacobi sweep."""
    ixx, iyy, izz, ixy, ixz, iyz = inertia
    a = [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]]
    v = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]
    for _ in range(50):
        p, q = max(((0, 1), (0, 2), (1, 2)), key=lambda pq: abs(a[pq[0]][pq[1]]))
        if abs(a[p][q]) < 1e-18:
            break
        th = 0.5 * math.atan2(2 * a[p][q], a[q][q] - a[p][p])
        c, s = math.cos(th), math.sin(th)
        for k in range(3):  # A = A J
            akp, akq = a[k][p], a[k][q]
            a[k][p], a[k][q] = c * akp - s * akq, s * akp + c * akq
        for k in range(3):  # A = J^T A
            apk, aqk = a[p][k], a[q][k]
            a[p][k], a[q][k] = c * apk - s * aqk, s * apk + c * aqk
        for k in range(3):
            vkp, vkq = v[k][p], v[k][q]
            v[k][p], v[k][q] = c * vkp - s * vkq, s * vkp + c * vkq
    det = (v[0][0] * (v[1][1] * v[2][2] - v[1][2] * v[2][1]) - v[0][1] * (v[1][0] * v[2][2] - v[1][2] * v[2][0])
           + v[0][2] * (v[1][0] * v[2][1] - v[1][1] * v[2][0]))
    if det < 0:
        for k in range(3):
            v[k][2] = -v[k][2]
    return [a[0][0], a[1][1], a[2][2]], mat_quat(v)


def mimics(asm):
    """{mimic joint: {"joint": leader, "multiplier": m, "offset": o}} of an assembly (in link order)."""
    return {l["joint"]["name"]: l["joint"]["mimic"] for l in asm["links"] if l["joint"].get("mimic")}


def with_mimics(asm, values=None):
    """Joint values with every mimic joint derived from its leader: a copy of `values` ({joint: value},
    m or rad; numbers or numpy arrays) where each mimic joint = multiplier * leader + offset, the
    leader's value taken from `values`, else its `initial`. A mimic joint's own entry is ignored (it
    always follows). Every place that resolves joint values (forward kinematics, item joints, the
    motion evaluator, the seams' poses) goes through here."""
    out = dict(values or {})
    mm = mimics(asm)
    if not mm:
        return out
    initial = {l["joint"]["name"]: l["joint"].get("initial", 0.0) for l in asm["links"]}
    for name, m in mm.items():
        lead = out.get(m["joint"])
        if lead is None:
            lead = initial.get(m["joint"], 0.0)
        out[name] = m["multiplier"] * lead + m["offset"]
    return out


def frames(asm, joint_values=None):
    """Forward kinematics at the given joint values (default: each joint's `initial`; mimic joints
    follow their leader). Returns {link: (pos, quat)} in the assembly root frame."""
    out = {}
    joint_values = with_mimics(asm, joint_values)
    for link in asm["links"]:  # parents always precede children
        pos, quat = link["pos"], link["quat"]
        if link["parent"]:
            ppos, pq = out[link["parent"]]
            pos, quat = [a + b for a, b in zip(ppos, qrot(pq, pos))], qmul(pq, quat)
        j = link["joint"]
        if j["type"] in ("revolute", "prismatic"):
            v = joint_values.get(j["name"], j["initial"])
            if j["type"] == "revolute":
                quat = qmul(quat, axis_angle(j["axis"], v))
            else:
                pos = [a + b for a, b in zip(pos, qrot(quat, [x * v for x in j["axis"]]))]
        out[link["name"]] = (pos, quat)
    return out


# --------------------------------------------------------------------------- readers


def _num(x, dim, key, pos, lenient):
    if isinstance(x, Q):
        dims = dim if isinstance(dim[0], tuple) else (dim,)
        if x.d not in dims:
            names = [dim_name(d) for d in dims]
            raise OrbError(f"{key} must be {'an' if names[0][0] in 'aeiou' else 'a'} {' or '.join(names)}, "
                           f"got {dim_name(x.d)}", pos)
        return x.v
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        d0 = dim[0] if isinstance(dim[0], tuple) else dim
        if d0 != DIMLESS and x != 0 and not lenient:
            raise OrbError(f"{key} needs a unit ({HINT.get(d0, dim_name(d0))}), got a bare number", pos)
        return float(x)
    raise OrbError(f"{key} must be a number, got {type_name(x)}", pos)


def num(blk, key, dim=DIMLESS, default=_MISSING, lenient=False):
    if key not in blk.attrs:
        if default is _MISSING:
            raise OrbError(f"{blk.path} needs `{key}`", blk.pos)
        return default
    return _num(blk.attrs[key], dim, key, blk.attr_pos[key], lenient)


def vec(blk, key, dim, n, default=_MISSING):
    if key not in blk.attrs:
        if default is _MISSING:
            raise OrbError(f"{blk.path} needs `{key}`", blk.pos)
        return default
    v, pos = blk.attrs[key], blk.attr_pos[key]
    if not isinstance(v, list) or len(v) != n:
        raise OrbError(f"{key} must be a list of {n}", pos)
    return [_num(x, dim, key, pos, False) for x in v]


def ref(blk, key, types, default=None, ir=None):
    if key not in blk.attrs:
        return default
    v = blk.attrs[key]
    if not isinstance(v, Block) or v.type not in types:
        art = "an" if types[0][0] in "aeiou" else "a"
        hint = {"item": ' item["<label>"] (an output\'s items: item["<output>/<path>"])',
                "camera": " camera.<name>", "timeline": " timeline.<name>"}.get(types[0], "")
        got = type_name(v)
        raise OrbError(f"{key} must reference {art} {' or '.join(types)}, got {'a ' if got == 'scene' else ''}{got}"
                       + (f"; write{hint}" if hint else ""), blk.attr_pos[key])
    if getattr(v, "placed", False) and ir is not None:  # item["<output>/..."], site["<output>/..."].zone.x
        check_placed(v, ir, blk.attr_pos[key])     # (ir None: checked by whoever reads it later)
    return v


def check_placed(v, ir, pos):
    """A reference to what an output placed (item, site, or a placed site's zone or fixture) exists."""
    def missing(what, label, have):
        have = sorted(have)
        under = [h for h in have if h.endswith("/" + label)]  # item["c"] for what output w placed as w/c
        if under:
            return OrbError(f"no {what.split()[-1]} {label!r}: an output's are named <output>/<instance path>; "
                            f"write {what.split()[-1]}[\"{under[0]}\"]", pos)
        return OrbError(f"no output placed {what} {label!r}" + did_you_mean(label, have)
                        + (f" (have: {', '.join(have[:8])}{', ...' if len(have) > 8 else ''})" if have else ""), pos)
    placed = lambda d: {n for n, x in d.items() if isinstance(x.get("scene"), dict)}  # noqa: E731
    if v.type == "item":
        if v.label not in ir["items"]:
            raise missing("an item", v.label, placed(ir["items"]))
        return
    parts = v.path.split(".")  # site.<label>[.zone|fixture.<name>]: placed labels hold no "."
    sites = {n: ir["sites"][n] for n in placed(ir["sites"])}
    if parts[1] not in sites:
        raise missing("a site", parts[1], sites)
    if len(parts) > 3:
        sec = sites[parts[1]].get(parts[2] + "s") or {}
        if parts[3] not in sec:
            raise OrbError(f"site {parts[1]} has no {parts[2]} {parts[3]!r}{did_you_mean(parts[3], sec)} "
                           f"(has: {', '.join(sec) or 'none'})", pos)


def boolean(blk, key, default):
    v = blk.attrs.get(key, default)
    if not isinstance(v, bool):
        raise OrbError(f"{key} must be true or false, got {type_name(v)}", blk.attr_pos.get(key, blk.pos))
    return v


def has_rot(blk):
    return any(k in blk.attrs for k in ("quat", "rpy", "yaw"))


def rot(blk):
    if "quat" in blk.attrs:
        q = vec(blk, "quat", DIMLESS, 4)
        n = math.sqrt(sum(x * x for x in q))
        return [x / n for x in q]
    if "rpy" in blk.attrs:
        return rpy_quat(*vec(blk, "rpy", ANG, 3))
    return rpy_quat(0, 0, num(blk, "yaw", ANG, 0.0))


def joint_value(j, name, v, pos):
    """A joint's value in SI (m or rad), checked against its range; the error says it in units."""
    lin = j["type"] == "prismatic"
    x = _num(v, L if lin else ANG, name, pos, False)
    r = j.get("range")
    if r and not r[0] - 1e-9 <= x <= r[1] + 1e-9:
        f = (lambda a: f"{a:.4g} m") if lin else (lambda a: f"{math.degrees(a):.4g} deg")
        raise OrbError(f"{name} = {f(x)} is outside its range [{f(r[0])}, {f(r[1])}]", pos)
    return x


def mimic_error(j, name, pos):
    """A mimic joint cannot be set (item `joints`, a timeline's `joints` keys): it follows its leader."""
    if j.get("mimic"):
        lead = j["mimic"]["joint"]
        raise OrbError(f"{name} is a mimic joint: it follows {lead} ({name} = {j['mimic']['multiplier']:g} * {lead} "
                       f"+ offset); set {lead} instead", pos)


def jsonify(v):
    if isinstance(v, Q):
        return float(f"{v.v:.12g}")
    if isinstance(v, float):
        return float(f"{v:.12g}")
    if isinstance(v, Block):
        return v.path
    if isinstance(v, list):
        return [jsonify(x) for x in v]
    if isinstance(v, dict):
        return {k: jsonify(x) for k, x in v.items()}
    return v


def generic(blk):
    """Any block -> {attrs..., children: [{type, name, ...}]} for pass-through."""
    out = {k: jsonify(v) for k, v in blk.attrs.items()}
    if blk.children:
        out["children"] = [{"type": c.type, "name": c.label, **generic(c)} for c in blk.children]
    return out


def color(blk):
    if "color" not in blk.attrs:
        return None
    c, pos = blk.attrs["color"], blk.attr_pos["color"]
    if isinstance(c, str):
        if c.startswith("#") and len(c) == 7:
            return [int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)] + [1.0]
        if c not in COLORS:
            raise OrbError(f"unknown color {c!r} (use #rrggbb or one of {', '.join(COLORS)})", pos)
        return COLORS[c]
    if isinstance(c, list) and len(c) in (3, 4):
        return [float(x) for x in c] + ([1.0] if len(c) == 3 else [])
    raise OrbError("color must be a name, #rrggbb or [r, g, b(, a)]", pos)


# --------------------------------------------------------------------------- normalizers


def project_path(p, root):
    """`p` relative to the project `root`, as written: `..` is folded lexically and symlinks are
    not followed, so the same project gives the same IR whether its files are real or links (Orb
    Studio checks unsaved edits in a temp mirror whose meshes/ and blobs/ are symlinks). Only when
    folding `..` would land on a different file than the OS opens (a `..` after a symlinked
    folder) is the real path used."""
    from .lang import MIRROR_OF, import_base

    if os.path.isfile(os.path.join(root, MIRROR_OF)):  # the editor's unsaved-edit mirror of a project
        mirrored = import_base(os.path.join(root, "x"))
        rel = os.path.relpath(os.path.normpath(os.path.abspath(p)), root)
        if not rel.startswith(".."):  # inside the mirror (its links): as in the project
            return rel
        root = str(mirrored)  # another project's file (an import): as the project itself sees it
    lexical = os.path.normpath(os.path.abspath(p))
    real = os.path.realpath(p)
    if os.path.realpath(lexical) != real:
        lexical = real
        root = os.path.realpath(root)
    rel = os.path.relpath(lexical, root)
    # a path climbing out of a project reached through a symlink (/tmp -> /private/tmp): the OS
    # follows the link before the `..`, so check what it opens, and use the real paths if it differs
    if rel.startswith("..") and os.path.realpath(os.path.join(root, rel)) != real:
        rel = os.path.relpath(real, os.path.realpath(root))
        if os.path.realpath(os.path.join(root, rel)) != real:
            rel = real
    return rel


def _clean(x):
    """A composed pose component: 12 significant digits, float noise (6e-17) snapped to 0."""
    return 0.0 if abs(x) < 1e-12 else float(f"{x:.12g}")


def selected_outputs(outputs=None):
    """Which `output` blocks to place: the given names, else $ORB_OUTPUT (comma separated), else all."""
    if outputs is None:
        env = os.environ.get("ORB_OUTPUT", "").strip()
        outputs = [o.strip() for o in env.split(",") if o.strip()] if env else None
    return None if outputs is None else list(outputs)


class World:
    def __init__(self, blocks, root, outputs=None):
        # lexical, not resolve(): symlinks inside the project (the editor's unsaved-edit mirror
        # links meshes/ and blobs/ to the real project) must not change the paths the IR records
        self.root = Path(os.path.abspath(root))
        self.blocks = blocks
        self.ir = {"orb": IR_VERSION, "units": "SI (m, kg, s, rad); quat = [w, x, y, z]",
                   "root": str(self.root), "parts": {}, "assemblies": {}, "sites": {}, "items": {},
                   "lifecycles": {}, "processes": {}, "seams": {}, "other": {},
                   "timelines": {}, "cameras": {}, "shots": [], "geo": {},
                   "includes": list(getattr(blocks, "includes", None) or []), "sources": {}}
        self.warnings = []
        self.item_blocks = {}
        self.outputs = selected_outputs(outputs)
        self.defs = {}  # (type, label) -> the Block compiled under that label (imported ones on demand)
        self.seam_names = set()
        self.placed_sites = {}  # a site block's label -> the labels of its placed copies
        self.scene_motion = {}  # timeline name -> [(item, scenes.LeafMotion)]: animated scenes, lowered by timeline()

    def build(self):
        seams = {"mujoco", "isaac", "glb", "usd", "minecraft", "step"} | {b.label for b in self.blocks if b.type == "seam"}
        self.seam_names = seams
        for b in list(self.blocks) + list(getattr(self.blocks, "natives", None) or []):
            self.scope_attrs(b, seams)
        if self.outputs is not None:
            have = {b.label for b in self.blocks if b.type == "output"}
            for o in self.outputs:
                if o not in have:
                    raise OrbError(f"no output {o!r} (have: {', '.join(sorted(have)) or 'none'})")
        if any(b.type == "output" for b in self.blocks):
            self.ir["outputs"] = {}
        phases = [{"part": self.part, "site": self.site, "lifecycle": self.lifecycle, "seam": self.seam,
                   "source": self.source}, {"assembly": self.assembly, "geo": self.geo}, {"process": self.process},
                  {"item": self.item}, {"output": self.output}, {"camera": self.camera},
                  {"timeline": self.timeline}, {"shot": self.shot}]
        known = {t for p in phases for t in p}
        for b in self.blocks:
            self.plain_values(b, known)
        for handlers in phases:  # later phases reference earlier ones
            for b in self.blocks:
                if b.type in handlers:
                    handlers[b.type](b)
        for b in self.blocks:
            if b.type not in known:
                self.ir["other"].setdefault(b.type, {})[b.label] = generic(b)
        for nb in getattr(self.blocks, "natives", None) or []:
            self.top_native(nb)
        for nb in getattr(self.blocks, "imported_natives", None) or []:
            self.imported_native(nb)
        for sec in ("items", "sites"):  # what each output placed (a decompiled world says it per entry)
            for name, d in self.ir[sec].items():
                o = (d.get("scene") or {}).get("output") if isinstance(d.get("scene"), dict) else None
                if o in self.ir.get("outputs", {}):
                    self.ir["outputs"][o][sec].append(name)
        for rec in self.ir.get("outputs", {}).values():  # sorted: the same whatever order blocks come in
            rec["items"].sort()
            rec["sites"].sort()
        if self.outputs is not None:  # every output compiled (references checked), then the others left out
            self.ir = select_outputs(self.ir, self.outputs)
        return self.ir

    def plain_values(self, b, known, top=True):
        """Attributes hold data: a scene or a function in one is a mistake (only an output's
        `value` places a scene) the world graph could not even write down. A misspelt top-level block
        type is a warning: unknown types pass through for third-party seams."""
        import difflib

        if top and b.type not in known:
            near = difflib.get_close_matches(b.type, sorted(known), 1, 0.75)
            if near:
                self.warnings.append(f"{b.pos}: {b.type} {b.label!r} is not a block Orb knows (did you mean "
                                     f"{near[0]}?); it passes through under other.{b.type}")

        def bad(v):
            if isinstance(v, (list, tuple)):
                return next((x for x in map(bad, v) if x is not None), None)
            if isinstance(v, dict):
                return next((x for x in map(bad, v.values()) if x is not None), None)
            t = type_name(v)
            return v if t == "scene" or t.startswith("fn ") or t.startswith("project ") else None

        for k, v in b.attrs.items():
            if top and b.type == "output" and k == "value":
                continue
            found = bad(v)
            if found is None:
                continue
            t = type_name(found)
            if t == "scene" and k in ("to", "follow", "of", "installed_in"):
                hint = f'{k} names a block: item["<output>/<instance path>"] for what an output placed'
            elif t == "scene":
                hint = ("only an output's `value = ...` places a scene" +
                        (" (is this block meant to be an output?)" if top and b.type not in known else ""))
            elif t.startswith("fn "):
                hint = "call it: a function value cannot be stored in the world"
            else:
                hint = "name one of its blocks (alias.part.x, alias.site.y, ...)"
            what = "a scene" if t == "scene" else f"the {'function' if t.startswith('fn ') else 'project'} {t.split(' ', 1)[1]}"
            raise OrbError(f"{b.type} {b.label}: {k} is {what}; {hint}", b.attr_pos.get(k, b.pos))
        for c in b.children:
            self.plain_values(c, known, top=False)

    # ------------------------------------------------------------------ only / skip, natives
    def scope_attrs(self, blk, seams, parent=None):
        """`only` / `skip` on any block: a list of seam names (a single string is a list of one);
        not both; not on the steps of something (keys, paths, detach, show, hide, set, natives)."""
        for k in SCOPE:
            if k not in blk.attrs:
                continue
            v, pos = blk.attrs[k], blk.attr_pos[k]
            if blk.type in NO_SCOPE and (parent is not None or blk.type == "native"):
                raise OrbError(f"`{k}` cannot go on a {blk.type} block: put it on the "
                               f"{'block that carries the native' if blk.type == 'native' else parent.type}", pos)
            v = [v] if isinstance(v, str) else v
            if not isinstance(v, list) or not v or not all(isinstance(x, str) for x in v):
                raise OrbError(f'{k} must be a list of seam names, e.g. {k} = ["isaac"]', pos)
            for x in v:
                if x not in seams:
                    self.warnings.append(f"{pos}: {k} names seam {x!r}, which is neither built in nor declared")
            blk.attrs[k] = v
        if all(k in blk.attrs for k in SCOPE):
            raise OrbError("a block takes `only` or `skip`, not both", blk.attr_pos["skip"])
        for c in blk.children:
            self.scope_attrs(c, seams, blk)

    @staticmethod
    def scope_of(blk):
        return {k: blk.attrs[k] for k in SCOPE if k in blk.attrs}

    def native_entry(self, nb):
        """One `native "<seam>" { ... }` block -> (seam, entry). Keys stay as written; values in SI;
        file keys (usd, mjcf, glb, schematic, mesh, ...) resolved against the declaring file and
        stored relative to the project root."""
        seam = nb.label
        if not seam:
            raise OrbError('native needs the seam it is for: native "isaac" { usd = "..." }', nb.pos)
        e = {}
        for k, v in nb.attrs.items():
            if k == "target":
                continue
            if k in NATIVE_FILES and isinstance(v, str):
                e[k] = self.file(nb, k)
            elif k == "at":
                e[k] = vec(nb, "at", L, 3)
            elif k == "rpy":
                e[k] = vec(nb, "rpy", ANG, 3)
            elif k == "yaw":
                e[k] = num(nb, "yaw", ANG)
            elif k == "quat":
                e[k] = rot(nb)
            elif k == "scale":
                e[k] = vec(nb, "scale", DIMLESS, 3) if isinstance(v, list) else num(nb, "scale")
            else:
                e[k] = jsonify(v)
        if "replace" in e and e["replace"] not in REPLACE:
            raise OrbError(f"replace is one of {' | '.join(REPLACE)}, got {e['replace']!r}", nb.attr_pos["replace"])
        return seam, e

    def natives(self, blk, d):
        """The `native "<seam>"` children of blk into d["native"] = {seam: entry}."""
        for nb in blk.kids("native"):
            seam, e = self.native_entry(nb)
            nat = d.setdefault("native", {})
            if seam in nat:
                raise OrbError(f"{blk.path} has two natives for {seam}", nb.pos)
            nat[seam] = e
        return d

    def ir_block(self, tgt, pos):
        """The IR dict of a block reference (a native's `target`)."""
        parts = tgt.path.split(".")
        try:
            if tgt.type == "item":
                return self.ir["items"][tgt.label]
            if tgt.type == "part":
                return self.ir["parts"][tgt.label]
            if tgt.type == "assembly":
                return self.ir["assemblies"][tgt.label]
            if tgt.type == "site":
                return self.ir["sites"][tgt.label]
            if tgt.type == "fixture" and parts[0] == "site":
                return self.ir["sites"][parts[1]]["fixtures"][tgt.label]
            if tgt.type == "fixture" and parts[0] == "geo":  # a geo block's context fixture
                return self.ir["sites"][self.ir["geo"][parts[1]]["site"]]["fixtures"][tgt.label]
            if tgt.type == "link" and parts[0] == "assembly":
                return next(l for l in self.ir["assemblies"][parts[1]]["links"] if l["name"] == tgt.label)
        except (KeyError, StopIteration):
            pass
        raise OrbError(f"a native's target is a {', '.join(NATIVE_ON[:-1])} or {NATIVE_ON[-1]}, got {tgt.path}", pos)

    def top_native(self, nb):
        """A top-level `native "<seam>" { target = ... }`: merges into that block's native map."""
        tgt = nb.attrs.get("target")
        if not isinstance(tgt, Block):
            raise OrbError('a top-level native needs `target = item["x"]` (or part.x, assembly.x, site.s.fixture.f)',
                           nb.attr_pos.get("target", nb.pos))
        d = self.ir_block(tgt, nb.attr_pos["target"])
        seam, e = self.native_entry(nb)
        nat = d.setdefault("native", {})
        if seam in nat:
            raise OrbError(f"{tgt.path} already has a native for {seam}", nb.pos)
        nat[seam] = e

    def imported_native(self, nb):
        """A top-level native of an imported project: it goes with its target wherever this world has
        it (a part or assembly a scene uses, each placed copy of a site); an item of that project
        is never in this world, so its natives are not either."""
        tgt = nb.attrs.get("target")
        if not isinstance(tgt, Block):
            raise OrbError('a top-level native needs `target = item["x"]` (or part.x, assembly.x, site.s.fixture.f)',
                           nb.attr_pos.get("target", nb.pos))
        if tgt.type in ("part", "assembly", "site", "item"):
            owner = (tgt.type, tgt.label)
        else:  # a child (site.s.fixture.f, assembly.a.link.l): labels may hold dots, so match known owners
            known = list(self.defs) + [("site", lab) for lab in self.placed_sites]
            hits = [k for k in known if tgt.path.startswith(f"{k[0]}.{k[1]}.")]
            parts = tgt.path.split(".")
            owner = max(hits, key=lambda k: len(k[1])) if hits else (parts[0], parts[1])
        if owner[0] in ("part", "assembly"):
            if self.defs.get(owner) is None:
                return
            dicts = [self.ir_block(tgt, nb.attr_pos["target"])]
        elif owner[0] == "site":
            placed = self.placed_sites.get(owner[1], [])
            dicts = [self.ir["sites"][lab] if tgt.type == "site" else self.ir["sites"][lab]["fixtures"][tgt.label]
                     for lab in placed]
        else:
            return
        if not dicts:  # nothing of it in this world: its asset need not exist here
            return
        seam, e = self.native_entry(nb)
        for d in dicts:
            nat = d.setdefault("native", {})
            if seam in nat:
                raise OrbError(f"{tgt.path} already has a native for {seam}", nb.pos)
            nat[seam] = dict(e)

    def file(self, blk, key):
        f = blk.attrs.get(key)
        if not isinstance(f, str):
            raise OrbError(f"{blk.path} needs `{key}` (a path relative to the .orb file)", blk.pos)
        raw = Path(os.path.abspath(blk.attr_pos[key].file)).parent / f
        if not raw.exists():  # the editor unsaved-edit mirror holds the project files only: a path into a
            # sibling library (../shared-library/...) is where the mirrored project has it
            from .lang import import_base
            alt = import_base(blk.attr_pos[key].file) / f
            if alt.exists():
                raw = alt
        if not raw.exists():
            raise OrbError(f"file not found: {raw.resolve()}", blk.attr_pos[key])
        return project_path(raw, self.root)

    def geom(self, gb, role):
        shape = gb.label
        if shape not in SHAPES:
            raise OrbError(f"unknown shape {shape!r} (use one of {', '.join(SHAPES)})", gb.pos)
        g = {"role": role, "shape": shape, "pos": vec(gb, "at", L, 3, [0.0] * 3), "quat": rot(gb)}
        if shape == "box":
            g["size"] = vec(gb, "size", L, 3)
        elif shape == "sphere":
            g["radius"] = num(gb, "radius", L)
        elif shape in ("cylinder", "capsule"):
            g["radius"], g["length"] = num(gb, "radius", L), num(gb, "length", L)
        else:
            g["file"] = self.file(gb, "file")
            g["scale"] = vec(gb, "scale", DIMLESS, 3, [1.0] * 3)
        if (c := color(gb)) is not None:
            g["color"] = c
        if "friction" in gb.attrs:
            g["friction"] = num(gb, "friction")
        g.update(self.scope_of(gb))
        return g

    def geoms(self, blk):
        return [self.geom(c, c.type) for c in blk.children if c.type in ("visual", "collision")]

    def inertial(self, blk, fallback=None):
        src = blk if "mass" in blk.attrs else fallback
        if src is None or "mass" not in src.attrs:
            return None
        get = lambda k, d, n, dflt: (vec(blk, k, d, n) if k in blk.attrs else vec(src, k, d, n, dflt))
        mass = num(src, "mass", M)
        if mass <= 0:
            raise OrbError("mass must be positive", src.attr_pos["mass"])
        inertia = get("inertia", INERTIA, 6, None)
        if inertia is None:  # solid sphere guess; flagged so nobody mistakes it for CAD data
            inertia = [mass * 0.004] * 3 + [0.0] * 3
            self.warnings.append(f"{blk.path}: no inertia given, using a placeholder")
        d, _ = principal(inertia)
        if min(d) <= 0 or d[0] + d[1] < d[2] * (1 - 1e-6) or d[1] + d[2] < d[0] * (1 - 1e-6) or d[0] + d[2] < d[1] * (1 - 1e-6):
            raise OrbError(f"{blk.path}: inertia {inertia} is not physically valid (triangle inequality)", blk.pos)
        return {"mass": mass, "com": get("com", L, 3, [0.0] * 3), "inertia": inertia}

    def part(self, b):
        self.defs[("part", b.label)] = b
        p = {k: jsonify(v) for k, v in b.attrs.items() if k not in ("mass", "com", "inertia")}
        p["inertial"] = self.inertial(b)
        p["geoms"] = self.geoms(b)
        self.ir["parts"][b.label] = self.natives(b, p)

    def joint(self, lb, is_root, fixed_base):
        js = lb.kids("joint")
        if len(js) > 1:
            raise OrbError(f"link {lb.label} has more than one joint", js[1].pos)
        if not js:
            return {"name": f"{lb.label}_joint", "type": "free" if is_root and not fixed_base else "fixed"}
        jb = js[0]
        jt = jb.attrs.get("type", "revolute")
        if jt not in JOINTS:
            raise OrbError(f"unknown joint type {jt!r} (use one of {', '.join(JOINTS)})", jb.pos)
        if jt == "free" and not is_root:
            raise OrbError("only the root link may have a free joint", jb.pos)
        j = {"name": jb.label or f"{lb.label}_joint", "type": jt}
        if jt not in ("revolute", "prismatic"):
            return j
        lin = jt == "prismatic"
        axis = vec(jb, "axis", DIMLESS, 3)
        n = math.sqrt(sum(x * x for x in axis))
        if n == 0:
            raise OrbError("axis must be non-zero", jb.attr_pos["axis"])
        j["axis"] = [x / n for x in axis]
        j["range"] = vec(jb, "range", L if lin else ANG, 2, None)
        if j["range"] and j["range"][0] >= j["range"][1]:
            raise OrbError("range must be [lower, upper] with lower < upper", jb.attr_pos["range"])
        j["initial"] = num(jb, "initial", L if lin else ANG, 0.0)
        mimic = jb.attrs.get("mimic")
        if mimic is None:
            for k in ("multiplier", "offset"):
                if k in jb.attrs:
                    raise OrbError(f"{k} belongs to a mimic joint: add mimic = \"<leader joint>\" or drop {k}",
                                   jb.attr_pos[k])
        if mimic is not None and (not isinstance(mimic, str) or not mimic):
            raise OrbError(f"mimic must name a joint of this assembly in quotes (mimic = \"finger_joint1\"), "
                           f"got {type_name(mimic)}", jb.attr_pos["mimic"])
        if mimic is not None:  # checked against the assembly's joints, initial derived: assembly()
            for k in ("kp", "kd"):
                if k in jb.attrs:
                    raise OrbError(f"joint {j['name']} mimics {mimic} and has no drive of its own: drop {k} "
                                   f"(the leader's drive moves it)", jb.attr_pos[k])
        elif j["range"] and not j["range"][0] <= j["initial"] <= j["range"][1]:
            raise OrbError(f"initial {j['initial']:.4g} is outside range {j['range']}", jb.attr_pos["initial"])
        j["effort"] = num(jb, "effort", FORCE if lin else TORQUE, None)
        j["velocity"] = num(jb, "velocity", VEL if lin else ANGVEL, None)
        # dynamics gains: proper units (Nm/rad, Nm*s/rad, ...) or bare numbers read as SI
        stiff, damp = ((1, 0, -2, 0, 0, 0), (1, 0, -1, 0, 0, 0)) if lin else ((1, 2, -2, 0, 0, -1), (1, 2, -1, 0, 0, -1))
        dims = {"damping": damp, "armature": M if lin else INERTIA, "friction": FORCE if lin else TORQUE}
        for k, d in dims.items():
            j[k] = num(jb, k, d, 0.0, lenient=True)
        if "kp" in jb.attrs:
            j["drive"] = {"mode": "position", "kp": num(jb, "kp", stiff, lenient=True),
                          "kd": num(jb, "kd", damp, 0.0, lenient=True)}
        if mimic is not None:
            j["mimic"] = {"joint": mimic, "multiplier": num(jb, "multiplier", DIMLESS, 1.0),
                          "offset": num(jb, "offset", L if lin else ANG, 0.0)}
        return j

    def resolve_mimics(self, b, links, jblocks):
        """Check every mimic joint against its assembly and derive its `initial` from the leader's."""
        byname = {l["joint"]["name"]: l["joint"] for l in links}
        movable = [n for n, j in byname.items() if j["type"] in ("revolute", "prismatic")]
        for name, j in byname.items():
            mm = j.get("mimic")
            if not mm:
                continue
            jb = jblocks[name]
            at, lead = jb.attr_pos["mimic"], mm["joint"]
            if lead == name:
                raise OrbError(f"joint {name} cannot mimic itself", at)
            lj = byname.get(lead)
            if lj is None:
                raise OrbError(f"joint {name}: mimic {lead!r} is not a joint of assembly {b.label}"
                               f"{did_you_mean(lead, movable)} (movable joints: {', '.join(movable) or 'none'})", at)
            if lj["type"] not in ("revolute", "prismatic"):
                raise OrbError(f"joint {name}: mimic {lead!r} is a {lj['type']} joint; a mimic joint follows a "
                               f"revolute or prismatic one", at)
            if lj.get("mimic"):
                raise OrbError(f"joint {name}: mimic {lead!r} is itself a mimic joint (it follows "
                               f"{lj['mimic']['joint']}); mimic {lj['mimic']['joint']} directly", at)
            lin = j["type"] == "prismatic"
            fmt = (lambda a: f"{a:.4g} m") if lin else (lambda a: f"{math.degrees(a):.4g} deg")
            v = mm["multiplier"] * lj["initial"] + mm["offset"]
            if "initial" in jb.attrs and abs(j["initial"] - v) > 1e-9:
                raise OrbError(f"joint {name}: initial {fmt(j['initial'])} contradicts its mimic: {name} = "
                               f"{mm['multiplier']:g} * {lead} + offset = {fmt(v)}; drop initial (a mimic joint's "
                               f"is derived)", jb.attr_pos["initial"])
            r = j.get("range")
            if r and not r[0] - 1e-9 <= v <= r[1] + 1e-9:
                raise OrbError(f"joint {name}: its initial value {fmt(v)} (= {mm['multiplier']:g} * {lead}.initial + "
                               f"offset) is outside its range [{fmt(r[0])}, {fmt(r[1])}]", at)
            j["initial"] = v
            free = [n for n, x in ((name, j), (lead, lj)) if x["type"] == "revolute" and not x.get("range")]
            if free:  # no limit is injected: Isaac just won't hold it
                self.warnings.append(f"{at}: joint {name} mimics {lead} with unlimited revolute joint(s) "
                                     f"{' and '.join(free)}: Isaac's PhysX refuses a mimic between joints unless "
                                     f"every revolute one has a finite range, so there it does not follow {lead} "
                                     f"(MuJoCo and every other seam do)")

    def assembly(self, b):
        self.defs[("assembly", b.label)] = b

        def imported_parts(lb):  # parts compile first, so a link's part not compiled yet is imported
            p = lb.attrs.get("part")
            if isinstance(p, Block) and p.type == "part" and ("part", p.label) not in self.defs:
                self.ensure_def(p, lb.attr_pos.get("part", lb.pos))
            for c in lb.kids("link"):
                imported_parts(c)
        for root in b.kids("link"):
            imported_parts(root)
        roots = b.kids("link")
        if len(roots) != 1:
            raise OrbError(f"assembly {b.label} needs exactly one root link, has {len(roots)}", b.pos)
        fixed_base = bool(b.attrs.get("fixed_base", False))
        links, seen, jblocks = [], {}, {}

        def walk(lb, parent):
            if lb.label in seen:
                raise OrbError(f"duplicate link {lb.label!r} (first at {seen[lb.label]})", lb.pos)
            seen[lb.label] = lb.pos
            part = ref(lb, "part", ("part",))
            link = {"name": lb.label, "parent": parent, "part": part.label if part else None,
                    "pos": vec(lb, "at", L, 3, [0.0] * 3), "quat": rot(lb),
                    "inertial": self.inertial(lb, part),
                    "joint": self.joint(lb, parent is None, fixed_base),
                    "geoms": (self.geoms(part) if part else []) + self.geoms(lb),
                    "sensors": [{"name": s.label, "type": s.attrs.get("type", s.label),
                                 "pos": vec(s, "at", L, 3, [0.0] * 3), "quat": rot(s), **self.scope_of(s)}
                                for s in lb.kids("sensor")]}
            if lb.kids("joint"):
                jblocks[link["joint"]["name"]] = lb.kids("joint")[0]
            link.update(self.scope_of(lb))
            self.natives(lb, link)
            if link["inertial"] is None and link["joint"]["type"] != "fixed":
                raise OrbError(f"link {lb.label} moves but has no mass (set mass or part)", lb.pos)
            links.append(link)
            for c in lb.kids("link"):
                walk(c, lb.label)

        walk(roots[0], None)
        joints = [l["joint"]["name"] for l in links]
        if len(set(joints)) != len(joints):
            raise OrbError(f"assembly {b.label} has duplicate joint names", b.pos)
        self.resolve_mimics(b, links, jblocks)
        asm = {k: jsonify(v) for k, v in b.attrs.items()}
        asm.update(root=links[0]["name"], links=links,
                   mass=sum(l["inertial"]["mass"] for l in links if l["inertial"]))
        self.ir["assemblies"][b.label] = self.natives(b, asm)

    def site(self, b):
        self.ir["sites"][b.label] = self.site_ir(b)

    def site_ir(self, b):
        s = {k: jsonify(v) for k, v in b.attrs.items() if k not in ("floor", "at")}
        s["pos"] = vec(b, "at", L, 3, [0.0] * 3)
        s["floor"] = vec(b, "floor", L, 2, None)
        s["zones"] = {}
        for z in b.kids("zone"):
            s["zones"][z.label] = {**{k: jsonify(v) for k, v in z.attrs.items() if k not in ("at", "size")},
                                   "pos": [a + c for a, c in zip(s["pos"], vec(z, "at", L, 3, [0.0] * 3))],
                                   "size": vec(z, "size", L, len(z.attrs.get("size") or [0, 0]), None),
                                   "color": color(z)}
        s["fixtures"] = {}
        for f in b.kids("fixture"):
            s["fixtures"][f.label] = self.fixture(f, s["pos"])
        return self.natives(b, s)

    def fixture(self, f, base, turn=None):
        """A fixture block -> its IR dict, posed at `base` + its `at` (turned by `turn`, a quat)."""
        at, q = vec(f, "at", L, 3, [0.0] * 3), rot(f)
        if turn is not None:
            at, q = qrot(turn, at), qmul(turn, q)
        fx = {"pos": [a + c for a, c in zip(base, at)], "quat": q, "geoms": self.geoms(f)}
        # anything else (source/provenance, labels, owner...) passes through for seams
        extra = {k: jsonify(v) for k, v in f.attrs.items() if k not in POSE + SCOPE}
        fx.update(self.scope_of(f))
        self.natives(f, fx)
        kids = [c for c in f.children if c.type not in ("visual", "collision", "native")]
        if kids:
            extra["children"] = [{"type": c.type, "name": c.label, **generic(c)} for c in kids]
        if extra:
            fx["attrs"] = extra
        return fx

    def geo(self, b):
        """`geo "x" { site, lat, lon, at, north, elevation, fixture ... }`: where a site is on the Earth,
        and the real-world context around it (orb geo fetch). Its fixtures are placed in the geo frame
        (X east, Y north, Z up, origin at lat/lon and `elevation`) and join the site's fixtures, so
        every seam draws them like any other fixture. docs/geo.md"""
        site = ref(b, "site", ("site",))
        if site is None:
            if len(self.ir["sites"]) != 1:
                raise OrbError(f"geo {b.label} needs `site = site[\"...\"]` (the project has "
                               f"{len(self.ir['sites'])} sites)", b.pos)
            sname = next(iter(self.ir["sites"]))
        else:
            sname = site.label
        s = self.ir["sites"][sname]

        def degrees(k, lo, hi):
            if k not in b.attrs:
                raise OrbError(f"geo {b.label} needs `{k}` (WGS84 degrees)", b.pos)
            v, pos = b.attrs[k], b.attr_pos[k]
            if isinstance(v, Q) and v.d == ANG:
                v = math.degrees(v.v)
            elif isinstance(v, Q) or not isinstance(v, (int, float)) or isinstance(v, bool):
                raise OrbError(f"{k} is in degrees: a bare number (30.3614) or an angle (30.3614 deg)", pos)
            if not lo <= v <= hi:
                raise OrbError(f"{k} = {v} is outside [{lo}, {hi}]", pos)
            return float(v)

        north = num(b, "north", ANG) if "north" in b.attrs else float(s.get("north") or 0.0)
        if not isinstance(north, (int, float)):
            raise OrbError(f"geo {b.label}: north must be an angle", b.pos)
        g = {k: jsonify(v) for k, v in b.attrs.items() if k not in ("site", "lat", "lon", "at", "north", "elevation") + SCOPE}
        g.update(site=sname, lat=degrees("lat", -90, 90), lon=degrees("lon", -180, 180),
                 at=vec(b, "at", L, 3, [s["pos"][0], s["pos"][1], s["pos"][2]]),
                 north=north, elevation=num(b, "elevation", L, 0.0), fixtures=[])
        g.update(self.scope_of(b))
        turn = [math.cos(north / 2), 0.0, 0.0, math.sin(north / 2)]
        for f in b.kids("fixture"):
            if f.label in s["fixtures"]:
                raise OrbError(f"geo {b.label}: site {sname} already has a fixture {f.label!r}", f.pos)
            fx = self.fixture(f, g["at"], turn)
            if not any(k in fx for k in SCOPE):  # the geo block's only/skip reaches its fixtures
                fx.update(self.scope_of(b))
            fx.setdefault("attrs", {})["geo"] = b.label
            s["fixtures"][f.label] = fx
            g["fixtures"].append(f.label)
        self.ir["geo"][b.label] = g

    def lifecycle(self, b):
        self.defs[("lifecycle", b.label)] = b
        stages = b.attrs.get("stages")
        if not isinstance(stages, list) or not stages or not all(isinstance(x, str) for x in stages):
            raise OrbError("lifecycle needs `stages = [\"...\", ...]`", b.pos)
        moves = [[a, c] for a, c in zip(stages, stages[1:])]
        for pair in b.attrs.get("allow", []):
            if not (isinstance(pair, list) and len(pair) == 2 and all(p in stages for p in pair)):
                raise OrbError(f"allow entries are [from, to] stage pairs, got {pair!r}", b.attr_pos["allow"])
            moves.append(pair)
        self.ir["lifecycles"][b.label] = {**generic(b), "stages": stages, "transitions": moves}

    def process(self, b):
        p = {k: jsonify(v) for k, v in b.attrs.items()}
        p["fidelity"] = b.attrs.get("fidelity", "logical")
        if p["fidelity"] not in ("logical", "kinematic", "physics"):
            raise OrbError("fidelity is one of logical | kinematic | physics", b.attr_pos.get("fidelity", b.pos))
        asm = ref(b, "assembly", ("assembly",))
        links = {l["name"] for l in self.ir["assemblies"][asm.label]["links"]} if asm else set()
        st = ref(b, "station", ("zone",))
        if st:
            p["station_pos"] = self.ir["sites"][st.path.split(".")[1]]["zones"][st.label]["pos"]
        t, steps = 0.0, []
        for s in b.kids("step"):
            dur = num(s, "takes", T, 0.0)
            for name in s.attrs.get("attach", []):
                if name not in links:
                    raise OrbError(f"step {s.label} attaches {name!r}, not a link of the process assembly",
                                   s.attr_pos["attach"])
            steps.append({"name": s.label, "start": t, "takes": dur, **generic(s)})
            t += dur
        p["steps"], p["duration"] = steps, t
        self.ir["processes"][b.label] = p

    def seam(self, b):
        self.ir["seams"][b.label] = generic(b)

    # ------------------------------------------------------------------ outputs: scenes into the world
    def output(self, b):
        """`output "<name>" { value = <scene> }`: where a scene value becomes part of the world. Its
        parts and assemblies become items, its sites sites, labelled by instance path
        (`<prefix>/<id>/...`, prefix defaults to the output's name) and posed in the world frame."""
        from .scenes import Scene, flatten, leaf_motion

        for k in b.attrs:
            if k not in ("value", "prefix", "label", "timeline") + SCOPE:
                raise OrbError(f"output has no attribute {k!r}{did_you_mean(k, ('value', 'prefix', 'label', 'timeline'))}"
                               " (use value, prefix, label, timeline, only, skip)", b.attr_pos[k])
        v = b.attrs.get("value")
        if not isinstance(v, Scene):
            hint = (" (wrap a list in compose([...]))" if isinstance(v, list) else
                    " (place a block with instance(...))" if isinstance(v, Block) else "")
            raise OrbError(f"output {b.label} needs `value = <a scene>` (from instance(), compose() or a fn "
                           f"returning one), got {type_name(v) if 'value' in b.attrs else 'nothing'}{hint}",
                           b.attr_pos.get("value", b.pos))
        prefix = b.attrs.get("prefix", b.label)
        if not isinstance(prefix, str) or "." in prefix:
            raise OrbError('prefix is a string without "." ("" for none)', b.attr_pos.get("prefix", b.pos))
        rec = {"label": str(b.attrs.get("label", v.label or b.label)), "prefix": prefix, "items": [], "sites": []}
        tlb = ref(b, "timeline", ("timeline",))
        moving = []
        for leaf in flatten(v, prefix, inherited=self.scope_of(b) or None):
            m = leaf_motion(leaf)  # (a site in a moving group is an error)
            if leaf.node.kind == "site":
                self.place_site(leaf, b)
            elif m.any():
                moving.append((self.place_item(leaf, b), m))
            else:
                self.place_item(leaf, b)
        if moving:
            self.scene_motion.setdefault(self.output_timeline(b, tlb, moving), []).extend(moving)
        self.ir["outputs"][b.label] = rec  # items / sites: filled in by build() from their `scene` records

    def output_timeline(self, b, tlb, moving):
        """The timeline an animated scene plays in: the output's `timeline = timeline.x`, else the
        project's only one."""
        if tlb is not None:
            return tlb.label
        names = [x.label for x in self.blocks if x.type == "timeline"]
        item, m = moving[0]
        vis = m.appear or m.vanish
        src = m.spans[0][:2] if m.spans else (vis[1], vis[2]) if vis else (item, m.at)
        where = f"instance {src[0]} at {src[1]}" if src[1] else f"instance {src[0]}"
        if not names:
            raise OrbError(f"output {b.label}: the scene moves ({where}) but the project has no timeline to play it in; "
                           f'declare one: timeline "film" {{ duration = 30 s }}', b.pos)
        if len(names) > 1:
            raise OrbError(f"output {b.label}: the scene moves ({where}) and the project has {len(names)} timelines "
                           f"({', '.join(names)}); say which one plays it: timeline = timeline.{names[0]}", b.pos)
        return names[0]

    def scene_timeline(self, name, dur, tl, written):
        """Lower the animated scenes that play in this timeline (output() collected them): each moving
        leaf gets a track (world keys, composed through its groups), a joints track and visibility
        steps, exactly what hand-written blocks would give. written: {(kind, item): pos} of the
        timeline's own track / joints / show-hide blocks (one item cannot have both)."""
        from .scenes import TIME

        for item, m in self.scene_motion.get(name, ()):
            for path, at, t0, t1, d, what in m.spans:
                if t0 < -1e-9 or t1 > dur + 1e-9:
                    dl = f" (its delay of {d:g} s included)" if d else ""
                    raise OrbError(f"instance {path}: its {what} runs from {t0:g} s to {t1:g} s{dl}, outside timeline "
                                   f"{name}'s [0, {dur:g}] s", at)
            for key, rec in (("appear", m.appear), ("vanish", m.vanish)):
                if rec is not None and rec[0] > dur + 1e-9:
                    raise OrbError(f"instance {rec[1]}: {key} at {rec[0]:g} s is after the end of timeline {name} "
                                   f"({dur:g} s)", rec[2])
            for kind, have, what in (("tracks", m.track, "track"), ("joints", m.joints, "joints"),
                                     ("visible", m.visible, "show/hide")):
                if have and (kind, item) in written:
                    attr = {"tracks": "motion", "joints": "joint_motion", "visible": "appear/vanish"}[kind]
                    raise OrbError(f"{item} is animated twice in timeline {name}: by its {what} block at "
                                   f"{written[kind, item]} and by its scene's {attr} at {m.at}; keep one",
                                   written[kind, item])
            if m.track:
                tl["tracks"][item] = m.track
            if m.joints:
                c = Block("joints", item, f"joints.{item}", m.at)
                for i, (t, obj) in enumerate(m.joints):
                    k = Block("key", "", f"joint_motion[{i}]", m.at)
                    k.attrs = {**obj, "t": Q(t, TIME)}
                    k.attr_pos = {kk: m.at for kk in k.attrs}
                    c.children.append(k)
                try:
                    tl["joints"][item] = self.joint_track(c, name, dur)[1]
                except OrbError as e:
                    raise OrbError(f"instance {item}: joint_motion: {e.msg}", e.pos) from None
            if m.visible:
                tl["visible"][item] = m.visible

    def ensure_def(self, blk, at):
        """A part / assembly / lifecycle a scene uses: compiled under its label unless it already is (an
        imported project's definitions are compiled only when a scene places them)."""
        key = (blk.type, blk.label)
        have = self.defs.get(key)
        if have is blk:
            return
        if have is not None:
            raise OrbError(f"two different {blk.type}s are labelled {blk.label!r} ({have.pos} and {blk.pos})", at)
        self.scope_attrs(blk, self.seam_names)
        if blk.type == "assembly":  # its links' parts are in the world too (the BOM; decompile links them)
            def walk(lb):
                p = lb.attrs.get("part")
                if isinstance(p, Block) and p.type == "part":
                    self.ensure_def(p, at)
                for c in lb.kids("link"):
                    walk(c)
            for root in blk.kids("link"):
                walk(root)
        {"part": self.part, "assembly": self.assembly, "lifecycle": self.lifecycle}[blk.type](blk)

    def scene_record(self, out, leaf):
        t, at = leaf.node.target, leaf.node.at
        where = f"{project_path(at.file, self.root)}:{at.line}:{at.col}" if at else None
        return {"output": out.label, "path": leaf.path, "of": f"{t.type}.{t.label}", "at": where}

    def place_item(self, leaf, out):
        n, label = leaf.node, leaf.path
        if label in self.ir["items"]:
            raise OrbError(f"output {out.label}: there is already an item {label!r}; give the output another "
                           f"prefix or the instance another id", n.at)
        for k in ("location", "installed_in", "made_by"):
            if k in leaf.attrs:
                raise OrbError(f"instance {label}: `{k}` does not apply to a placed instance (the scene places it)", n.at)
        self.ensure_def(n.target, n.at)
        lc = leaf.attrs.get("lifecycle")
        if isinstance(lc, Block) and lc.type == "lifecycle":
            self.ensure_def(lc, n.at)
        blk = Block("item", label, f"item.{label}", n.at)
        blk.attrs = {"of": n.target, **leaf.attrs}
        blk.attr_pos = {k: n.at for k in blk.attrs}
        self.scope_attrs(blk, self.seam_names)
        self.item(blk)
        it = self.ir["items"][label]
        it.update(location=f"output.{out.label}", pos=[_clean(x) for x in leaf.pos],
                  quat=[_clean(x) for x in leaf.quat], scene=self.scene_record(out, leaf))
        return label

    def place_site(self, leaf, out):
        """A site instance: the site's floor, zones and fixtures moved into the instance frame. A turned
        site carries `quat` (zones and the floor are drawn turned with it)."""
        n, label = leaf.node, leaf.path
        if label in self.ir["sites"]:
            raise OrbError(f"output {out.label}: there is already a site {label!r}", n.at)
        bad = [k for k in leaf.attrs if k in ("floor", "at", "yaw", "rpy", "quat")]
        if bad:
            raise OrbError(f"instance {label}: set {bad[0]} on the site itself (the instance places it)", n.at)
        self.scope_attrs(n.target, self.seam_names)
        s = self.site_ir(n.target)
        extra = Block("instance", label, f"instance.{label}", n.at)  # only = "mujoco": a list, checked
        extra.attrs, extra.attr_pos = dict(leaf.attrs), {k: n.at for k in leaf.attrs}
        self.scope_attrs(extra, self.seam_names)
        if any(k in extra.attrs for k in SCOPE):  # the instance's scope replaces the site's own
            for k in SCOPE:
                s.pop(k, None)
        s.update({k: jsonify(v) for k, v in extra.attrs.items()})
        p0, q0 = leaf.pos, leaf.quat

        def world(p):
            r = qrot(list(q0), p)
            return [_clean(a + c) for a, c in zip(p0, r)]
        s["pos"] = world(s["pos"])
        for z in s["zones"].values():
            z["pos"] = world(z["pos"])
        for fx in s["fixtures"].values():
            fx["pos"] = world(fx["pos"])
            fx["quat"] = [_clean(x) for x in qmul(list(q0), fx["quat"])]
        s["quat"] = [_clean(x) for x in qmul(list(q0), s.get("quat") or [1.0, 0.0, 0.0, 0.0])]  # (a rebuilt
        # site may already be turned: decompiled from a composed world)
        s["scene"] = self.scene_record(out, leaf)
        self.ir["sites"][label] = s
        self.placed_sites.setdefault(n.target.label, []).append(label)
        return label

    def item(self, b):
        of = ref(b, "of", ("assembly", "part"))
        if of is None:
            raise OrbError(f"item {b.label} needs `of = assembly.<x>` or `part.<x>`", b.pos)
        if (of.type, of.label) not in self.defs:  # an imported project's definition: compiled when used
            self.ensure_def(of, b.attr_pos["of"])
        lc0 = b.attrs.get("lifecycle")
        if isinstance(lc0, Block) and lc0.type == "lifecycle" and ("lifecycle", lc0.label) not in self.defs:
            self.ensure_def(lc0, b.attr_pos["lifecycle"])
        loc = ref(b, "location", ("site", "zone", "output"))
        if loc is not None and getattr(loc, "placed", False):
            raise OrbError(f"item {b.label} stands outside every output, so it cannot be located in the placed "
                           f"{loc.path}: put it in the scene (instance(..., at = ...)) the output places",
                           b.attr_pos["location"])
        base = [0.0] * 3
        if loc is not None and loc.type != "output":  # an output's frame is the world frame
            site = self.ir["sites"][loc.path.split(".")[1]]
            base = site["zones"][loc.label]["pos"] if loc.type == "zone" else site["pos"]
        it = {k: jsonify(v) for k, v in b.attrs.items() if k not in ("at", "rpy", "yaw", "quat", "joints")}
        it.update(kind=of.type, of=of.label, location=loc.path if loc else None,
                  pos=[a + c for a, c in zip(base, vec(b, "at", L, 3, [0.0] * 3))], quat=rot(b))
        ref(b, "installed_in", ("item",))
        ref(b, "made_by", ("process",))
        if "joints" in b.attrs:
            asm = self.ir["assemblies"].get(of.label) if of.type == "assembly" else None
            js = b.attrs["joints"]
            if asm is None or not isinstance(js, dict):
                raise OrbError("joints = { name = value } needs an assembly item", b.attr_pos["joints"])
            known = {l["joint"]["name"]: l["joint"] for l in asm["links"]
                     if l["joint"]["type"] in ("revolute", "prismatic")}
            for k in js:
                if k not in known:
                    raise OrbError(f"{of.label} has no movable joint {k!r} (has: {', '.join(known) or 'none'})",
                                   b.attr_pos["joints"])
                mimic_error(known[k], k, b.attr_pos["joints"])
            it["joints"] = {k: joint_value(known[k], k, v, b.attr_pos["joints"]) for k, v in js.items()}
        lc = ref(b, "lifecycle", ("lifecycle",))
        if lc is not None:
            stages = self.ir["lifecycles"][lc.label]["stages"]
            if b.attrs.get("stage") not in stages:
                raise OrbError(f"stage must be one of {stages}", b.attr_pos.get("stage", b.pos))
        if "body" in b.attrs and b.attrs["body"] not in BODIES:
            raise OrbError(f"body is one of {' | '.join(BODIES)}, got {b.attrs['body']!r}", b.attr_pos["body"])
        self.item_blocks[b.label] = b
        self.ir["items"][b.label] = self.natives(b, it)

    # ------------------------------------------------------------------ sources (orb pull)
    def source(self, b):
        """`source "<name>" { git = "..."  ref = "main"  generate = [...] }`; `sha` from orb.lock."""
        if not isinstance(b.attrs.get("git"), str):
            raise OrbError(f'source {b.label} needs `git = "https://..."` (a repository git can clone)', b.pos)
        gen = b.attrs.get("generate")
        if gen is not None and not (isinstance(gen, str) or (isinstance(gen, list) and gen
                                                            and all(isinstance(x, str) for x in gen))):
            raise OrbError('generate is a command: ["python3", "tools/gen.py"] (or one string)', b.attr_pos["generate"])
        if not isinstance(b.attrs.get("ref", "main"), str):
            raise OrbError("ref is a branch, tag or commit sha (a string)", b.attr_pos["ref"])
        s = {k: jsonify(v) for k, v in b.attrs.items()}
        s.setdefault("ref", "main")
        sha = None
        lock = self.root / "orb.lock"
        if lock.is_file():
            try:
                import json

                sha = (json.loads(lock.read_text()).get("sources") or {}).get(b.label, {}).get("sha")
            except (ValueError, AttributeError):
                self.warnings.append(f"{lock}: not valid JSON; sources show sha = null")
        s["sha"] = sha
        self.ir["sources"][b.label] = s

    # ------------------------------------------------------------------ cameras and shots
    @staticmethod
    def use_view(b, key, names):
        """`view = overview(scene)` (a camera's or shot's), `view_to = ...` (a shot's end): fills the
        attributes `names` (eye, look-at, fov) the block does not set itself. A block that sets its own
        fov (and not its eye) keeps the view's framing: the eye moves along the line of sight by the
        ratio of the two fovs' tangents."""
        if key not in b.attrs:
            return
        v, at = b.attrs[key], b.attr_pos[key]
        if not isinstance(v, dict) or not {"at", "look_at"} <= set(v) or set(v) - {"at", "look_at", "fov"}:
            raise OrbError(f"{key} takes a view {{at, look_at, fov}} (from overview(...)), got {type_name(v)}", at)
        eye_n, look_n, fov_n = names
        refit = (fov_n in b.attrs and eye_n not in b.attrs and "fov" in v
                 and isinstance(b.attrs[fov_n], Q) and isinstance(v["fov"], Q))
        for src, dst in zip(("at", "look_at", "fov"), names):
            if src in v and dst not in b.attrs:
                b.attrs[dst], b.attr_pos[dst] = v[src], at
        if refit:  # the block's own fov keeps the view's framing: the eye slides along its line of sight
            f0, f1 = float(v["fov"].v), float(b.attrs[fov_n].v)
            if 0 < f0 < math.pi and 0 < f1 < math.pi:
                k = math.tan(f0 / 2) / math.tan(f1 / 2)
                look = b.attrs[look_n] if look_n in b.attrs else v["look_at"]
                f = lambda x: float(getattr(x, "v", x))
                b.attrs[eye_n] = [Q(f(lk) + k * (f(e) - f(lk)), L) for e, lk in zip(v["at"], look)]

    def camera(self, b):
        self.use_view(b, "view", ("at", "look_at", "fov"))
        c = {k: jsonify(v) for k, v in b.attrs.items() if k not in ("at", "look_at", "fov", "view")}
        c.update(pos=vec(b, "at", L, 3), look_at=vec(b, "look_at", L, 3), fov=num(b, "fov", ANG, math.radians(60)))
        if not 0 < c["fov"] < math.pi:
            raise OrbError("fov must be between 0 and 180 deg", b.attr_pos["fov"])
        if math.dist(c["pos"], c["look_at"]) < 1e-6:
            raise OrbError(f"camera {b.label} stands on the point it looks at (at = look_at)", b.attr_pos["at"])
        self.ir["cameras"][b.label] = c

    def shot(self, b):
        self.use_view(b, "view", ("at", "look_at", "fov"))
        self.use_view(b, "view_to", ("move_to", "look_to", "fov_to"))
        tls = self.ir["timelines"]
        tlb = ref(b, "timeline", ("timeline",))
        tl = tlb.label if tlb else next(iter(tls), None)
        if tl is None:
            raise OrbError(f"shot {b.label} needs a timeline, and the project has none", b.pos)
        if tl not in tls:  # a timeline scoped out at compile time cannot happen; keep the error honest
            raise OrbError(f"shot {b.label}: no timeline {tl!r}", b.pos)
        dur = tls[tl]["duration"]
        t = vec(b, "t", T, 2, [0.0, dur])
        if not 0 <= t[0] < t[1] <= dur + 1e-9:
            raise OrbError(f"t = [{t[0]:g}, {t[1]:g}] s must be a range inside the timeline's [0, {dur:g}] s",
                           b.attr_pos.get("t", b.pos))
        cam = ref(b, "camera", ("camera",))
        follow = ref(b, "follow", ("item",), ir=self.ir)
        pos, look = vec(b, "at", L, 3, None), vec(b, "look_at", L, 3, None)
        if cam is None and follow is None and pos is None:
            raise OrbError(f"shot {b.label} needs camera = camera.<x>, at + look_at, or follow = item[\"...\"]", b.pos)
        if pos is not None and look is None and follow is None:
            raise OrbError("at needs look_at (or follow = item[...])", b.attr_pos["at"])
        fov = num(b, "fov", ANG, None)
        if fov is None:
            fov = self.ir["cameras"][cam.label]["fov"] if cam else math.radians(60)
        known = ("timeline", "t", "camera", "follow", "at", "look_at", "offset", "relative", "fov", "label",
                 "view", "view_to")
        shot = {k: jsonify(v) for k, v in b.attrs.items() if k not in known}
        shot.update(name=b.label, timeline=tl, t0=t[0], t1=t[1], camera=cam.label if cam else None, pos=pos,
                    look_at=look, follow=follow.label if follow else None, offset=vec(b, "offset", L, 3, None),
                    relative=boolean(b, "relative", False), fov=fov, label=str(b.attrs.get("label", b.label)))
        # the camera move (motion.Motion.camera_at): checked here, so every player reads SI values
        for k in ("move_to", "look_to", "offset_to", "look_offset"):
            if k in b.attrs:
                shot[k] = vec(b, k, L, 3)
        for k, dim in (("fov_to", ANG), ("orbit", ANG), ("lag", T)):
            if k in b.attrs:
                shot[k] = num(b, k, dim)
        if "ease" in b.attrs and b.attrs["ease"] not in ("linear", "smooth"):
            raise OrbError(f"a shot's ease is linear | smooth, got {b.attrs['ease']!r}", b.attr_pos["ease"])
        for k in ("fov", "fov_to"):
            if shot.get(k) is not None and not 0 < shot[k] < math.pi:
                raise OrbError(f"{k} must be between 0 and 180 deg", b.attr_pos.get(k, b.pos))
        if pos is not None and look is not None and math.dist(pos, look) < 1e-6:
            raise OrbError(f"shot {b.label} stands on the point it looks at (at = look_at)", b.attr_pos["at"])
        if "offset_to" in shot and shot["offset"] is None:
            raise OrbError("offset_to needs offset (a follow shot's offset at its start)", b.attr_pos["offset_to"])
        if shot.get("lag", 0) < 0:
            raise OrbError("lag is a time >= 0 s", b.attr_pos["lag"])
        # blend (motion.Motion.camera_at): the shot's own, else its timeline's `blend` (clipped to the shot)
        tlblk = tlb or next((x for x in self.blocks if x.type == "timeline" and x.label == tl), None)
        if "blend" in b.attrs:
            shot["blend"] = num(b, "blend", T)
            if not 0 <= shot["blend"] <= t[1] - t[0] + 1e-9:
                raise OrbError(f"blend is a time from 0 s to the shot's length ({t[1] - t[0]:g} s), "
                               f"got {shot['blend']:g} s", b.attr_pos["blend"])
        elif tlblk is not None and "blend" in tlblk.attrs:
            dflt = num(tlblk, "blend", T)
            if dflt < 0:
                raise OrbError("a timeline's blend (its shots' default) is a time >= 0 s", tlblk.attr_pos["blend"])
            if dflt > 0:
                shot["blend"] = min(dflt, t[1] - t[0])
        if any(s["name"] == b.label for s in self.ir["shots"]):
            raise OrbError(f"duplicate shot {b.label!r}", b.pos)
        self.ir["shots"].append(shot)

    # ------------------------------------------------------------------ timelines
    def tl_item(self, c, tl):
        if c.label not in self.ir["items"]:
            raise OrbError(f"timeline {tl}: no item {c.label!r} (have: {', '.join(self.ir['items']) or 'none'})", c.pos)
        return c.label

    @staticmethod
    def tl_time(c, dur, key="t", default=_MISSING):
        t = num(c, key, T, default)
        if t is not None and dur is None and t < -1e-9:  # scene motion: the end is checked when lowered
            raise OrbError(f"{key} = {t:g} s is before 0 s", c.attr_pos.get(key, c.pos))
        if t is not None and dur is not None and not -1e-9 <= t <= dur + 1e-9:
            raise OrbError(f"{key} = {t:g} s is outside the timeline's [0, {dur:g}] s", c.attr_pos.get(key, c.pos))
        return None if t is None else float(t)

    @staticmethod
    def ease(c, default):
        e = c.attrs.get("ease", default)
        if e not in EASES:
            raise OrbError(f"ease is one of {' | '.join(EASES)}, got {e!r}", c.attr_pos.get("ease", c.pos))
        return e

    @staticmethod
    def allow(c, keys):
        for k in c.attrs:
            if k not in keys and k not in SCOPE:
                raise OrbError(f"{c.type} has no attribute {k!r} (use {', '.join(keys)})", c.attr_pos[k])

    def loc_base(self, item):
        """The frame an item's `at` (and its keys' `at`, its paths' `via`) is in: its location."""
        loc = self.ir["items"][item].get("location")
        if not loc or not loc.startswith("site."):  # unplaced, or placed by an output: the world frame
            return [0.0, 0.0, 0.0]
        p = loc.split(".")
        site = self.ir["sites"][p[1]]
        return list(site["zones"][p[3]]["pos"] if len(p) > 3 else site["pos"])

    def track(self, c, tl, dur):
        """Keys and paths -> [{"t", "pos", "quat", "ease"}] (world frame)."""
        item = self.tl_item(c, tl)
        it = self.ir["items"][item]
        return item, track_keys(c, it["pos"], it["quat"], self.loc_base(item), dur)

    def joint_track(self, c, tl, dur):
        """Joint keys -> [{"t", "values": {joint: v}, "ease"}]; every key carries every joint the
        track names (a joint a key leaves out holds its previous value)."""
        item = self.tl_item(c, tl)
        it = self.ir["items"][item]
        asm = self.ir["assemblies"].get(it["of"]) if it["kind"] == "assembly" else None
        if asm is None:
            raise OrbError(f"joints {item!r}: {item} is not an assembly item", c.pos)
        movable = {l["joint"]["name"]: l["joint"] for l in asm["links"] if l["joint"]["type"] in ("revolute", "prismatic")}
        default_ease = self.ease(c, "linear")
        self.allow(c, ("ease",))
        rows, last = [], None
        for k in c.children:
            if k.type != "key":
                raise OrbError(f"joints holds key blocks, not {k.type}", k.pos)
            t = self.tl_time(k, dur)
            if last is not None and t < last - 1e-9:
                raise OrbError(f"key times must not decrease ({t:g} s after {last:g} s)", k.attr_pos["t"])
            last = t
            given = {kk: (v, k.attr_pos[kk]) for kk, v in k.attrs.items() if kk not in ("t", "ease", "values") + SCOPE}
            if isinstance(k.attrs.get("values"), dict):
                given.update({kk: (v, k.attr_pos["values"]) for kk, v in k.attrs["values"].items()})
            vals = {}
            for name, (v, pos) in given.items():
                j = movable.get(name)
                if j is None:
                    raise OrbError(f"{it['of']} has no movable joint {name!r}{did_you_mean(name, movable)} "
                                   f"(has: {', '.join(movable) or 'none'})", pos)
                mimic_error(j, name, pos)
                vals[name] = joint_value(j, name, v, pos)
            rows.append((t, self.ease(k, default_ease), vals))
        names = list(dict.fromkeys(n for _, _, v in rows for n in v))
        cur = {n: float((it.get("joints") or {}).get(n, movable[n]["initial"])) for n in names}
        keys = []
        for t, e, vals in rows:
            cur.update(vals)
            keys.append({"t": t, "values": dict(cur), "ease": e})
        return item, keys

    def timeline(self, b):
        """A timeline block -> the normalized form seams read (contract 4c)."""
        from .motion import Motion, compose, inverse

        name = b.label
        dur = float(num(b, "duration", T))
        if dur <= 0:
            raise OrbError("duration must be positive", b.attr_pos["duration"])
        known = ("duration", "loop", "physics", "label")
        tl = {k: jsonify(v) for k, v in b.attrs.items() if k not in known}
        tl.update(label=str(b.attrs.get("label", name)), duration=dur, loop=boolean(b, "loop", True),
                  physics=boolean(b, "physics", True), tracks={}, joints={}, attach=[], visible={}, channels={},
                  phases=[], events=[], bodies={})
        scope, moves, vis, written = {}, {}, {}, {}
        for i, c in enumerate(b.children):
            if c.type in ("track", "joints", "show", "hide") and c.label in self.ir["items"]:
                written.setdefault(({"track": "tracks", "joints": "joints"}.get(c.type, "visible"), c.label), c.pos)
            if c.type == "track":
                item, keys = self.track(c, name, dur)
                if item in tl["tracks"]:
                    raise OrbError(f"timeline {name}: a second track for {item}", c.pos)
                tl["tracks"][item] = keys
                if self.scope_of(c):
                    scope.setdefault("tracks", {})[item] = self.scope_of(c)
            elif c.type == "joints":
                item, keys = self.joint_track(c, name, dur)
                if item in tl["joints"]:
                    raise OrbError(f"timeline {name}: a second joints track for {item}", c.pos)
                tl["joints"][item] = keys
                if self.scope_of(c):
                    scope.setdefault("joints", {})[item] = self.scope_of(c)
            elif c.type in ("attach", "detach"):
                item = self.tl_item(c, name)
                self.allow(c, ("t", "to", "link", "at", "yaw", "rpy", "quat") if c.type == "attach" else ("t",))
                moves.setdefault(item, []).append((self.tl_time(c, dur), 0 if c.type == "detach" else 1, i, c))
            elif c.type in ("show", "hide"):
                item = self.tl_item(c, name)
                self.allow(c, ("t",))
                vis.setdefault(item, []).append((self.tl_time(c, dur), i, c.type == "show"))
            elif c.type == "set":
                if not c.label:
                    raise OrbError('set needs a channel name: set "beam" { t = 30 s  value = 1 }', c.pos)
                self.allow(c, ("t", "value"))
                v = c.attrs.get("value")
                if isinstance(v, bool):
                    v = 1.0 if v else 0.0
                if not isinstance(v, (int, float, Q)):
                    raise OrbError("set needs value = <number>", c.attr_pos.get("value", c.pos))
                tl["channels"].setdefault(c.label, []).append([self.tl_time(c, dur), float(_val_si(v))])
            elif c.type == "phase":
                st = ref(c, "station", ("zone",), ir=self.ir)
                ph = {k: jsonify(v) for k, v in c.attrs.items() if k not in ("t", "label", "station")}
                ph.update(name=c.label, label=str(c.attrs.get("label", c.label)), t0=self.tl_time(c, dur), t1=None,
                          station=st.path if st else None)
                tl["phases"].append(ph)
            elif c.type == "event":
                ev = {k: jsonify(v) for k, v in c.attrs.items() if k not in ("t", "text")}
                ev.update(name=c.label, t=self.tl_time(c, dur), text=str(c.attrs.get("text", "")))
                tl["events"].append(ev)
            else:
                raise OrbError(f"a timeline holds {', '.join(TIMELINE_ENTRIES)}; not {c.type}", c.pos)
        for ch in tl["channels"].values():
            ch.sort(key=lambda s: s[0])
        tl["phases"].sort(key=lambda p: p["t0"])
        for a, nxt in zip(tl["phases"], tl["phases"][1:] + [None]):
            a["t1"] = nxt["t0"] if nxt else dur
        tl["events"].sort(key=lambda e: e["t"])
        for item, steps in vis.items():
            steps.sort()
            out = [[t, show] for t, _, show in steps]
            if out[0][1] and out[0][0] > 0:  # first a `show`: hidden until then
                out.insert(0, [0.0, False])
            tl["visible"][item] = out
        self.scene_timeline(name, dur, tl, written)
        # attach / detach -> intervals; attaching while attached is a transfer
        records = []
        for item, evs in moves.items():
            open_ = None
            for t, kind, _, c in sorted(evs, key=lambda e: e[:3]):
                if kind == 0:  # detach
                    if open_ is None:
                        if any(e[1] == 1 and abs(e[0] - t) < 1e-12 for e in evs):
                            raise OrbError(f"{item} is attached and detached at the same time ({t:g} s)", c.pos)
                        raise OrbError(f"detach {item} at {t:g} s: it is not attached then", c.pos)
                    open_["t1"] = t
                    open_ = None
                    continue
                to = ref(c, "to", ("item",), ir=self.ir)
                if to is None:
                    raise OrbError(f'attach needs to = item["<carrier>"]', c.pos)
                if to.label == item:
                    raise OrbError(f"{item} cannot carry itself", c.attr_pos["to"])
                carrier = self.ir["items"][to.label]
                asm = self.ir["assemblies"].get(carrier["of"]) if carrier["kind"] == "assembly" else None
                link = c.attrs.get("link")
                if asm is None:
                    if link is not None:
                        raise OrbError(f"{to.label} is a part item: it has no links", c.attr_pos["link"])
                else:
                    names = [l["name"] for l in asm["links"]]
                    if link is None:
                        link = asm["root"]
                    elif link not in names:
                        raise OrbError(f"{carrier['of']} has no link {link!r} (has: {', '.join(names)})",
                                       c.attr_pos["link"])
                here = not ("at" in c.attrs or has_rot(c))
                rec = {"item": item, "to": to.label, "link": link, "t0": t, "t1": None,
                       "pos": None if here else [float(x) for x in vec(c, "at", L, 3, [0.0] * 3)],
                       "quat": None if here else rot(c),
                       **self.scope_of(c)}
                if open_ is not None:
                    open_["t1"] = t
                open_ = rec
                records.append(rec)
            for r in records:
                if r["item"] == item and r["t1"] is not None and r["t1"] <= r["t0"]:
                    raise OrbError(f"{item} is attached and detached at the same time ({r['t0']:g} s)", b.pos)
        records.sort(key=lambda r: r["t0"])
        for r in records:  # attach "here": the payload's pose relative to the carrier link, then
            if r["pos"] is not None:
                continue
            ir = {**self.ir, "timelines": {name: {**tl, "attach": [x for x in records if x["pos"] is not None]}}}
            m = Motion(ir, name)
            rel = compose(inverse(m.link_pose(r["to"], r["link"], r["t0"])), m.pose_at(r["item"], r["t0"]))
            r["pos"] = [float(f"{x:.12g}") + 0.0 for x in rel[0]]
            r["quat"] = [float(f"{x:.12g}") + 0.0 for x in rel[1]]
        tl["attach"] = records
        # bodies
        payloads = {r["item"] for r in records}
        carriers = {r["to"] for r in records}
        for item, it in self.ir["items"].items():
            explicit = it.get("body")
            if item in tl["tracks"]:
                if explicit not in (None, "kinematic"):
                    raise OrbError(f"{item} has a track in timeline {name}: its body is kinematic, not {explicit}",
                                   self.item_blocks[item].attr_pos["body"])
                body = "kinematic"
            elif item in tl["joints"]:
                body = explicit or "kinematic"
            elif item in payloads:
                if explicit == "static":
                    raise OrbError(f"{item} is carried in timeline {name}: a static body cannot be",
                                   self.item_blocks[item].attr_pos["body"])
                body = explicit or "dynamic"
            elif item in carriers or item in tl["visible"] or explicit:
                body = explicit or "static"
            else:
                continue
            tl["bodies"][item] = "kinematic" if body == "dynamic" and not tl["physics"] else body
        if scope:
            tl["scope"] = scope
        self.ir["timelines"][name] = tl


def track_keys(c, pos, quat, base, dur, label_errors=False):
    """A track block's `key` and `path` children -> [{"t", "pos", "quat", "ease"}]: from the pose
    (pos, quat), in a frame offset by `base` (the item's location). One definition for timeline
    tracks and scene motion (orb/scenes.py: `instance(..., motion = [...])`, whose keys are blocks
    synthesized from its objects); dur None: no end to check yet (scene motion is checked when an
    output lowers it). label_errors: prefix each error with the child's path (`motion[2]: ...`)."""
    from .motion import yaw_of

    W = World
    cur_p, cur_q = list(pos), list(quat)
    default_ease = W.ease(c, "linear")
    W.allow(c, ("ease",))
    keys, last = [], None

    def add(t, e):
        keys.append({"t": float(f"{t:.12g}"), "pos": [float(f"{x:.12g}") for x in cur_p],
                     "quat": [float(f"{x:.12g}") for x in cur_q], "ease": e})

    for k in c.children:
        try:
            if k.type == "key":
                W.allow(k, ("t", "at", "yaw", "rpy", "quat", "ease"))
                t = W.tl_time(k, dur)
                if last is not None and t < last - 1e-9:
                    raise OrbError(f"key times must not decrease ({t:g} s after {last:g} s)", k.attr_pos["t"])
                if "at" in k.attrs:
                    v = k.attrs["at"]  # [x, y] keeps the height, as a path's via points do
                    xyz = vec(k, "at", L, 2) + [cur_p[2] - base[2]] if isinstance(v, list) and len(v) == 2 \
                        else vec(k, "at", L, 3)
                    cur_p = [a + b for a, b in zip(base, xyz)]
                if has_rot(k):
                    cur_q = rot(k)
                add(t, W.ease(k, default_ease))
                last = t
            elif k.type == "path":
                W.allow(k, ("t", "speed", "via", "face", "turn_rate", "ease"))
                t = W.tl_time(k, dur, default=None)
                if t is None:
                    t = last if last is not None else 0.0
                if last is not None and t < last - 1e-9:
                    raise OrbError(f"path starts at {t:g} s, before the key at {last:g} s", k.attr_pos["t"])
                if last is None or t > last + 1e-9:
                    add(t, "linear")  # hold until the path starts
                speed = num(k, "speed", VEL)
                rate = num(k, "turn_rate", ANGVEL, math.radians(120))
                if speed <= 0 or rate <= 0:
                    raise OrbError("speed and turn_rate must be positive", k.pos)
                face = k.attrs.get("face", "travel")
                if face not in ("travel", "reverse", "keep"):
                    raise OrbError(f'face is "travel", "reverse" or "keep", got {face!r}', k.attr_pos["face"])
                e = W.ease(k, "linear")
                via, vpos = k.attrs.get("via"), k.attr_pos.get("via", k.pos)
                if not isinstance(via, list) or not via:
                    raise OrbError("path needs via = [[x, y, z], ...] m (or [x, y] points)", vpos)
                if all(not isinstance(v, list) for v in via):
                    via = [via]  # one point
                for v in via:
                    if not isinstance(v, list) or len(v) not in (2, 3):
                        raise OrbError("each via point is [x, y] or [x, y, z]", vpos)
                    xyz = [_num(x, L, "via", vpos, False) for x in v]
                    p = [base[0] + xyz[0], base[1] + xyz[1], base[2] + xyz[2] if len(xyz) == 3 else cur_p[2]]
                    d = [a - b for a, b in zip(p, cur_p)]
                    dist = math.sqrt(sum(x * x for x in d))
                    if dist < 1e-9:
                        continue
                    if face != "keep" and math.hypot(d[0], d[1]) > 1e-9:
                        heading = math.atan2(d[1], d[0]) + (math.pi if face == "reverse" else 0.0)
                        dyaw = heading - yaw_of(cur_q)
                        dyaw = math.atan2(math.sin(dyaw), math.cos(dyaw))
                        if abs(dyaw) > 1e-9:
                            t += abs(dyaw) / rate
                            q = qmul(axis_angle([0.0, 0.0, 1.0], dyaw), cur_q)
                            n = math.sqrt(sum(x * x for x in q))
                            cur_q = [x / n for x in q]
                            add(t, "linear")
                    t += dist / speed
                    cur_p = p
                    add(t, e)
                if dur is not None and t > dur + 1e-9:
                    raise OrbError(f"path ends at {t:.3f} s, after the timeline's {dur:g} s", k.pos)
                last = t
            else:
                raise OrbError(f"a track holds key and path blocks, not {k.type}", k.pos)
        except OrbError as e:
            if not label_errors or e.msg.startswith(k.path):
                raise
            raise OrbError(f"{k.path}: {e.msg}", e.pos) from None
    return keys


def hidden_capture_boxes(site, cfg):
    """Fixture names whose boxes collide but are not drawn: in a site that carries a scan mesh
    (`kind = "scan_mesh"`), the boxes `orb capture` derived from it (`source.capture` set) sit
    exactly on the scanned surfaces and would hide it. Seam setting `show_capture_boxes = true`
    draws them again."""
    if cfg.get("show_capture_boxes", False):
        return set()
    fxs = site["fixtures"]
    if not any(fx.get("attrs", {}).get("kind") == "scan_mesh" for fx in fxs.values()):
        return set()
    out = set()
    for name, fx in fxs.items():
        attrs = fx.get("attrs", {})
        src = attrs.get("source")
        if attrs.get("kind") != "scan_mesh" and isinstance(src, dict) and src.get("capture"):
            out.add(name)
    return out


def _val_si(v):
    return v.v if isinstance(v, Q) else v


def compile_world(blocks, root, outputs=None):
    """outputs: which `output` blocks to place (default: $ORB_OUTPUT, else every one)."""
    from .motion import check

    w = World(blocks, root, outputs)
    ir = w.build()
    ir["warnings"] = w.warnings
    errs = check(ir)
    if errs:
        raise OrbError("; ".join(errs))
    return ir


# --------------------------------------------------------------------------- seam scoping


def in_seam(block, seam):
    """Is an IR block (a dict that may carry `only` / `skip` lists) present in `seam`?"""
    if not isinstance(block, dict):
        return True
    only, skip = block.get("only"), block.get("skip")
    if isinstance(only, list) and seam not in only:
        return False
    if isinstance(skip, list) and seam in skip:
        return False
    return True


def _prune(x, seam):
    """Drop every dict (a section entry, a list element) that is not for `seam`, at any depth, and
    reduce every `native` map to that seam's entry."""
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if isinstance(v, dict) and not in_seam(v, seam):
                continue
            if k == "native" and isinstance(v, dict):
                if isinstance(v.get(seam), dict):
                    out[k] = {seam: _prune(v[seam], seam)}
                continue
            out[k] = _prune(v, seam)
        return out
    if isinstance(x, list):
        return [_prune(v, seam) for v in x if not (isinstance(v, dict) and not in_seam(v, seam))]
    return x


def _unscope(tl, seam):
    """A timeline without the keyed entries (tracks, joints, visible, channels) its `scope` map
    leaves out of `seam`; the map itself goes."""
    if not isinstance(tl, dict) or "scope" not in tl:
        return tl
    tl = dict(tl)
    for kind, scoped in (tl.pop("scope") or {}).items():
        if isinstance(tl.get(kind), dict):
            tl[kind] = {k: v for k, v in tl[kind].items() if in_seam((scoped or {}).get(k), seam)}
    return tl


def movable_joints(asm):
    return {l["joint"]["name"] for l in asm["links"] if l["joint"]["type"] in ("revolute", "prismatic")}


def for_seam(ir, seam):
    """The world graph as `seam` sees it (contract v0.3, 1a): a deep copy without the blocks whose
    `only` / `skip` leave the seam out (at any depth: parts, geoms, links and their subtrees, sites,
    zones, fixtures, items, timelines, cameras, shots, phases, events, attach records, ...), without
    the references to what was removed (items of a removed part or assembly, tracks, attach records,
    shots that name a removed item, link, camera or timeline), and with every `native` map reduced
    to `{seam: {...}}` (or gone when the block has no native for this seam). The copy carries
    `"seam": seam`; scoping it again for the same seam changes nothing."""
    out = {}
    for k, v in ir.items():  # only/skip on tracks, joints, visibility, channels: the timeline's scope map
        if k == "timelines" and isinstance(v, dict):
            v = {name: _unscope(tl, seam) for name, tl in v.items()}
        out[k] = v
    out = _prune(out, seam)
    # assemblies: a removed link takes its subtree with it; no root link left -> no assembly
    for name in list(out.get("assemblies") or {}):
        asm = out["assemblies"][name]
        kept, links = set(), []
        for link in asm.get("links") or []:
            if link["parent"] is None or link["parent"] in kept:
                kept.add(link["name"])
                links.append(link)
        if not links or links[0]["parent"] is not None:
            del out["assemblies"][name]
            continue
        asm["links"], asm["root"] = links, links[0]["name"]
        if "mass" in asm:
            asm["mass"] = sum(l["inertial"]["mass"] for l in links if l.get("inertial"))
    _drop_references(out)
    out["seam"] = seam
    return out


def select_outputs(ir, names):
    """The world with only the outputs `names` placed (`--output`): the other outputs' items and sites
    go, with every reference to them (tracks, attach records, follow shots, phase stations), and so do
    the imported definitions only they used."""
    drop = set(ir.get("outputs") or {}) - set(names)
    if not drop:
        return ir
    out = dict(ir)
    gone = lambda d: isinstance(d.get("scene"), dict) and d["scene"].get("output") in drop  # noqa: E731
    out["items"] = {k: v for k, v in (ir.get("items") or {}).items() if not gone(v)}
    out["sites"] = {k: v for k, v in (ir.get("sites") or {}).items() if not gone(v)}
    out["outputs"] = {k: v for k, v in ir["outputs"].items() if k not in drop}
    out["timelines"] = copy.deepcopy(ir.get("timelines") or {})
    out["shots"] = copy.deepcopy(ir.get("shots") or [])
    # imported definitions (labelled alias/...) that only the dropped outputs used
    used = {("part" if v.get("kind") == "part" else "assembly", v.get("of")) for v in out["items"].values()}
    used |= {("lifecycle", v.get("lifecycle")) for v in out["items"].values()}
    used |= {("process", v.get("made_by")) for v in out["items"].values()}
    for name, asm in (ir.get("assemblies") or {}).items():
        if ("assembly", name) in used:
            used |= {("part", l.get("part")) for l in asm.get("links") or []}
    for sec, kind in (("parts", "part"), ("assemblies", "assembly"), ("lifecycles", "lifecycle"),
                      ("processes", "process")):
        if isinstance(ir.get(sec), dict):
            out[sec] = {k: v for k, v in ir[sec].items() if "/" not in k or (kind, k) in used}
    _drop_references(out)
    return out


def _drop_references(out):
    """Remove what names something no longer in `out`: items of a missing part or assembly, joints
    an assembly no longer has, tracks / attach records / phase stations / shots naming what is gone."""
    sections = {"part": out.get("parts") or {}, "assembly": out.get("assemblies") or {}}
    items = out.get("items") or {}
    for name in list(items):
        it = items[name]
        if it.get("kind") in sections and it.get("of") not in sections[it["kind"]]:
            del items[name]
        elif it.get("kind") == "assembly" and it.get("joints"):
            ok = movable_joints(sections["assembly"][it["of"]])
            it["joints"] = {k: v for k, v in it["joints"].items() if k in ok}

    def links_of(item):
        it = items.get(item) or {}
        asm = sections["assembly"].get(it.get("of")) if it.get("kind") == "assembly" else None
        return {l["name"] for l in asm["links"]} if asm else set()

    zones = {f"site.{s}.zone.{z}" for s, site in (out.get("sites") or {}).items() for z in site.get("zones") or {}}
    for tl in (out.get("timelines") or {}).values():
        for kind in ("tracks", "joints", "visible", "bodies"):
            if isinstance(tl.get(kind), dict):
                tl[kind] = {k: v for k, v in tl[kind].items() if k in items}
        for item, keys in list((tl.get("joints") or {}).items()):
            ok = movable_joints(sections["assembly"][items[item]["of"]]) if items[item].get("kind") == "assembly" \
                else set()
            keys = [{**k, "values": {j: v for j, v in (k.get("values") or {}).items() if j in ok}} for k in keys]
            keys = [k for k in keys if k["values"]]
            if keys:
                tl["joints"][item] = keys
            else:
                del tl["joints"][item]
        tl["attach"] = [r for r in tl.get("attach") or [] if r.get("item") in items and r.get("to") in items
                        and (not r.get("link") or r["link"] in links_of(r["to"]))]
        phases = sorted(tl.get("phases") or [], key=lambda p: p["t0"])
        for a, b in zip(phases, phases[1:] + [None]):
            a["t1"] = b["t0"] if b else tl.get("duration", a.get("t1"))
            if a.get("station") and a["station"] not in zones:
                a["station"] = None
        tl["phases"] = phases
    timelines, cameras = out.get("timelines") or {}, out.get("cameras") or {}
    if "shots" in out:
        out["shots"] = [s for s in out["shots"] if (s.get("timeline") is None or s["timeline"] in timelines)
                        and (s.get("camera") is None or s["camera"] in cameras)
                        and (s.get("follow") is None or s["follow"] in items)]


def scope_report(ir, seam):
    """What `for_seam(ir, seam)` leaves out and which natives the seam gets, for `orb plan` and
    `orb validate`: {"out": ["item.horse-01", ...], "geoms_out": n, "natives": ["item.truck-01 (usd)", ...]}."""
    s = for_seam(ir, seam)
    out = []
    for sec, word in (("parts", "part"), ("assemblies", "assembly"), ("sites", "site"), ("items", "item"),
                      ("lifecycles", "lifecycle"), ("processes", "process"), ("timelines", "timeline"),
                      ("cameras", "camera")):
        out += [f"{word}.{k}" for k in ir.get(sec) or {} if k not in (s.get(sec) or {})]
    for name, asm in (s.get("assemblies") or {}).items():
        kept = {l["name"] for l in asm["links"]}
        out += [f"assembly.{name}.link.{l['name']}" for l in ir["assemblies"][name]["links"] if l["name"] not in kept]
    for name, site in (s.get("sites") or {}).items():
        for kind in ("zones", "fixtures"):
            out += [f"site.{name}.{kind[:-1]}.{k}" for k in ir["sites"][name].get(kind) or {}
                    if k not in site.get(kind) or {}]
    kept_shots = {x.get("name") for x in s.get("shots") or []}
    out += [f"shot.{x.get('name')}" for x in ir.get("shots") or [] if x.get("name") not in kept_shots]
    for typ, blocks in (ir.get("other") or {}).items():
        if isinstance(blocks, dict):
            out += [f"{typ}.{k}" for k in blocks if k not in ((s.get("other") or {}).get(typ) or {})]

    def count_geoms(x):
        n = 0
        for p in (x.get("parts") or {}).values():
            n += len(p.get("geoms") or [])
        for a in (x.get("assemblies") or {}).values():
            n += sum(len(l.get("geoms") or []) for l in a["links"])
        for site in (x.get("sites") or {}).values():
            n += sum(len(f.get("geoms") or []) for f in (site.get("fixtures") or {}).values())
        return n

    # geoms that went on their own (not with a removed block): count against the same blocks unscoped
    same = {"parts": {k: ir["parts"][k] for k in s.get("parts") or {}},
            "assemblies": {k: {**ir["assemblies"][k], "links": [l for l in ir["assemblies"][k]["links"]
                                                                if l["name"] in {m["name"] for m in v["links"]}]}
                           for k, v in (s.get("assemblies") or {}).items()},
            "sites": {k: {"fixtures": {f: ir["sites"][k]["fixtures"][f] for f in v.get("fixtures") or {}}}
                      for k, v in (s.get("sites") or {}).items()}}
    natives = []

    def nat(addr, blk):
        e = (blk.get("native") or {}).get(seam)
        if isinstance(e, dict):
            files = [str(e[k]) for k in NATIVE_FILES if isinstance(e.get(k), str)]
            what = files or [f"{k} {e[k]}" for k in ("entity", "block") if e.get(k)] or ["native"]
            natives.append(f"{addr} ({', '.join(what)})")

    for k, v in (s.get("parts") or {}).items():
        nat(f"part.{k}", v)
    for k, v in (s.get("assemblies") or {}).items():
        nat(f"assembly.{k}", v)
        for l in v["links"]:
            nat(f"assembly.{k}.link.{l['name']}", l)
    for k, v in (s.get("sites") or {}).items():
        nat(f"site.{k}", v)
        for f, fx in (v.get("fixtures") or {}).items():
            nat(f"site.{k}.fixture.{f}", fx)
    for k, v in (s.get("items") or {}).items():
        nat(f"item.{k}", v)
    return {"out": out, "geoms_out": count_geoms(same) - count_geoms(s), "natives": natives}
