"""Scene values: what OrbLang functions build, return and pass to each other.

A Scene is an immutable arrangement of placed things. Each node is one instance with a local id
and a rigid pose (metres, [w, x, y, z]) relative to the scene's frame:

* a `part` or `assembly` (a block of this project or of an imported one): becomes an item;
* a `site` (a room: floor, zones, fixtures, a captured scan): becomes a site;
* another Scene: a group, so a returned scene can be reused (twice, rotated, inside a bigger one).

Building a Scene places nothing. An `output "<name>" { value = <scene> }` block is where a scene
becomes part of the world: orb/world.py walks it (flatten) into ordinary items and sites, so every
seam, timeline and `only`/`skip` works on composed scenes unchanged. Their labels are the instance
paths, `<output>/<id>/<id>`: deterministic, unique however often a scene is reused.
"""
from __future__ import annotations

import difflib
import math
import re

from .builtins import builtin, no_kw
from .lang import Block, OrbError, Q, did_you_mean, interpolated, type_name

ID = re.compile(r"[^./\s\"]+")
IDENT = (1.0, 0.0, 0.0, 0.0)
POSE_KEYS = ("at", "yaw", "rpy", "quat", "face")
# names people reach for that are not instance()'s
MISNAMED = {"pos": "at", "position": "at", "translation": "at", "offset": "at", "rotation": "yaw, rpy or quat",
            "orientation": "yaw, rpy or quat", "heading": "yaw", "angle": "yaw", "rot": "yaw, rpy or quat",
            "look_at": "face", "facing": "face", "toward": "face", "towards": "face", "aim": "face"}
SCOPE = ("only", "skip")
LENGTH, ANGLE, TIME = (0, 1, 0, 0, 0, 0), (0, 0, 0, 0, 0, 1), (0, 0, 1, 0, 0, 0)
# animated scenes: instance() attributes that say how the instance moves and when it is there
# (never item attributes; lowered into a timeline by orb/world.py)
ANIM = ("motion", "joint_motion", "appear", "vanish", "delay")
KEY_KEYS = ("t", "at", "yaw", "rpy", "quat", "ease")
PATH_KEYS = ("t", "via", "speed", "face", "turn_rate", "ease")
ANIM_MISNAMED = {"animation": "motion", "track": "motion", "keys": "motion", "path": "motion",
                 "joint_track": "joint_motion", "show": "appear", "hide": "vanish", "disappear": "vanish",
                 "start": "delay", "t0": "delay"}
# composing a moving group with what is in it: where the result is not exactly a lerp/slerp
# between keys (a group turning carries its children along arcs), it is sampled this often
SUB_DT, SUB_ANGLE = 0.25, math.radians(5)


def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw)


def qrot(q, v):
    w, x, y, z = qmul(qmul(q, (0.0, *v)), (q[0], -q[1], -q[2], -q[3]))
    return (x, y, z)


def compose_pose(parent, child):
    """parent (pos, quat) * child (pos, quat)."""
    (pp, pq), (cp, cq) = parent, child
    return tuple(a + b for a, b in zip(pp, qrot(pq, cp))), qmul(pq, cq)


def rpy_quat(r, p, y):
    cr, sr, cp, sp, cy, sy = (math.cos(r / 2), math.sin(r / 2), math.cos(p / 2), math.sin(p / 2),
                              math.cos(y / 2), math.sin(y / 2))
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)


def _length(x, key, pos):
    if isinstance(x, Q) and x.d == LENGTH:
        return float(x.v)
    if isinstance(x, (int, float)) and not isinstance(x, bool) and x == 0:
        return 0.0
    raise OrbError(f"{key} takes lengths (m, mm, ...), got {type_name(x)}", pos)


def _angle(x, key, pos):
    if isinstance(x, Q) and x.d == ANGLE:
        return float(x.v)
    if isinstance(x, (int, float)) and not isinstance(x, bool) and x == 0:
        return 0.0
    raise OrbError(f"{key} takes an angle (deg or rad), got {type_name(x)}", pos)


def point_like(x, key, pos):
    """A point to aim at, turn about or ring around: [x, y(, z)] lengths, an object with a `center`
    (bounds() returns one), or a scene, block or box ({min, max}): where it stands, the middle of its
    footprint at its base. -> (x, y, z) floats."""
    if isinstance(x, list):
        if len(x) not in (2, 3):
            raise OrbError(f"{key} takes a point [x, y] or [x, y, z] (lengths), got a list of {len(x)}", pos)
        return tuple(_length(v, key, pos) for v in x) + ((0.0,) if len(x) == 2 else ())
    if isinstance(x, dict) and "center" in x:
        return point_like(x["center"], key, pos)
    if isinstance(x, (Scene, Block, dict)):
        from .layout import box_of

        lo, hi = box_of(x, key, pos)  # where it stands: the middle of its footprint, at its base
        return ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2])
    raise OrbError(f"{key} takes a point ([x, y, z] m), a scene, a block or a box, got {type_name(x)}", pos)


def heading(q):
    """The yaw (about +z) of a rotation, radians."""
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def pose(kw, pos):
    """at / yaw / rpy / quat / face keyword arguments -> (pos, quat); missing: identity. `face = point`
    turns +x toward the point (about z, from `at`); a `yaw` beside it is an extra turn."""
    p = (0.0, 0.0, 0.0)
    if "at" in kw:
        v = kw["at"]
        if not isinstance(v, list) or len(v) not in (2, 3):
            raise OrbError(f"at takes [x, y] or [x, y, z] lengths, got {type_name(v)}", pos)
        p = tuple(_length(x, "at", pos) for x in v) + ((0.0,) if len(v) == 2 else ())
    rots = [k for k in ("yaw", "rpy", "quat") if k in kw]
    if len(rots) > 1:
        raise OrbError(f"give one rotation: {' or '.join(rots)}", pos)
    if "face" in kw:
        if rots and rots[0] != "yaw":
            raise OrbError(f"face turns about z: give it a yaw for an extra turn, not {rots[0]}", pos)
        t = point_like(kw["face"], "face", pos)
        if math.isclose(t[0], p[0], abs_tol=1e-12) and math.isclose(t[1], p[1], abs_tol=1e-12):
            raise OrbError("face: the point is straight above or below `at`, so it has no heading", pos)
        extra = _angle(kw["yaw"], "yaw", pos) if "yaw" in kw else 0.0
        return p, rpy_quat(0, 0, math.atan2(t[1] - p[1], t[0] - p[0]) + extra)
    q = IDENT
    if "yaw" in kw:
        q = rpy_quat(0, 0, _angle(kw["yaw"], "yaw", pos))
    elif "rpy" in kw:
        v = kw["rpy"]
        if not isinstance(v, list) or len(v) != 3:
            raise OrbError("rpy takes [roll, pitch, yaw] angles", pos)
        q = rpy_quat(*(_angle(x, "rpy", pos) for x in v))
    elif "quat" in kw:
        v = kw["quat"]
        if not isinstance(v, list) or len(v) != 4 or not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
            raise OrbError("quat takes [w, x, y, z] numbers", pos)
        n = math.sqrt(sum(x * x for x in v))
        if n == 0:
            raise OrbError("quat must not be all zeros", pos)
        q = tuple(x / n for x in v)
    return p, q


class Anim:
    """How one instance moves and when it is there, in its parent's frame and its own time (before
    any `delay`): keys [{t, pos, quat, ease}] compiled from `motion` (None: it stands still), joint
    keys [(t, {joint: value, ease}), ...] from `joint_motion`, appear / vanish (s or None), delay (s)."""
    __slots__ = ("keys", "joints", "appear", "vanish", "delay")

    def __init__(self, keys=None, joints=None, appear=None, vanish=None, delay=0.0):
        self.keys, self.joints, self.appear, self.vanish, self.delay = keys, joints, appear, vanish, delay

    def moved(self, parent):
        """The same animation in a frame `parent` (pos, quat) moves into (move() moves the keys too)."""
        if not self.keys:
            return self
        keys = []
        for k in self.keys:
            p, q = compose_pose(parent, (k["pos"], k["quat"]))
            keys.append({**k, "pos": list(p), "quat": list(q)})
        return Anim(keys, self.joints, self.appear, self.vanish, self.delay)


class Node:
    """One placed thing in a scene. kind: part | assembly | site | scene. anim: an Anim or None."""
    __slots__ = ("id", "kind", "target", "pos", "quat", "attrs", "at", "anim")

    def __init__(self, id, kind, target, pos, quat, attrs, at, anim=None):
        self.id, self.kind, self.target, self.pos, self.quat = id, kind, target, tuple(pos), tuple(quat)
        self.attrs, self.at, self.anim = dict(attrs), at, anim

    def moved(self, parent):
        p, q = compose_pose(parent, (self.pos, self.quat))
        return Node(self.id, self.kind, self.target, p, q, self.attrs, self.at,
                    self.anim.moved(parent) if self.anim else None)

    def __repr__(self):
        return f"<{self.kind} {self.id}>"


class Scene:
    """An immutable arrangement of nodes (see the module docstring)."""
    orb_type = "scene"

    def __init__(self, nodes, label="", at=None):
        self.nodes, self.label, self.at = tuple(nodes), label, at
        seen = {}
        for n in self.nodes:
            if n.id in seen:
                raise OrbError(f"two instances named {n.id!r} in one scene (first placed at {seen[n.id]}); "
                               f"give each its own id = \"...\"", n.at)
            seen[n.id] = n.at

    def node(self, id):
        return next((n for n in self.nodes if n.id == id), None)

    def orb_get(self, name, pos):
        """scene.ids, scene.label, scene.count; scene.<id> or scene["<id>"]: that instance (a scene of one)."""
        if name == "ids":
            return [n.id for n in self.nodes]
        if name == "label":
            return self.label
        if name == "count":
            return len(self.nodes)
        n = self.node(name) if isinstance(name, str) else None
        if n is None:
            have = ", ".join(n.id for n in self.nodes) or "none"
            if not isinstance(name, str):
                raise OrbError(f"a scene is indexed by instance id, not {type_name(name)} (s.ids[0] is the first "
                               f"id; has: {have})", pos)
            raise OrbError(f"scene has no instance {name!r}{did_you_mean(name, [n.id for n in self.nodes])} "
                           f"(has: {have})", pos)
        return Scene([n], n.id, n.at)

    def __repr__(self):
        return f"<scene {self.label or ''} {[n.id for n in self.nodes]}>"


def default_id(of):
    if isinstance(of, Block):
        return of.label.rsplit("/", 1)[-1]
    if isinstance(of, Scene):
        return of.label
    return ""


def make_node(of, kw, pos):
    """instance(...) arguments -> a Node."""
    extras = {k: v for k, v in kw.items() if k not in POSE_KEYS and k != "id" and k not in ANIM}
    misnamed = {**MISNAMED, **ANIM_MISNAMED}
    for k in extras:  # item attributes pass through; a misspelt pose or id would silently do nothing
        if k in ("joints", "body"):
            continue
        near = difflib.get_close_matches(k, list(misnamed), n=1, cutoff=0.8) if len(k) > 3 else []
        hint = (misnamed.get(k) or (misnamed[near[0]] if near else "")
                or (did_you_mean(k, POSE_KEYS + ("id", "joints", "body")) if len(k) > 2 else "")
                or ("" if len(k) <= 3 or not difflib.get_close_matches(k, ANIM, n=1, cutoff=0.8)  # not location
                    else f" (did you mean {difflib.get_close_matches(k, ANIM, n=1, cutoff=0.8)[0]!r}?)"))
        if hint:
            raise OrbError(f"instance() has no {k!r}{hint if hint.startswith(' (') else f' (use {hint})'}; "
                           "other attributes go on the item", pos)
    if isinstance(of, Block) and of.type in ("part", "assembly"):
        kind, target = of.type, of
    elif isinstance(of, Block) and of.type == "item":  # an item lends its definition and attributes, not its place
        target = of.attrs.get("of")
        if not isinstance(target, Block) or target.type not in ("part", "assembly"):
            raise OrbError(f"{of.path} has no `of = part.x | assembly.x` to instance", pos)
        kind = target.type
        extras = {**{k: v for k, v in of.attrs.items()
                     if k not in ("of", "location", "installed_in", "made_by") + POSE_KEYS}, **extras}
    elif isinstance(of, Block) and of.type == "site":
        kind, target = "site", of
    elif isinstance(of, Scene):
        kind, target = "scene", of
        bad = sorted(set(extras) - set(SCOPE))
        if bad:
            raise OrbError(f"instance of a scene takes id, at, a rotation, only, skip, motion, appear, vanish and "
                           f"delay; not {bad[0]!r}", pos)
    else:
        hint = ""
        if isinstance(of, Block) and of.type == "output":
            ns, _, name = of.label.rpartition("/")
            hint = (f" (another project's output is a scene: {ns.replace('/', '.')}.outputs.{name})" if ns else
                    " (an output places its scene; pass the scene itself)")
        raise OrbError(f"instance() places a part, assembly, item, site or scene; got {type_name(of)}{hint}", pos)
    id = kw.get("id", default_id(of))
    if not isinstance(id, str) or not ID.fullmatch(id):
        raise OrbError(f"instance id must be a non-empty string without '.', '/' or spaces, got {id!r}"
                       + ("" if id else ' (pass id = "...")'), pos)
    p, q = pose(kw, pos)
    anim = make_anim({k: kw[k] for k in ANIM if k in kw}, kind, target, p, q, pos)
    return Node(id, kind, target, p, q, extras, pos, anim)


# --------------------------------------------------------------------------- animated scenes


def _time(x, key, pos):
    """A time >= 0 in seconds (a bare 0 is 0)."""
    if isinstance(x, Q) and x.d == TIME:
        v = float(x.v)
    elif isinstance(x, (int, float)) and not isinstance(x, bool) and x == 0:
        v = 0.0
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        raise OrbError(f"{key} needs a time unit ({x:g} s, ms, min), got a bare number", pos)
    else:
        raise OrbError(f"{key} takes a time (s, ms, min), got {type_name(x)}", pos)
    if v < -1e-12:
        raise OrbError(f"{key} = {v:g} s is before 0 s", pos)
    return max(v, 0.0)


def make_anim(kw, kind, target, p, q, pos):
    """instance()'s motion / joint_motion / appear / vanish / delay -> an Anim (None without them)."""
    if not kw:
        return None
    if kind == "site":
        raise OrbError(f"a site stands still and is always there: instance() of a site takes no "
                       f"{sorted(kw)[0]} (animate what stands in it)", pos)
    if "joint_motion" in kw and kind != "assembly":
        what = f"{target.type} {target.label}" if isinstance(target, Block) else "a scene"
        raise OrbError(f"joint_motion moves an assembly's joints; {what} has none"
                       + (" (give it to the assembly's own instance inside the scene)" if kind == "scene" else ""),
                       pos)
    delay = _time(kw["delay"], "delay", pos) if "delay" in kw else 0.0
    appear = _time(kw["appear"], "appear", pos) if "appear" in kw else None
    vanish = _time(kw["vanish"], "vanish", pos) if "vanish" in kw else None
    if appear is not None and vanish is not None and vanish <= appear:
        raise OrbError(f"vanish = {vanish:g} s must come after appear = {appear:g} s", pos)
    keys = motion_keys(kw["motion"], p, q, pos) if "motion" in kw else None
    joints = joint_motion_keys(kw["joint_motion"], pos) if "joint_motion" in kw else None
    return Anim(keys, joints, appear, vanish, delay)


def _entries(v, what, example, pos):
    if not isinstance(v, list) or not v:
        raise OrbError(f"{what} is a list of keys: {example}, got {type_name(v)}"
                       + (" (wrap one key in [ ])" if isinstance(v, dict) else ""), pos)
    for i, e in enumerate(v):
        if not isinstance(e, dict):
            raise OrbError(f"{what}[{i}] is an object like {{t = 5 s, ...}}, got {type_name(e)}", pos)
    return v


def motion_keys(v, p, q, pos):
    """`motion = [{t, at, yaw, ease}, {via, speed, ...}, ...]` -> keys in the parent frame, from the
    instance's own pose. Each object becomes a timeline `key` (or, with `via`, a `path`) block, and
    orb.world.track_keys compiles them: the semantics of a timeline track exactly."""
    from .world import track_keys

    _entries(v, "motion", "[{t = 0 s, at = [x, y, z] m}, {t = 5 s, at = [...] m, yaw = 90 deg}]", pos)
    kids = []
    for i, e in enumerate(v):
        kind, have, other = ("path", PATH_KEYS, KEY_KEYS) if "via" in e else ("key", KEY_KEYS, PATH_KEYS)
        for k in e:
            if k in have:
                continue
            if k in other:
                raise OrbError(f"motion[{i}]: `{k}` belongs to a {'key' if kind == 'path' else 'path'}; this "
                               f"{kind} takes {', '.join(have)}" + (" (a path goes from the previous pose through "
                               "`via`)" if kind == "path" else " (add `via` for a path)"), pos)
            raise OrbError(f"motion[{i}] has no {k!r}{did_you_mean(k, KEY_KEYS + PATH_KEYS)} (a key takes "
                           f"{', '.join(KEY_KEYS)}; a path takes via, {', '.join(PATH_KEYS[:1] + PATH_KEYS[2:])})", pos)
        b = Block(kind, "", f"motion[{i}]", pos)
        b.attrs, b.attr_pos = dict(e), {k: pos for k in e}
        kids.append(b)
    track = Block("track", "", "motion", pos)
    track.children = kids
    return track_keys(track, p, q, (0.0, 0.0, 0.0), None, label_errors=True)


def joint_motion_keys(v, pos):
    """`joint_motion = [{t = 0 s, knee = 20 deg, ease = "smooth"}, ...]` -> [(t, object), ...]; the
    joints are checked against the assembly when an output lowers them (orb.world joint_track)."""
    from .world import EASES

    _entries(v, "joint_motion", "[{t = 0 s, knee = 0 deg}, {t = 2 s, knee = 40 deg}]", pos)
    out, last = [], None
    for i, e in enumerate(v):
        if "t" not in e:
            raise OrbError(f"joint_motion[{i}] needs `t` (a time: t = 2 s)", pos)
        t = _time(e["t"], f"joint_motion[{i}].t", pos)
        if last is not None and t < last - 1e-9:
            raise OrbError(f"joint_motion[{i}]: key times must not decrease ({t:g} s after {last:g} s)", pos)
        if e.get("ease", "linear") not in EASES:
            raise OrbError(f"joint_motion[{i}]: ease is one of {' | '.join(EASES)}, got {e.get('ease')!r}", pos)
        if not any(k not in ("t", "ease") for k in e):
            raise OrbError(f"joint_motion[{i}] sets no joint (write {{t = .. s, <joint> = <value>}})", pos)
        out.append((t, dict(e)))
        last = t
    return out


def scenes_of(args, pos, what):
    out = []
    for a in args:
        if isinstance(a, list):
            out.extend(scenes_of(a, pos, what))
        elif isinstance(a, Scene):
            out.append(a)
        elif isinstance(a, Block):
            out.append(Scene([make_node(a, {}, pos)], default_id(a), pos))
        else:
            raise OrbError(f"{what} takes scenes (or blocks to place as they are), got {type_name(a)}", pos)
    return out


# --------------------------------------------------------------------------- builtins


@builtin("instance")
def _instance(prog, args, kw, pos):
    """instance(of, id = "x", at = [x, y, z] m, yaw = 90 deg | rpy = [...] | quat = [...] | face = point, motion = [...],
    joint_motion = [...], appear = 4 s, vanish = 20 s, delay = 3 s, ...attrs): a scene of one placed part,
    assembly, item, site or scene. face = point (or a scene or block: where it stands) turns its +x toward
    that point; a yaw beside it is an extra turn. motion: keys {t, at, yaw | rpy | quat, ease} and paths
    {via, speed, ...} in the parent's frame; joint_motion: joint keys {t, <joint> = value, ease}
    (assemblies); appear / vanish: shown from / hidden from; delay: shifts every time of this instance and
    everything in it. Other attributes (joints, body, stage, lifecycle, serial, only, skip, ...) go on the
    item it becomes."""
    if len(args) > 1 or (args and "of" in kw):
        raise OrbError("instance() takes one thing to place", pos)
    of = args[0] if args else kw.pop("of", None)
    kw = {k: v for k, v in kw.items() if k != "of"}
    if of is None:
        raise OrbError("instance() needs something to place: instance(part.x, ...)", pos)
    n = make_node(of, kw, pos)
    return Scene([n], n.id, pos)


@builtin("compose")
def _compose(prog, args, kw, pos):
    """compose(scene, scene, ... | [scenes], label = "x"): one scene holding all their instances
    (ids must differ)."""
    no_kw("compose", kw, pos, ("label",))
    label = kw.get("label", "")
    if not isinstance(label, str):
        raise OrbError("compose() label is a string", pos)
    nodes = [n for s in scenes_of(args, pos, "compose()") for n in s.nodes]
    return Scene(nodes, label, pos)


# --------------------------------------------------------------------------- another project's outputs


def output_scene(prog, name, pos):
    """`robots.outputs.<name>` (prog: the imported project): the scene its `output "<name>"` places,
    reusable in this project like any scene (instance it, compose it, move it). Its instance ids are
    the paths under the output's prefix (`site/block0/unit` -> `block0/unit`), its label the output's.

    An output written as OrbLang gives its `value` (its only/skip on every instance that does not say
    its own). A project `orb import` rebuilt from a world graph (decompiled: an export whose sources
    imported other projects) has an empty `output` block and the placed items and sites as plain
    blocks (`location = output.<name>`, `scene = {output, path, ...}`, poses in the world frame): they
    are grouped back by path into the same tree, so the instance paths and world poses come back."""
    out = prog.block("output", name, pos)
    v = out.attrs.get("value")
    if not isinstance(v, Scene):
        raise OrbError(f"output {name!r} of the imported project has no scene `value` (got "
                       f"{type_name(v) if v is not None else 'nothing'})", pos)
    scope = {k: out.attrs[k] for k in SCOPE if k in out.attrs}
    nodes = [n if not scope or any(k in n.attrs for k in SCOPE) else
             Node(n.id, n.kind, n.target, n.pos, n.quat, {**n.attrs, **scope}, n.at, n.anim) for n in v.nodes]
    prefix = out.attrs.get("prefix", name)
    tree = {}  # id -> Node (a leaf) or dict (a group, by the next path segment)
    for typ in ("item", "site"):
        for label in [l for t, l in prog.decls if t == typ]:
            b = prog.block(typ, label, pos)
            rec = b.attrs.get("scene")
            if not isinstance(rec, dict) or rec.get("output") != name or not isinstance(rec.get("path"), str):
                continue
            path = rec["path"]
            if prefix:
                if not path.startswith(prefix + "/"):
                    continue
                path = path[len(prefix) + 1:]
            segs = path.split("/")
            if not all(ID.fullmatch(s) for s in segs):
                continue
            if typ == "item":
                target = b.attrs.get("of")
                if not isinstance(target, Block) or target.type not in ("part", "assembly"):
                    raise OrbError(f"{b.path} has no `of = part.x | assembly.x` to place", b.pos)
                p, q = pose({k: b.attrs[k] for k in POSE_KEYS if k in b.attrs and k != "face"}, b.pos)
                attrs = {k: x for k, x in b.attrs.items()
                         if k not in ("of", "location", "installed_in", "made_by", "scene") + POSE_KEYS}
                leaf = Node(segs[-1], target.type, target, p, q, attrs, b.pos)
            else:  # its floor, zones and fixtures are already in the world frame (the site carries its quat)
                leaf = Node(segs[-1], "site", b, (0.0, 0.0, 0.0), IDENT, {}, b.pos)
            level = tree
            for s in segs[:-1]:
                level = level.setdefault(s, {})
                if not isinstance(level, dict):
                    break
            else:
                level.setdefault(segs[-1], leaf)

    def build(level):
        return [Node(k, "scene", Scene(build(x), k, pos), (0.0, 0.0, 0.0), IDENT, {}, pos) if isinstance(x, dict)
                else x for k, x in level.items()]
    have = {n.id for n in nodes}
    nodes += [n for n in build(tree) if n.id not in have]
    label = out.attrs.get("label", v.label or name)
    return Scene(nodes, label if isinstance(label, str) else name, out.pos)


def _decl_labels(prog, typ):
    return [l for t, l in prog.decls if t == typ]


def _placed_by_output(b, outputs):
    rec = b.attrs.get("scene")
    return isinstance(rec, dict) and rec.get("output") in outputs


def world_scene(prog, alias, pos):
    """`room.world` (prog: the imported project): its whole world as one scene, the way `orb apply`
    in that project would build it, ready to place in this one:

    * each site it declares (id: its label), at its own `at`, with its zones and fixtures;
    * each item it declares (id: its label) where its `location` and `at` put it in that world (a zone's
      or site's position plus `at`), turned as declared, with its attributes (joints, serial, stage,
      lifecycle, body...). Where it stood (`location`, `installed_in`, `made_by`) stays behind: in this
      world it is placed by the scene, like any instance;
    * each output's scene as a group named by the output's prefix (`prefix = ""`: its instances
      directly), so `room.world` holds `site/block0/unit` just as that project's world does.

    The project's seams, timelines, cameras, sources and `geo` context are its own and stay out."""
    outputs = _decl_labels(prog, "output")
    nodes = []

    def ident(label, what, at):
        if not ID.fullmatch(label):
            raise OrbError(f"{alias}.world: {what} {label!r} cannot be an instance id (no '.', '/' or spaces); "
                           f"place it on its own with instance({alias}.{what}[\"{label}\"], id = \"...\")", at)
        return label

    sites = {}
    for label in _decl_labels(prog, "site"):
        b = prog.block("site", label, pos)
        if _placed_by_output(b, outputs):
            continue  # an output placed it (a project rebuilt from a world graph): it comes with the output
        sites[b.path] = b
        nodes.append(Node(ident(label, "site", b.pos), "site", b, (0.0, 0.0, 0.0), IDENT, {}, b.pos))
    for label in _decl_labels(prog, "item"):
        b = prog.block("item", label, pos)
        if _placed_by_output(b, outputs):
            continue
        target = b.attrs.get("of")
        if not isinstance(target, Block) or target.type not in ("part", "assembly"):
            raise OrbError(f"{b.path} has no `of = part.x | assembly.x` to place", b.pos)
        base = (0.0, 0.0, 0.0)
        loc = b.attrs.get("location")
        if isinstance(loc, Block) and loc.type in ("site", "zone"):
            site = sites.get(loc.path if loc.type == "site" else loc.path.rsplit(".zone.", 1)[0])
            if site is not None:
                base = _v3(site.attrs.get("at"))
                if loc.type == "zone":
                    base = tuple(a + c for a, c in zip(base, _v3(loc.attrs.get("at"))))
        p, q = pose({k: b.attrs[k] for k in ("at", "yaw", "rpy", "quat") if k in b.attrs}, b.pos)
        attrs = {k: x for k, x in b.attrs.items()
                 if k not in ("of", "location", "installed_in", "made_by", "scene") + POSE_KEYS}
        nodes.append(Node(ident(label, "item", b.pos), target.type, target,
                          tuple(a + c for a, c in zip(base, p)), q, attrs, b.pos))
    for name in outputs:
        s = prog.outputs(f"{alias}.outputs").lookup(name, pos)
        prefix = prog.block("output", name, pos).attrs.get("prefix", name)
        if not prefix:
            nodes.extend(s.nodes)
            continue
        segs = prefix.split("/")
        group = Node(segs[-1], "scene", s, (0.0, 0.0, 0.0), IDENT, {}, s.at or pos)
        for seg in reversed(segs[:-1]):  # prefix = "east/site": a group in a group
            group = Node(seg, "scene", Scene([group], seg, pos), (0.0, 0.0, 0.0), IDENT, {}, pos)
        nodes.append(group)
    merged = {}
    for n in nodes:  # two outputs under one group path (prefix "east/a", "east/b") share the group
        have = merged.get(n.id)
        if have is not None and have.kind == "scene" and n.kind == "scene" and not have.attrs and not n.attrs:
            merged[n.id] = Node(n.id, "scene", Scene(list(have.target.nodes) + list(n.target.nodes), n.id, pos),
                                have.pos, have.quat, {}, have.at)
        elif have is not None:
            raise OrbError(f"{alias}.world: two things are named {n.id!r} (at {have.at} and {n.at})", pos)
        else:
            merged[n.id] = n
    return Scene(list(merged.values()), alias, pos)  # instance(room.world) is placed as `room`


# --------------------------------------------------------------------------- lowering (orb/world.py)


class Leaf:
    """One part/assembly/site of a flattened scene: its instance path and world pose (at rest). chain:
    ((path, node, delay), ...) from the outermost instance down to this one, each with the total
    delay its times shift by (its own and its groups'), for leaf_motion()."""
    __slots__ = ("path", "node", "pos", "quat", "attrs", "chain")

    def __init__(self, path, node, pos, quat, attrs, chain=()):
        self.path, self.node, self.pos, self.quat, self.attrs, self.chain = path, node, pos, quat, attrs, chain


def flatten(scene, prefix="", parent=((0.0, 0.0, 0.0), IDENT), inherited=None, chain=(), delay=0.0):
    """Every leaf of a scene, depth first in declaration order. A group's only/skip applies to the
    instances inside it that do not say their own."""
    for n in scene.nodes:
        T = compose_pose(parent, (n.pos, n.quat))
        path = f"{prefix}/{n.id}" if prefix else n.id
        attrs = dict(n.attrs)
        if inherited and not any(k in attrs for k in SCOPE):
            attrs.update(inherited)
        d = delay + (n.anim.delay if n.anim else 0.0)
        ch = chain + ((path, n, d),)
        if n.kind == "scene":
            yield from flatten(n.target, path, T, {k: attrs[k] for k in SCOPE if k in attrs} or None, ch, d)
        else:
            yield Leaf(path, n, T[0], T[1], attrs, ch)


# --------------------------------------------------------------------------- animated scenes: lowering


def _q12(x):
    return float(f"{x:.12g}") + 0.0


class LeafMotion:
    """What one leaf does over time, in the world frame and absolute (timeline) time:
    track: [{t, pos, quat, ease}] or None; joints: [(t, {joint: value, ease}), ...] or None;
    visible: [[t, bool], ...] or None; spans: [(path, at, t0, t1, delay, what)] for range checks;
    appear / vanish: (t, path, at) of the instance that decides it, or None."""
    __slots__ = ("track", "joints", "visible", "spans", "appear", "vanish", "at")

    def any(self):
        return bool(self.track or self.joints or self.visible)


class _Level:
    __slots__ = ("keys", "times", "pos", "quat")

    def __init__(self, keys, pos, quat):
        self.keys, self.pos, self.quat = keys, tuple(pos), tuple(quat)
        self.times = [k["t"] for k in keys] if keys else None

    def pose(self, t, left=False):
        """The level's pose (in its parent's frame) at t; left: the limit just before t (before a
        jump: a `step` key, or two keys at one time)."""
        from .motion import ease_u, lerp, segment, slerp

        if not self.keys:
            return self.pos, self.quat
        ts, ks = self.times, self.keys
        if left:
            import bisect

            i = bisect.bisect_left(ts, t - 1e-9)
            if i < len(ts) and abs(ts[i] - t) <= 1e-9:
                k = ks[0] if i == 0 else (ks[i - 1] if ks[i]["ease"] == "step" else ks[i])
                return tuple(k["pos"]), tuple(k["quat"])
        i, j, u = segment(ts, t)
        a = ks[i]
        if i == j:
            return tuple(a["pos"]), tuple(a["quat"])
        b = ks[j]
        u = ease_u(u, b.get("ease", "linear"))
        return tuple(lerp(a["pos"], b["pos"], u)), tuple(slerp(a["quat"], b["quat"], u))

    def segment_end(self, a):
        """The key that ends the segment running after time a (None: it holds from a on)."""
        import bisect

        if not self.keys:
            return None
        j = bisect.bisect_right(self.times, a + 1e-9)
        return j if 0 < j < len(self.keys) else None


def _differ(p, q):
    from .motion import qangle

    return any(abs(x - y) > 1e-9 for x, y in zip(p[0], q[0])) or qangle(p[1], q[1]) > 1e-9


def compose_track(levels):
    """[_Level] from the outermost instance down to a leaf -> the leaf's world keys. The world pose
    is level_0(t) ∘ level_1(t) ∘ ... ∘ leaf(t). Keys go at the union of every level's key times;
    between two of them the result is exact when one ease shapes every moving level over exactly
    that span and at most one level turns with nothing offset or moving below it (then it is the
    lerp/slerp of the composed end poses). Elsewhere (a turning group carrying its children along
    arcs, a child moving inside a moving group with another ease) it is sampled every SUB_DT
    seconds and every SUB_ANGLE of turn, linear between samples."""
    from .motion import qangle

    def world(t, left=False):
        P = ((0.0, 0.0, 0.0), IDENT)
        for lv in levels:
            P = compose_pose(P, lv.pose(t, left))
        return P

    times = sorted({k["t"] for lv in levels if lv.keys for k in lv.keys})
    merged = []
    for t in times:
        if not merged or t - merged[-1] > 1e-9:
            merged.append(t)
    out = []

    def emit(t, P, ease):
        n = math.sqrt(sum(x * x for x in P[1])) or 1.0
        out.append({"t": _q12(t), "pos": [_q12(x) for x in P[0]], "quat": [_q12(x / n) for x in P[1]], "ease": ease})

    def jump(t, before):
        after = world(t)
        if _differ(before, after):
            emit(t, after, "step")

    t0 = merged[0]
    before = world(t0, left=True)
    emit(t0, before, "linear")
    jump(t0, before)
    for a, b in zip(merged, merged[1:]):
        end = world(b, left=True)
        moving = [i for i, lv in enumerate(levels) if lv.keys and _differ(lv.pose(a), lv.pose(b, left=True))]
        if not moving:
            emit(b, end, "linear")
            jump(b, end)
            continue
        eases, exact = set(), True
        for i in moving:
            lv = levels[i]
            j = lv.segment_end(a)
            e = lv.keys[j]["ease"] if j is not None else "linear"
            eases.add(e)
            if e != "linear" and (j is None or abs(lv.times[j - 1] - a) > 1e-9 or abs(lv.times[j] - b) > 1e-9):
                exact = False  # a smoothstep cut into pieces is no smoothstep of its own
        turning = [i for i in moving if qangle(levels[i].pose(a)[1], levels[i].pose(b, left=True)[1]) > 1e-9]
        if len(eases) > 1 or len(turning) > 1:
            exact = False
        elif turning:
            r = turning[0]
            below = ((0.0, 0.0, 0.0), IDENT)
            for i in range(r + 1, len(levels)):
                if i in moving:
                    exact = False
                below = compose_pose(below, levels[i].pose(a))
            if any(abs(x) > 1e-9 for x in below[0]):
                exact = False  # a turning frame carries an offset child along an arc
        if exact:
            emit(b, end, eases.pop())
        else:
            turn = sum(qangle(levels[i].pose(a)[1], levels[i].pose(b, left=True)[1]) for i in turning)
            n = min(4000, max(1, math.ceil((b - a) / SUB_DT - 1e-9), math.ceil(turn / SUB_ANGLE - 1e-9)))
            for k in range(1, n):
                t = a + (b - a) * k / n
                emit(t, world(t), "linear")
            emit(b, end, "linear")
        jump(b, end)
    return out


def leaf_motion(leaf):
    """A flattened leaf's motion, composed through every group above it (see LeafMotion)."""
    m = LeafMotion()
    m.spans, m.appear, m.vanish, m.at = [], None, None, leaf.node.at
    levels = []
    for path, n, d in leaf.chain:
        a = n.anim
        keys = None
        if a is not None and a.keys:
            keys = [{**k, "t": _q12(k["t"] + d)} for k in a.keys]
            m.spans.append((path, n.at, keys[0]["t"], keys[-1]["t"], d, "motion"))
        levels.append(_Level(keys, n.pos, n.quat))
        if a is not None and a.appear is not None and (m.appear is None or a.appear + d > m.appear[0]):
            m.appear = (_q12(a.appear + d), path, n.at)
        if a is not None and a.vanish is not None and (m.vanish is None or a.vanish + d < m.vanish[0]):
            m.vanish = (_q12(a.vanish + d), path, n.at)
    moving = [p for (p, n, _), lv in zip(leaf.chain, levels) if lv.keys]
    if leaf.node.kind == "site":
        if moving:
            raise OrbError(f"instance {moving[0]} moves, but the site {leaf.path} in it cannot (a site stands "
                           f"still): place the site outside the moving group", leaf.node.at)
        m.track = m.joints = m.visible = None
        return m
    m.track = compose_track(levels) if moving else None
    path, n, d = leaf.chain[-1]
    m.joints = None
    if n.anim is not None and n.anim.joints:
        m.joints = [(_q12(t + d), obj) for t, obj in n.anim.joints]
        m.spans.append((path, n.at, m.joints[0][0], m.joints[-1][0], d, "joint_motion"))
    A = m.appear[0] if m.appear else 0.0
    V = m.vanish[0] if m.vanish else None
    if V is not None and V <= A:
        m.visible = [[0.0, False]]
    else:
        steps = [[0.0, False], [A, True]] if A > 0 else []
        if V is not None:
            steps.append([V, False])
        m.visible = steps or None
    return m


# --------------------------------------------------------------------------- presentation: views of scenes
# Views are values: overview(scene) -> {at, look_at, fov}, for a `camera` or `shot` (`view = ...`,
# `view_to = ...`). They drive the cameras videos and players use (camera previews,
# the editor). External seams without camera support may ignore view values.


def _f(x):
    return float(x.v) if isinstance(x, Q) else float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else 0.0


def _v3(x):
    if not isinstance(x, list) or len(x) not in (2, 3):
        return (0.0, 0.0, 0.0)
    v = [_f(a) for a in x]
    return tuple(v + [0.0] * (3 - len(v)))


def _rot(b):
    """A block's own rotation (quat, rpy or yaw), identity when it has none or it is malformed."""
    q = b.attrs.get("quat")
    if isinstance(q, list) and len(q) == 4:
        v = [_f(x) for x in q]
        n = math.sqrt(sum(x * x for x in v))
        if n > 0:
            return tuple(x / n for x in v)
    r = b.attrs.get("rpy")
    if isinstance(r, list) and len(r) == 3:
        return tuple(rpy_quat(*[_f(x) for x in r]))
    return tuple(rpy_quat(0.0, 0.0, _f(b.attrs.get("yaw", 0))))


def _mesh_box(g):
    """(lo, hi) of an OBJ geom's vertices, read once while the file is unchanged; None for other
    formats or a missing file (the geom then counts by its origin)."""
    import os
    from pathlib import Path

    from .seams.meshio import memo

    f, fp = g.attrs.get("file"), g.attr_pos.get("file")
    if not isinstance(f, str) or fp is None or not f.lower().endswith(".obj"):
        return None
    path = (Path(os.path.abspath(fp.file)).parent / f).resolve()
    if not path.is_file():
        return None

    def read():
        lo, hi = [math.inf] * 3, [-math.inf] * 3
        with open(path, "rb") as fh:
            for line in fh:
                if line[:2] in (b"v ", b"v\t"):
                    try:
                        xyz = [float(t) for t in line.split()[1:4]]
                    except ValueError:
                        continue
                    for k in range(3):
                        if xyz[k] < lo[k]:
                            lo[k] = xyz[k]
                        if xyz[k] > hi[k]:
                            hi[k] = xyz[k]
        return ((tuple(lo), tuple(hi)) if lo[0] <= hi[0] else None), ()

    return memo(("aabb", str(path)), [path], read)


def _box_corners(lo, hi):
    return [tuple(hi[i] if (k >> i) & 1 else lo[i] for i in range(3)) for k in range(8)]


def _geom_points(g, pos, quat):
    """The corners of a geom's box, turned by the geom's and its frame's rotation; an OBJ mesh by its
    vertices' box (scaled). Framing, not physics: a turned box's corners, not its hull."""
    s = g.label
    if s == "box":
        h = [x / 2 for x in _v3(g.attrs.get("size"))]
        lo, hi = [-x for x in h], h
    elif s == "sphere":
        r = _f(g.attrs.get("radius", 0))
        lo, hi = [-r] * 3, [r] * 3
    elif s in ("cylinder", "capsule"):
        r, ln = _f(g.attrs.get("radius", 0)), _f(g.attrs.get("length", 0))
        z = ln / 2 + (r if s == "capsule" else 0.0)
        lo, hi = [-r, -r, -z], [r, r, z]
    else:
        box = _mesh_box(g) if s == "mesh" else None
        sc = g.attrs.get("scale")
        sc = [_f(x) for x in sc] if isinstance(sc, list) and len(sc) == 3 else [1.0, 1.0, 1.0]
        if box is None:
            lo = hi = [0.0, 0.0, 0.0]
        else:
            a = [box[0][i] * sc[i] for i in range(3)]
            b = [box[1][i] * sc[i] for i in range(3)]
            lo, hi = [min(x, y) for x, y in zip(a, b)], [max(x, y) for x, y in zip(a, b)]
    at, q = _v3(g.attrs.get("at")), qmul(quat, _rot(g))
    c = tuple(p + d for p, d in zip(pos, qrot(quat, at)))
    return [tuple(a + b for a, b in zip(c, qrot(q, p))) for p in _box_corners(lo, hi)]


def _geoms(b, pos, quat=(1.0, 0.0, 0.0, 0.0)):
    pts = [tuple(pos)]
    for g in b.kids("visual") + b.kids("collision"):
        pts += _geom_points(g, pos, quat)
    return pts


def block_points(b):
    """Points that bound a part, assembly (links at their rest pose) or site in its own frame."""
    if b.type == "part":
        return _geoms(b, (0.0, 0.0, 0.0))
    if b.type == "assembly":
        pts = []

        def walk(link, pos, quat):
            p = tuple(a + c for a, c in zip(pos, qrot(quat, _v3(link.attrs.get("at")))))
            q = qmul(quat, _rot(link))
            part = link.attrs.get("part")
            if isinstance(part, Block) and part.type == "part":
                pts.extend(_geoms(part, p, q))
            pts.extend(_geoms(link, p, q))
            for c in link.kids("link"):
                walk(c, p, q)
        for root in b.kids("link"):
            walk(root, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
        return pts or [(0.0, 0.0, 0.0)]
    if b.type == "site":
        o = _v3(b.attrs.get("at"))
        top = _f(b.attrs.get("ceiling_height", 0))
        pts = [o]
        fl = b.attrs.get("floor")
        if isinstance(fl, list) and len(fl) >= 2:
            w, d = _f(fl[0]), _f(fl[1])
            pts += [(o[0] + sx * w / 2, o[1] + sy * d / 2, o[2] + z) for sx in (-1, 1) for sy in (-1, 1)
                    for z in (0.0, top)]
        for z in b.kids("zone"):
            pts.append(tuple(a + c for a, c in zip(o, _v3(z.attrs.get("at")))))
        for fx in b.kids("fixture"):
            pts += _geoms(fx, tuple(a + c for a, c in zip(o, _v3(fx.attrs.get("at")))), _rot(fx))
        if top > 0:  # the room as one stands in it: a capture's stairwell walls above the ceiling don't count
            pts = [(x, y, min(z, o[2] + top)) for x, y, z in pts]
        return pts
    return [(0.0, 0.0, 0.0)]


def scene_box(scene, path, pos):
    """(lo, hi) in the scene's frame of every instance (or of those at/under `path`)."""
    pts = []
    for leaf in flatten(scene):
        if path and leaf.path != path and not leaf.path.startswith(path + "/"):
            continue
        for p in block_points(leaf.node.target):
            pts.append(tuple(a + c for a, c in zip(leaf.pos, qrot(leaf.quat, p))))
    if not pts:
        have = ", ".join(lf.path for lf in flatten(scene))
        raise OrbError(f"no instance {path!r} in the scene (have: {have or 'none'})" if path else
                       "an empty scene has no bounds", pos)
    return [min(p[i] for p in pts) for i in range(3)], [max(p[i] for p in pts) for i in range(3)]


def _lengths(v):
    return [Q(float(x), LENGTH) for x in v]


def _scene_arg(name, args, kw, pos, keys):
    no_kw(name, kw, pos, keys)
    if len(args) != 1 or not isinstance(args[0], Scene):
        raise OrbError(f"{name}() takes one scene, got {type_name(args[0]) if args else 'nothing'}", pos)
    path = kw.get("path", "")
    if not isinstance(path, str):
        raise OrbError(f"{name}() path is an instance path string (\"line/cell_0/robot\")", pos)
    return args[0], path


@builtin("bounds")
def _bounds(prog, args, kw, pos):
    """bounds(scene, path = "a/b"): {min, max, center, size} of the scene (or one instance in it),
    lengths in the scene's frame: every geom's box (turned), OBJ meshes by their vertices, site
    floors, zones and fixtures up to the ceiling."""
    scene, path = _scene_arg("bounds", args, kw, pos, ("path",))
    lo, hi = scene_box(scene, path, pos)
    return {"min": _lengths(lo), "max": _lengths(hi), "center": _lengths([(a + b) / 2 for a, b in zip(lo, hi)]),
            "size": _lengths([b - a for a, b in zip(lo, hi)])}


@builtin("overview")
def _overview(prog, args, kw, pos):
    """overview(scene, path = "a/b", yaw = -50 deg, pitch = 35 deg, fov = 60 deg, margin = 0.1,
    aspect = 16/9): a view {at, look_at, fov} that frames the scene (or one instance in it) from
    the azimuth `yaw` (0: from +x, 90 deg: from +y) and elevation `pitch` (90 deg: straight down)."""
    from .seams.framing import fit_distance, view_dir

    scene, path = _scene_arg("overview", args, kw, pos, ("path", "yaw", "pitch", "fov", "margin", "aspect"))
    lo, hi = scene_box(scene, path, pos)
    az = math.degrees(_angle(kw["yaw"], "yaw", pos)) if "yaw" in kw else -50.0
    el = math.degrees(_angle(kw["pitch"], "pitch", pos)) if "pitch" in kw else 35.0
    fov = _angle(kw["fov"], "fov", pos) if "fov" in kw else math.radians(60)
    if not 0 < fov < math.pi or not -90 <= el <= 90:
        raise OrbError("overview(): fov is between 0 and 180 deg, pitch between -90 and 90 deg", pos)
    el = max(-89.9, min(89.9, el))  # straight down still has an up: the top of the frame faces the yaw
    margin, aspect = kw.get("margin", 0.1), kw.get("aspect", 16 / 9)
    plain = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool)
    if not plain(margin) or margin <= -1:
        raise OrbError(f"overview(): margin is a plain number above -1 (0.1: 10% room around the scene), got "
                       f"{interpolated(margin, pos) if plain(margin) or isinstance(margin, Q) else type_name(margin)}", pos)
    if not plain(aspect) or aspect <= 0:
        raise OrbError("overview(): aspect is a plain number above 0 (width / height, 16/9 by default)", pos)
    aim = [(a + b) / 2 for a, b in zip(lo, hi)]
    corners = [[(hi if (k >> i) & 1 else lo)[i] for i in range(3)] for k in range(8)]
    d = view_dir(az, el)
    tan_v = math.tan(fov / 2)
    dist = fit_distance(corners, aim, d, tan_v * aspect, tan_v, margin, min_dist=0.5)
    return {"at": _lengths([a + dist * c for a, c in zip(aim, d)]), "look_at": _lengths(aim), "fov": Q(fov, ANGLE)}


from . import layout  # noqa: E402,F401  (registers move, row, grid, ring, frame, separate, ...: it builds on this module)
