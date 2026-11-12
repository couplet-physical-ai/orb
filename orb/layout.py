"""Orientation, vector math, clusters and cluster editing: the builtins that lay out a site.

* math on points and angles: `polar`, `rotate`, `yaw_to`, `rpy_to`, `dot`, `cross`, `norm`, `distance`,
  `normalize`, `lerp`, `linspace`, `bezier`, `path_length`, `clamp`, `wrap`, `pow`, `exp`, `log`;
* clusters, one value for many placed things: `cluster` (at any list of points), `row`, `grid`,
  `arc`, `ring` and `along` (a polyline or curve), each returning an ordinary scene;
* editing a scene by instance path (globs: `row*/unit_1?`, `**/module_n`): `move(..., path, about, face)`,
  `without`, `frame` and `transform`; boxes in plan: `overlaps`, `clashes`, `separate`, `beside`.

Everything is pure, like every builtin: a call returns a new scene and never changes the one it was
given, so a cluster held in a `let` can be placed, moved and edited as often as needed.
"""
from __future__ import annotations

import math
import re

from .builtins import BUILTINS, arity, builtin, no_kw
from .lang import DIMLESS, Block, Builtin, OrbError, Q, _dims, _q, _val, arith, dim_name, type_name
from .scenes import (ANGLE, IDENT, LENGTH, Node, Scene, _angle, _geoms, _length, _rot, block_points, compose_pose,
                     default_id, flatten, heading, make_node, point_like, pose, qmul, qrot, rpy_quat, scenes_of)

ORIGIN = ((0.0, 0.0, 0.0), IDENT)
MAX_COUNT = 10000


def _plain(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _whole(x, what, pos, lo=0):
    if isinstance(x, float) and x.is_integer():
        x = int(x)
    if not isinstance(x, int) or isinstance(x, bool) or x < lo:
        raise OrbError(f"{what} is a whole number{f' (at least {lo})' if lo else ''}, got "
                       f"{x if _plain(x) else type_name(x)}", pos)
    if x > MAX_COUNT:
        raise OrbError(f"{what} of {x} is too many (at most {MAX_COUNT})", pos)
    return x


def _L(v):
    return [Q(float(x), LENGTH) for x in v]


def _A(x):
    return Q(float(x), ANGLE)


def _qinv(q):
    return (q[0], -q[1], -q[2], -q[3])


def _inverse(T):
    p, q = T
    qi = _qinv(q)
    return tuple(-x for x in qrot(qi, p)), qi


def _quat_rpy(q):
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    return roll, pitch, heading(q)


def _rotation(kw, pos):
    """yaw / rpy / quat -> a quaternion (identity when none is given)."""
    return pose({k: kw[k] for k in ("yaw", "rpy", "quat") if k in kw}, pos)[1]


def _vec(v, what, pos):
    if not isinstance(v, list) or not v:
        raise OrbError(f"{what} takes a vector (a list of numbers or quantities), got {type_name(v)}", pos)
    for x in v:
        if not (_plain(x) or isinstance(x, Q)):
            raise OrbError(f"{what} takes a vector of numbers or quantities, got a {type_name(x)} in it", pos)
    return v


def _same_unit(v, what, pos):
    """One dimension for every component (a bare 0 fits any) -> that dimension."""
    ds = {_dims(x) for x in v if _val(x) != 0} or {_dims(v[0])}
    if len(ds) > 1:
        raise OrbError(f"{what}: the components have different units", pos)
    return ds.pop()


def _norm_f(v, what, pos):
    d = _same_unit(_vec(v, what, pos), what, pos)
    return math.sqrt(sum(float(_val(x)) ** 2 for x in v)), d


# --------------------------------------------------------------------------- vector and angle math


@builtin("dot")
def _dot(prog, args, kw, pos):
    """dot(a, b): the dot product of two vectors (units multiply: dot([1, 0] m, [2, 0] N) is 2 Nm)"""
    no_kw("dot", kw, pos)
    arity("dot", args, 2, pos=pos)
    a, b = (_vec(x, "dot()", pos) for x in args)
    if len(a) != len(b):
        raise OrbError(f"dot(): vectors of {len(a)} and {len(b)}", pos)
    out = 0
    for x, y in zip(a, b):
        out = arith("+", out, arith("*", x, y, pos), pos)
    return out


@builtin("cross")
def _cross(prog, args, kw, pos):
    """cross(a, b): the cross product of two 3-vectors (units multiply)"""
    no_kw("cross", kw, pos)
    arity("cross", args, 2, pos=pos)
    a, b = (_vec(x, "cross()", pos) for x in args)
    if len(a) != 3 or len(b) != 3:
        raise OrbError("cross() takes two [x, y, z] vectors", pos)
    m = lambda x, y: arith("*", x, y, pos)
    s = lambda x, y: arith("-", x, y, pos)
    return [s(m(a[1], b[2]), m(a[2], b[1])), s(m(a[2], b[0]), m(a[0], b[2])), s(m(a[0], b[1]), m(a[1], b[0]))]


@builtin("norm")
def _norm(prog, args, kw, pos):
    """norm(v): the length of a vector, in its unit (norm([3, 4, 0] m) is 5 m)"""
    no_kw("norm", kw, pos)
    arity("norm", args, 1, pos=pos)
    n, d = _norm_f(args[0], "norm()", pos)
    return _q(n, d)


@builtin("distance")
def _distance(prog, args, kw, pos):
    """distance(a, b): how far apart two points are (scenes and blocks count by their centre)"""
    no_kw("distance", kw, pos)
    arity("distance", args, 2, pos=pos)
    a, b = (point_like(x, "distance()", pos) for x in args)
    return Q(math.dist(a, b), LENGTH)


@builtin("normalize")
def _normalize(prog, args, kw, pos):
    """normalize(v): the direction of a vector, as plain numbers of length 1 (normalize([0, 2, 0] m) is [0, 1, 0])"""
    no_kw("normalize", kw, pos)
    arity("normalize", args, 1, pos=pos)
    n, _ = _norm_f(args[0], "normalize()", pos)
    if n == 0:
        raise OrbError("normalize() of a zero vector has no direction", pos)
    return [float(_val(x)) / n for x in args[0]]


@builtin("lerp")
def _lerp(prog, args, kw, pos):
    """lerp(a, b, t): a + (b - a) * t, of numbers, quantities or points (t = 0.5: halfway)"""
    no_kw("lerp", kw, pos)
    arity("lerp", args, 3, pos=pos)
    a, b, t = args
    if not _plain(t):
        raise OrbError(f"lerp(a, b, t): t is a plain number (0 at a, 1 at b), got {type_name(t)}", pos)
    return arith("+", a, arith("*", arith("-", b, a, pos), t, pos), pos)


@builtin("linspace")
def _linspace(prog, args, kw, pos):
    """linspace(a, b, n): n evenly spaced values from a to b, both included; numbers, quantities or
    points (linspace(0 m, 10 m, 3) is [0, 5, 10] m)"""
    no_kw("linspace", kw, pos)
    arity("linspace", args, 3, pos=pos)
    a, b, n = args
    n = _whole(n, "linspace() n", pos, 1)
    if n == 1:
        return [a]
    return [_lerp(prog, [a, b, i / (n - 1)], {}, pos) for i in range(n)]


@builtin("bezier")
def _bezier(prog, args, kw, pos):
    """bezier(points, n): n points along the Bezier curve with these control points (the first and
    last are on it); feed it to along() or cluster() for a curved row"""
    no_kw("bezier", kw, pos)
    arity("bezier", args, 2, pos=pos)
    ctrl, n = args
    if not isinstance(ctrl, list) or len(ctrl) < 2 or not all(isinstance(c, list) for c in ctrl):
        raise OrbError("bezier() takes a list of at least two control points ([[0, 0, 0] m, [5, 5, 0] m, ...])", pos)
    n = _whole(n, "bezier() n", pos, 2)
    out = []
    for i in range(n):
        t, pts = i / (n - 1), ctrl
        while len(pts) > 1:
            pts = [_lerp(prog, [pts[k], pts[k + 1], t], {}, pos) for k in range(len(pts) - 1)]
        out.append(pts[0])
    return out


def _points(xs, what, pos):
    if not isinstance(xs, list) or not all(isinstance(p, list) for p in xs):
        got = (f"a list holding {type_name(next(p for p in xs if not isinstance(p, list)))}"
               if isinstance(xs, list) else type_name(xs))
        raise OrbError(f"{what} takes a list of points ([[0, 0, 0] m, [10, 0, 0] m, ...]), got {got}", pos)
    return [point_like(p, what, pos) for p in xs]


@builtin("path_length")
def _path_length(prog, args, kw, pos):
    """path_length(points): the length of the polyline through these points"""
    no_kw("path_length", kw, pos)
    arity("path_length", args, 1, pos=pos)
    pts = _points(args[0], "path_length()", pos)
    return Q(sum(math.dist(a, b) for a, b in zip(pts, pts[1:])), LENGTH)


@builtin("polar")
def _polar(prog, args, kw, pos):
    """polar(r, angle, center = [0, 0, 0] m): the point r away from center at this angle (0: +x, 90 deg:
    +y); center may be a scene or a block (its centre). Points around a statue: polar(25 m, i * 30 deg, center = rock)"""
    no_kw("polar", kw, pos, ("center",))
    arity("polar", args, 2, pos=pos)
    r = _length(args[0], "polar() r", pos)
    a = _angle(args[1], "polar() angle", pos)
    c = point_like(kw["center"], "polar() center", pos) if "center" in kw else (0.0, 0.0, 0.0)
    return _L([c[0] + r * math.cos(a), c[1] + r * math.sin(a), c[2]])


def _about(kw, pos, default=(0.0, 0.0, 0.0)):
    return point_like(kw["about"], "about", pos) if "about" in kw else default


@builtin("rotate")
def _rotate(prog, args, kw, pos):
    """rotate(x, yaw = a | rpy = [...] | quat = [...], about = point): a point turned about another
    (default the origin), or a scene turned (rotate(scene, ...) is move(scene, yaw = ..., about = ...),
    path too). about may be a scene or block: its centre."""
    arity("rotate", args, 1, pos=pos)
    x = args[0]
    if isinstance(x, (Scene, Block)):
        no_kw("rotate", kw, pos, ("yaw", "rpy", "quat", "about", "path"))
        return _move(prog, [x], kw, pos)
    no_kw("rotate", kw, pos, ("yaw", "rpy", "quat", "about"))
    if not isinstance(x, list):
        raise OrbError(f"rotate() turns a point or a scene, got {type_name(x)}", pos)
    p = point_like(x, "rotate()", pos)
    c, q = _about(kw, pos), _rotation(kw, pos)
    r = qrot(q, tuple(a - b for a, b in zip(p, c)))
    return _L([a + b for a, b in zip(c, r)][:3])


@builtin("yaw_to")
def _yaw_to(prog, args, kw, pos):
    """yaw_to(from, to): the yaw that turns +x at `from` toward `to` (points, scenes or blocks)"""
    no_kw("yaw_to", kw, pos)
    arity("yaw_to", args, 2, pos=pos)
    a, b = (point_like(x, "yaw_to()", pos) for x in args)
    if math.isclose(a[0], b[0], abs_tol=1e-12) and math.isclose(a[1], b[1], abs_tol=1e-12):
        same = all(math.isclose(x, y, abs_tol=1e-12) for x, y in zip(a, b))
        raise OrbError("yaw_to(): " + ("from and to are the same point" if same else "the points are one above the "
                       "other") + ", so there is no heading", pos)
    return _A(math.atan2(b[1] - a[1], b[0] - a[0]))


@builtin("rpy_to")
def _rpy_to(prog, args, kw, pos):
    """rpy_to(from, to, roll = 0 deg): the rpy that points +x at `from` straight at `to`, tilting up or
    down (a dish, a camera, a solar tracker’s normal); rpy = rpy_to(...) on a link, geom or instance"""
    no_kw("rpy_to", kw, pos, ("roll",))
    arity("rpy_to", args, 2, pos=pos)
    a, b = (point_like(x, "rpy_to()", pos) for x in args)
    d = [y - x for x, y in zip(a, b)]
    if not any(abs(c) > 1e-12 for c in d):
        raise OrbError("rpy_to(): from and to are the same point", pos)
    roll = _angle(kw["roll"], "roll", pos) if "roll" in kw else 0.0
    return [_A(roll), _A(-math.atan2(d[2], math.hypot(d[0], d[1]))), _A(math.atan2(d[1], d[0]))]


@builtin("wrap")
def _wrap(prog, args, kw, pos):
    """wrap(angle): the same heading between -180 and 180 deg"""
    no_kw("wrap", kw, pos)
    arity("wrap", args, 1, pos=pos)
    a = float(args[0]) if _plain(args[0]) else _angle(args[0], "wrap()", pos)  # a plain number: radians
    return _A((a + math.pi) % (2 * math.pi) - math.pi)


@builtin("clamp")
def _clamp(prog, args, kw, pos):
    """clamp(x, lo, hi): x kept between lo and hi (units must agree)"""
    no_kw("clamp", kw, pos)
    arity("clamp", args, 3, pos=pos)
    x, lo, hi = args
    if arith(">", lo, hi, pos):
        raise OrbError("clamp(x, lo, hi): lo is above hi", pos)
    return lo if arith("<", x, lo, pos) else hi if arith(">", x, hi, pos) else x


@builtin("pow")
def _pow(prog, args, kw, pos):
    """pow(x, n): x to the power n; a quantity takes a whole n (pow(2 m, 2) is 4 m2)"""
    no_kw("pow", kw, pos)
    arity("pow", args, 2, pos=pos)
    x, n = args
    if not (_plain(x) or isinstance(x, Q)) or not _plain(n):
        raise OrbError("pow(x, n): x is a number or quantity, n a plain number", pos)
    d = _dims(x)
    if d != DIMLESS and not float(n).is_integer():
        raise OrbError(f"pow() of {dim_name(d)} takes a whole power", pos)
    try:
        v = float(_val(x)) ** n
    except (OverflowError, ZeroDivisionError):
        raise OrbError(f"pow({_val(x):g}, {n:g}) is undefined", pos) from None
    if isinstance(v, complex):
        raise OrbError(f"pow({_val(x):g}, {n:g}) is not a real number", pos)
    return _q(v, tuple(int(e * n) for e in d))


def _plain_unary(name, f, doc):
    def impl(prog, args, kw, pos):
        no_kw(name, kw, pos)
        arity(name, args, 1, 2 if name == "log" else 1, pos)
        if not all(_plain(a) for a in args):
            raise OrbError(f"{name}() takes plain numbers", pos)
        try:
            return f(*args)
        except (ValueError, OverflowError, ZeroDivisionError):
            raise OrbError(f"{name}({', '.join(f'{a:g}' for a in args)}) is undefined", pos) from None
    BUILTINS[name] = Builtin(name, impl, doc)


_plain_unary("exp", math.exp, "exp(x): e to the power x")
_plain_unary("log", math.log, "log(x), log(x, base): the natural (or base) logarithm")


# --------------------------------------------------------------------------- binding arguments


def _bind(what, args, kw, names, pos, required=()):
    """Positional arguments by `names`, then the named ones -> one dict."""
    if len(args) > len(names):
        raise OrbError(f"{what}() takes at most {len(names)} positional arguments ({', '.join(names)}), "
                       f"got {len(args)}", pos)
    out = dict(kw)
    for n, v in zip(names, args):
        if n in out:
            raise OrbError(f"{what}() got {n!r} twice", pos)
        out[n] = v
    for n in required:
        if n not in out:
            raise OrbError(f"{what}() needs {n!r}", pos)
    return out


# --------------------------------------------------------------------------- clusters

CLUSTER_KEYS = ("prefix", "label", "yaw", "face")
NOT_IN_CLUSTER = {"at": "move the whole cluster: move(row(...), at = ...)",
                  "rpy": "give the definition its own rotation, or turn the cluster with move()",
                  "quat": "turn the cluster with move()", "id": "ids are prefix + index; set prefix = \"...\""}


# what each cluster takes; a layout word it does not take is a misspelling, not an item attribute
PARAMS = {"cluster": ("of", "points"), "row": ("of", "n", "pitch", "centered"),
          "grid": ("of", "cols", "rows", "pitch", "count", "centered", "stagger", "row_prefix"),
          "arc": ("of", "n", "radius", "center", "from", "to"), "ring": ("of", "n", "radius", "center", "start"),
          "along": ("of", "path", "n", "spacing")}
LAYOUT_WORDS = {"n", "count", "number", "num", "pitch", "spacing", "step", "radius", "r", "center", "centre",
                "centered", "centred", "from", "to", "start", "end", "stop", "begin", "angle", "sweep", "span",
                "cols", "rows", "columns", "stagger", "row_prefix", "points", "path", "origin", "direction",
                "around", "about", "diameter"}
SAY = {("arc", "start"): "from", ("arc", "end"): "to", ("arc", "stop"): "to", ("arc", "begin"): "from",
       ("ring", "from"): "start", ("ring", "begin"): "start", ("along", "count"): "n", ("row", "count"): "n",
       ("arc", "count"): "n", ("ring", "count"): "n", ("along", "step"): "spacing", ("along", "pitch"): "spacing",
       ("row", "spacing"): "pitch", ("grid", "spacing"): "pitch", ("row", "step"): "pitch",
       ("arc", "around"): "center", ("ring", "around"): "center", ("arc", "centre"): "center",
       ("ring", "centre"): "center", ("row", "centred"): "centered", ("grid", "centred"): "centered",
       ("arc", "r"): "radius", ("ring", "r"): "radius", ("grid", "columns"): "cols"}


def _check_params(what, kw, pos):
    for k in kw:
        if k in LAYOUT_WORDS and k not in PARAMS[what] and k not in CLUSTER_KEYS:
            use = SAY.get((what, k))
            takes = ", ".join(PARAMS[what] + CLUSTER_KEYS)
            raise OrbError(f"{what}() has no {k!r}" + (f" (use {use})" if use else "") + f"; {what}() takes {takes} "
                           "(other names go on every instance as item attributes)", pos)


def _of_list(of, pos, what):
    ofs = of if isinstance(of, list) else [of]
    if not ofs:
        raise OrbError(f"{what}() needs something to place", pos)
    for o in ofs:
        if not isinstance(o, (Block, Scene)):
            raise OrbError(f"{what}() places parts, assemblies, items, sites or scenes (or a list of them, used "
                           f"in turn), got {type_name(o)}", pos)
    return ofs


def _face_yaw(face, p, pos, radial=None, tangent=None, what="face"):
    """The heading (radians) of an instance at p: face = "none" | "in" | "out" | "along" | a point."""
    if face is None or face == "none":
        return 0.0
    if isinstance(face, str):
        if face == "in" and radial is not None:
            return radial + math.pi
        if face == "out" and radial is not None:
            return radial
        if face == "along" and tangent is not None:
            return tangent
        ok = ["\"none\""] + (["\"in\"", "\"out\""] if radial is not None else []) + (["\"along\""] if tangent is not None else [])
        raise OrbError(f"{what} = {face!r}: here face is {', '.join(ok)} or a point (a scene or block: where it "
                       "stands); \"in\" and \"out\" are for arc() and ring(), \"along\" for arc(), ring() and "
                       "along(): turn a row with move(row(...), yaw = ...)", pos)
    t = point_like(face, "face", pos)
    if math.isclose(t[0], p[0], abs_tol=1e-9) and math.isclose(t[1], p[1], abs_tol=1e-9):
        return 0.0  # standing on the point it faces: no heading to take
    return math.atan2(t[1] - p[1], t[0] - p[0])


def _cluster_scene(what, of, places, kw, pos):
    """places: [(point, heading)] in the cluster's frame -> a scene of len(places) instances, ids prefix + i.
    kw: prefix, label, yaw (an extra turn, or one per place) and item attributes for every instance."""
    ofs = _of_list(of, pos, what)
    _check_params(what, kw, pos)
    for k, hint in NOT_IN_CLUSTER.items():
        if k in kw:
            raise OrbError(f"{what}() has no {k!r} ({hint})", pos)
    extras = {k: v for k, v in kw.items() if k not in CLUSTER_KEYS}
    prefix = kw.get("prefix")
    if prefix is None:
        prefix = ((default_id(ofs[0]) if len(ofs) == 1 else "") or "unit") + "_"
    if not isinstance(prefix, str):
        raise OrbError(f"{what}() prefix is a string (ids are prefix + index: \"unit_\" -> unit_0, unit_1, ...)", pos)
    label = kw.get("label", "")
    if not isinstance(label, str):
        raise OrbError(f"{what}() label is a string", pos)
    yaw = kw.get("yaw", 0)
    if isinstance(yaw, list):
        if len(yaw) != len(places):
            raise OrbError(f"{what}() yaw is one angle, or a list of one per instance ({len(places)}), got {len(yaw)}", pos)
        extra = [_angle(y, "yaw", pos) for y in yaw]
    else:
        extra = [_angle(yaw, "yaw", pos)] * len(places)
    if len(places) > MAX_COUNT:
        raise OrbError(f"{what}() of {len(places)} instances is too many (at most {MAX_COUNT}): nest clusters", pos)
    protos, nodes = {}, []
    for i, ((p, h), e) in enumerate(zip(places, extra)):
        o = ofs[i % len(ofs)]
        proto = protos.get(id(o))
        if proto is None:
            try:  # one check of the definition and attributes, then copies; it checks the prefix too
                proto = protos[id(o)] = make_node(o, {"id": prefix + "0", **extras}, pos)
            except OrbError as e:
                if not e.msg.startswith("instance() has no"):
                    raise
                raise OrbError(f"{what}()" + e.msg[len("instance()"):].replace("go on the item", "go on every instance"),
                               pos) from None
        nodes.append(renode(proto, p, rpy_quat(0, 0, h + e), f"{prefix}{i}"))
    return Scene(nodes, label, pos)


def _length_kw(d, k, what, pos, default=None):
    if k not in d:
        if default is None:
            raise OrbError(f"{what}() needs {k!r} (a length)", pos)
        return default
    return _length(d[k], f"{what}() {k}", pos)


def _bool_kw(d, k, what, pos):
    v = d.get(k, False)
    if not isinstance(v, bool):
        raise OrbError(f"{what}() {k} is true or false", pos)
    return v


@builtin("cluster")
def _cluster(prog, args, kw, pos):
    """cluster(of, points, yaw = 0 deg | [yaws], face = point, prefix = "<label>_", label = "", ...attrs): one
    scene of `of` placed at every point (ids prefix + index: unit_0, unit_1, ...). `of` may be a list, used in
    turn. face turns each toward a point (a scene or block: its centre); yaw is an extra turn, or one per
    point. Other attributes (stage, only, skip, ...) go on every instance."""
    d = _bind("cluster", args, kw, ("of", "points"), pos, ("of", "points"))
    of, pts = d.pop("of"), _points(d.pop("points"), "cluster() points", pos)
    face = d.get("face")
    return _cluster_scene("cluster", of, [(p, _face_yaw(face, p, pos)) for p in pts], d, pos)


@builtin("row")
def _row(prog, args, kw, pos):
    """row(of, n, pitch, centered = false, face = point, yaw, prefix, label, ...attrs): n instances along
    +x, `pitch` apart, the first at the origin (centered: the middle). Turn or move the whole row with
    move(row(...), yaw = ..., at = ...)."""
    d = _bind("row", args, kw, ("of", "n", "pitch"), pos, ("of", "n", "pitch"))
    n = _whole(d.pop("n"), "row() n", pos)
    pitch = _length_kw(d, "pitch", "row", pos)
    d.pop("pitch")
    x0 = -(n - 1) * pitch / 2 if _bool_kw(d, "centered", "row", pos) else 0.0
    d.pop("centered", None)
    face = d.get("face")
    pts = [(x0 + i * pitch, 0.0, 0.0) for i in range(n)]
    return _cluster_scene("row", d.pop("of"), [(p, _face_yaw(face, p, pos)) for p in pts], d, pos)


@builtin("grid")
def _grid(prog, args, kw, pos):
    """grid(of, cols, rows, pitch = [dx, dy] m, count = cols * rows, centered = false, stagger = 0 m,
    row_prefix = "row", face, yaw, prefix, label, ...attrs): rows of `cols` along +x, row r at y = r * dy.
    Each row is its own instance (row0/unit_0, row0/unit_1, ...), so move(g, path = "row3", ...) moves one row.
    count stops early (row by row); stagger shifts every other row along x."""
    d = _bind("grid", args, kw, ("of", "cols", "rows", "pitch"), pos, ("of", "cols", "rows", "pitch"))
    cols, rows = _whole(d.pop("cols"), "grid() cols", pos), _whole(d.pop("rows"), "grid() rows", pos)
    pt = d.pop("pitch")
    if isinstance(pt, list) and len(pt) == 2:
        dx, dy = (_length(v, "grid() pitch", pos) for v in pt)
    else:
        dx = dy = _length(pt, "grid() pitch (a length, or [dx, dy])", pos)
    count = _whole(d.pop("count", cols * rows), "grid() count", pos)
    stagger = _length_kw(d, "stagger", "grid", pos, 0.0)
    d.pop("stagger", None)
    centered = _bool_kw(d, "centered", "grid", pos)
    d.pop("centered", None)
    rp = d.pop("row_prefix", "row")
    if not isinstance(rp, str):
        raise OrbError("grid() row_prefix is a string", pos)
    label = d.pop("label", "")
    of, face = d.pop("of"), d.get("face")
    if cols * rows > MAX_COUNT:
        raise OrbError(f"grid() of {cols * rows} instances is too many (at most {MAX_COUNT}): nest grids", pos)
    x0 = -(cols - 1) * dx / 2 if centered else 0.0
    y0 = -(rows - 1) * dy / 2 if centered else 0.0
    nodes = []
    for r in range(rows):
        k = min(cols, count - r * cols)
        if k <= 0:
            break
        off = (x0 + (stagger if r % 2 else 0.0), y0 + r * dy, 0.0)
        places = [((i * dx, 0.0, 0.0), _face_yaw(face, (off[0] + i * dx, off[1], 0.0), pos)) for i in range(k)]
        sub = _cluster_scene("grid", of, places, d, pos)
        nodes.append(Node(f"{rp}{r}", "scene", sub, off, IDENT, {}, pos))
    return Scene(nodes, label, pos) if nodes else Scene([], label, pos)


def _arc_places(n, radius, c, angles, face, sweep, pos):
    out = []
    for a in angles:
        p = (c[0] + radius * math.cos(a), c[1] + radius * math.sin(a), c[2])
        out.append((p, _face_yaw(face, p, pos, radial=a, tangent=a + math.copysign(math.pi / 2, sweep))))
    return out


@builtin("arc")
def _arc(prog, args, kw, pos):
    """arc(of, n, radius, center = [0, 0, 0] m, from = 0 deg, to = 180 deg, face = "none" | "in" | "out" |
    "along" | point, yaw, prefix, label, ...attrs): n instances on a circular arc, both ends included.
    face "in" turns each +x toward the centre, "out" away, "along" along the arc (from -> to)."""
    d = _bind("arc", args, kw, ("of", "n", "radius"), pos, ("of", "n", "radius"))
    n = _whole(d.pop("n"), "arc() n", pos)
    r = _length(d.pop("radius"), "arc() radius", pos)
    c = point_like(d.pop("center"), "arc() center", pos) if "center" in d else (0.0, 0.0, 0.0)
    a0 = _angle(d.pop("from"), "arc() from", pos) if "from" in d else 0.0
    a1 = _angle(d.pop("to"), "arc() to", pos) if "to" in d else math.pi
    angles = [a0] if n == 1 else [a0 + (a1 - a0) * i / (n - 1) for i in range(n)]
    return _cluster_scene("arc", d.pop("of"), _arc_places(n, r, c, angles, d.get("face"), a1 - a0 or 1.0, pos), d, pos)


@builtin("ring")
def _ring(prog, args, kw, pos):
    """ring(of, n, radius, center = [0, 0, 0] m, start = 0 deg, face = "none" | "in" | "out" | "along" |
    point, yaw, prefix, label, ...attrs): n instances evenly around a full circle (counter-clockwise from
    `start`). Units around a statue: ring(part.unit, 12, 30 m, center = rock, face = "in")"""
    d = _bind("ring", args, kw, ("of", "n", "radius"), pos, ("of", "n", "radius"))
    n = _whole(d.pop("n"), "ring() n", pos, 1)
    r = _length(d.pop("radius"), "ring() radius", pos)
    c = point_like(d.pop("center"), "ring() center", pos) if "center" in d else (0.0, 0.0, 0.0)
    a0 = _angle(d.pop("start"), "ring() start", pos) if "start" in d else 0.0
    angles = [a0 + 2 * math.pi * i / n for i in range(n)]
    return _cluster_scene("ring", d.pop("of"), _arc_places(n, r, c, angles, d.get("face"), 1.0, pos), d, pos)


@builtin("along")
def _along(prog, args, kw, pos):
    """along(of, path, n = | spacing =, face = "none" | "along" | point, yaw, prefix, label, ...attrs):
    instances along a polyline of points (a curve: bezier(...)): n evenly from its start to its end, or
    one every `spacing` from the start. face "along" turns each +x along the path."""
    d = _bind("along", args, kw, ("of", "path"), pos, ("of", "path"))
    pts = _points(d.pop("path"), "along() path", pos)
    if len(pts) < 2:
        raise OrbError("along() path takes at least two points", pos)
    segs = [(a, b, math.dist(a, b)) for a, b in zip(pts, pts[1:]) if math.dist(a, b) > 0]
    total = sum(s[2] for s in segs)
    if total == 0:
        raise OrbError("along(): the path has no length", pos)
    if ("n" in d) == ("spacing" in d):
        raise OrbError("along() takes n (how many, evenly from start to end) or spacing (how far apart), one of them", pos)
    if "n" in d:
        n = _whole(d.pop("n"), "along() n", pos, 1)
        ss = [0.0] if n == 1 else [total * i / (n - 1) for i in range(n)]
    else:
        sp = _length(d.pop("spacing"), "along() spacing", pos)
        if sp <= 0:
            raise OrbError("along() spacing is above 0 m", pos)
        n = int(total / sp + 1e-9) + 1
        if n > MAX_COUNT:
            raise OrbError(f"along(): {n} instances is too many (at most {MAX_COUNT})", pos)
        ss = [i * sp for i in range(n)]
    face, places, k, base = d.get("face"), [], 0, 0.0
    for s in ss:
        while k < len(segs) - 1 and s > base + segs[k][2] + 1e-12:
            base += segs[k][2]
            k += 1
        a, b, ln = segs[k]
        t = min(1.0, max(0.0, (s - base) / ln))
        p = tuple(x + (y - x) * t for x, y in zip(a, b))
        places.append((p, _face_yaw(face, p, pos, tangent=math.atan2(b[1] - a[1], b[0] - a[0]))))
    return _cluster_scene("along", d.pop("of"), places, d, pos)


# --------------------------------------------------------------------------- instance paths and boxes


def renode(n, p, q, id=None):
    """Node n at a new pose in its parent's frame; its motion (an animated instance's keys) moves with it."""
    anim = getattr(n, "anim", None)
    more = () if anim is None else (anim.moved(compose_pose((p, q), _inverse((n.pos, n.quat)))),)
    return Node(id or n.id, n.kind, n.target, p, q, n.attrs, n.at, *more)


class Glob:
    """Instance paths to match: (regex, innermost only) pairs, any of which matches."""

    def __init__(self, parts):
        self.parts = parts

    def fullmatch(self, p, node=None):
        return any(rx.fullmatch(p) and not (inner and node is not None and node.kind == "scene")
                   for rx, inner in self.parts)


def glob_rx(path, what, pos):
    """An instance path with globs: * within one level, ? one character, ** any number of levels (at the
    end: the innermost instances, the parts and assemblies, at any depth); a list of them matches any."""
    if isinstance(path, list) and path:
        return Glob([pr for p in path for pr in glob_rx(p, what, pos).parts])
    if not isinstance(path, str) or not path:
        raise OrbError(f"{what} path is an instance path string (\"row3\", \"row*/unit_1?\", \"**/module_n\")", pos)
    out, i = "", 0
    while i < len(path):
        if path.startswith("**/", i):
            out, i = out + "(?:[^/]+/)*", i + 3
        elif path.startswith("**", i):
            out, i = out + ".*", i + 2
        elif path[i] == "*":
            out, i = out + "[^/]*", i + 1
        elif path[i] == "?":
            out, i = out + "[^/]", i + 1
        else:
            out, i = out + re.escape(path[i]), i + 1
    return Glob([(re.compile(out), path == "**" or path.endswith("/**"))])


def all_paths(scene, prefix=""):
    for n in scene.nodes:
        p = f"{prefix}/{n.id}" if prefix else n.id
        yield p
        if n.kind == "scene":
            yield from all_paths(n.target, p)


def matches(scene, rx, P=ORIGIN, prefix=""):
    """(path, node, parent pose in the scene's frame) of every instance the path matches, outermost first:
    the instances inside a match move with it and are not matched again."""
    for n in scene.nodes:
        p = f"{prefix}/{n.id}" if prefix else n.id
        if rx.fullmatch(p, n):
            yield p, n, P
        elif n.kind == "scene":
            yield from matches(n.target, rx, compose_pose(P, (n.pos, n.quat)), p)


def _no_match(scene, path, what, pos):
    from .lang import did_you_mean

    have = list(all_paths(scene))
    shown = ", ".join(have[:12]) + (f", ... ({len(have)} in all)" if len(have) > 12 else "")
    raise OrbError(f"{what}: no instance matches path {path!r}{did_you_mean(path, have)} (has: {shown or 'none'})", pos)


def edit(scene, rx, fn, P=ORIGIN, prefix=""):
    """A new scene with every matched node replaced by fn(path, node, parent pose) (a list: none drops
    it); only the groups on the way to a match are rebuilt. -> (scene, matches)."""
    out, hit = [], 0
    for n in scene.nodes:
        p = f"{prefix}/{n.id}" if prefix else n.id
        if rx.fullmatch(p, n):
            out.extend(fn(p, n, P))
            hit += 1
        elif n.kind == "scene":
            sub, h = edit(n.target, rx, fn, compose_pose(P, (n.pos, n.quat)), p)
            out.append(Node(n.id, "scene", sub, n.pos, n.quat, n.attrs, n.at,
                            *(() if getattr(n, "anim", None) is None else (n.anim,))) if h else n)
            hit += h
        else:
            out.append(n)
    return (Scene(out, scene.label, scene.at) if hit else scene), hit


_BOX_CACHE = {}


def _local_box(block):
    """A block's box in its own frame (cached per block value: blocks are immutable once built)."""
    k = id(block)
    hit = _BOX_CACHE.get(k)
    if hit is not None and hit[0] is block:
        return hit[1]
    pts = block_points(block)
    box = ([min(p[i] for p in pts) for i in range(3)], [max(p[i] for p in pts) for i in range(3)])
    if len(_BOX_CACHE) > 4096:
        _BOX_CACHE.clear()
    _BOX_CACHE[k] = (block, box)
    return box


def _turned_box(lo, hi, T):
    p, q = T
    if q == IDENT:
        return [a + b for a, b in zip(lo, p)], [a + b for a, b in zip(hi, p)]
    cs = [qrot(q, tuple(hi[i] if (k >> i) & 1 else lo[i] for i in range(3))) for k in range(8)]
    return [min(c[i] for c in cs) + p[i] for i in range(3)], [max(c[i] for c in cs) + p[i] for i in range(3)]


def _hull(pts):
    """The convex hull of 2D points, counter-clockwise (monotone chain)."""
    pts = sorted(set(pts))
    if len(pts) <= 2:
        return pts

    def half(ps):
        out = []
        for p in ps:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (p[1] - out[-2][1])
                                     - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])) <= 1e-12:
                out.pop()
            out.append(p)
        return out
    lower, upper = half(pts), half(reversed(pts))
    return lower[:-1] + upper[:-1]


def _part_box(lo, hi, T):
    """A part's box placed by T, for clash tests: (lo, hi, footprint), the footprint the turned box's
    outline in plan (a polygon), None when it is not turned about z (its box is then exact)."""
    p, q = T
    if q == IDENT or (abs(q[1]) < 1e-12 and abs(q[2]) < 1e-12 and abs(q[3]) < 1e-12):
        return [a + b for a, b in zip(lo, p)], [a + b for a, b in zip(hi, p)], None
    cs = [qrot(q, tuple(hi[i] if (k >> i) & 1 else lo[i] for i in range(3))) for k in range(8)]
    cs = [tuple(c[i] + p[i] for i in range(3)) for c in cs]
    lo2, hi2 = [min(c[i] for c in cs) for i in range(3)], [max(c[i] for c in cs) for i in range(3)]
    return lo2, hi2, _hull([(round(c[0], 9), round(c[1], 9)) for c in cs])


def _union(boxes):
    boxes = list(boxes)
    if not boxes:
        return None
    return [min(b[0][i] for b in boxes) for i in range(3)], [max(b[1][i] for b in boxes) for i in range(3)]


def node_box(node, P=ORIGIN):
    """The box (in the scene's frame) of one instance and everything in it; None for an empty group."""
    T = compose_pose(P, (node.pos, node.quat))
    if node.kind != "scene":
        return _turned_box(*_local_box(node.target), T)
    return _union(_turned_box(*_local_box(lf.node.target), (lf.pos, lf.quat)) for lf in flatten(node.target, "", T))


def box_of(x, what, pos):
    """The (lo, hi) box of a scene, a part/assembly/site/fixture block (a fixture in its site's frame), a
    box ({min, max}, as bounds() returns) or a point."""
    if isinstance(x, Scene):
        b = _union(filter(None, (node_box(n) for n in x.nodes)))
        if b is None:
            raise OrbError(f"{what}: an empty scene has no box", pos)
        return b
    if isinstance(x, Block) and x.type in ("part", "assembly", "site"):
        return _local_box(x)
    if isinstance(x, Block) and x.type == "fixture":
        from .scenes import _v3

        pts = _geoms(x, _v3(x.attrs.get("at")), _rot(x))
        return [min(p[i] for p in pts) for i in range(3)], [max(p[i] for p in pts) for i in range(3)]
    if isinstance(x, dict) and "min" in x and "max" in x:
        return list(point_like(x["min"], what, pos)), list(point_like(x["max"], what, pos))
    if isinstance(x, list) and x and all(isinstance(v, (Q, int, float)) and not isinstance(v, bool) for v in x):
        p = list(point_like(x, what, pos))
        return p, list(p)
    raise OrbError(f"{what} takes a scene, a block, a box ({{min, max}}) or a point, got {type_name(x)}", pos)


def _site_parts(block):
    """What of a site can clash: its fixtures (its floor and zones are ground), in the site's frame."""
    from .scenes import _v3

    o, out = _v3(block.attrs.get("at")), []
    for fx in block.kids("fixture"):
        pts = _geoms(fx, tuple(a + c for a, c in zip(o, _v3(fx.attrs.get("at")))), _rot(fx))
        out.append(([min(p[i] for p in pts) for i in range(3)], [max(p[i] for p in pts) for i in range(3)]))
    return out


def _parts(block):
    """The boxes of a placed block that can clash with others, in its own frame."""
    return _site_parts(block) if block.type == "site" else [_local_box(block)]


def node_shape(node, P=ORIGIN):
    """The boxes (in the scene's frame) of every part in one instance: what clashes, finer than its
    one box (a ring of units around a rock does not clash with the rock)."""
    T = compose_pose(P, (node.pos, node.quat))
    if node.kind != "scene":
        return [_part_box(lo, hi, T) for lo, hi in _parts(node.target)]
    return [_part_box(lo, hi, (lf.pos, lf.quat)) for lf in flatten(node.target, "", T) for lo, hi in _parts(lf.node.target)]


def _is_point(x):
    return isinstance(x, list) and bool(x) and all(isinstance(v, (Q, int, float)) and not isinstance(v, bool) for v in x)


def shape_of(x, what, pos):
    """The boxes of an obstacle: a scene (every part in it), a block, a box {min, max}, a point, or a list
    of them."""
    if isinstance(x, Scene):
        return [b for n in x.nodes for b in node_shape(n)]
    if isinstance(x, Block) and x.type in ("part", "assembly", "site"):
        return _parts(x)
    if isinstance(x, list) and x and not _is_point(x):
        return [b for o in x for b in shape_of(o, what, pos)]
    return [box_of(x, what, pos)]


class Hash:
    """Boxes in a uniform grid over x, y, for finding the ones a box comes within `gap` of."""

    def __init__(self, gap, axes, sample=()):
        sizes = sorted(max(b[1][0] - b[0][0], b[1][1] - b[0][1]) for b in sample) or [1.0]
        self.cell = max(sizes[len(sizes) // 2] + gap, 0.05)
        self.gap, self.axes, self.cells, self.big, self.boxes = gap, axes, {}, [], []

    def _span(self, b, g):
        c = self.cell
        return (math.floor((b[0][0] - g) / c), math.floor((b[1][0] + g) / c),
                math.floor((b[0][1] - g) / c), math.floor((b[1][1] + g) / c))

    def add(self, b, tag=None):
        i = len(self.boxes)
        self.boxes.append((b, tag))
        x0, x1, y0, y1 = self._span(b, self.gap)
        if (x1 - x0 + 1) * (y1 - y0 + 1) > 256:  # a big box (a long wall): checked against everything
            self.big.append(i)
            return
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                self.cells.setdefault((x, y), []).append(i)

    def near(self, b):
        """(box, tag) of every stored box b comes within gap of."""
        x0, x1, y0, y1 = self._span(b, 0.0)
        cand = set(self.big)
        if (x1 - x0 + 1) * (y1 - y0 + 1) > 256:
            cand.update(range(len(self.boxes)))
        else:
            for x in range(x0, x1 + 1):
                for y in range(y0, y1 + 1):
                    cand.update(self.cells.get((x, y), ()))
        return [self.boxes[i] for i in sorted(cand) if hit(b, self.boxes[i][0], self.gap, self.axes)]


def _hits(shape, h):
    return [(b, o) for b in shape for o, _ in h.near(b)]


def _axes(d, what, pos):
    a = d.get("axes", "xy")
    if a not in ("xy", "xyz"):
        raise OrbError(f"{what} axes is \"xy\" (footprints, the default) or \"xyz\" (boxes in 3D)", pos)
    return (0, 1) if a == "xy" else (0, 1, 2)


def _gap(d, what, pos):
    g = _length_kw(d, "gap", what, pos, 0.0)
    if g < 0:
        raise OrbError(f"{what} gap is 0 m or more", pos)
    return g


def _rect(b):
    (x0, y0), (x1, y1) = b[0][:2], b[1][:2]
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _apart(P, Q, gap):
    """Are two convex footprints at least gap apart along one of their edge normals? (Exact for gap 0;
    with a gap, corner to corner, it errs toward a clash.)"""
    for poly in (P, Q):
        for i in range(len(poly)):
            (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
            ax, ay = y2 - y1, x1 - x2
            ln = math.hypot(ax, ay)
            if ln < 1e-12:
                continue
            ax, ay = ax / ln, ay / ln
            pa = [x * ax + y * ay for x, y in P]
            qa = [x * ax + y * ay for x, y in Q]
            if max(pa) + gap - 1e-9 <= min(qa) or max(qa) + gap - 1e-9 <= min(pa):
                return True
    return False


def hit(a, b, gap, axes):
    """Do two boxes come closer than gap (touching is not a clash)? A turned part counts by its
    footprint in plan (and, with axes "xyz", its height), not by the bigger box around it."""
    if not all(a[0][i] < b[1][i] + gap - 1e-9 and b[0][i] < a[1][i] + gap - 1e-9 for i in axes):
        return False
    pa, pb = (a[2] if len(a) > 2 else None), (b[2] if len(b) > 2 else None)
    if pa is None and pb is None:
        return True
    return not _apart(pa or _rect(a), pb or _rect(b), gap)


# --------------------------------------------------------------------------- editing scenes


def _pivot(about, N, box, pos):
    if about == "self":
        return N[0]
    if about == "origin":
        return (0.0, 0.0, 0.0)
    if about == "center":
        if box is None:
            raise OrbError("about = \"center\" of an empty group", pos)
        return tuple((a + b) / 2 for a, b in zip(*box))
    return about


def _turn(kw, N, d, pos):
    """The rotation (in the scene's frame) a move gives an instance whose pose is N."""
    if "face" not in kw:
        return _rotation(kw, pos)
    if any(k in kw for k in ("rpy", "quat")):
        raise OrbError("face turns about z: give it a yaw for an extra turn, not rpy or quat", pos)
    p = tuple(a + b for a, b in zip(N[0], d))
    want = _face_yaw(kw["face"], p, pos) + (_angle(kw["yaw"], "yaw", pos) if "yaw" in kw else 0.0)
    return rpy_quat(0, 0, want - heading(N[1]))


MOVE_KEYS = ("at", "yaw", "rpy", "quat", "face", "about", "path")


@builtin("move")
def _move(prog, args, kw, pos):
    """move(scene, at = [dx, dy, dz] m, yaw = a | rpy | quat | face = point, about = "self" | "center" |
    point, path = "glob"): the same instances moved by `at` and turned, adding no level to their paths.
    Without path the whole scene moves (turning about its origin); with path only the instances it
    matches ("row3", "cluster_b", "*/module_n", "**/unit_1?"), each turned about itself. about = "center"
    turns about the bounds' centre, a point (or a scene or block: its centre) revolves about it.
    face turns each to look at a point (+x toward it; yaw: an extra turn)."""
    no_kw("move", kw, pos, MOVE_KEYS)
    if len(args) != 1:
        raise OrbError("move() takes one scene", pos)
    (s,) = scenes_of(args, pos, "move()")
    d = pose({"at": kw["at"]}, pos)[0] if "at" in kw else (0.0, 0.0, 0.0)
    about = kw.get("about", "self")
    if not (isinstance(about, str) and about in ("self", "center", "origin")):
        if isinstance(about, str):
            raise OrbError(f"about is \"self\", \"center\", \"origin\" or a point, got {about!r}", pos)
        about = point_like(about, "about", pos)

    def moved(N, box):
        R = _turn(kw, N, d, pos)
        c = _pivot(about, N, box, pos)
        p = tuple(dd + cc + r for dd, cc, r in zip(d, c, qrot(R, tuple(a - b for a, b in zip(N[0], c)))))
        return p, qmul(R, N[1])

    if "path" not in kw:
        box = box_of(s, "move()", pos) if about == "center" else None
        p, q = moved(ORIGIN, box)  # the scene's own frame, moved
        return Scene([n.moved((p, q)) for n in s.nodes], s.label, pos)
    rx = glob_rx(kw["path"], "move()", pos)

    def fn(path, n, P):
        N = compose_pose(P, (n.pos, n.quat))
        p, q = compose_pose(_inverse(P), moved(N, node_box(n, P) if about == "center" else None))
        return [renode(n, p, q)]

    out, k = edit(s, rx, fn)
    if not k:
        _no_match(s, kw["path"], "move()", pos)
    return Scene(out.nodes, s.label, pos)


@builtin("without")
def _without(prog, args, kw, pos):
    """without(scene, path = "*", overlapping = obstacle, gap = 0 m, axes = "xy"): the scene without the
    instances the path matches; with overlapping (a scene, block, box, point or a list of them), only
    those with a part closer than gap to it: lay out a full grid, then clear the rock."""
    d = _bind("without", args, kw, ("scene", "path"), pos, ("scene",))
    no_kw("without", {k: v for k, v in d.items() if k != "scene"}, pos, ("path", "overlapping", "gap", "axes"))
    (s,) = scenes_of([d["scene"]], pos, "without()")
    path = d.get("path", "*")
    rx = glob_rx(path, "without()", pos)
    if "overlapping" not in d:
        out, k = edit(s, rx, lambda p, n, P: [])
        if not k:
            _no_match(s, path, "without()", pos)
        return out
    obs = shape_of(d["overlapping"], "without() overlapping", pos)
    gap, axes = _gap(d, "without()", pos), _axes(d, "without()", pos)
    if not any(True for _ in matches(s, rx)):
        _no_match(s, path, "without()", pos)
    h = Hash(gap, axes, obs)
    for b in obs:
        h.add(b)
    return edit(s, rx, lambda p, n, P: [] if _hits(node_shape(n, P), h) else [n])[0]


@builtin("frame")
def _frame(prog, args, kw, pos):
    """frame(scene, path): where one instance stands in the scene, {at, yaw, rpy, quat}: place things
    relative to it with transform([5, 0, 0] m, frame(site, "rock")) and yaw = f.yaw"""
    d = _bind("frame", args, kw, ("scene", "path"), pos, ("scene", "path"))
    no_kw("frame", {k: v for k, v in d.items() if k not in ("scene", "path")}, pos)
    (s,) = scenes_of([d["scene"]], pos, "frame()")
    found = list(matches(s, glob_rx(d["path"], "frame()", pos)))
    if not found:
        _no_match(s, d["path"], "frame()", pos)
    if len(found) > 1:
        raise OrbError(f"frame(): path {d['path']!r} matches {len(found)} instances ({', '.join(f[0] for f in found[:5])}"
                       f"{', ...' if len(found) > 5 else ''}); name one", pos)
    _, n, P = found[0]
    p, q = compose_pose(P, (n.pos, n.quat))
    return {"at": _L(p), "yaw": _A(heading(q)), "rpy": [_A(a) for a in _quat_rpy(q)], "quat": list(q)}


@builtin("transform")
def _transform(prog, args, kw, pos):
    """transform(point, frame): a point given in a frame ({at, yaw | rpy | quat}, as frame() returns) in
    the frame's parent: transform([5, 0, 0] m, f) is 5 m ahead of f along its +x"""
    no_kw("transform", kw, pos)
    arity("transform", args, 2, pos=pos)
    p, f = args
    if not isinstance(f, dict):
        raise OrbError(f"transform(point, frame): frame is an object {{at, yaw}} (frame() returns one), got {type_name(f)}", pos)
    rot = next(({k: f[k]} for k in ("quat", "rpy", "yaw") if k in f), {})
    at, q = pose({**({"at": f["at"]} if "at" in f else {}), **rot}, pos)
    r = qrot(q, point_like(p, "transform()", pos))
    return _L([a + b for a, b in zip(at, r)])


@builtin("overlaps")
def _overlaps(prog, args, kw, pos):
    """overlaps(a, b, gap = 0 m, axes = "xy"): does any part of a come closer than gap to any part of b?
    a and b are scenes, blocks, boxes or points (b may be a list); "xy" compares footprints, "xyz" boxes
    in 3D. Touching is not overlapping."""
    no_kw("overlaps", kw, pos, ("gap", "axes"))
    arity("overlaps", args, 2, pos=pos)
    gap, axes = _gap(kw, "overlaps()", pos), _axes(kw, "overlaps()", pos)
    a, b = shape_of(args[0], "overlaps()", pos), shape_of(args[1], "overlaps()", pos)
    if not a or not b or not hit(_union(a), _union(b), gap, axes):
        return False
    h = Hash(gap, axes, b)
    for x in b:
        h.add(x)
    return bool(_hits(a, h))


@builtin("clashes")
def _clashes(prog, args, kw, pos):
    """clashes(scene, path = "*", gap = 0 m, axes = "xy"): every pair [a, b] of the instances the path
    matches with parts closer than gap ("*": the top-level clusters, "*/*": the units in them; a
    cluster's own units never clash with each other here)"""
    d = _bind("clashes", args, kw, ("scene", "path"), pos, ("scene",))
    no_kw("clashes", {k: v for k, v in d.items() if k != "scene"}, pos, ("path", "gap", "axes"))
    (s,) = scenes_of([d["scene"]], pos, "clashes()")
    gap, axes = _gap(d, "clashes()", pos), _axes(d, "clashes()", pos)
    found = [(p, node_shape(n, P)) for p, n, P in matches(s, glob_rx(d.get("path", "*"), "clashes()", pos))]
    if not found and s.nodes:
        _no_match(s, d.get("path", "*"), "clashes()", pos)
    h = Hash(gap, axes, [b for _, sh in found for b in sh])
    pairs = set()
    for i, (_, sh) in enumerate(found):
        for b in sh:
            pairs.update((j, i) for _, j in h.near(b) if j != i)
        for b in sh:
            h.add(b, i)
    return [[found[i][0], found[j][0]] for i, j in sorted(pairs)]


DIRS = {"+x": [(0, 1)], "-x": [(0, -1)], "+y": [(1, 1)], "-y": [(1, -1)], "x": [(0, 1), (0, -1)],
        "y": [(1, 1), (1, -1)], "xy": [(0, 1), (0, -1), (1, 1), (1, -1)]}


def _moved_box(b, d):
    """A clash box slid by d = [dx, dy, dz]."""
    lo, hi = [x + y for x, y in zip(b[0], d)], [x + y for x, y in zip(b[1], d)]
    poly = b[2] if len(b) > 2 else None
    return lo, hi, (None if poly is None else [(x + d[0], y + d[1]) for x, y in poly])


def _shift(shape, axis, s):
    d = [0.0, 0.0, 0.0]
    d[axis] = s
    return [_moved_box(b, d) for b in shape]


def _push(shape, h, axis, sign):
    """How far to slide a shape along one axis, only that way, until no part of it comes within gap of
    the boxes in h. Each round clears every pair it found for good, so it ends."""
    total, gap = 0.0, h.gap
    for _ in range(1000):
        hits = _hits(shape, h)
        if not hits:
            return total
        s = (max(o[1][axis] + gap - b[0][axis] for b, o in hits) if sign > 0 else
             min(o[0][axis] - gap - b[1][axis] for b, o in hits))
        shape = _shift(shape, axis, s)
        total += s
    whole = _union(shape)  # a pathological tangle: clear the shape's whole box of everything
    hi = max(o[1][axis] for o, _ in h.boxes) + gap if sign > 0 else min(o[0][axis] for o, _ in h.boxes) - gap
    return total + (hi - whole[0][axis] if sign > 0 else hi - whole[1][axis])


def _clear(shape, h, dirs):
    """The shortest slide, along one of dirs, that clears a shape -> [dx, dy, dz]."""
    best = None
    for axis, sign in dirs:
        t = _push(shape, h, axis, sign)
        if best is None or abs(t) < abs(best[1]) - 1e-12:
            best = (axis, t)
    d = [0.0, 0.0, 0.0]
    d[best[0]] = best[1]
    return d


@builtin("separate")
def _separate(prog, args, kw, pos):
    """separate(a, b, gap = 0 m, along = "xy"): a slid the shortest way (along "xy": +x, -x, +y or -y;
    or "x", "y", "+x", ...) until no part of it comes within gap of b (a scene, block, box, point or a
    list). separate(scene, path = "*", keep = "glob", gap, along): the matched instances cleared of each
    other in order: the first stays, each later one slides clear of those before it (and of what keep
    matches, which never moves)."""
    second = args[1] if len(args) == 2 else None
    paths = isinstance(second, str) or (isinstance(second, list) and bool(second) and all(isinstance(x, str) for x in second))
    if len(args) == 2 and not paths:
        no_kw("separate", kw, pos, ("gap", "along", "axes"))
        (a,) = scenes_of(args[:1], pos, "separate()")
        d, obs, mode = kw, shape_of(args[1], "separate()", pos), "pair"
    else:
        d = _bind("separate", args, kw, ("scene", "path"), pos, ("scene",))
        no_kw("separate", {k: v for k, v in d.items() if k != "scene"}, pos, ("path", "keep", "gap", "along", "axes"))
        (a,) = scenes_of([d["scene"]], pos, "separate()")
        mode = "all"
    gap, axes = _gap(d, "separate()", pos), _axes(d, "separate()", pos)
    along = d.get("along", "xy")
    if not isinstance(along, str) or along not in DIRS:
        raise OrbError(f"separate() along is one of {', '.join(repr(k) for k in DIRS)}, got "
                       f"{repr(along) if isinstance(along, str) else type_name(along)}", pos)
    if mode == "pair":
        shape = shape_of(a, "separate()", pos)
        h = Hash(gap, axes, shape + obs)
        for b in obs:
            h.add(b)
        sh = _clear(shape, h, DIRS[along]) if obs else [0.0, 0.0, 0.0]
        return Scene([n.moved((tuple(sh), IDENT)) for n in a.nodes], a.label, pos) if any(sh) else a
    path = d.get("path", "*")
    rx = glob_rx(path, "separate()", pos)
    keep = glob_rx(d["keep"], "separate() keep", pos) if "keep" in d else None
    kept = [(p, node_shape(n, P)) for p, n, P in matches(a, keep)] if keep else []
    found = [(p, node_shape(n, P)) for p, n, P in matches(a, rx) if not (keep and keep.fullmatch(p, n))]
    if not found and not kept:
        _no_match(a, path, "separate()", pos)
    h = Hash(gap, axes, [b for _, sh in kept + found for b in sh])
    for _, sh in kept:
        for b in sh:
            h.add(b)
    shifts = {}
    for p, shape in found:
        sh = _clear(shape, h, DIRS[along]) if h.boxes and shape else [0.0, 0.0, 0.0]
        if any(sh):
            shifts[p] = sh
            shape = [_moved_box(b, sh) for b in shape]
        for b in shape:
            h.add(b)
    if not shifts:
        return a

    def fn(p, n, P):
        if p not in shifts:
            return [n]
        N = compose_pose(P, (n.pos, n.quat))
        q, r = compose_pose(_inverse(P), (tuple(x + y for x, y in zip(N[0], shifts[p])), N[1]))
        return [renode(n, q, r)]
    return edit(a, rx, fn)[0]


@builtin("beside")
def _beside(prog, args, kw, pos):
    """beside(a, b, side = "+x" | "-x" | "+y" | "-y", gap = 0 m, align = "center" | "min" | "max" | "none"):
    a moved to sit on that side of b, gap between their boxes, lined up with b across (align)"""
    no_kw("beside", kw, pos, ("side", "gap", "align"))
    arity("beside", args, 2, pos=pos)
    (a,) = scenes_of(args[:1], pos, "beside()")
    A, B = box_of(a, "beside()", pos), box_of(args[1], "beside()", pos)
    side, align = kw.get("side", "+x"), kw.get("align", "center")
    if side not in ("+x", "-x", "+y", "-y"):
        raise OrbError(f"beside() side is \"+x\", \"-x\", \"+y\" or \"-y\", got {side!r}", pos)
    if align not in ("center", "min", "max", "none"):
        raise OrbError(f"beside() align is \"center\", \"min\", \"max\" or \"none\", got {align!r}", pos)
    gap = _gap(kw, "beside()", pos)
    ax, cross = (0, 1) if side[1] == "x" else (1, 0)
    d = [0.0, 0.0, 0.0]
    d[ax] = B[1][ax] + gap - A[0][ax] if side[0] == "+" else B[0][ax] - gap - A[1][ax]
    if align == "center":
        d[cross] = (B[0][cross] + B[1][cross] - A[0][cross] - A[1][cross]) / 2
    elif align == "min":
        d[cross] = B[0][cross] - A[0][cross]
    elif align == "max":
        d[cross] = B[1][cross] - A[1][cross]
    return Scene([n.moved((tuple(d), IDENT)) for n in a.nodes], a.label, pos)
