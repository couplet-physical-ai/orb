"""The functions OrbLang provides: `range(3)`, `len(xs)`, `sqrt(2 m2)`, ... and the scene builtins
(`instance`, `compose`, `move`, ...; orb/scenes.py). Every builtin is pure: it reads its arguments
and returns a value, never touches files, simulators or the world being built."""
from __future__ import annotations

import math

from .lang import DIMLESS, Block, Builtin, Function, Group, OrbError, Q, _dims, _q, _val, dim_name, type_name

BUILTINS = {}
_ANG = (0, 0, 0, 0, 0, 1)


def builtin(name, doc=""):
    def deco(fn):
        BUILTINS[name] = Builtin(name, fn, doc or (fn.__doc__ or "").strip())
        return fn
    return deco


def arity(name, args, lo, hi=None, pos=None):
    hi = lo if hi is None else hi
    if not lo <= len(args) <= hi:
        want = f"{lo}" if lo == hi else f"{lo} to {hi}"
        raise OrbError(f"{name}() takes {want} argument{'s' * (hi != 1)}, got {len(args)}", pos)


def no_kw(name, kw, pos, allowed=()):
    extra = sorted(set(kw) - set(allowed))
    if extra:
        from .lang import did_you_mean

        raise OrbError(f"{name}() has no parameter {extra[0]!r}" + (did_you_mean(extra[0], allowed) or
                       (f" (has: {', '.join(allowed)})" if allowed else " (it takes none)")), pos)


def is_num(x):
    return isinstance(x, (int, float, Q)) and not isinstance(x, bool)


def need_num(name, x, pos):
    if not is_num(x):
        raise OrbError(f"{name}() needs a number or quantity, got {type_name(x)}", pos)
    return x


# --------------------------------------------------------------------------- lists and objects


@builtin("range")
def _range(prog, args, kw, pos):
    """range(n), range(a, b), range(a, b, step): integers (plain numbers)."""
    no_kw("range", kw, pos)
    arity("range", args, 1, 3, pos)
    args = [int(a) if isinstance(a, float) and a.is_integer() else a for a in args]  # 3.0: a whole number
    if not all(isinstance(a, int) and not isinstance(a, bool) for a in args):
        raise OrbError("range() takes whole numbers; for fractional steps scale a whole range: "
                       "[for i in range(5) : i * 0.25]", pos)
    if len(args) == 3 and args[2] == 0:
        raise OrbError("range() step must not be 0", pos)
    out = list(range(*args))
    if len(out) > 10000:
        raise OrbError(f"range() of {len(out)} numbers is too many (at most 10000)", pos)
    return out


@builtin("len")
def _len(prog, args, kw, pos):
    """len(list | object | string | group)"""
    no_kw("len", kw, pos)
    arity("len", args, 1, pos=pos)
    x = args[0]
    if isinstance(x, Group):
        return len(x.labels)
    if isinstance(x, (list, dict, str)):
        return len(x)
    if getattr(x, "orb_type", None) == "scene":
        return len(x.nodes)
    raise OrbError(of_wrong("len", x, "a list, object, string or group"), pos)


@builtin("concat")
def _concat(prog, args, kw, pos):
    """concat(list, list, ...) -> one list"""
    no_kw("concat", kw, pos)
    out = []
    for a in args:
        if not isinstance(a, list):
            raise OrbError(f"concat() takes lists, got {type_name(a)}", pos)
        out.extend(a)
    return out


@builtin("merge")
def _merge(prog, args, kw, pos):
    """merge(object, object, ...) -> one object; later keys win"""
    no_kw("merge", kw, pos)
    out = {}
    for a in args:
        if not isinstance(a, dict):
            raise OrbError(f"merge() takes objects, got {type_name(a)}", pos)
        out.update(a)
    return out


def of_wrong(name, x, takes):
    """`len(scene)`-style mistakes: what the builtin takes, and for a scene how to list its instances."""
    hint = (" (a scene's instances: s.ids, s.count, s[\"<id>\"])" if type_name(x) == "scene" else "")
    return f"{name}() takes {takes}, got {type_name(x)}{hint}"


@builtin("keys")
def _keys(prog, args, kw, pos):
    """keys(object | group) -> the keys (labels of a group: keys(site.office.zone))"""
    no_kw("keys", kw, pos)
    arity("keys", args, 1, pos=pos)
    x = args[0]
    if isinstance(x, Group):
        return list(x.labels)
    if isinstance(x, dict):
        return list(x)
    raise OrbError(of_wrong("keys", x, "an object or group"), pos)


@builtin("values")
def _values(prog, args, kw, pos):
    """values(object | group)"""
    no_kw("values", kw, pos)
    arity("values", args, 1, pos=pos)
    x = args[0]
    if isinstance(x, Group):
        return [x.lookup(l, pos) for l in x.labels]
    if isinstance(x, dict):
        return list(x.values())
    raise OrbError(of_wrong("values", x, "an object or group"), pos)


@builtin("contains")
def _contains(prog, args, kw, pos):
    """contains(list | object | string, x)"""
    no_kw("contains", kw, pos)
    arity("contains", args, 2, pos=pos)
    c, x = args
    if isinstance(c, str):
        if not isinstance(x, str):
            raise OrbError(f"contains(string, x) looks for a string, got {type_name(x)}", pos)
        return x in c
    if isinstance(c, dict):
        return x in c
    if isinstance(c, list):
        from .lang import equal

        def same(y):
            try:
                return equal(y, x, pos)
            except OrbError:  # another unit: not this one
                return False
        return any(same(y) for y in c)
    raise OrbError(of_wrong("contains", c, "a list, object or string"), pos)


@builtin("lookup")
def _lookup(prog, args, kw, pos):
    """lookup(object, key, default)"""
    no_kw("lookup", kw, pos)
    arity("lookup", args, 2, 3, pos)
    if not isinstance(args[0], dict):
        raise OrbError(f"lookup() takes an object, got {type_name(args[0])}", pos)
    if args[1] in args[0]:
        return args[0][args[1]]
    if len(args) == 3:
        return args[2]
    raise OrbError(f"object has no key {args[1]!r}", pos)


@builtin("label")
def _label(prog, args, kw, pos):
    """label(block) -> its label as written (without an import prefix)"""
    no_kw("label", kw, pos)
    arity("label", args, 1, pos=pos)
    if not isinstance(args[0], Block):
        raise OrbError(of_wrong("label", args[0], "a block"), pos)
    return args[0].label.rsplit("/", 1)[-1]


@builtin("error")
def _error(prog, args, kw, pos):
    """error("message"): stop with this message (inside a ternary: c ? x : error("..."))"""
    no_kw("error", kw, pos)
    arity("error", args, 1, pos=pos)
    from .lang import interpolated

    try:
        msg = interpolated(args[0], pos)
    except OrbError:
        msg = type_name(args[0])
    raise OrbError(msg or "error() (with an empty message)", pos)


@builtin("call")
def _call(prog, args, kw, pos):
    """call(f, args...): call a function value (fns are values: pass them to other fns)"""
    if not args or not isinstance(args[0], (Function, Builtin)):
        raise OrbError("call() takes a function first", pos)
    return prog.call(args[0], [(None, a, pos) for a in args[1:]] + [(k, v, pos) for k, v in kw.items()], pos)


# --------------------------------------------------------------------------- math


def _unary(name, f, angle_in=False, angle_out=False, keep=False, doc=""):
    def impl(prog, args, kw, pos):
        no_kw(name, kw, pos)
        arity(name, args, 1, pos=pos)
        x = need_num(name, args[0], pos)
        if angle_in and _dims(x) not in (_ANG, DIMLESS):
            raise OrbError(f"{name}() takes an angle, got {dim_name(_dims(x))}", pos)
        if not (angle_in or keep) and _dims(x) != DIMLESS:
            raise OrbError(f"{name}() takes a plain number, got {dim_name(_dims(x))}", pos)
        try:
            v = f(_val(x))
        except ValueError:
            raise OrbError(f"{name}({_val(x):g}) is undefined", pos) from None
        if keep and _dims(x) != DIMLESS:
            return _q(float(v), _dims(x))
        return _q(v, _ANG) if angle_out else v
    BUILTINS[name] = Builtin(name, impl, doc)


_unary("sin", math.sin, angle_in=True, doc="sin(angle)")
_unary("cos", math.cos, angle_in=True, doc="cos(angle)")
_unary("tan", math.tan, angle_in=True, doc="tan(angle)")
_unary("asin", math.asin, angle_out=True, doc="asin(x) -> angle")
_unary("acos", math.acos, angle_out=True, doc="acos(x) -> angle")
_unary("abs", abs, keep=True, doc="abs(x) (keeps the unit)")
_unary("floor", math.floor, keep=True, doc="floor(x) -> the whole number at or below x (keeps the unit)")
_unary("ceil", math.ceil, keep=True, doc="ceil(x) -> the whole number at or above x (keeps the unit)")


@builtin("round")
def _round(prog, args, kw, pos):
    """round(x), round(x, step): the nearest whole number, or multiple of step (keeps the unit)"""
    no_kw("round", kw, pos)
    arity("round", args, 1, 2, pos)
    x = need_num("round", args[0], pos)
    if len(args) == 1:
        v = round(_val(x))
        return _q(float(v), _dims(x)) if isinstance(x, Q) else int(v)
    s = need_num("round", args[1], pos)
    if _dims(s) != _dims(x) or _val(s) == 0:
        raise OrbError("round(x, step): step must be non-zero and have x's unit", pos)
    return _q(round(_val(x) / _val(s)) * _val(s), _dims(x))


@builtin("sqrt")
def _sqrt(prog, args, kw, pos):
    """sqrt(x): units halve (sqrt(4 m2) = 2 m)"""
    no_kw("sqrt", kw, pos)
    arity("sqrt", args, 1, pos=pos)
    x = need_num("sqrt", args[0], pos)
    d = _dims(x)
    if any(e % 2 for e in d):
        raise OrbError(f"sqrt() of {dim_name(d)}: the unit has no square root", pos)
    if _val(x) < 0:
        raise OrbError("sqrt() of a negative number", pos)
    return _q(math.sqrt(_val(x)), tuple(e // 2 for e in d))


@builtin("atan2")
def _atan2(prog, args, kw, pos):
    """atan2(y, x) -> angle"""
    no_kw("atan2", kw, pos)
    arity("atan2", args, 2, pos=pos)
    y, x = (need_num("atan2", a, pos) for a in args)
    if _dims(x) != _dims(y) and _val(x) != 0 and _val(y) != 0:
        raise OrbError("atan2(y, x) needs y and x in the same unit", pos)
    return Q(math.atan2(_val(y), _val(x)), _ANG)


def _extreme(name, pick):
    def impl(prog, args, kw, pos):
        no_kw(name, kw, pos)
        xs = args[0] if len(args) == 1 and isinstance(args[0], list) else args
        if not xs:
            raise OrbError(f"{name}() of nothing", pos)
        for x in xs:
            need_num(name, x, pos)
        d = {_dims(x) for x in xs if _val(x) != 0} or {DIMLESS}
        if len(d) > 1:
            raise OrbError(f"{name}() of different units", pos)
        return pick(xs, key=_val)
    BUILTINS[name] = Builtin(name, impl, f"{name}(a, b, ...) or {name}(list) (units must agree)")


_extreme("min", min)
_extreme("max", max)


@builtin("sum")
def _sum(prog, args, kw, pos):
    """sum(list) (units must agree)"""
    from .lang import arith

    no_kw("sum", kw, pos)
    arity("sum", args, 1, pos=pos)
    if not isinstance(args[0], list):
        raise OrbError(f"sum() takes a list, got {type_name(args[0])}", pos)
    out = 0
    for x in args[0]:
        out = arith("+", out, x, pos)
    return out


BUILTINS["pi"] = math.pi  # a constant, not a function


def _load_scene_builtins():
    from . import scenes  # noqa: F401  (registers instance, compose, move, row, grid, ...; scenes loads layout)


_load_scene_builtins()
