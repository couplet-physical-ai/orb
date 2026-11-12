"""Orb language: lexer, parser, units and evaluator.

An Orb program is a directory of .orb files. Top-level blocks (`part "x" {}`) are
order-independent, like Terraform: references are resolved lazily, so `part.x.mass`
works no matter which file or line declares `part "x"`.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path


# --------------------------------------------------------------------------- errors


@dataclass(frozen=True)
class Pos:
    file: str
    line: int
    col: int

    def __str__(self):
        return f"{self.file}:{self.line}:{self.col}"


class OrbError(Exception):
    def __init__(self, msg, pos=None):
        self.msg, self.pos = msg, pos
        super().__init__(f"{pos}: {msg}" if pos else msg)


# --------------------------------------------------------------------------- units

# Dimension vector: (kg, m, s, A, K, rad). Angles are a dimension so `range = [-60, 60] deg`
# can't be confused with a length.
_M, _L, _T = (1, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0)
_ANG = (0, 0, 0, 0, 0, 1)
DIMLESS = (0, 0, 0, 0, 0, 0)

UNITS = {
    "kg": (1, _M), "g": (1e-3, _M),
    "m": (1, _L), "cm": (1e-2, _L), "mm": (1e-3, _L), "km": (1e3, _L),
    "inch": (0.0254, _L), "ft": (0.3048, _L),
    "s": (1, _T), "ms": (1e-3, _T), "min": (60, _T), "h": (3600, _T),
    "N": (1, (1, 1, -2, 0, 0, 0)), "Nm": (1, (1, 2, -2, 0, 0, 0)), "J": (1, (1, 2, -2, 0, 0, 0)),
    "W": (1, (1, 2, -3, 0, 0, 0)), "Pa": (1, (1, -1, -2, 0, 0, 0)), "Hz": (1, (0, 0, -1, 0, 0, 0)),
    "A": (1, (0, 0, 0, 1, 0, 0)), "V": (1, (1, 2, -3, -1, 0, 0)), "K": (1, (0, 0, 0, 0, 1, 0)),
    "rad": (1, _ANG), "deg": (math.pi / 180, _ANG), "rpm": (2 * math.pi / 60, (0, 0, -1, 0, 0, 1)),
}

_DIM_NAMES = {
    _M: "mass", _L: "length", _T: "time", _ANG: "angle", DIMLESS: "number",
    (1, 2, 0, 0, 0, 0): "inertia", (1, 2, -2, 0, 0, 0): "torque/energy", (1, 1, -2, 0, 0, 0): "force",
    (0, 0, -1, 0, 0, 1): "angular velocity", (0, 1, -1, 0, 0, 0): "velocity",
}


def dim_name(d):
    if d in _DIM_NAMES:
        return _DIM_NAMES[d]
    sym = ("kg", "m", "s", "A", "K", "rad")
    return "*".join(f"{s}^{e}" if e != 1 else s for s, e in zip(sym, d) if e) or "number"


def interpolated(v, pos):
    """How a value reads inside "${...}": numbers with their unit (1.5 m, 90 deg), true/false/null."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, str):
        return v
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, Q):
        if v.d == DIMLESS:
            return f"{v.v:g}"
        if v.d == _ANG:
            return f"{math.degrees(v.v):g} deg"
        unit = {_L: "m", _M: "kg", _T: "s"}.get(v.d) or dim_name(v.d)
        return f"{v.v:g} {unit}"
    raise OrbError(f"cannot interpolate {type_name(v)} into a string", pos)


def unit_term(text):
    """'m2' -> (1, m^2); 'kg' -> (1, kg). None if not a unit."""
    m = re.fullmatch(r"([A-Za-z]+?)(\d*)", text)
    if not m or m.group(1) not in UNITS:
        return None
    f, d = UNITS[m.group(1)]
    e = int(m.group(2) or 1)
    return f**e, tuple(x * e for x in d)


@dataclass(frozen=True)
class Q:
    """A physical quantity, stored in SI."""
    v: float
    d: tuple

    def __str__(self):
        return f"{self.v:g} ({dim_name(self.d)})"


def _q(v, d):
    return v if d == DIMLESS else Q(v, d)


def _dims(x):
    return x.d if isinstance(x, Q) else DIMLESS


def _val(x):
    return x.v if isinstance(x, Q) else x


def _is_num(x):
    return isinstance(x, (int, float, Q)) and not isinstance(x, bool)


def equal(a, b, pos):
    """`a == b`: lists and objects compare whole (same length/keys, equal members), numbers by value
    within 1e-9 (0.1 m + 0.2 m == 0.3 m) and only in the same unit (bare 0 equals 0 of any unit);
    anything else by identity of kind and value (true is not 1)."""
    if isinstance(a, list) or isinstance(b, list):
        return isinstance(a, list) and isinstance(b, list) and len(a) == len(b) and \
            all(equal(x, y, pos) for x, y in zip(a, b))
    if isinstance(a, dict) or isinstance(b, dict):
        return isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys() and \
            all(equal(a[k], b[k], pos) for k in a)
    if _is_num(a) and _is_num(b):
        da, db = _dims(a), _dims(b)
        va, vb = _val(a), _val(b)
        if da != db and not (va == 0 and not isinstance(a, Q)) and not (vb == 0 and not isinstance(b, Q)):
            raise OrbError(f"unit mismatch: cannot compare {dim_name(da)} and {dim_name(db)}"
                           f"{' (give the number a unit)' if DIMLESS in (da, db) else ''}", pos)
        return math.isclose(va, vb, rel_tol=1e-9, abs_tol=1e-12)
    if _is_num(a) or _is_num(b) or isinstance(a, bool) != isinstance(b, bool):
        return False
    return type(a) is type(b) and a == b if not isinstance(a, Block) else a is b


def arith(op, a, b, pos):
    if op in ("==", "!="):
        return equal(a, b, pos) == (op == "==")
    if (isinstance(a, list) or isinstance(b, list)) and op in ("<", ">", "<=", ">="):
        raise OrbError(f"'{op}' compares numbers, not lists (compare members: p[0] {op} q[0])", pos)
    if isinstance(a, list) or isinstance(b, list):  # elementwise, scalars broadcast
        n = len(a) if isinstance(a, list) else len(b)
        if isinstance(a, list) and isinstance(b, list) and len(a) != len(b):
            raise OrbError(f"vector length mismatch: {len(a)} vs {len(b)}", pos)
        aa = a if isinstance(a, list) else [a] * n
        bb = b if isinstance(b, list) else [b] * n
        return [arith(op, x, y, pos) for x, y in zip(aa, bb)]
    if isinstance(a, str) or isinstance(b, str):
        if op == "+" and isinstance(a, str) and isinstance(b, str):
            return a + b
        raise OrbError(f"cannot apply '{op}' to {type_name(a)} and {type_name(b)}", pos)
    for x in (a, b):
        if not isinstance(x, (int, float, Q)) or isinstance(x, bool):
            raise OrbError(f"cannot apply '{op}' to {type_name(x)}", pos)
    da, db = _dims(a), _dims(b)
    va, vb = _val(a), _val(b)
    if op in ("+", "-", "<", ">", "<=", ">="):
        if da != db and not (va == 0 and op in "+-") and not (vb == 0 and op in "+-"):
            raise OrbError(f"unit mismatch: cannot {op} {dim_name(da)} and {dim_name(db)}", pos)
        d = da if da != DIMLESS else db
        if op == "+":
            return _q(va + vb, d)
        if op == "-":
            return _q(va - vb, d)
        return {"<": va < vb, ">": va > vb, "<=": va <= vb, ">=": va >= vb}[op]
    if op == "*":
        return _q(va * vb, tuple(x + y for x, y in zip(da, db)))
    if op == "/":
        if vb == 0:
            raise OrbError("division by zero", pos)
        return _q(va / vb, tuple(x - y for x, y in zip(da, db)))
    if op == "%":  # the remainder, with the divisor's sign (7 % 3 = 1, 2.5 m % 1 m = 0.5 m)
        if db != DIMLESS and da != db:
            raise OrbError(f"unit mismatch: cannot {dim_name(da)} % {dim_name(db)}", pos)
        if vb == 0:
            raise OrbError("modulo by zero", pos)
        r = math.fmod(va, vb)
        if r and (r < 0) != (vb < 0):
            r += vb
        return _q(int(r) if isinstance(va, int) and isinstance(vb, int) else r, da)
    raise OrbError(f"unknown operator {op}", pos)


BLOCK_TYPES = ("part", "assembly", "site", "item", "lifecycle", "process", "seam", "output", "camera", "timeline",
               "shot", "source", "capture")


def did_you_mean(word, options):
    """' (did you mean 'x'?)' for the closest of `options`, or ''."""
    import difflib

    m = difflib.get_close_matches(str(word), sorted({str(o) for o in options}), n=1, cutoff=0.7)
    return f" (did you mean {m[0]!r}?)" if m else ""


def type_name(v):
    if isinstance(v, Block):
        return f"block {v.path}"
    if isinstance(v, (Function, Builtin)):
        return f"fn {v.name}"
    if isinstance(v, Namespace):
        return f"project {v.alias}"
    if getattr(v, "orb_type", None):
        return v.orb_type
    if isinstance(v, Q):
        return dim_name(v.d)
    return {bool: "bool", int: "number", float: "number", str: "string", list: "list",
            dict: "object", type(None): "null"}.get(type(v), type(v).__name__)


# --------------------------------------------------------------------------- lexer

_TOKEN = re.compile(r"""
  (?P<ws>[ \t\r\n]+) | (?P<comment>\#[^\n]*|//[^\n]*)
| (?P<num>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)
| (?P<str>"(?:[^"\\\n$]|\\.|\$(?!\{)|\$\{(?:[^{}"\n]|"(?:[^"\\\n]|\\.)*"|\{[^{}\n]*\})*\})*")
| (?P<id>[A-Za-z_][A-Za-z0-9_]*)
| (?P<op>==|!=|<=|>=|&&|\|\||[-+*/%!=<>?:,.\[\]{}()])
""", re.X)


@dataclass
class Tok:
    kind: str
    text: str
    pos: Pos
    start: int
    end: int


def lex(src, file, line0=1, col0=1):
    toks, i, line, lstart = [], 0, line0, 0
    while i < len(src):
        m = _TOKEN.match(src, i)
        if not m:
            at = Pos(file, line, i - lstart + col0)
            if src[i] == '"':
                rest = src[i:src.find("\n", i) if "\n" in src[i:] else len(src)]
                raise OrbError("a string with an unclosed ${ (close it with })" if "${" in rest else
                               "a string that does not end on its line (missing \")", at)
            if src[i] == "'":
                raise OrbError("strings use double quotes: \"...\"", at)
            raise OrbError(f"unexpected character {src[i]!r}", at)
        kind = m.lastgroup
        if kind not in ("ws", "comment"):
            toks.append(Tok(kind, m.group(), Pos(file, line, i - lstart + col0), m.start(), m.end()))
        nl = m.group().count("\n")
        if nl:
            line += nl
            lstart = m.start() + m.group().rindex("\n") + 1
            col0 = 1
        i = m.end()
    toks.append(Tok("eof", "", Pos(file, line, i - lstart + col0), i, i))
    return toks


# --------------------------------------------------------------------------- parser
# AST nodes are tuples: (kind, ..., pos). Kept deliberately small.

# types a fn parameter or result may declare (checked when the function is called)
TYPES = ("any", "number", "string", "bool", "list", "object", "fn", "scene", "block", "part", "assembly",
         "site", "item", "zone", "fixture", "length", "mass", "time", "angle", "point")

_ESC = {"n": "\n", "t": "\t", '"': '"', "\\": "\\", "$": "$"}


class Parser:
    def __init__(self, toks, file):
        self.t, self.i, self.file = toks, 0, file

    def peek(self, k=0):
        return self.t[min(self.i + k, len(self.t) - 1)]

    def next(self):
        tok = self.t[self.i]
        self.i += 1
        return tok

    def accept(self, text):
        if self.peek().kind in ("op", "id") and self.peek().text == text:
            return self.next()
        return None

    def expect(self, text=None, kind=None):
        tok = self.peek()
        ok = (text is None or (tok.text == text and tok.kind in ("op", "id"))) and (kind is None or tok.kind == kind)
        if not ok:
            want = repr(text) if text else kind
            got = repr(tok.text) if tok.kind != "eof" else "end of file"
            raise OrbError(f"expected {want}, got {got}", tok.pos)
        return self.next()

    # statements -------------------------------------------------------------
    def program(self):
        out = []
        while self.peek().kind != "eof":
            out.append(self.stmt(top=True))
        return out

    def body(self, in_module=False):
        self.expect("{")
        out = []
        while not self.accept("}"):
            if self.peek().kind == "eof":
                raise OrbError("unclosed '{'", self.peek().pos)
            out.append(self.stmt(in_module=in_module))
        return out

    def stmt(self, top=False, in_module=False):
        tok = self.expect(kind="id")
        if tok.text == "let":
            name = self.expect(kind="id").text
            self.expect("=")
            return ("let", name, self.expr(), tok.pos)
        if tok.text == "module" and top:
            name = self.expect(kind="id").text
            stmts = self.body(in_module=True)
            params = [(s[1], s[2]) for s in stmts if s[0] == "param"]
            return ("module", name, params, [s for s in stmts if s[0] != "param"], tok.pos)
        if tok.text == "fn" and top and self.peek().kind == "id":
            return self.fn_decl(tok)
        if tok.text == "import" and top and self.peek().kind == "str" and self.peek(1).text == "as":
            lit = self.string(self.next())
            if len(lit[1]) != 1 or not isinstance(lit[1][0], str):
                raise OrbError("import takes a plain string path", tok.pos)
            self.next()  # as
            alias = self.expect(kind="id").text
            return ("import", lit[1][0], alias, tok.pos)
        if tok.text == "include" and top and self.peek().kind == "str":
            lit = self.string(self.next())
            if len(lit[1]) != 1 or not isinstance(lit[1][0], str):
                raise OrbError("include takes a plain string path", tok.pos)
            return ("include", lit[1][0], tok.pos)
        if tok.text == "param" and in_module:
            name = self.expect(kind="id").text
            return ("param", name, self.expr() if self.accept("=") else None, tok.pos)
        if self.accept("="):
            return ("attr", tok.text, self.expr(), tok.pos)
        labels = []
        while self.peek().kind == "str":
            labels.append(self.string(self.next()))
        if self.peek().text != "{":
            raise OrbError(f"expected '=' or '{{' after {tok.text!r}", self.peek().pos)
        return ("block", tok.text, labels, self.body(), tok.pos)

    def type_ann(self):
        tok = self.expect(kind="id")
        if tok.text not in TYPES:
            hint = did_you_mean(tok.text, TYPES)
            raise OrbError(f"unknown type {tok.text!r}" + (hint or f" (use one of {', '.join(TYPES)})"), tok.pos)
        return tok.text

    def fn_decl(self, tok):
        """fn name(a, b: length, c = 1 m) -> scene { let x = ...  return expr }"""
        name = self.expect(kind="id").text
        self.expect("(")
        params, seen = [], set()
        while not self.accept(")"):
            p = self.expect(kind="id")
            if p.text in seen:
                raise OrbError(f"fn {name} has two parameters named {p.text!r}", p.pos)
            seen.add(p.text)
            ty = self.type_ann() if self.accept(":") else None
            default = self.expr() if self.accept("=") else None
            if default is None and params and params[-1][2] is not None:
                raise OrbError(f"parameter {p.text!r} needs a default: it follows one that has a default", p.pos)
            params.append((p.text, ty, default, p.pos))
            if not self.accept(","):
                self.expect(")")
                break
        rtype = None
        if self.peek().text == "-" and self.peek(1).text == ">" and self.peek().end == self.peek(1).start:
            self.i += 2
            rtype = self.type_ann()
        self.expect("{")
        lets, ret = [], None
        while not self.accept("}"):
            t = self.peek()
            if t.kind == "eof":
                raise OrbError("unclosed '{'", t.pos)
            if ret is not None:
                raise OrbError(f"nothing may follow `return` in fn {name}", t.pos)
            if t.kind == "id" and t.text == "let":
                self.next()
                n = self.expect(kind="id")
                if n.text in seen:
                    raise OrbError(f"let {n.text} in fn {name} shadows a parameter or an earlier let", n.pos)
                seen.add(n.text)
                self.expect("=")
                lets.append((n.text, self.expr(), t.pos))
            elif t.kind == "id" and t.text == "return":
                self.next()
                ret = self.expr()
            else:
                raise OrbError(f"a function body holds `let` statements and one `return`, got {t.text!r}", t.pos)
        if ret is None:
            raise OrbError(f"fn {name} needs `return <expression>`", tok.pos)
        return ("fn", name, params, lets, ret, rtype, tok.pos)

    # expressions --------------------------------------------------------------
    def expr(self):
        cond = self.logic("||", "or", lambda: self.logic("&&", "and", self.compare))
        if self.accept("?"):
            a = self.expr()
            self.expect(":")
            return ("tern", cond, a, self.expr(), cond[-1])
        return cond

    def logic(self, text, kind, operand):
        """a || b, a && b: bools only, short-circuit (&& binds tighter)."""
        a = operand()
        while self.peek().kind == "op" and self.peek().text == text:
            op = self.next()
            a = (kind, a, operand(), op.pos)
        return a

    def compare(self):
        a = self.additive()
        while self.peek().kind == "op" and self.peek().text in ("==", "!=", "<", ">", "<=", ">="):
            op = self.next()
            a = ("bin", op.text, a, self.additive(), op.pos)
        return a

    def additive(self):
        a = self.term()
        while self.peek().kind == "op" and self.peek().text in "+-":
            op = self.next()
            a = ("bin", op.text, a, self.term(), op.pos)
        return a

    def term(self):
        a = self.unary()
        while self.peek().kind == "op" and self.peek().text in ("*", "/", "%"):
            op = self.next()
            a = ("bin", op.text, a, self.unary(), op.pos)
        return a

    def unary(self):
        if self.peek().text == "-" and self.peek().kind == "op":
            op = self.next()
            return ("neg", self.unary(), op.pos)
        if self.peek().text == "!" and self.peek().kind == "op":
            op = self.next()
            return ("not", self.unary(), op.pos)
        return self.postfix()

    def postfix(self):
        e = self.primary()
        while True:
            tok = self.peek()
            if tok.text == "." and tok.kind == "op":
                self.next()
                e = ("get", e, self.expect(kind="id").text, tok.pos)
            elif tok.text == "[" and tok.kind == "op" and self.t[self.i - 1].end == tok.start:
                self.next()
                idx = self.expr()
                self.expect("]")
                e = ("index", e, idx, tok.pos)
            elif tok.text == "(" and tok.kind == "op" and self.t[self.i - 1].end == tok.start:
                self.next()  # a call: f(x, y = 2 m); the ( touches the callee, like [ of an index
                args = []
                while not self.accept(")"):
                    if self.peek().kind == "id" and self.peek(1).kind == "op" and self.peek(1).text == "=":
                        k = self.next()
                        self.next()
                        args.append((k.text, self.expr(), k.pos))
                    else:
                        if args and args[-1][0] is not None:
                            raise OrbError("positional arguments go before named ones", self.peek().pos)
                        args.append((None, self.expr(), self.peek().pos))
                    if not self.accept(","):
                        self.expect(")")
                        break
                e = ("call", e, args, tok.pos)
            elif (tok.kind == "id" and tok.pos.line == self.t[self.i - 1].pos.line and unit_term(tok.text)
                  and self.peek(1).text != "="):  # `x = 1  m = 2`: m is the next attribute
                e = ("unit", e, self.unit(), tok.pos)
            else:
                return e

    def unit(self):
        """kg | m/s2 | Nm*s/rad -- operators must touch their operands (no spaces)."""
        tok = self.next()
        f, d = unit_term(tok.text)
        while True:
            op, nxt = self.peek(), self.peek(1)
            if (op.kind == "op" and op.text in "*/" and op.start == self.t[self.i - 1].end
                    and nxt.kind == "id" and nxt.start == op.end and unit_term(nxt.text)):
                self.i += 2
                f2, d2 = unit_term(nxt.text)
                sgn = 1 if op.text == "*" else -1
                f, d = (f * f2 if sgn > 0 else f / f2), tuple(a + sgn * b for a, b in zip(d, d2))
            else:
                return f, d

    def primary(self):
        tok = self.next()
        if tok.kind == "num":
            v = float(tok.text)
            return ("lit", int(v) if v.is_integer() and not re.search(r"[.eE]", tok.text) else v, tok.pos)
        if tok.kind == "str":
            return self.string(tok)
        if tok.kind == "id":
            if tok.text in ("true", "false", "null"):
                return ("lit", {"true": True, "false": False, "null": None}[tok.text], tok.pos)
            return ("name", tok.text, tok.pos)
        if tok.text == "(":
            e = self.expr()
            self.expect(")")
            return e
        if tok.text == "[" and self.peek().kind == "id" and self.peek().text == "for" and self.peek(1).kind == "id":
            self.next()  # [for x in xs : expr if cond]   [for i, x in xs : expr]
            a = self.expect(kind="id").text
            b = self.expect(kind="id").text if self.accept(",") else None
            self.expect("in")
            src = self.expr()
            if self.peek().text == "if" and self.peek().kind == "id":
                raise OrbError("a comprehension's filter goes last: [for x in xs : <value> if <condition>]",
                               self.peek().pos)
            self.expect(":")
            body = self.expr()
            cond = self.expr() if self.accept("if") else None
            self.expect("]")
            return ("for", a, b, src, body, cond, tok.pos)
        if tok.text == "[":
            items = []
            while not self.accept("]"):
                items.append(self.expr())
                if not self.accept(","):
                    self.expect("]")
                    break
            return ("list", items, tok.pos)
        if tok.text == "{":
            pairs = []
            while not self.accept("}"):
                k = self.next()
                if k.kind not in ("id", "str"):
                    raise OrbError("expected object key", k.pos)
                key = k.text if k.kind == "id" else self.string(k)
                self.expect("=")
                pairs.append((key, self.expr()))
                self.accept(",")
            return ("obj", pairs, tok.pos)
        got = repr(tok.text) if tok.kind != "eof" else "end of file"
        raise OrbError(f"unexpected {got}", tok.pos)

    def string(self, tok):
        """String literal with ${expr} interpolation -> ('str', [str | expr, ...])."""
        raw, parts, buf, i = tok.text[1:-1], [], "", 0
        while i < len(raw):
            c = raw[i]
            if c == "\\" and i + 1 < len(raw):
                buf += _ESC.get(raw[i + 1], raw[i + 1])
                i += 2
            elif raw.startswith("${", i):
                j = _interp_end(raw, i + 2)
                if j < 0:
                    raise OrbError("unclosed ${ in string (close it with })", tok.pos)
                if not raw[i + 2:j].strip():
                    raise OrbError("empty ${} in string (write \\${ for a literal ${)", tok.pos)
                if buf:
                    parts.append(buf)
                    buf = ""
                sub = Parser(lex(raw[i + 2:j], self.file, tok.pos.line, tok.pos.col + i + 3), self.file)
                parts.append(sub.expr())
                sub.expect(kind="eof")
                i = j + 1
            else:
                buf += c
                i += 1
        if buf or not parts:
            parts.append(buf)
        return ("str", parts, tok.pos)


def _interp_end(raw, i):
    """The index of the `}` closing a `${` whose expression starts at i (strings and braces inside
    it skipped), or -1."""
    depth, n = 0, len(raw)
    while i < n:
        c = raw[i]
        if c == '"':
            i += 1
            while i < n and raw[i] != '"':
                i += 2 if raw[i] == "\\" else 1
        elif c == "{":
            depth += 1
        elif c == "}":
            if depth == 0:
                return i
            depth -= 1
        i += 1
    return -1


def parse(src, file="<input>"):
    return Parser(lex(src, file), file).program()


# --------------------------------------------------------------------------- evaluator


class Block:
    """An evaluated block. `path` is its address, e.g. site.office.zone.bench."""

    def __init__(self, type, label, path, pos):
        self.type, self.label, self.path, self.pos = type, label, path, pos
        self.attrs, self.attr_pos, self.children = {}, {}, []

    def child(self, type, label=None):
        for c in self.children:
            if c.type == type and (label is None or c.label == label):
                return c
        return None

    def kids(self, type):
        return [c for c in self.children if c.type == type]

    def __repr__(self):
        return f"<{self.path}>"


def relabel(blk, prefix):
    """Prefix a top-level block's label (and every path under it) with an import prefix."""
    old = blk.path
    blk.label = prefix + blk.label
    new = f"{blk.type}.{blk.label}"

    def walk(b):
        if b.path == old or b.path.startswith(old + "."):
            b.path = new + b.path[len(old):]
        for c in b.children:
            walk(c)
    walk(blk)


class Group:
    """`part` or `site.office.zone`: a set of blocks addressable by label. `stub(label)`: what a
    label no block declares means (item["<output>/<id>"]: an item an output places), else an error."""

    def __init__(self, path, lookup, labels, stub=None):
        self.path, self.lookup, self.labels, self.stub = path, lookup, labels, stub


class Function:
    """A `fn` declared in some program; called with that program's globals in scope (lexical)."""

    def __init__(self, st, prog):
        self.st, self.prog, self.name = st, prog, st[1]

    def __repr__(self):
        return f"<fn {self.name}>"


class Builtin:
    """A function the language provides: impl(prog, args, kwargs, pos)."""

    def __init__(self, name, impl, doc=""):
        self.name, self.impl, self.doc = name, impl, doc

    def __repr__(self):
        return f"<builtin {self.name}>"


MIRROR_OF = ".orb-mirror-of"  # the editor checks unsaved edits in a temp mirror; this file names the project


def import_base(file):
    """The folder an `import "<path>"` in `file` is relative to: the file's own, or for a file in Orb
    the editor's unsaved-edit mirror (the editor project-mirror implementation) the mirrored project's, so
    `import "../robots"` finds the project's real neighbours."""
    base = Path(file).parent
    marker = base / MIRROR_OF
    if marker.is_file():
        orig = Path(marker.read_text(encoding="utf-8").strip())
        if orig.is_absolute() and orig.is_dir():
            return orig
    return base


EXPORT_EXT = (".glb", ".gltf", ".usd", ".usda", ".usdc", ".usdz")  # Orb exports `import` can name
IMPORTS_DIR = Path(".orb") / "imports"  # where an imported export file is unpacked (by content hash)
EXPORT_MARK = ".orb-export"  # in an unpacked export: "<sha256>\n<the export file>"


class _Quiet:
    """A reporter that says nothing: unpacking an imported export is part of loading."""

    def __getattr__(self, name):
        return lambda *a, **k: None


def file_sha256(path):
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def unpack_export(path, base, rel, pos):
    """`import "../a/export/a.glb" as a`: an Orb export (GLB/USD) is a project too. Its OrbLang sources
    (or, when they imported other projects, the canonical OrbLang rebuilt from its world graph) and
    meshes are unpacked once per file content into `<base>/.orb/imports/<name>-<sha12>/` (the importing
    project's; ~/.cache/orb/imports when that is not writable) and read from there. -> (dir, sha256)."""
    import shutil
    import tempfile

    from .config import slugify

    sha = file_sha256(path)
    name = f"{slugify(Path(path).stem)}-{sha[:12]}"
    roots = [Path(base) / IMPORTS_DIR, Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "orb" / "imports"]
    for root in roots:  # already unpacked (by this project, or into the user cache)
        mark = root / name / EXPORT_MARK
        if mark.is_file() and mark.read_text(encoding="utf-8").split("\n", 1)[0] == sha:
            return root / name, sha
    from .importer import NotOrb, import_export

    last = None
    for root in roots:
        try:
            root.mkdir(parents=True, exist_ok=True)
            tmp = Path(tempfile.mkdtemp(prefix=f".{name}.", dir=root))
        except OSError as e:
            last = e
            continue
        try:
            try:
                import_export(path, Path(path).stem, into=tmp, force=True, rep=_Quiet())
            except NotOrb as e:
                raise OrbError(f"import {rel!r}: {Path(path).name} is not an Orb export ({e}); a scan becomes a project "
                               "with `orb capture`, then import that project's folder", pos) from None
            except OrbError as e:
                raise OrbError(f"import {rel!r}: cannot read {Path(path).name} as a project: {e.msg}", pos) from None
            (tmp / EXPORT_MARK).write_text(f"{sha}\n{os.path.realpath(path)}\n", encoding="utf-8")
            dest = root / name
            if dest.exists():  # a stale or half-written copy (no matching mark): replaced
                shutil.rmtree(dest, ignore_errors=True)
            try:
                os.replace(tmp, dest)
            except OSError:  # another load unpacked it meanwhile
                if not (dest / EXPORT_MARK).is_file():
                    raise
            return dest, sha
        finally:
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)
    raise OrbError(f"import {rel!r}: nowhere to unpack {Path(path).name} ({last})", pos)


class Namespace:
    """`import "<dir>" as name`: another Orb project, read on its own. `name.part.x`, `name.fn(...)`,
    `name.some_let`, `name.outputs.<output>` (the scene an output of it places), `name.world` (its
    whole world as one scene). Nothing of it is in this world until a scene places it."""

    def __init__(self, alias, prog):
        self.alias, self.prog = alias, prog

    def __repr__(self):
        return f"<project {self.alias}>"


MAX_DEPTH = 64  # nested fn calls (recursion included)
_CALLS = []


def check_type(v, ty, what, pos):
    """Is v of the declared parameter/result type? (No declaration: anything.)"""
    if ty is None or ty == "any":
        return
    dims = {"length": (0, 1, 0, 0, 0, 0), "mass": (1, 0, 0, 0, 0, 0), "time": (0, 0, 1, 0, 0, 0),
            "angle": (0, 0, 0, 0, 0, 1)}
    if ty in dims:
        ok = (isinstance(v, Q) and v.d == dims[ty]) or (v == 0 and isinstance(v, (int, float)) and not isinstance(v, bool))
    elif ty == "point":
        ok = isinstance(v, list) and len(v) == 3 and all(
            (isinstance(x, Q) and x.d == dims["length"]) or (x == 0 and not isinstance(x, (bool, Q))) for x in v)
    elif ty == "number":
        ok = isinstance(v, (int, float)) and not isinstance(v, bool)
    elif ty in ("part", "assembly", "site", "item", "zone", "fixture"):
        ok = isinstance(v, Block) and v.type == ty
    else:
        ok = {"string": lambda: isinstance(v, str), "bool": lambda: isinstance(v, bool),
              "list": lambda: isinstance(v, list), "object": lambda: isinstance(v, dict),
              "fn": lambda: isinstance(v, (Function, Builtin)), "block": lambda: isinstance(v, Block),
              "scene": lambda: getattr(v, "orb_type", None) == "scene"}[ty]()
    if not ok:
        unit = {"length": "m", "mass": "kg", "time": "s", "angle": "deg"}.get(ty)
        hint = (f" (write {v:g} {unit})" if unit and isinstance(v, (int, float)) and not isinstance(v, bool) else
                " (a point is [x, y, z] in length units, e.g. [1, 2, 0] m)" if ty == "point" and isinstance(v, list) else "")
        art = "" if ty == "any" else "an " if ty[0] in "aeiou" else "a "
        raise OrbError(f"{what} must be {art}{ty}, got {type_name(v)}{hint}", pos)


class Env:
    def __init__(self, parent=None, vars=None, prog=None):
        self.parent, self.vars, self.prog = parent, dict(vars or {}), prog

    def lookup(self, name, pos):
        env = self
        while env:
            if name in env.vars:
                return env.vars[name]
            if env.parent is None and env.prog:
                try:
                    return env.prog.resolve_global(name, pos)
                except OrbError as e:
                    if e.msg == f"unknown name {name!r}":
                        raise OrbError(e.msg + did_you_mean(name, self.names()), pos) from None
                    raise
            env = env.parent
        raise OrbError(f"unknown name {name!r}" + did_you_mean(name, self.names()), pos)

    def names(self):
        """Every name visible here: locals out to the project's globals and the builtins."""
        out, env = set(), self
        while env:
            out |= set(env.vars)
            if env.parent is None and env.prog:
                out |= env.prog.global_names()
            env = env.parent
        return out


BUILTIN_SEAMS = ("mujoco", "isaac", "glb", "usd", "minecraft", "step")
SKIP_DIRS = {"build", "export", "node_modules", "__pycache__"}


class Blocks(list):
    """The evaluated top-level blocks, plus what is not a block of the world: the libraries the
    program included (absolute paths) and its top-level `native` blocks (with a `target`)."""
    includes: list = []
    natives: list = []


def orb_files(d):
    """The .orb files of an included library directory, recursively, in a stable order; hidden
    directories (.orb/sources, .git) and build outputs are not part of it."""
    d = Path(d)
    out = []
    for f in sorted(d.rglob("*.orb")):
        rel = f.relative_to(d).parts[:-1]
        if not any(p.startswith(".") or p in SKIP_DIRS for p in rel) and f.is_file():
            out.append(f)
    return out


def addendum_seam(file, seams):
    """`trucks.isaac.orb` -> "isaac" when isaac is a seam (built in or declared): a seam addendum,
    whose top-level blocks are only in that seam."""
    parts = Path(str(file)).name.split(".")
    return parts[-2] if len(parts) >= 3 and parts[-1] == "orb" and parts[-2] in seams else None


class Program:
    def __init__(self, prefix="", imported=None, parent=None, dir=None):
        # prefix: "robots/" for a project imported `as robots` (its top-level labels become
        # robots/<label>); imported: realpath -> Program, shared by every program of one load (a
        # project two others import is one project: one set of definitions, not one per path to it)
        self.prefix, self.imported = prefix, ({} if imported is None else imported)
        self.parent, self.dir = parent, dir  # the importing program; this project's real directory
        self.fns, self.fn_vals, self.imports, self.ns_vals = {}, {}, {}, {}
        self.lets, self.modules, self.decls = {}, {}, {}
        self.let_vals, self.blocks, self.busy = {}, {}, set()
        self.env = Env(prog=self)
        self.natives, self.includes, self.loaded = [], [], set()
        self._top_natives = None
        self.library_sources = []  # (label, file) of `source` blocks in included files: not this project's
        self.decl_file, self._seams = {}, None
        self.export_sha = None  # an imported Orb export file: its sha256 (prog.ident is the file)
        self._output_scenes, self._world_scene = {}, None
        self.from_include = set()  # ("fn" | "let", name) declared in an included file
        self._names_known = True  # check_fns: may an unknown name be reported (not in an included fn)

    # loading ------------------------------------------------------------------
    def add_file(self, f, included=False):
        """Read one .orb file (once: a file reached twice through includes is read once)."""
        key = os.path.realpath(f)
        if key in self.loaded:
            return
        self.loaded.add(key)
        self.add_source(Path(f).read_text(encoding="utf-8-sig"), str(f), included)  # a BOM (Notepad) is not source

    def include(self, rel, file, pos):
        """`include "<path>"`: a directory of .orb files (recursive) or one file, relative to the
        including file (for a file in the editor unsaved-edit mirror, the mirrored project file). Its blocks
        merge into the program as if declared here."""
        target = Path(os.path.normpath(Path(file).parent / rel))
        if not target.exists():  # the editor unsaved-edit mirror holds the project files only: a sibling
            # library (../shared-library) is where the mirrored project has it. Project-local includes stay
            # in the mirror, so their unsaved edits are what gets read (and read once).
            alt = Path(os.path.normpath(import_base(file) / rel))
            if alt.exists():
                target = alt
        if not target.exists():
            raise OrbError(f"include {rel!r}: no such file or directory ({os.path.abspath(target)})", pos)
        files = orb_files(target) if target.is_dir() else [target]
        if not files:
            raise OrbError(f"include {rel!r}: no .orb files in {os.path.abspath(target)}", pos)
        ab = os.path.abspath(target)
        if ab not in self.includes:
            self.includes.append(ab)
        for f in files:
            self.add_file(f, included=True)

    def add_source(self, src, file, included=False):
        """Parse one file's text into the program. `included`: it was reached through `include`, so its
        `source` blocks belong to that library (pulled on its own), not to this project."""
        for st in parse(src, file):
            kind, pos = st[0], st[-1]
            if kind == "include":
                self.include(st[1], file, pos)
            elif kind == "fn":
                self._unique(self.fns, st[1], st, f"fn {st[1]}")
                if included:
                    self.from_include.add(("fn", st[1]))
            elif kind == "import":
                self._unique(self.imports, st[2], (st[1], file, pos), f"import {st[2]}")
            elif kind == "block" and st[1] == "source" and included:  # a library's sources are its own
                label = st[2][0][1][0] if st[2] and len(st[2][0][1]) == 1 else ""
                self.library_sources.append((label, file))
            elif kind == "block" and st[1] == "native":  # top-level native: attaches to its `target`
                if len(st[2]) > 1 or any(len(l[1]) != 1 or not isinstance(l[1][0], str) for l in st[2]):
                    raise OrbError("top-level blocks take one plain string label", pos)
                self.natives.append((st, file))
            elif kind == "let":
                self._unique(self.lets, st[1], st, f"let {st[1]}")
                if included:
                    self.from_include.add(("let", st[1]))
            elif kind == "module":
                self._unique(self.modules, st[1], st, f"module {st[1]}")
            elif kind == "block":
                if st[1] in self.modules:
                    raise OrbError("module instances must live inside a block", pos)
                if len(st[2]) > 1 or any(len(l[1]) != 1 or not isinstance(l[1][0], str) for l in st[2]):
                    raise OrbError("top-level blocks take one plain string label", pos)
                label = st[2][0][1][0] if st[2] else ""
                self._unique(self.decls, (st[1], label), st, f"{st[1]} {label!r}")
                self.decl_file[(st[1], label)] = file
            else:
                raise OrbError("attributes must live inside a block", pos)

    @staticmethod
    def _unique(table, key, st, what):
        if key in table:
            raise OrbError(f"duplicate {what} (first declared at {table[key][-1]})", st[-1])
        table[key] = st

    def seams(self):
        """Seam names a file suffix may name: the built-in ones and every declared `seam`."""
        if self._seams is None:
            self._seams = set(BUILTIN_SEAMS) | {label for t, label in self.decls if t == "seam"}
        return self._seams

    def evaluate(self):
        for t, label in list(self.decls):
            self.block(t, label, self.decls[(t, label)][-1])
        out = Blocks(self.blocks[k] for k in self.decls)
        out.includes = list(self.includes)
        out.imports = self.import_records()
        out.natives = self.top_natives()
        # every project this one imports (at any depth): their top-level natives travel with their
        # definitions; the world applies the ones whose target it places
        out.imported_natives = []
        done = set()
        while True:
            todo = [p for k, p in self.imported.items() if k not in done]
            if not todo:
                break
            for k, p in list(self.imported.items()):
                if k not in done:
                    done.add(k)
                    out.imported_natives += p.top_natives()
        return out

    # imports, read up front --------------------------------------------------------
    def resolve_imports(self):
        """Read every project this one imports, at any depth, breadth first in declaration order: a
        missing or cyclic import is an error even before anything uses it, and a project reached
        along several paths is read once, labelled by its nearest import."""
        queue, seen = [self], {id(self)}
        while queue:
            p = queue.pop(0)
            for alias, (_, _, ipos) in list(p.imports.items()):
                prog = p.namespace(alias, ipos).prog
                if id(prog) not in seen:
                    seen.add(id(prog))
                    queue.append(prog)
        self.check_import_cycles()
        return self

    def check_import_cycles(self):
        """An import cycle anywhere in the graph is an error, whichever path first reached its projects
        (the check while reading follows only the path that read each one)."""
        state = {}  # id(program) -> "open" while on the walk, "done" after

        def walk(p):
            state[id(p)] = "open"
            for alias, (rel, _, ipos) in p.imports.items():
                q = p.ns_vals[alias].prog
                if state.get(id(q)) == "open":
                    raise OrbError(f"import cycle: {rel!r} imports a project that imports it", ipos)
                if id(q) not in state:
                    walk(q)
            state[id(p)] = "done"
        walk(self)

    def evaluate_lets(self):
        """Evaluate every top-level `let` (`orb validate`): an unused one that cannot be evaluated is an
        error too, not only when something reads it."""
        for name, st in self.lets.items():
            self.resolve_global(name, st[-1])

    def import_records(self):
        """One record per imported project: the label prefix its definitions get, where it is (relative
        to this project) and a sha256 over its .orb files (included ones too), so `orb plan` can say
        when an imported project changed under this one."""
        import hashlib

        out = []
        for prog in sorted(self.imported.values(), key=lambda p: p.prefix):
            h = hashlib.sha256()
            for f in sorted(prog.loaded):
                rel = os.path.relpath(f, prog.dir).replace(os.sep, "/")
                h.update(rel.encode() + b"\0" + hashlib.sha256(Path(f).read_bytes()).hexdigest().encode() + b"\n")
            digest = prog.export_sha or h.hexdigest()  # an export file: the file itself
            base = import_base(Path(self.dir) / "x") if self.dir else None  # Studio's mirror: the real project
            path = os.path.relpath(prog.ident, base) if base else prog.ident
            out.append({"label": prog.prefix.rstrip("/"), "path": path.replace(os.sep, "/"),
                        "sha256": digest})
        return out

    def check_fns(self):
        """What a call would find wrong in a fn body, found without calling it: a name nothing
        defines, and an expression without names that cannot be evaluated (`1 m + 1 kg`). Run on
        every project of a load, so a broken function fails `orb validate` even if nothing calls it.
        Top-level `let`s get the same checks."""
        glob = self.global_names()
        for name, st in self.lets.items():  # top-level lets: the same checks (they are evaluated lazily)
            self._names_known = ("let", name) not in self.from_include
            self._check_expr(st[2], set(), glob)
        for fname, st in self.fns.items():
            _, _, params, lets, ret, _, _ = st
            # a shared library's fn may name what only some of the projects including it declare (a
            # let, sites): there an unknown name is the includer's to supply, found when called
            self._names_known = ("fn", fname) not in self.from_include
            scope = set()
            try:
                for pn, _, default, _ in params:
                    if default is not None:
                        self._check_expr(default, scope, glob)
                    scope.add(pn)
                for ln, ex, _ in lets:
                    self._check_expr(ex, scope, glob)
                    scope.add(ln)
                self._check_expr(ret, scope, glob)
            except OrbError as e:
                raise OrbError(f"{e.msg} (in fn {fname})", e.pos) from None

    def _check_expr(self, e, scope, glob, fold=True):
        """fold: evaluate name-free parts now. Off inside a branch a named condition guards (`c ? a : b`,
        `c && x`, `c || x`): code that never runs there is not an error."""
        kind, pos = e[0], e[-1]
        if kind == "lit":
            return
        if not _has_names(e):  # nothing to look up: evaluating it now is what a call would do
            if fold:
                self.eval(e, Env())
            return
        if kind == "name":
            n = e[1]
            if n not in scope and n not in glob and n not in BLOCK_TYPES and self._names_known:
                raise OrbError(f"unknown name {n!r}" + did_you_mean(n, scope | glob), pos)
            return
        if kind == "tern" and _has_names(e[1]):
            self._check_expr(e[1], scope, glob, fold)
            self._check_expr(e[2], scope, glob, False)
            self._check_expr(e[3], scope, glob, False)
            return
        if kind in ("and", "or") and _has_names(e[1]):
            self._check_expr(e[1], scope, glob, fold)
            self._check_expr(e[2], scope, glob, False)
            return
        if kind == "for":
            _, a, b, src, body, cond = e[:6]
            self._check_expr(src, scope, glob, fold)
            inner = scope | {a} | ({b} if b else set())
            self._check_expr(body, inner, glob, fold)
            if cond is not None:
                self._check_expr(cond, inner, glob, fold)
            return
        for sub in _subexprs(e):
            self._check_expr(sub, scope, glob, fold)

    def top_natives(self):
        """The top-level `native` blocks, evaluated (their `target` resolved in this project)."""
        if self._top_natives is None:
            self._top_natives = []
            for st, file in self.natives:
                blk = self.eval_block(st, self.env, "")
                seam = addendum_seam(file, self.seams())
                if seam and blk.label and blk.label != seam:
                    raise OrbError(f"native {blk.label!r} in {Path(file).name}, an addendum for {seam} (natives "
                                   f"there are for {seam})", blk.pos)
                if seam and not blk.label:
                    blk.label = seam
                self._top_natives.append(blk)
        return self._top_natives

    # lazy globals ---------------------------------------------------------------
    def resolve_global(self, name, pos):
        if name in self.fns:
            if name in self.lets or name in self.imports:
                raise OrbError(f"{name!r} is both a fn and a {'let' if name in self.lets else 'import'}", pos)
            if name not in self.fn_vals:
                self.fn_vals[name] = Function(self.fns[name], self)
            return self.fn_vals[name]
        if name in self.imports:
            return self.namespace(name, pos)
        if name in self.lets:
            if name not in self.let_vals:
                if ("let", name) in self.busy:
                    raise OrbError(f"cycle while evaluating let {name}", pos)
                self.busy.add(("let", name))
                self.let_vals[name] = self.eval(self.lets[name][2], self.env)
                self.busy.discard(("let", name))
            return self.let_vals[name]
        labels = [l for t, l in self.decls if t == name]
        placed = name in ("item", "site") and any(t == "output" for t, _ in self.decls)
        if labels or placed:
            return Group(name, lambda label, p: self.block(name, label, p), labels,
                         (self.placed_item if name == "item" else self.placed_site) if placed else None)
        if name == "outputs":  # this project's own outputs, each the scene it places
            return self.outputs("outputs")
        from .builtins import BUILTINS

        if name in BUILTINS:
            return BUILTINS[name]
        if name in BLOCK_TYPES:
            raise OrbError(f"this project has no {name} blocks ({name}.x names a {name} \"x\" declared at the top "
                           "level)", pos)
        raise OrbError(f"unknown name {name!r}", pos)

    def global_names(self):
        from .builtins import BUILTINS

        return (set(self.lets) | set(self.fns) | set(self.imports) | {t for t, _ in self.decls} | set(BUILTINS)
                | {"outputs"})

    def owns(self, name):
        """Does this project define `name` itself (a let, fn, import or block type)? Then it wins over
        what the language provides under that name (`outputs`, `<alias>.world`)."""
        return name in self.lets or name in self.fns or name in self.imports or any(t == name for t, _ in self.decls)

    def placed_site(self, label, pos):
        """site["<output>/<id>"]: a site an output places (its .zone.x / .fixture.y too). The world checks
        that it exists."""
        if not isinstance(label, str):
            raise OrbError(f"sites are named by strings, not {type_name(label)}", pos)
        b = Block("site", label, f"site.{label}", pos)
        b.placed = True
        return b

    def placed_item(self, label, pos):
        """item["<output>/<id>/..."]: an item an output places (`item["<id>"]` under `prefix = ""`). The
        world checks that it exists (and says which declared or placed items there are)."""
        if not isinstance(label, str):
            raise OrbError(f"items are named by strings, not {type_name(label)}", pos)
        b = Block("item", label, f"item.{label}", pos)
        b.placed = True
        return b

    def namespace(self, alias, pos):
        """The project `import "<path>" as alias` names, read once (on first use) as its own program."""
        if alias in self.ns_vals:
            return self.ns_vals[alias]
        rel, file, ipos = self.imports[alias]
        target = Path(os.path.normpath(import_base(file) / rel))
        if not target.exists():
            raise OrbError(f"import {rel!r}: no such file or directory ({os.path.abspath(target)})", ipos)
        prefix = f"{self.prefix}{alias}/"
        real = os.path.realpath(target)
        chain, p = [], self
        while p is not None:  # every program up the import chain, by folder and (one-file imports) by file
            chain += [p.dir, getattr(p, "ident", None)]
            p = p.parent
        if real in chain:
            raise OrbError(f"import cycle: {rel!r} imports a project that imports it", ipos)
        export = target.is_file() and target.suffix.lower() in EXPORT_EXT
        if target.is_file() and not export and target.suffix != ".orb":
            raise OrbError(f"import {rel!r}: an import names an Orb project (a folder), one .orb file or an Orb "
                           f"export ({', '.join(EXPORT_EXT)}), not a {target.suffix or 'file without a suffix'}", ipos)
        key = real  # by project, not by path to it: `load()` reads the nearest import first, so a shared
        # project is labelled by its shallowest alias however many projects import it
        prog = self.imported.get(key)
        if prog is None:
            try:
                if export:  # an Orb export: the project it carries, unpacked once per file content
                    folder, sha = unpack_export(target, import_base(file), rel, ipos)
                    prog = load_program(folder, prefix=prefix, imported=self.imported, parent=self)
                    prog.ident, prog.export_sha = real, sha
                else:
                    prog = load_program(target, prefix=prefix, imported=self.imported, parent=self)
            except OrbError as e:
                if "(imported as" in e.msg:
                    raise
                raise OrbError(f"{e.msg} (imported as {alias} at {ipos})", e.pos) from None
            self.imported[key] = prog
        self.ns_vals[alias] = Namespace(alias, prog)
        return self.ns_vals[alias]

    # functions ------------------------------------------------------------------
    def call(self, f, args, pos):
        """Call a fn or builtin. args: [(name or None, value, pos)], evaluated in the caller."""
        if isinstance(f, Builtin):
            pos_args = [v for k, v, _ in args if k is None]
            kw = {}
            for k, v, p in args:
                if k is not None:
                    if k in kw:
                        raise OrbError(f"{f.name}() got {k!r} twice", p)
                    kw[k] = v
            return f.impl(self, pos_args, kw, pos)
        if not isinstance(f, Function):
            raise OrbError(f"cannot call {type_name(f)}: only functions take arguments", pos)
        return f.prog.invoke(f, args, pos)

    def invoke(self, f, args, pos):
        _, name, params, lets, ret, rtype, fpos = f.st
        names = [p[0] for p in params]
        bound = {}
        positional = [(v, p) for k, v, p in args if k is None]
        if len(positional) > len(params):
            raise OrbError(f"fn {name} takes {len(params)} argument{'s' * (len(params) != 1)}, got {len(positional)}", pos)
        for pn, (v, _) in zip(names, positional):
            bound[pn] = v
        for k, v, p in args:
            if k is None:
                continue
            if k not in names:
                raise OrbError(f"fn {name} has no parameter {k!r}" + (did_you_mean(k, names) or
                               f" (has: {', '.join(names) or 'none'})"), p)
            if k in bound:
                raise OrbError(f"fn {name} got {k!r} twice", p)
            bound[k] = v
        if len(_CALLS) >= MAX_DEPTH:
            raise OrbError(f"fn {name}: calls nested deeper than {MAX_DEPTH} (unbounded recursion?)", pos)
        _CALLS.append(name)
        try:
            scope = Env(self.env)  # lexical: this program's globals and the parameters, not the caller's lets
            for pn, ty, default, ppos in params:
                if pn in bound:
                    v = bound[pn]
                elif default is not None:
                    try:
                        v = self.eval(default, scope)
                    except OrbError as e:
                        later = [q for q in names[names.index(pn) + 1:] if e.msg.startswith(f"unknown name {q!r}")]
                        if later:
                            raise OrbError(f"the default of {pn!r} uses {later[0]!r}, a later parameter: defaults "
                                           "may use only the parameters before them", e.pos) from None
                        raise
                else:
                    raise OrbError(f"fn {name} needs argument {pn!r}", pos)
                check_type(v, ty, f"argument {pn!r} of fn {name}", pos)
                scope.vars[pn] = v
            for ln, ex, _ in lets:
                scope.vars[ln] = self.eval(ex, scope)
            out = self.eval(ret, scope)
            check_type(out, rtype, f"the result of fn {name}", fpos)
            return out
        except OrbError as e:
            if "(in fn " in e.msg:
                if len(_CALLS) == 1 and "calls nested deeper" in e.msg and "outermost call" not in e.msg:
                    raise OrbError(f"{e.msg}; the outermost call is at {pos}", e.pos) from None
                raise
            raise OrbError(f"{e.msg} (in fn {name}, called at {pos})", e.pos) from None
        finally:
            _CALLS.pop()

    def block(self, type, label, pos):
        key = (type, label)
        if key in self.blocks:
            return self.blocks[key]
        if key not in self.decls:
            raise OrbError(f"no {type} named {label!r}", pos)
        if key in self.busy:
            raise OrbError(f"reference cycle through {type}.{label}", pos)
        self.busy.add(key)
        blk = self.eval_block(self.decls[key], self.env, "")
        seam = addendum_seam(self.decl_file.get(key, ""), self.seams())
        if seam:  # every top-level block of `x.<seam>.orb` is only in that seam
            for k in ("only", "skip"):
                if k in blk.attrs:
                    raise OrbError(f"{Path(blk.pos.file).name} is an addendum for {seam}: its blocks are only in "
                                   f"{seam} already (drop `{k}`)", blk.attr_pos[k])
            blk.attrs["only"], blk.attr_pos["only"] = [seam], blk.pos
        if self.prefix:  # an imported project's blocks: robots/robot (its own references keep their source labels)
            relabel(blk, self.prefix)
        self.blocks[key] = blk
        self.busy.discard(key)
        return self.blocks[key]

    # blocks -------------------------------------------------------------------
    def eval_block(self, st, env, parent_path):
        _, typ, label_asts, body, pos = st
        labels = [self.eval(l, env) for l in label_asts]
        if len(labels) > 1:
            raise OrbError(f"{typ} takes at most one label", pos)
        label = labels[0] if labels else ""
        if not isinstance(label, str):
            raise OrbError("block labels must be strings", pos)
        path = ".".join(p for p in (parent_path, typ, label) if p)
        blk = Block(typ, label, path, pos)
        self.fill(blk, body, Env(env))
        return blk

    def fill(self, blk, body, env):
        for st in body:
            kind, pos = st[0], st[-1]
            if kind == "let":
                env.vars[st[1]] = self.eval(st[2], env)
            elif kind == "attr":
                if st[1] in blk.attrs:
                    raise OrbError(f"duplicate attribute {st[1]!r} (first set at {blk.attr_pos[st[1]]})", pos)
                blk.attrs[st[1]] = self.eval(st[2], env)
                blk.attr_pos[st[1]] = pos
            elif kind == "block" and st[1] in self.modules:
                self.expand(st, blk, env)
            elif kind == "block":
                blk.children.append(self.eval_block(st, env, blk.path))
            else:
                raise OrbError(f"{kind} is not allowed here", pos)

    def expand(self, st, blk, env):
        """Module instance: params are evaluated in the caller, the body is spliced into blk."""
        _, typ, label_asts, body, pos = st
        mod = self.modules[typ]
        labels = [self.eval(l, env) for l in label_asts]
        args = {}
        for s in body:
            if s[0] != "attr":
                raise OrbError(f"module {typ} takes only parameters", s[-1])
            args[s[1]] = self.eval(s[2], env)
        names = [p for p, _ in mod[2]]
        if unknown := set(args) - set(names):
            raise OrbError(f"module {typ} has no parameter {sorted(unknown)[0]!r} (has: {', '.join(names)})", pos)
        scope = Env(self.env, {"name": labels[0] if labels else typ})
        for p, default in mod[2]:
            if p in args:
                scope.vars[p] = args[p]
            elif default is not None:
                scope.vars[p] = self.eval(default, scope)
            else:
                raise OrbError(f"module {typ} needs parameter {p!r}", pos)
        try:
            self.fill(blk, mod[3], scope)
        except OrbError as e:
            if "(in module" in e.msg:
                raise
            raise OrbError(f"{e.msg} (in module {typ} {labels[0] if labels else ''}, used at {pos})", e.pos) from None

    # expressions --------------------------------------------------------------
    def eval(self, e, env):
        kind, pos = e[0], e[-1]
        if kind == "lit":
            return e[1]
        if kind == "str":
            out = ""
            for p in e[1]:
                v = p if isinstance(p, str) else self.eval(p, env)
                out += interpolated(v, pos)
            return out
        if kind == "name":
            return env.lookup(e[1], pos)
        if kind == "list":
            return [self.eval(x, env) for x in e[1]]
        if kind == "obj":
            return {(k if isinstance(k, str) else self.eval(k, env)): self.eval(v, env) for k, v in e[1]}
        if kind == "neg":
            return arith("*", self.eval(e[1], env), -1, pos)
        if kind == "bin":
            return arith(e[1], self.eval(e[2], env), self.eval(e[3], env), pos)
        if kind in ("and", "or", "not"):
            def truth(x, side):
                v = self.eval(x, env)
                if not isinstance(v, bool):
                    sym = {"and": "&&", "or": "||", "not": "!"}[kind]
                    raise OrbError(f"'{sym}' takes bools, got {type_name(v)} on the {side}", pos)
                return v
            if kind == "not":
                return not truth(e[1], "right")
            a = truth(e[1], "left")
            if a == (kind == "or"):  # short-circuit: the right side is not evaluated
                return a
            return truth(e[2], "right")
        if kind == "tern":
            c = self.eval(e[1], env)
            if not isinstance(c, bool):
                raise OrbError(f"condition must be a bool, got {type_name(c)}", pos)
            return self.eval(e[2] if c else e[3], env)
        if kind == "unit":
            return self.apply_unit(self.eval(e[1], env), e[2], pos)
        if kind == "call":
            f = self.eval(e[1], env)
            if not isinstance(f, (Function, Builtin)) and e[1][0] == "name":
                shown = interpolated(f, pos) if isinstance(f, (int, float, Q, str)) and not isinstance(f, bool) else None
                t = type_name(f)
                art = "" if t.startswith(("block ", "project ")) else "an " if t[0] in "aeiou" else "a "
                raise OrbError(f"{e[1][1]!r} is {art}{t}" + (f" ({shown})" if shown is not None else "")
                               + ", not a function", pos)
            return self.call(f, [(k, self.eval(a, env), p) for k, a, p in e[2]], pos)
        if kind == "for":
            _, a, b, src, body, cond = e[:6]
            seq = self.eval(src, env)
            if isinstance(seq, dict):
                pairs = list(seq.items())
            elif isinstance(seq, list):
                pairs = list(enumerate(seq))
            elif isinstance(seq, Group):
                pairs = [(l, seq.lookup(l, pos)) for l in seq.labels]
            else:
                raise OrbError(f"for ... in needs a list, an object or a group of blocks, got {type_name(seq)}"
                               + (" (a scene's instances: [for id in s.ids : s[id]])" if type_name(seq) == "scene" else ""),
                               pos)
            out = []
            for k, v in pairs:
                scope = Env(env, {a: v} if b is None else {a: k, b: v})
                if cond is not None:
                    c = self.eval(cond, scope)
                    if not isinstance(c, bool):
                        raise OrbError(f"for ... if needs a bool, got {type_name(c)}", pos)
                    if not c:
                        continue
                out.append(self.eval(body, scope))
            return out
        if kind == "get":
            return self.get(self.eval(e[1], env), e[2], pos)
        if kind == "index":
            obj, idx = self.eval(e[1], env), self.eval(e[2], env)
            if isinstance(obj, list):
                if not isinstance(idx, int) or not -len(obj) <= idx < len(obj):
                    if not isinstance(idx, int) or isinstance(idx, bool):
                        shown = f" {interpolated(idx, pos)}" if isinstance(idx, (float, Q)) else ""
                        raise OrbError(f"a list index is a whole number, got {type_name(idx)}{shown}", pos)
                    raise OrbError(f"list index {idx} is out of range (the list has {len(obj)} item"
                                   f"{'s' * (len(obj) != 1)})", pos)
                return obj[idx]
            return self.get(obj, idx, pos)
        raise OrbError(f"cannot evaluate {kind}", pos)

    def apply_unit(self, v, unit, pos):
        f, d = unit
        if isinstance(v, list):
            return [self.apply_unit(x, unit, pos) for x in v]
        if isinstance(v, Q) or not isinstance(v, (int, float)) or isinstance(v, bool):
            raise OrbError(f"cannot attach a unit to {type_name(v)}", pos)
        return _q(v * f, d)

    def outputs(self, path):
        """`outputs` (this project's) or `<alias>.outputs` (an imported one's): the outputs by name, each
        the scene it places, built when first read. `path` names the group in messages."""
        def scene(label, pos):
            if label not in self._output_scenes:
                from .scenes import output_scene

                self._output_scenes[label] = output_scene(self, label, pos)
            return self._output_scenes[label]
        return Group(path, scene, [l for t, l in self.decls if t == "output"])

    def world_scene(self, alias, pos):
        """`<alias>.world`: everything the imported project's own world holds, as one scene."""
        if self._world_scene is None:
            from .scenes import world_scene

            self._world_scene = world_scene(self, alias, pos)
        return self._world_scene

    def get(self, obj, name, pos):
        if isinstance(obj, Namespace):
            if not isinstance(name, str):
                raise OrbError(f"cannot index project {obj.alias} with {type_name(name)}", pos)
            if name == "outputs" and not obj.prog.owns(name):
                return obj.prog.outputs(f"{obj.alias}.outputs")
            if name == "world" and not obj.prog.owns(name):
                return obj.prog.world_scene(obj.alias, pos)
            try:
                return obj.prog.env.lookup(name, pos)
            except OrbError as e:
                if e.msg.startswith(f"unknown name {name!r}"):
                    kinds = sorted({t for t, l in obj.prog.decls if l == name})
                    hint = (f" (did you mean {obj.alias}.{kinds[0]}.{name}?)" if kinds else
                            did_you_mean(name, obj.prog.global_names()))
                    raise OrbError(f"project {obj.alias} has no {name!r}{hint}", pos) from None
                raise
        if hasattr(obj, "orb_get"):
            return obj.orb_get(name, pos)
        if isinstance(obj, Group):
            if name not in obj.labels and obj.stub is not None:
                return obj.stub(name, pos)
            if name not in obj.labels:
                raise OrbError(f"{obj.path} has no {name!r}" + (did_you_mean(name, obj.labels) or
                               f" (has: {', '.join(map(str, obj.labels)) or 'none'})"), pos)
            return obj.lookup(name, pos)
        if isinstance(obj, Block):
            if getattr(obj, "placed", False) and obj.type == "site" and name in ("zone", "fixture"):
                def child(label, p, site=obj, kind=name):
                    c = Block(kind, label, f"{site.path}.{kind}.{label}", p)
                    c.placed = True
                    return c
                return Group(f"{obj.path}.{name}", child, [], child)
            if getattr(obj, "placed", False):
                raise OrbError(f"{obj.path} is placed by an output: shots, tracks, attach and follow can name it, "
                               f"but its {name!r} exists only in the world (read the scene instead: bounds(s, path = "
                               "\"<id>\"), s.<id>)", pos)
            if name in obj.attrs:
                return obj.attrs[name]
            kids = obj.kids(name)
            if kids:
                return Group(f"{obj.path}.{name}", lambda label, p: obj.child(name, label), [k.label for k in kids])
            raise OrbError(f"{obj.path} has no attribute or child {name!r}"
                           + did_you_mean(name, list(obj.attrs) + [c.type for c in obj.children]), pos)
        if isinstance(obj, dict):
            if name not in obj:
                raise OrbError(f"object has no key {name!r}" + (did_you_mean(name, obj) or
                               f" (has: {', '.join(map(str, obj)) or 'none'})"), pos)
            return obj[name]
        raise OrbError(f"cannot read .{name} of {type_name(obj)}", pos)


def load_program(path, prefix="", imported=None, parent=None, exclude=()):
    """Parse a .orb file or a directory of them (and everything they include); nothing evaluated
    yet (`orb pull` evaluates only the `source` blocks). `exclude`: project file names to leave out
    (`orb geo` regenerates geo.orb from the rest of the project)."""
    path = Path(path)
    # project files: top-level *.orb files (not the .orb/ directory `orb pull` keeps sources in)
    files = [f for f in sorted(path.glob("*.orb")) if f.is_file() and not f.name.startswith(".")
             and f.name not in exclude] if path.is_dir() else [path]
    if not files:
        raise OrbError(f"no .orb files in {path}")
    prog = Program(prefix, imported, parent, os.path.realpath(path if path.is_dir() else path.parent))
    prog.ident = os.path.realpath(path)  # what an import names: the project's folder, or one .orb file
    for f in files:
        prog.add_file(f)
    return prog


def load(path, exclude=()):
    """Load a .orb file or a directory of them; return evaluated top-level blocks (a `Blocks`
    list: `.includes`, the top-level `.natives` and the `.imports` records ride along). Every import
    is read and every fn body checked first."""
    prog = load_program(path, exclude=exclude).resolve_imports()
    for p in [prog, *prog.imported.values()]:
        p.check_fns()
    out = prog.evaluate()
    out.program = prog
    return out


def _subexprs(e):
    """The expressions directly inside expression `e` (not its names, units or literal keys)."""
    kind = e[0]
    if kind in ("lit", "name"):
        return []
    if kind == "str":
        return [p for p in e[1] if not isinstance(p, str)]
    if kind == "list":
        return list(e[1])
    if kind == "obj":
        return [k for k, _ in e[1] if not isinstance(k, str)] + [v for _, v in e[1]]
    if kind in ("neg", "not", "unit"):
        return [e[1]]
    if kind == "bin":
        return [e[2], e[3]]
    if kind in ("and", "or"):
        return [e[1], e[2]]
    if kind == "tern":
        return [e[1], e[2], e[3]]
    if kind == "call":
        return [e[1]] + [a for _, a, _ in e[2]]
    if kind == "for":
        return [x for x in (e[3], e[4], e[5]) if x is not None]
    if kind == "get":
        return [e[1]]
    if kind == "index":
        return [e[1], e[2]]
    return []


def _has_names(e):
    """Does expression `e` look anything up (a name, so a call too)?"""
    return e[0] in ("name", "for", "call") or any(_has_names(x) for x in _subexprs(e))
