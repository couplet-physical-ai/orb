"""OrbLang functions: fn declarations, calls, return values, lexical scope, comprehensions,
builtins and `import ... as` namespaces."""
import math

import pytest

from orb.lang import OrbError, Program, Q, load, load_program


def prog(src, file="t.orb"):
    p = Program()
    p.add_source(src, file)
    return p


def val(src, name="out"):
    return prog(src).resolve_global(name, None)


L = (0, 1, 0, 0, 0, 0)


def test_call_and_return():
    assert val("fn double(x) { return x * 2 }\nlet out = double(21)") == 42


def test_named_args_defaults_and_units():
    src = """
fn box_volume(w: length, d: length, h: length = 1 m) { let a = w * d  return a * h }
let out = box_volume(2 m, d = 300 cm)
"""
    v = val(src)
    assert isinstance(v, Q) and v.v == pytest.approx(6.0) and v.d == (0, 3, 0, 0, 0, 0)


def test_results_feed_other_functions():
    src = """
fn offset(p: point, dx: length) -> point { return [p[0] + dx, p[1], p[2]] }
fn twice(p, dx) { return offset(offset(p, dx), dx) }
let out = twice([1, 0, 0] m, 50 cm)
"""
    assert [x.v for x in val(src)] == pytest.approx([2.0, 0.0, 0.0])


def test_functions_are_values():
    src = """
fn inc(x) { return x + 1 }
fn apply(f: fn, x) { return f(x) }
let out = [apply(inc, 1), call(inc, 5)]
"""
    assert val(src) == [2, 6]


def test_forward_reference_across_files():
    p = Program()
    p.add_source("let out = later(2)", "a.orb")
    p.add_source("fn later(x) { return x * 10 }", "b.orb")
    assert p.resolve_global("out", None) == 20


def test_lexical_scope_not_callers_lets():
    src = """
fn peek() { return secret }
part "p" { let secret = 1  x = peek() }
"""
    with pytest.raises(OrbError, match="unknown name 'secret'"):
        prog(src).evaluate()


def test_globals_visible_inside():
    assert val("let g = 9.81\nfn f() { return g }\nlet out = f()") == 9.81


def test_function_results_go_into_blocks():
    src = """
fn leg_mass(n) { return n * 0.5 kg }
part "robot" { mass = leg_mass(4) }
"""
    (b,) = prog(src).evaluate()
    assert b.attrs["mass"].v == 2.0


@pytest.mark.parametrize("src,msg", [
    ("fn f(a) { return a }\nlet out = f()", "needs argument 'a'"),
    ("fn f(a) { return a }\nlet out = f(1, 2)", "takes 1 argument, got 2"),
    ("fn f(a) { return a }\nlet out = f(b = 1)", "has no parameter 'b'"),
    ("fn f(a) { return a }\nlet out = f(1, a = 2)", "got 'a' twice"),
    ("fn f(a: length) { return a }\nlet out = f(2 kg)", "argument 'a' of fn f must be a length, got mass"),
    ("fn f(a) -> string { return a }\nlet out = f(2)", "result of fn f must be a string"),
    ("fn f(n) { return f(n + 1) }\nlet out = f(0)", "nested deeper than"),
    ("let x = 1\nlet out = x(2)", "'x' is a number \\(1\\), not a function"),
    ("fn f(a) { return a }\nfn f(b) { return b }", "duplicate fn f"),
])
def test_call_errors(src, msg):
    with pytest.raises(OrbError, match=msg):
        p = prog(src)
        p.resolve_global("out", None)


@pytest.mark.parametrize("src,msg", [
    ("fn f(a) { let x = 1 }", "needs `return"),
    ("fn f(a) { return 1  return 2 }", "nothing may follow `return`"),
    ("fn f(a, a) { return a }", "two parameters named 'a'"),
    ("fn f(a = 1, b) { return a }", "needs a default"),
    ("fn f(a: widget) { return a }", "unknown type 'widget'"),
    ("fn f(a) { x = 1  return a }", "holds `let` statements and one `return`"),
    ("fn f(a) { let a = 2  return a }", "shadows a parameter"),
    ("let out = f(a = 1, 2)", "positional arguments go before named ones"),
])
def test_parse_errors(src, msg):
    with pytest.raises(OrbError, match=msg):
        prog(src)


def test_error_points_at_the_call():
    src = "fn f(a) { return a + 1 kg }\nlet out = f(1 m)"
    with pytest.raises(OrbError, match=r"unit mismatch.*\(in fn f, called at t.orb:2:12\)"):
        val(src)


def test_recursion_bounded():
    src = "fn fact(n) { return n <= 1 ? 1 : n * fact(n - 1) }\nlet out = fact(10)"
    assert val(src) == 3628800


def test_unused_call_does_nothing():
    p = prog("fn boom() { return error(\"never\") }\nlet unused = boom()\npart \"p\" { mass = 1 kg }")
    assert [b.label for b in p.evaluate()] == ["p"]


def test_comprehensions():
    assert val("let out = [for i in range(4) : i * i if i != 2]") == [0, 1, 9]
    assert val("let out = [for i, x in [\"a\", \"b\"] : \"${i}${x}\"]") == ["0a", "1b"]
    assert val("let out = [for k, v in { a = 1, b = 2 } : \"${k}=${v}\"]") == ["a=1", "b=2"]
    src = 'site "s" { zone "a" { at = [1, 0, 0] m }  zone "b" {} }\nlet out = [for z in site.s.zone : label(z)]'
    assert val(src) == ["a", "b"]


def test_builtins():
    assert val("let out = len([1, 2, 3])") == 3
    assert val("let out = concat([1], [2, 3])") == [1, 2, 3]
    assert val("let out = merge({ a = 1 }, { a = 2, b = 3 })") == {"a": 2, "b": 3}
    assert val("let out = max([1 m, 3 m, 2 m])").v == 3
    assert val("let out = sqrt(16 m2)") == Q(4.0, L)
    assert val("let out = cos(60 deg)") == pytest.approx(0.5)
    assert val("let out = atan2(1 m, 1 m)").v == pytest.approx(math.pi / 4)
    assert val("let out = round(1.26 m, 10 cm)").v == pytest.approx(1.3)
    assert val("let out = sum([1 m, 50 cm])").v == pytest.approx(1.5)
    assert val("let out = lookup({ a = 1 }, \"b\", 7)") == 7
    assert val("let out = keys({ a = 1, b = 2 })") == ["a", "b"]
    assert val("let out = pi") == pytest.approx(math.pi)
    with pytest.raises(OrbError, match="sqrt.*no square root"):
        val("let out = sqrt(2 m)")
    with pytest.raises(OrbError, match="custom stop"):
        val('let out = 1 > 2 ? 1 : error("custom stop")')


def test_declarations_win_over_builtins():
    # a project that happens to name a let like a builtin keeps its own
    assert val("let len = 5\nlet out = len") == 5


def test_legacy_programs_unchanged():
    src = """
module leg { param side  link "${name}_hip" { at = [0, side * 0.1, 0] m } }
part "p" { mass = 1 kg }
assembly "a" { link "base" { part = part.p  leg "FL" { side = 1 } } }
"""
    blocks = prog(src).evaluate()
    assert [b.label for b in blocks] == ["p", "a"]


def test_import_namespace(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "lib.orb").write_text("""
let gap = 2 m
part "box" { mass = 1 kg  visual "box" { size = [1, 1, 1] m } }
site "room" { floor = [4, 4] m  zone "bench" { at = [1, 1, 0] m } }
fn spaced(n, step: length = gap) { return [for i in range(n) : i * step] }
""")
    main = tmp_path / "main"
    main.mkdir()
    (main / "main.orb").write_text("""
import "../lib" as lib
let steps = lib.spaced(3)
let box = lib.part.box
let bench = lib.site.room.zone.bench
part "box" { mass = 2 kg }
""")
    p = load_program(main)
    assert [x.v for x in p.resolve_global("steps", None)] == [0, 2, 4]  # the lib's own `gap`
    box = p.resolve_global("box", None)
    assert box.label == "lib/box" and box.path == "part.lib/box"
    assert p.resolve_global("bench", None).path == "site.lib/room.zone.bench"
    # importing places nothing: only this project's own blocks are in its world
    assert [b.path for b in load(main)] == ["part.box"]


def test_import_errors(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "a.orb").write_text('import "../b" as b\nlet x = b.y')
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "b.orb").write_text('import "../a" as a\nlet y = a.x')
    with pytest.raises(OrbError, match="import cycle"):
        load_program(tmp_path / "a").resolve_global("x", None)
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "c.orb").write_text('import "../nope" as n\nlet x = n.y')
    with pytest.raises(OrbError, match="no such file"):
        load_program(tmp_path / "c").resolve_global("x", None)
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "d.orb").write_text('import "../b" as b\nlet x = b.missing')
    with pytest.raises(OrbError, match="project b has no 'missing'"):
        load_program(tmp_path / "d").resolve_global("x", None)


def test_nested_import_prefixes(tmp_path):
    for name, src in {"c": 'part "bolt" { mass = 1 g }',
                      "b": 'import "../c" as hw\nlet bolt = hw.part.bolt',
                      "a": 'import "../b" as kit\nlet bolt = kit.bolt'}.items():
        (tmp_path / name).mkdir()
        (tmp_path / name / f"{name}.orb").write_text(src)
    assert load_program(tmp_path / "a").resolve_global("bolt", None).label == "kit/hw/bolt"


@pytest.mark.parametrize("expr,want", [
    ("7 % 3", 1), ("-7 % 3", 2), ("7 % -3", -2), ("6 % 3 == 0", True),
    ("true || false && false", True), ("!(1 > 2) && 2 > 1", True), ("!true || !false", True),
    ("false && (1 / 0 > 1)", False), ("true || 1 / 0 > 1", True),  # short-circuit
    ("1 + 2 * 3 % 4", 3), ("[for i in range(8) : i if i % 3 == 0 && i > 0]", [3, 6]),
])
def test_logic_and_remainder(expr, want):
    assert val(f"let out = {expr}") == want


def test_remainder_keeps_units():
    v = val("let out = 2.5 m % 1 m")
    assert v.v == pytest.approx(0.5) and v.d == (0, 1, 0, 0, 0, 0)


@pytest.mark.parametrize("src,msg", [
    ("let out = 1 && true", "'&&' takes bools, got number on the left"),
    ("let out = !3", "'!' takes bools, got number"),
    ("let out = 3 m % 2 kg", "unit mismatch"),
    ("let out = 3 % 0", "modulo by zero"),
    ("let out = compse(1)", "unknown name 'compse' \\(did you mean 'compose'\\?\\)"),
    ("let out = [1, 2][5]", "list index 5 is out of range \\(the list has 2 items\\)"),
    ("let out = { alpha = 1 }.alpah", "did you mean 'alpha'"),
    ("fn f(n: lenght) -> number { return 1 }\nlet out = f(1 m)", "did you mean 'length'"),
    ("fn f(count: number) -> number { return count }\nlet out = f(cuont = 2)", "did you mean 'count'"),
    ("fn f(count: number) -> number { return cuont }\nlet out = f(2)", "unknown name 'cuont' \\(did you mean 'count'"),
    ("let out = [for i in range(3) if i > 0 : i]", "filter goes last"),
    ("let out = range(3, stepp = 1)", "range\\(\\) has no parameter 'stepp'"),
])
def test_diagnostics_suggest_fixes(src, msg):
    with pytest.raises(OrbError, match=msg):
        val(src)
