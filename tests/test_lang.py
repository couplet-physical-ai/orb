import pytest

from orb.cli import diff, state_of
from orb.lang import OrbError, Program, Q
from orb.world import compile_world, principal, qrot


def run(src):
    p = Program()
    p.add_source(src, "t.orb")
    return {b.path: b for b in p.evaluate()}


def err(src):
    with pytest.raises(OrbError) as e:
        run(src)
    return e.value.msg


def test_units_normalize_to_si():
    b = run('part "p" { mass = 678 g  len = [1, 2] mm  w = 30.1 rad/s  kd = 2 Nm*s/rad  a = 90 deg }')["part.p"]
    assert b.attrs["mass"] == Q(0.678, (1, 0, 0, 0, 0, 0))
    assert [x.v for x in b.attrs["len"]] == [0.001, 0.002]
    assert b.attrs["kd"].d == (1, 2, -1, 0, 0, -1)
    assert abs(b.attrs["a"].v - 1.5707963) < 1e-6


def test_unit_mismatch_is_an_error():
    assert "cannot + mass and length" in err('part "p" { x = 1 kg + 2 m }')


def test_unit_name_can_still_be_an_attribute():
    b = run('part "p" { x = 1  m = 2  s = "ok" }')["part.p"]
    assert b.attrs == {"x": 1, "m": 2, "s": "ok"}


def test_refs_are_order_independent():
    b = run('part "a" { m = part.b.mass * 2 }\npart "b" { mass = 1 kg }')["part.a"]
    assert b.attrs["m"].v == 2


def test_modules_splice_and_interpolate():
    bs = run('''
module leg {
  param side
  param len = 10 mm
  link "${name}_hip" { at = [0, side * len, 0] }
}
assembly "a" { link "base" { leg "FL" { side = 1 }  leg "FR" { side = -1  len = 5 mm } } }
''')
    base = bs["assembly.a"].children[0]
    assert [c.label for c in base.children] == ["FL_hip", "FR_hip"]
    assert base.children[1].attrs["at"][1].v == pytest.approx(-0.005)


def test_module_rejects_unknown_param():
    assert "no parameter 'x'" in err('module m { param a }\npart "p" { m "i" { x = 1 } }')


def test_errors_carry_positions():
    with pytest.raises(OrbError) as e:
        run('part "p" {\n  mass = 1 kg +\n}')
    assert str(e.value).startswith("t.orb:3:")


def test_cycles_are_reported():
    assert "cycle" in err('part "a" { x = part.b.x }\npart "b" { x = part.a.x }')


def test_bare_numbers_need_units_for_physical_attrs():
    p = Program()
    p.add_source('part "p" { mass = 2 }', "t.orb")
    with pytest.raises(OrbError, match="needs a unit"):
        compile_world(p.evaluate(), ".")


def test_inertia_must_be_physical():
    p = Program()
    p.add_source('part "p" { mass = 1 kg  inertia = [1, 1, 5, 0, 0, 0] kg*m2 }', "t.orb")
    with pytest.raises(OrbError, match="triangle"):
        compile_world(p.evaluate(), ".")


def test_principal_axes_reconstruct_inertia():
    inertia = [0.02448, 0.098077, 0.107, 0.00012166, 0.0014849, -0.0000312]
    d, q = principal(inertia)
    # I = R diag(d) R^T, checked on each basis vector
    for k, e in enumerate(([1, 0, 0], [0, 1, 0], [0, 0, 1])):
        col = [0.0, 0.0, 0.0]
        for i in range(3):
            ri = qrot(q, [1 if j == i else 0 for j in range(3)])
            proj = sum(a * b for a, b in zip(ri, e))
            col = [c + d[i] * proj * r for c, r in zip(col, ri)]
        full = [[inertia[0], inertia[3], inertia[4]], [inertia[3], inertia[1], inertia[5]], [inertia[4], inertia[5], inertia[2]]]
        assert col == pytest.approx([full[r][k] for r in range(3)], abs=1e-9)


LIFECYCLE = '''
lifecycle "l" { stages = ["received", "in_service", "retired"] }
site "s" { zone "z" {} }
part "p" { mass = 1 kg }
item "i" { of = part.p  lifecycle = lifecycle.l  stage = "%s"  location = site.s.zone.z }
'''


def ir_of(src):
    p = Program()
    p.add_source(src, "t.orb")
    return state_of(compile_world(p.evaluate(), "."))


def test_plan_allows_next_stage_and_blocks_skips():
    old = ir_of(LIFECYCLE % "received")
    lines, errors, counts = diff(old, ir_of(LIFECYCLE % "in_service"))
    assert counts["~"] == 1 and not errors
    assert '      stage: "received" -> "in_service"' in lines
    _, errors, _ = diff(old, ir_of(LIFECYCLE % "retired"))
    assert errors and "not allowed" in errors[0]
