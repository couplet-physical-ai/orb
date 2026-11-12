"""Orientation, vector math, clusters and editing scenes by path (orb/layout.py)."""
import math
import time

import pytest

from orb.lang import OrbError, Program, Q
from orb.scenes import Scene, flatten, heading

LIB = """
part "unit"   { mass = 2000 kg  collision "box" { size = [6, 2.4, 2.6] m  at = [0, 0, 1.3] m } }
part "rock" { mass = 1 kg     collision "box" { size = [8, 8, 5] m  at = [0, 0, 2.5] m } }
part "fan"  { mass = 5 kg     visual "box" { size = [0.5, 0.5, 0.5] m } }
"""


def run(src, name="out"):
    p = Program()
    p.add_source(LIB + src, "t.orb")
    return p.resolve_global(name, None)


def v(x):
    return [v(a) for a in x] if isinstance(x, list) else float(x.v) if isinstance(x, Q) else x


def close(a, b, tol=1e-9):
    a, b = v(a), v(b)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(close(x, y, tol) for x, y in zip(a, b))
    return math.isclose(a, b, abs_tol=tol)


def leaves(s):
    return {lf.path: lf for lf in flatten(s)}


def yaw_deg(lf):
    return math.degrees(heading(lf.quat))


# --------------------------------------------------------------------------- math


@pytest.mark.parametrize("expr, want", [
    ("polar(10 m, 90 deg, center = [1, 2, 0] m)", [1, 12, 0]),
    ("rotate([1, 0, 0] m, yaw = 90 deg, about = [1, 1, 0] m)", [2, 1, 0]),
    ("rotate([2, 0, 0] m, yaw = 180 deg)", [-2, 0, 0]),
    ("yaw_to([0, 0, 0] m, [1, 1, 0] m)", math.pi / 4),
    ("rpy_to([0, 0, 0] m, [1, 0, 1] m)", [0, -math.pi / 4, 0]),
    ("dot([1, 2, 3] m, [4, 5, 6] m)", 32),
    ("cross([1, 0, 0], [0, 1, 0])", [0, 0, 1]),
    ("norm([3, 4, 0] m)", 5),
    ("normalize([0, 2, 0] m)", [0, 1, 0]),
    ("distance([0, 0, 0] m, [3, 4, 0] m)", 5),
    ("lerp([0, 0, 0] m, [10, 20, 0] m, 0.25)", [2.5, 5, 0]),
    ("linspace(0 m, 10 m, 3)", [0, 5, 10]),
    ("bezier([[0, 0, 0] m, [5, 10, 0] m, [10, 0, 0] m], 3)", [[0, 0, 0], [5, 5, 0], [10, 0, 0]]),
    ("path_length([[0, 0, 0] m, [3, 4, 0] m, [3, 10, 0] m])", 11),
    ("clamp(12 m, 0 m, 10 m)", 10),
    ("wrap(270 deg)", -math.pi / 2),
    ("pow(2 m, 2)", 4),
    ("exp(0)", 1),
    ("log(100, 10)", 2),
])
def test_math(expr, want):
    assert close(run(f"let out = {expr}"), want)


def test_math_keeps_units():
    assert run("let out = pow(2 m, 2)").d == (0, 2, 0, 0, 0, 0)
    assert run("let out = norm([3, 4] m)").d == (0, 1, 0, 0, 0, 0)
    assert run("let out = dot([1, 0] m, [2, 0] N)").d == Q(1, (1, 2, -2, 0, 0, 0)).d  # N*m


@pytest.mark.parametrize("expr, msg", [
    ("normalize([0, 0, 0] m)", "zero vector"),
    ("norm([1 m, 1 kg])", "different units"),
    ("polar(10, 0 deg)", "lengths"),
    ("yaw_to([1, 1, 0] m, [1, 1, 5] m)", "no heading"),
    ("pow(2 m, 0.5)", "whole power"),
    ("linspace(0 m, 1 m, 0)", "at least 1"),
    ("clamp(1, 5, 2)", "lo is above hi"),
])
def test_math_errors(expr, msg):
    with pytest.raises(OrbError, match=msg):
        run(f"let out = {expr}")


# --------------------------------------------------------------------------- orientation


def test_instance_face_turns_toward_a_point_with_an_extra_yaw():
    ls = leaves(run('let out = compose(instance(part.unit, id = "a", at = [10, 0, 0] m, face = [0, 0, 0] m),'
                    ' instance(part.unit, id = "b", at = [0, 10, 0] m, face = [0, 0, 0] m, yaw = 90 deg))'))
    assert close(yaw_deg(ls["a"]), 180) and close(yaw_deg(ls["b"]), 0)


def test_face_a_scene_aims_at_where_it_stands():
    ls = leaves(run('let rock = instance(part.rock, at = [0, 20, 0] m)\n'
                    'let out = instance(part.unit, at = [0, 0, 0] m, face = rock)'))
    assert close(yaw_deg(ls["unit"]), 90)


def test_face_with_rpy_is_an_error():
    with pytest.raises(OrbError, match="face turns about z"):
        run("let out = instance(part.unit, face = [1, 0, 0] m, rpy = [0, 0, 0] deg)")


# --------------------------------------------------------------------------- clusters


def test_row_ids_positions_and_centered():
    s = run('let out = row(part.unit, 4, 10 m, centered = true, stage = "installed")')
    ls = leaves(s)
    assert list(ls) == ["unit_0", "unit_1", "unit_2", "unit_3"]
    assert close([ls[k].pos[0] for k in ls], [-15, -5, 5, 15])
    assert ls["unit_2"].attrs == {"stage": "installed"}


def test_grid_is_rows_of_units_with_count_and_stagger():
    s = run("let out = grid(part.unit, 3, 3, pitch = [10, 20] m, count = 7, stagger = 5 m, prefix = \"u\")")
    assert s.orb_get("ids", None) == ["row0", "row1", "row2"]
    ls = leaves(s)
    assert len(ls) == 7 and "row2/u0" in ls and "row2/u1" not in ls
    assert close(ls["row1/u2"].pos, (25, 20, 0)) and close(ls["row2/u0"].pos, (0, 40, 0))


def test_ring_faces_in_out_and_along():
    for face, at0 in (("in", 180), ("out", 0), ("along", 90)):
        ls = leaves(run(f'let out = ring(part.unit, 4, 25 m, center = [100, 50, 0] m, face = "{face}")'))
        assert close(ls["unit_0"].pos, (125, 50, 0)) and close(ls["unit_1"].pos, (100, 75, 0))
        assert close(yaw_deg(ls["unit_0"]), at0), face


def test_ring_around_a_rock_stands_on_the_ground():
    ls = leaves(run('let rock = instance(part.rock, at = [100, 50, 0] m)\n'
                    'let out = ring(part.unit, 6, 25 m, center = rock, face = "in")'))
    assert all(close(lf.pos[2], 0) for lf in ls.values())
    assert all(close(math.dist(lf.pos[:2], (100, 50)), 25) for lf in ls.values())


def test_arc_includes_both_ends():
    ls = leaves(run("let out = arc(part.unit, 3, 10 m, from = 0 deg, to = 90 deg)"))
    assert close(ls["unit_0"].pos, (10, 0, 0)) and close(ls["unit_2"].pos, (0, 10, 0))


def test_along_a_polyline_by_n_and_spacing():
    ls = leaves(run('let out = along(part.unit, [[0, 0, 0] m, [10, 0, 0] m, [10, 10, 0] m], n = 3, face = "along")'))
    assert close(ls["unit_1"].pos, (10, 0, 0)) and close(ls["unit_2"].pos, (10, 10, 0))
    assert close(yaw_deg(ls["unit_0"]), 0) and close(yaw_deg(ls["unit_2"]), 90)
    ls = leaves(run("let out = along(part.unit, [[0, 0, 0] m, [25, 0, 0] m], spacing = 10 m)"))
    assert close([lf.pos[0] for lf in ls.values()], [0, 10, 20])


def test_along_a_bezier_curve():
    s = run("let out = along(part.unit, bezier([[0, 0, 0] m, [50, 50, 0] m, [100, 0, 0] m], 40), n = 10)")
    assert len(s.nodes) == 10


def test_cluster_cycles_definitions_and_takes_a_yaw_per_point():
    ls = leaves(run('let out = cluster([part.unit, part.fan], [[0, 0] m, [5, 0] m, [10, 0] m], yaw = [0 deg, 90 deg, 0 deg])'))
    assert [lf.node.target.label for lf in ls.values()] == ["unit", "fan", "unit"]
    assert list(ls) == ["unit_0", "unit_1", "unit_2"] and close(yaw_deg(ls["unit_1"]), 90)


def test_cluster_of_a_scene_nests():
    s = run('fn unit() -> scene { return compose(instance(part.unit, id = "block"), instance(part.fan, id = "fan", at = [3, 0, 3] m)) }\n'
            'let out = row(unit(), 3, 10 m, prefix = "unit_")')
    assert "unit_2/fan" in leaves(s)


@pytest.mark.parametrize("src, msg", [
    ("row(part.unit, 3, 10 m, at = [1, 0, 0] m)", "move the whole cluster"),
    ("row(part.unit, 3)", "needs 'pitch'"),
    ("ring(part.unit, 3, 5 m, face = \"sideways\")", "\"in\", \"out\""),
    ("row(part.unit, 3, 10 m, face = \"in\")", "here face is \"none\" or a point"),
    ("along(part.unit, [[0, 0, 0] m, [1, 0, 0] m])", "n .* or spacing"),
    ("cluster(part.unit, [[0, 0] m], yaw = [0 deg, 1 deg])", "one per instance"),
    ("row(part.unit, 3, 1 m, prefix = \"a/b\")", "instance id"),
    ("grid(part.unit, 200, 200, pitch = 1 m)", "too many"),
])
def test_cluster_errors(src, msg):
    with pytest.raises(OrbError, match=msg):
        run(f"let out = {src}")


# --------------------------------------------------------------------------- editing by path


GRIDS = """
let a = grid(part.unit, 4, 3, pitch = [10, 20] m)
let site = compose(instance(part.rock, id = "rock", at = [100, 50, 0] m), instance(a, id = "a"),
                   instance(move(a, at = [25, 20.5, 0] m), id = "b"))
"""


def test_move_one_row_by_path_keeps_the_rest():
    s = run(GRIDS + 'let out = move(site, path = "a/row1", at = [0, 100, 0] m)')
    ls, before = leaves(s), leaves(run(GRIDS + "let out = site"))
    assert close(ls["a/row1/unit_0"].pos, (0, 120, 0))
    assert close(ls["a/row0/unit_0"].pos, before["a/row0/unit_0"].pos) and close(ls["b/row1/unit_0"].pos, before["b/row1/unit_0"].pos)


def test_move_turns_each_match_in_place_by_default():
    ls = leaves(run(GRIDS + 'let out = move(site, path = "a/*/unit_1", yaw = 90 deg)'))
    assert close(ls["a/row2/unit_1"].pos, (10, 40, 0)) and close(yaw_deg(ls["a/row2/unit_1"]), 90)
    assert close(yaw_deg(ls["a/row2/unit_0"]), 0)


def test_move_about_a_point_revolves_a_cluster():
    ls = leaves(run(GRIDS + 'let out = move(site, path = "b", yaw = 180 deg, about = site.rock)'))
    assert close(ls["b/row0/unit_0"].pos, (175, 79.5, 0))  # (25, 20.5) turned about (100, 50)


def test_move_about_center_and_glob_list():
    s = run(GRIDS + 'let out = move(site, path = ["a", "b"], yaw = 180 deg, about = "center")')
    ls = leaves(s)
    assert close(ls["a/row0/unit_0"].pos, (30, 40, 0))  # grid a spans x 0..30, y 0..40


def test_move_face_turns_every_unit_toward_the_rock():
    ls = leaves(run(GRIDS + 'let out = move(site, path = "*/*/*", face = site.rock)'))
    for p, lf in ls.items():
        if p != "rock":
            want = math.degrees(math.atan2(50 - lf.pos[1], 100 - lf.pos[0]))
            assert close(yaw_deg(lf), want, 1e-6), p


def test_move_without_path_is_unchanged():
    s = run('let out = move(row(part.unit, 2, 10 m), at = [0, 5, 0] m, yaw = 90 deg)')
    ls = leaves(s)
    assert close(ls["unit_1"].pos, (0, 15, 0))


def test_move_path_errors():
    with pytest.raises(OrbError, match="no instance matches path 'a/row9'.*did you mean"):
        run(GRIDS + 'let out = move(site, path = "a/row9", at = [1, 0, 0] m)')
    with pytest.raises(OrbError, match="has no parameter"):
        run(GRIDS + 'let out = move(site, by = [1, 0, 0] m)')


def test_the_given_scene_never_changes():
    p = Program()
    p.add_source(LIB + GRIDS + 'let moved = move(site, path = "a", at = [0, 500, 0] m)\nlet out = site', "t.orb")
    p.resolve_global("moved", None)
    assert close(leaves(p.resolve_global("out", None))["a/row0/unit_0"].pos, (0, 0, 0))


def test_frame_and_transform_place_relative_to_an_instance():
    out = run('let s = compose(instance(part.rock, id = "rock", at = [10, 0, 0] m, yaw = 90 deg))\n'
              'let f = frame(s, "rock")\nlet out = {f = f, p = transform([5, 0, 0] m, f)}')
    assert close(out["f"]["at"], [10, 0, 0]) and close(out["f"]["yaw"], math.pi / 2)
    assert close(out["p"], [10, 5, 0])


def test_frame_needs_one_match():
    with pytest.raises(OrbError, match="matches 3 instances"):
        run(GRIDS + 'let out = frame(site, "a/*")')


# --------------------------------------------------------------------------- clashes


def test_clashes_by_parts_not_by_cluster_boxes():
    out = run(GRIDS + 'let around = ring(part.unit, 8, 25 m, center = site.rock, face = "in")\n'
              'let s2 = compose(site, instance(around, id = "statue"))\n'
              'let out = {top = clashes(s2), units = clashes(s2, "*/*/*")}')
    assert out["top"] == [["a", "b"]]  # the ring around the rock does not clash with it
    assert ["a/row1/unit_3", "b/row0/unit_0"] in out["units"] or ["a/row1/unit_2", "b/row0/unit_0"] in out["units"]


def test_separate_all_clears_later_clusters():
    out = run(GRIDS + 'let s = separate(site, keep = "rock", gap = 5 m, along = "y")\n'
              'let out = {s = s, c = clashes(s, gap = 5 m)}')
    assert out["c"] == []
    ls = leaves(out["s"])
    assert close(ls["a/row0/unit_0"].pos, (0, 0, 0))  # the first stays
    assert ls["b/row0/unit_0"].pos[1] > 20.5 and ls["b/row0/unit_0"].pos[0] == pytest.approx(25)  # b slid north


def test_separate_pair_and_beside():
    out = run('let a = grid(part.unit, 3, 3, pitch = [10, 20] m)\n'
              'let b = move(a, at = [5, 1, 0] m)\n'
              'let out = {s = separate(b, a, gap = 2 m), bs = beside(b, a, side = "+x", gap = 10 m)}')
    s, bs = out["s"], out["bs"]
    assert min(lf.pos[0] for lf in leaves(bs).values()) == pytest.approx(23 + 10 + 3)  # a ends at x 23
    assert isinstance(s, Scene)
    p = Program()
    p.add_source(LIB + 'let a = grid(part.unit, 3, 3, pitch = [10, 20] m)\nlet b = move(a, at = [5, 1, 0] m)\n'
                 'let out = overlaps(separate(b, a, gap = 2 m), a, gap = 2 m)\nlet was = overlaps(b, a)', "t.orb")
    assert p.resolve_global("was", None) is True and p.resolve_global("out", None) is False


def test_without_overlapping_clears_a_rock():
    s = run('let rock = instance(part.rock, at = [20, 20, 0] m)\n'
            'let out = without(grid(part.unit, 6, 4, pitch = [10, 10] m), "*/*", overlapping = rock, gap = 1 m)')
    ls = leaves(s)
    assert len(ls) == 23 and "row2/unit_2" not in ls and "row2/unit_1" in ls


def test_without_by_path():
    s = run('let out = without(grid(part.unit, 3, 3, pitch = 10 m), "row1")')
    assert s.orb_get("ids", None) == ["row0", "row2"]


def test_overlaps_touching_is_not_a_clash():
    assert run("let out = overlaps(instance(part.unit), instance(part.unit, at = [6, 0, 0] m))") is False
    assert run("let out = overlaps(instance(part.unit), instance(part.unit, at = [6, 0, 0] m), gap = 0.5 m)") is True
    assert run("let out = overlaps(instance(part.unit), [3.2, 0, 0] m, gap = 0.5 m)") is True


# --------------------------------------------------------------------------- into the world


def test_clusters_lower_into_items(tmp_path):
    (tmp_path / "site.orb").write_text(LIB + GRIDS + """
let shown = separate(compose(site, instance(ring(part.unit, 8, 25 m, center = site.rock, face = "in"), id = "statue")),
                     keep = ["rock", "statue"], gap = 5 m)
output "site" { value = shown }
""")
    from orb.lang import load
    from orb.world import compile_world

    ir = compile_world(load(tmp_path), tmp_path)
    labels = list(ir["items"])
    assert "site/statue/unit_7" in labels and "site/b/row2/unit_3" in labels and len(labels) == 1 + 24 + 8


def test_a_thousand_units_is_quick():
    t = time.perf_counter()
    s = run('fn unit() -> scene { return compose(instance(part.unit, id = "block"), instance(part.fan, id = "fan", at = [3, 0, 3] m)) }\n'
            'let field = grid(unit(), 100, 10, pitch = [10, 20] m)\n'
            'let out = separate(compose(instance(field, id = "a"), instance(move(field, at = [500, 100, 0] m), id = "b")), gap = 5 m)')
    assert len(list(flatten(s))) == 4000
    assert time.perf_counter() - t < 10


@pytest.mark.parametrize("src, msg", [
    ("arc(part.unit, 5, 20 m, start = 0 deg, end = 90 deg)", r"arc\(\) has no 'start' \(use from\)"),
    ("ring(part.unit, 5, 20 m, from = 90 deg)", r"ring\(\) has no 'from' \(use start\)"),
    ("along(part.unit, [[0, 0, 0] m, [9, 0, 0] m], n = 3, count = 3)", r"along\(\) has no 'count' \(use n\)"),
    ("row(part.unit, 3, 1 m, spacing = 2 m)", r"row\(\) has no 'spacing' \(use pitch\)"),
    ("ring(part.unit, 5, 20 m, rotation = 30 deg)", r"ring\(\) has no 'rotation'.*every instance"),
    ("row(part.unit, 3, 10 m, face = \"in\")", "for arc\\(\\) and ring\\(\\)"),
    ("cluster(part.unit, [\"a\", \"b\"])", "a list holding string"),
    ("yaw_to([1, 1, 0] m, [1, 1, 0] m)", "the same point"),
])
def test_newcomer_mistakes(src, msg):
    with pytest.raises(OrbError, match=msg):
        run(f"let out = {src}")


def test_cluster_item_attributes_still_pass_through():
    ls = leaves(run('let out = ring(part.unit, 3, 5 m, stage = "planned", serial = "x")'))
    assert ls["unit_0"].attrs == {"stage": "planned", "serial": "x"}


def test_clashes_path_that_matches_nothing_is_an_error():
    with pytest.raises(OrbError, match="no instance matches path 'zzz'"):
        run(GRIDS + 'let out = clashes(site, "zzz")')


MIXED = """
let mixed = compose(instance(grid(part.unit, 3, 2, pitch = [10, 20] m), id = "g"),
                    instance(row(part.unit, 3, 10 m), id = "r", at = [3, 1, 0] m))
"""


def test_trailing_double_star_is_every_part_at_any_depth():
    out = run(MIXED + 'let out = {c = clashes(mixed, "**"), w = without(mixed, "**", overlapping = [12, 1, 0] m, gap = 0.5 m)}')
    assert out["c"] == [["g/row0/unit_0", "r/unit_0"], ["g/row0/unit_1", "r/unit_1"], ["g/row0/unit_2", "r/unit_2"]]
    assert set(leaves(out["w"])) == set(leaves(run(MIXED + "let out = mixed"))) - {"g/row0/unit_1", "r/unit_1"}
    ls = leaves(run(MIXED + 'let out = move(mixed, path = "g/**", yaw = 90 deg)'))
    assert close(ls["g/row1/unit_2"].pos, (20, 20, 0)) and close(yaw_deg(ls["g/row1/unit_2"]), 90)  # each in place
