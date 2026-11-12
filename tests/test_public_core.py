from __future__ import annotations

import json
from pathlib import Path

from orb.cli import main
from orb.importer import import_export
from orb.lang import load
from orb.seams import run as run_seam
from orb.world import compile_world

ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_examples_compile():
    for example in ("warehouse", "robot-cell", "solar-field"):
        project = ROOT / "examples" / example
        ir = compile_world(load(project), project)
        assert ir["parts"]
        assert ir["sites"]


def test_overview_and_glb_import_roundtrip(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "main.orb").write_text('''
part "crate" { mass = 1 kg  visual "box" { size = [0.4, 0.3, 0.2] m } }
site "room" { floor = [4, 4] m  fixture "bench" { collision "box" { size = [1, 1, 0.8] m } } }
let room = instance(site.room, id = "room")
camera "overview" { view = overview(room, pitch = 45 deg) }
''')
    ir = compile_world(load(project), project)
    assert ir["cameras"]["overview"]["fov"] > 0
    assert ir["cameras"]["overview"]["pos"]
    artifacts = run_seam("glb", ir, tmp_path / "glb", {})
    assert artifacts and artifacts[0].is_file()

    class QuietReporter:
        def stage(self, *args, **kwargs):
            pass

        def progress(self, *args, **kwargs):
            pass

    report = import_export(artifacts[0], "roundtrip", into=tmp_path / "roundtrip", rep=QuietReporter())
    assert report["world_matches"]
    assert (tmp_path / "roundtrip" / "main.orb").is_file()


def test_external_seam_receives_normalized_json(tmp_path):
    project = ROOT / "examples" / "warehouse"
    ir = compile_world(load(project), project)
    assert ir["orb"] == "0.1"
    assert ir["parts"]["tote"]["inertial"]["mass"] == 3
    out = tmp_path / "mujoco"
    artifacts = run_seam("mujoco", ir, out, {})
    assert artifacts and (out / "scene.xml").is_file()


def test_cli_validate_and_ir(capsys):
    project = ROOT / "examples" / "robot-cell"
    assert main(["validate", str(project)]) == 0
    assert "valid" in capsys.readouterr().out.lower()
    assert main(["ir", str(project)]) == 0
    graph = json.loads(capsys.readouterr().out)
    assert graph["orb"] == "0.1" and "cell" in graph["sites"]
