from __future__ import annotations

import pytest

from orb.lang import load
from orb.motion import Motion
from orb.world import compile_world, frames, mimics, with_mimics

PROJECT = '''
site "cell" { floor = [2, 2] m }
assembly "gripper" {
  fixed_base = true
  link "palm" {
    mass = 1 kg
    inertia = [0.01, 0.01, 0.01, 0, 0, 0] kg*m2
    link "left" {
      at = [0, 0.01, 0] m
      mass = 0.1 kg
      inertia = [0.0001, 0.0001, 0.0001, 0, 0, 0] kg*m2
      joint "leader" { type = "prismatic"  axis = [0, 1, 0]  range = [0, 55] mm  initial = 10 mm }
    }
    link "right" {
      at = [0, -0.01, 0] m
      mass = 0.1 kg
      inertia = [0.0001, 0.0001, 0.0001, 0, 0, 0] kg*m2
      joint "follower" { type = "prismatic"  axis = [0, -1, 0]  range = [0, 55] mm
                           mimic = "leader"  multiplier = 1 }
    }
  }
}
item "g" { of = assembly.gripper  location = site.cell  joints = { leader = 20 mm } }
timeline "grip" {
  duration = 2 s
  joints "g" { key { t = 0 s  leader = 0 mm }  key { t = 2 s  leader = 45 mm } }
}
'''


@pytest.fixture
def ir(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "main.orb").write_text(PROJECT)
    return compile_world(load(project), project)


def test_mimic_initial_and_joint_maps(ir):
    assembly = ir["assemblies"]["gripper"]
    assert mimics(assembly) == {"follower": {"joint": "leader", "multiplier": 1.0, "offset": 0.0}}
    assert with_mimics(assembly, {"leader": 0.04}) == {"leader": 0.04, "follower": pytest.approx(0.04)}
    assert frames(assembly, {"leader": 0.04})["right"][0][1] == pytest.approx(-0.05)
    assert ir["items"]["g"]["joints"] == {"leader": pytest.approx(0.02)}


def test_motion_derives_follower_pose(ir):
    motion = Motion(ir, "grip")
    joints = motion.joints_at("g", 1.0)
    assert joints["leader"] == pytest.approx(0.0225)
    assert joints["follower"] == pytest.approx(0.0225)
