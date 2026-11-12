"""Small adapters from the Orb world graph to simulators and interchange formats.

An external seam executable named ``orb-seam-<name>`` receives normalized JSON on
stdin and ``--out <directory>``. It writes artifacts under that directory and
prints their paths, one per line. A non-zero exit code indicates failure.
"""
from __future__ import annotations

import importlib
import json
import shlex
import shutil
import subprocess
from pathlib import Path

from ..lang import OrbError

# name -> (module, description, primary artifact)
BUILTIN = {
    "mujoco": ("mujoco", "MuJoCo XML scene", "scene.xml"),
    "glb": ("glb", "binary glTF scene", "scene.glb"),
}
DEFAULT = ("mujoco",)


def module(name: str):
    try:
        entry = BUILTIN[name]
    except KeyError as e:
        raise OrbError(f"unknown built-in seam {name!r}") from e
    return importlib.import_module(f".{entry[0]}", __name__)


def fidelity(name: str):
    return module(name).PROCESS_FIDELITY if name in BUILTIN else None


def main_artifact(name: str, out: Path):
    if name not in BUILTIN:
        return None
    return out / BUILTIN[name][2]


def run(name: str, ir: dict, out: Path, cfg: dict):
    if name in BUILTIN:
        return [Path(p) for p in module(name).emit(ir, out, cfg)]
    command = cfg.get("command") or shutil.which(f"orb-seam-{name}")
    if not command:
        raise OrbError(f"seam {name!r} is not built in ({', '.join(BUILTIN)}) and no executable is configured")
    out.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [*shlex.split(command), "--out", str(out)],
        input=json.dumps(ir), text=True, capture_output=True, check=False,
    )
    if result.returncode:
        raise OrbError(f"seam {name} failed: {result.stderr.strip() or result.returncode}")
    return [Path(line) for line in result.stdout.splitlines() if line.strip()]
