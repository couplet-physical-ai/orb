"""A small CLI for validating OrbLang projects and applying simulator seams."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import seams
from .lang import OrbError, load
from .world import compile_world

SECTIONS = ("parts", "assemblies", "sites", "items", "lifecycles", "processes", "seams", "timelines", "cameras", "other")


def state_of(ir: dict, imports=()):
    """Return the stable sections of a compiled world graph."""
    state = {key: ir.get(key, {}) for key in SECTIONS}
    if imports:
        state = {"imports": imports, **state}
    return state


def diff(old: dict, new: dict):
    """Compare lifecycle state transitions in two world graphs."""
    lines, errors = [], []
    counts = {"+": 0, "~": 0, "-": 0}
    before = old.get("items", {})
    after = new.get("items", {})
    for name in sorted(set(before) | set(after)):
        if name not in before:
            lines.append(f"  + item.{name}")
            counts["+"] += 1
        elif name not in after:
            lines.append(f"  - item.{name}")
            counts["-"] += 1
        elif before[name] != after[name]:
            lines.append(f"  ~ item.{name}")
            counts["~"] += 1
            old_stage, new_stage = before[name].get("stage"), after[name].get("stage")
            if old_stage != new_stage:
                lines.append(f"      stage: {json.dumps(old_stage)} -> {json.dumps(new_stage)}")
                lifecycle = after[name].get("lifecycle")
                if lifecycle:
                    label = lifecycle.split(".")[-1]
                    transitions = (new.get("lifecycles", {}).get(label) or {}).get("transitions", [])
                    if [old_stage, new_stage] not in transitions:
                        errors.append(f"item.{name}: stage {old_stage!r} -> {new_stage!r} is not allowed by {lifecycle}")
    return lines, errors, counts


def _project(path: str) -> Path:
    project = Path(path).expanduser().resolve()
    if not project.exists():
        raise OrbError(f"project does not exist: {path}")
    if not project.is_dir() and project.suffix != ".orb":
        raise OrbError(f"expected a project directory or .orb file: {path}")
    return project


def _compile(path: str):
    project = _project(path)
    return project, compile_world(load(project), project if project.is_dir() else project.parent)


def _parser():
    parser = argparse.ArgumentParser(prog="orb", description="Validate OrbLang projects and apply simulator seams.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("validate", "compile and validate a project"), ("ir", "print the normalized JSON world graph")):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("project", nargs="?", default=".")
    apply = sub.add_parser("apply", help="write outputs from one or more seams")
    apply.add_argument("project", nargs="?", default=".")
    apply.add_argument("--seam", action="append", dest="selected", help="seam name; may be repeated")
    seams_cmd = sub.add_parser("seams", help="list built-in seams and the external adapter protocol")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        if args.command == "seams":
            for name, (_, description, artifact) in seams.BUILTIN.items():
                print(f"{name}: {description} ({artifact})")
            print("External seams: orb-seam-NAME executables on PATH or a seam block command")
            return 0

        project, ir = _compile(args.project)
        if args.command == "validate":
            counts = ", ".join(f"{len(ir.get(section, {}))} {section}" for section in SECTIONS if ir.get(section))
            print(f"valid OrbLang project ({counts or 'no declarations'})")
        elif args.command == "ir":
            print(json.dumps(ir, indent=2, sort_keys=True))
        elif args.command == "apply":
            names = args.selected or list(ir.get("seams", {})) or list(seams.DEFAULT)
            for name in names:
                out = project / "build" / name
                artifacts = seams.run(name, ir, out, ir.get("seams", {}).get(name, {}))
                print(f"{name}: wrote {len(artifacts)} file(s) under {out}")
        return 0
    except (OrbError, OSError, ValueError) as error:
        print(f"orb: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
