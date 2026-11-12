"""Small local path helpers for generated Orb projects and imported exports."""
from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from pathlib import Path

PROJECTS = "~/.local/share/orb/projects"
_SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")


def slugify(name: str) -> str:
    """Convert a label to a portable project directory name."""
    value = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    value = re.sub(r"[^a-z0-9_-]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-_")[:63].rstrip("-_")
    return value if _SLUG.fullmatch(value) else "project-" + hashlib.sha256(str(name).encode()).hexdigest()[:8]


def projects_dir() -> Path:
    """Directory for projects imported without an explicit destination."""
    return Path(os.environ.get("ORB_PROJECTS", PROJECTS)).expanduser()


def project_dest(name: str) -> Path:
    return projects_dir() / name
