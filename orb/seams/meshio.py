"""Small, dependency-free Wavefront OBJ/MTL reader shared by the seams.

Handles what phone scans produce: `v x y z [r g b]` (vertex colors, 0-1 or 0-255), `vt`,
`f v/vt/vn` polygons (triangulated as fans), `mtllib` / `usemtl` with `map_Kd` textures.
Indices stay as written in the OBJ: positions and UVs are indexed separately, so UV seams
survive (nothing is welded here).

`load_obj(path)` is `Obj(path)` remembered for the rest of the process (until the OBJ, its
.mtl or a texture changes on disk): `orb apply --seam all` and `orb export --to all` run
several seams over the same meshes, and each would otherwise parse every OBJ again.
"""
from __future__ import annotations

import itertools
import math
import os
from collections import OrderedDict
from pathlib import Path

from ..lang import OrbError

# --------------------------------------------------------------------------- memo

_MEMO: OrderedDict = OrderedDict()  # key -> (dependency stamps, value, weight)
MEMO_BUDGET = 4_000_000  # remembered vertices + triangles (a 300k-face scan and its ceiling are ~1M)


def _stamp(path):
    try:
        st = os.stat(path)
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


def memo(key, deps, make, weight=lambda value: 1):
    """make() once per key while none of `deps` (paths) changes on disk (mtime, size, existence).
    make() returns (value, extra dependency paths it read). Least recently used entries go first
    once the remembered weight passes MEMO_BUDGET."""
    hit = _MEMO.get(key)
    if hit is not None and all(_stamp(p) == s for p, s in hit[0]):
        _MEMO.move_to_end(key)
        return hit[1]
    before = [(p, _stamp(p)) for p in deps]
    value, more = make()
    stamps = before + [(p, _stamp(p)) for p in more]
    w = weight(value)
    _MEMO[key] = (stamps, value, w)
    total = sum(e[2] for e in _MEMO.values())
    while total > MEMO_BUDGET and len(_MEMO) > 1:
        _, old = _MEMO.popitem(last=False)
        total -= old[2]
    return value


def clear_memo():
    _MEMO.clear()


def load_obj(path: Path):
    """Obj(path), parsed once per process while the OBJ, its .mtl files and textures are unchanged.
    The result is shared: treat it as read-only."""
    p = Path(path).resolve()

    def make():
        o = Obj(p)
        return o, o.deps

    return memo(("obj", str(p)), [p], make, weight=lambda o: len(o.pos) + o.ntris)


def parse_mtl(path: Path, deps=None):
    """-> {material: {"Kd": [r, g, b] | None, "map_Kd": Path | None}}. `deps` (a list) gets every
    texture path the materials name, found or not."""
    mats, cur = {}, None
    if not path.exists():
        return mats
    for line in path.read_text(errors="replace").splitlines():
        p = line.split()
        if not p or p[0].startswith("#"):
            continue
        if p[0] == "newmtl":
            cur = mats.setdefault(" ".join(p[1:]), {"Kd": None, "map_Kd": None})
        elif cur is None:
            continue
        elif p[0] == "Kd" and len(p) >= 4:
            cur["Kd"] = [float(x) for x in p[1:4]]
        elif p[0] == "map_Kd" and len(p) >= 2:
            # options (-s 1 1 1, -bm 2 ...) come first; the file name is last
            tex = (path.parent / p[-1].replace("\\", "/")).resolve()
            if deps is not None:
                deps.append(tex)
            cur["map_Kd"] = tex if tex.exists() else None
    return mats


class Obj:
    """pos: [(x, y, z)], col: [(r, g, b)] or None, uv: [(u, v)],
    tri_v / tri_t: flat corner indices (3 per triangle; tri_t is None without UVs),
    tri_m: material slot per triangle (None if the OBJ uses at most one material),
    tri_g: `o`/`g` group per triangle (index into `objects`),
    materials: slot -> name, mtl: parsed .mtl entries, deps: the .mtl and texture paths read.
    A line it cannot read is an OrbError naming the file and the line."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.deps = []
        pos, col, uv, tv, tt, tm, tg = [], [], [], [], [], [], []
        nrm, tn, has_n, nnrm = [], [], False, 0
        mtl, materials, slot = {}, [], {}
        objects, oslot, gcur = [None], {None: 0}, 0
        cur, has_t = 0, False
        npos, nuv = 0, 0
        for n, line in enumerate(self.path.read_text(errors="replace").splitlines(), 1):
            if len(line) < 2:
                continue
            c0, c1 = line[0], line[1]
            try:
                if c0 == "v":
                    if c1 == " " or c1 == "\t":
                        p = line.split()
                        pos.append((float(p[1]), float(p[2]), float(p[3])))
                        if len(p) >= 7:
                            col.append((float(p[4]), float(p[5]), float(p[6])))
                        npos += 1
                    elif c1 == "t":
                        p = line.split()
                        uv.append((float(p[1]), float(p[2]) if len(p) > 2 else 0.0))
                        nuv += 1
                    elif c1 == "n":
                        p = line.split()
                        nrm.append((float(p[1]), float(p[2]), float(p[3])))
                        nnrm += 1
                elif c0 == "f" and (c1 == " " or c1 == "\t"):
                    vs, ts, ns = [], [], []
                    for tok in line.split()[1:]:
                        s = tok.split("/")
                        k = int(s[0])
                        vs.append(k - 1 if k > 0 else npos + k)
                        if len(s) > 1 and s[1]:
                            k = int(s[1])
                            ts.append(k - 1 if k > 0 else nuv + k)
                            has_t = True
                        else:
                            ts.append(-1)
                        if len(s) > 2 and s[2]:
                            k = int(s[2])
                            ns.append(k - 1 if k > 0 else nnrm + k)
                            has_n = True
                        else:
                            ns.append(-1)
                    for i in range(1, len(vs) - 1):
                        tv += (vs[0], vs[i], vs[i + 1])
                        tt += (ts[0], ts[i], ts[i + 1])
                        tn += (ns[0], ns[i], ns[i + 1])
                        tm.append(cur)
                        tg.append(gcur)
                elif (c0 == "o" or c0 == "g") and (c1 == " " or c1 == "\t"):
                    name = line[2:].strip() or None
                    if name not in oslot:
                        oslot[name] = len(objects)
                        objects.append(name)
                    gcur = oslot[name]
                elif line.startswith("usemtl"):
                    name = line[6:].strip()
                    if name not in slot:
                        slot[name] = len(materials)
                        materials.append(name)
                    cur = slot[name]
                elif line.startswith("mtllib"):
                    for name in line[6:].split():
                        lib = (self.path.parent / name).resolve()
                        self.deps.append(lib)
                        mtl.update(parse_mtl(lib, self.deps))
            except (ValueError, IndexError):  # a short `v` line, a face index that is not a number, ...
                raise OrbError(f"{self.path}: line {n} is not valid OBJ: {line.strip()[:80]!r}") from None
        if not math.isfinite(sum(itertools.chain.from_iterable(pos))):  # any NaN/inf makes the sum one
            bad = next((i for i, v in enumerate(pos) if not (math.isfinite(v[0]) and math.isfinite(v[1])
                                                              and math.isfinite(v[2]))), None)
            if bad is not None:  # NaN/inf breaks every seam downstream (MuJoCo refuses, glTF is invalid)
                raise OrbError(f"{self.path}: vertex {bad + 1} is not finite ({' '.join(map(str, pos[bad]))}); "
                               "re-capture or repair the mesh")
        self.pos, self.uv, self.mtl = pos, uv, mtl
        self.col = col if col and len(col) == len(pos) else None
        if self.col and max(map(max, self.col)) > 1.0:  # 0-255 writers
            self.col = [(r / 255, g / 255, b / 255) for r, g, b in self.col]
        self.tri_v = tv
        self.tri_t = tt if has_t and uv else None
        # authored normals (`vn`, `f v/vt/vn`): kept only when every corner has one
        ok_n = has_n and nrm and all(0 <= k < len(nrm) for k in tn)
        self.nrm, self.tri_n = (nrm, tn) if ok_n else (None, None)
        self.materials = materials or [None]
        self.tri_m = tm if len(set(tm)) > 1 else None
        self.objects, self.tri_g, self._tm = objects, tg, tm

    def parts(self):
        """The OBJ's faces split by object/group (`o`, `g`) and material (`usemtl`), in file
        order: [{"object": name | None, "slot": material slot, "tris": [triangle index, ...]}].
        Loaders that keep one shape per file (MuJoCo keeps only the first `o`) need one file each."""
        out, at = [], {}
        for t, key in enumerate(zip(self.tri_g, self._tm)):
            if key not in at:
                at[key] = len(out)
                out.append({"object": self.objects[key[0]], "slot": key[1], "tris": []})
            out[at[key]]["tris"].append(t)
        return out

    def write_part(self, tris, path: Path, with_uv=True, min_vertices=0):
        """A self-contained OBJ (no mtllib) of these triangles: only the vertices and UVs they use,
        re-indexed; UVs only when every corner has one. Unused vertices a hair apart pad it to
        min_vertices (MuJoCo refuses a mesh with fewer than 4). Returns the triangle count."""
        corners = [i for t in tris for i in (3 * t, 3 * t + 1, 3 * t + 2)]
        cv = [self.tri_v[i] for i in corners]
        uv = with_uv and self.tri_t is not None and all(self.tri_t[i] >= 0 for i in corners)
        vs = list(dict.fromkeys(cv))  # each vertex once, in the order the faces first use it
        vmap = {v: k for k, v in enumerate(vs, 1)}
        fv = list(map(vmap.__getitem__, cv))
        ts = []
        if uv:
            ct = [self.tri_t[i] for i in corners]
            ts = list(dict.fromkeys(ct))
            tmap = {w: k for k, w in enumerate(ts, 1)}
            vt = [x for pair in zip(fv, map(tmap.__getitem__, ct)) for x in pair]
            fs = [("f %d/%d %d/%d %d/%d\n" * len(tris)) % tuple(vt)]
        else:
            fs = [("f %d %d %d\n" * len(tris)) % tuple(fv)]
        lines = [f"# orb: part of {self.path.name}\n"]
        lines += ["v %.9g %.9g %.9g\n" % self.pos[v] for v in vs]
        x, y, z = self.pos[vs[0]] if vs else (0.0, 0.0, 0.0)
        for k in range(len(vs), min_vertices):  # never referenced by a face
            d = 1e-5 * (k + 1)
            lines.append("v %.9g %.9g %.9g\n" % (x + d * (k % 3 == 0), y + d * (k % 3 == 1), z + d * (k % 3 == 2)))
        lines += ["vt %.9g %.9g\n" % self.uv[w] for w in ts]
        Path(path).write_text("".join(lines + fs))
        return len(tris)

    @property
    def ntris(self):
        return len(self.tri_v) // 3

    def texture(self, slot):
        name = self.materials[slot]
        return (self.mtl.get(name) or {}).get("map_Kd") if name is not None else None

    def kd(self, slot):
        name = self.materials[slot]
        return (self.mtl.get(name) or {}).get("Kd") if name is not None else None

    @property
    def textured(self):
        return self.tri_t is not None and any(self.texture(i) for i in range(len(self.materials)))

    @property
    def kind(self):
        return "textured" if self.textured else "vcolor" if self.col else "plain"


def _header(path: Path):
    """(has vt, has vertex colors, mtl paths) from the lines before the first face (remembered
    while the file is unchanged)."""
    p = Path(path).resolve()
    return memo(("header", str(p)), [p], lambda: (_read_header(p), ()))


def _read_header(path: Path):
    has_vt = has_col = False
    mtllibs = []
    with open(path, errors="replace") as fh:
        for line in fh:
            if line.startswith("vt"):
                has_vt = True
            elif line.startswith("v ") and not has_col and len(line.split()) >= 7:
                has_col = True
            elif line.startswith("mtllib"):
                mtllibs += [(Path(path).parent / n).resolve() for n in line[6:].split()]
            elif line.startswith("f ") and (has_vt or has_col or mtllibs):
                break
    return has_vt, has_col, mtllibs


def obj_stats(path: Path):
    """(object/material groups, vertices) of an OBJ without a full parse: how many groups
    (`o`/`g` x `usemtl`) its faces fall into (1 means a loader that keeps one shape sees every
    face) and how many `v` lines it has. Remembered while the file is unchanged."""
    p = Path(path).resolve()
    return memo(("stats", str(p)), [p], lambda: (_read_stats(p), ()))


def _read_stats(path: Path):
    keys, obj, mat, nv = set(), None, None, 0
    with open(path, errors="replace") as fh:
        for line in fh:
            c = line[:2]
            if c == "f " or c == "f\t":
                keys.add((obj, mat))
            elif c == "v " or c == "v\t":
                nv += 1
            elif c in ("o ", "g ", "o\t", "g\t"):
                obj = line[2:].strip() or None
            elif line.startswith("usemtl"):
                mat = line[6:].strip()
    return len(keys), nv


def part_count(path: Path):
    """How many object/material groups an OBJ's faces fall into (see obj_stats)."""
    return obj_stats(path)[0]


def first_texture(path: Path):
    """The first existing map_Kd texture of an OBJ's materials (with UVs), or None."""
    has_vt, _, mtllibs = _header(path)
    if not has_vt:
        return None
    for lib in mtllibs:
        for m in parse_mtl(lib).values():
            if m["map_Kd"]:
                return m["map_Kd"]
    return None


def first_kd(path: Path):
    """The first Kd colour of an OBJ's materials, or None (a single-material OBJ's colour, for
    a seam without vertex colours: `orb geo` trees and facades)."""
    for lib in _header(path)[2]:
        for m in parse_mtl(lib).values():
            if m["Kd"]:
                return m["Kd"]
    return None


def peek_kind(path: Path):
    """Cheap check without a full parse: 'textured' | 'vcolor' | 'plain'."""
    if first_texture(path) is not None:
        return "textured"
    return "vcolor" if _header(path)[1] else "plain"
