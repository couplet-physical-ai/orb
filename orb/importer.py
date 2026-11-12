"""`orb import`: the reverse door. An Orb export (GLB, USD/USDZ) back into an OrbLang project.

Exports made by the Orb carry their provenance: `asset.extras.orb` (glTF) or the layer's
`customLayerData["orb"]` (USD) holds the OrbLang sources and the world graph, and every mesh
says which project file it was (`extras.orb.file`, `orb:file`). Import writes the sources back
verbatim (or, if an export only kept the world graph, the canonical OrbLang from
`orb.decompile`), rebuilds each mesh file from the export's geometry (positions, UVs, vertex
colours, textures, and the OBJ's `o` groups, which the seams keep per primitive / GeomSubset),
then compiles the result and checks it against the export's world graph. Anything without that
provenance is a foreign scan and goes to `orb capture`; an Orb export that left its sources and
world graph out (`embed_sources = false`) is refused, since capturing it would turn a twin into a
"scan" of itself.
"""
from __future__ import annotations

import contextvars
import json
import re
import struct
from pathlib import Path

from .lang import OrbError, load
from .world import compile_world

OBJECT_FAMILY = "object"  # GeomSubset family the isaac/usd seams write for an OBJ's `o` groups
GLTF_EXT = (".glb", ".gltf")
USD_EXT = (".usd", ".usda", ".usdc", ".usdz")
COMP = {5120: "i1", 5121: "u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


class NotOrb(Exception):
    """The file has no Orb provenance: a foreign scan."""


DEPS = "deps"  # where an imported project's files go: ../robot-cell/meshes/x.obj -> deps/robot-cell/meshes/x.obj


_DEP_ROOTS = contextvars.ContextVar("dep_roots", default={})  # the export's `deps`: climb -> project name


def _safe_rel(fn):
    """A project-relative path from an export (a mesh, meshes/<x>/LICENSE): never absolute, never
    `..` inside it. A file of a project the exported one imported (`../robot-cell/meshes/...`) is
    packaged under deps/ (`deps/robot-cell/meshes/...`), so the import needs none of the original folders.
    The export says where each imported project starts (`deps`: `../../lib/robot-cell` is robot-cell),
    so a long climb still lands in deps/<project>/."""
    m = UNPACKED.match(str(fn))
    if m:  # a file of an export the project imported (unpacked in .orb/imports/<name>-<sha12>/): deps/<name>/
        fn = f"{DEPS}/{m.group(1)}/{m.group(2)}"
    for key, name in _DEP_ROOTS.get().items():
        if str(fn).startswith(key + "/") and name and "/" not in name and name not in (".", ".."):
            fn = f"{DEPS}/{name}/{str(fn)[len(key) + 1:]}"
            break
    p = Path(str(fn))
    lead = 0
    while lead < len(p.parts) and p.parts[lead] == "..":
        lead += 1
    if lead and lead < len(p.parts):  # (an export without `deps`): ../x and ../../x are different folders
        p = Path(DEPS, *(["up"] * (lead - 1)), *p.parts[lead:])
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise OrbError(f"refusing to write {fn!r} outside the project")
    return p


UNPACKED = re.compile(r"^\.orb/imports/([^/]+?)(?:-[0-9a-f]{12})?/(.+)$")  # orb/lang.py unpack_export
IMPORT_RE = re.compile(r'^\s*import\s+"', re.M)


def packaged(world):
    """The world graph with every file outside the project (an imported project's meshes) moved to
    where _safe_rel writes it. -> (world, moved?)."""
    moved = False

    def walk(x):
        nonlocal moved
        if isinstance(x, dict):
            out = {}
            for k, v in x.items():
                if k == "file" and isinstance(v, str) and (v.startswith("../") or UNPACKED.match(v)):
                    out[k] = _safe_rel(v).as_posix()
                    moved = True
                else:
                    out[k] = walk(v)
            return out
        if isinstance(x, list):
            return [walk(v) for v in x]
        return x
    return walk(world), moved


# --------------------------------------------------------------------------- glTF


def read_gltf(path: Path):
    """-> (json doc, binary chunk bytes or None). A file that is not glTF (truncated, another
    format with a .glb name) is an OrbError that says so."""
    data = path.read_bytes()
    try:
        return _read_gltf(path, data)
    except (struct.error, UnicodeDecodeError, json.JSONDecodeError) as e:
        what = "binary glTF (.glb)" if data[:4] == b"glTF" else "glTF"
        raise OrbError(f"{path.name} is not a readable {what} file ({e}); is the download or copy complete?") \
            from None


def _read_gltf(path: Path, data: bytes):
    if data[:4] == b"glTF":
        _, _, total = struct.unpack("<4sII", data[:12])
        off, doc, binc = 12, None, None
        while off + 8 <= min(total, len(data)):
            n, kind = struct.unpack("<I4s", data[off:off + 8])
            chunk = data[off + 8:off + 8 + n]
            if kind == b"JSON":
                doc = json.loads(chunk.decode("utf-8"))
            elif kind == b"BIN\0":
                binc = chunk
            off += 8 + n
        if not isinstance(doc, dict):
            raise OrbError(f"{path.name} is not a readable binary glTF (.glb) file (no JSON chunk); "
                           "is the download or copy complete?")
        return doc, binc
    doc = json.loads(data.decode("utf-8"))
    if not isinstance(doc, dict):
        raise json.JSONDecodeError("not a glTF JSON object", "", 0)
    binc = None
    bufs = doc.get("buffers") or []
    if bufs and bufs[0].get("uri"):
        uri = bufs[0]["uri"]
        if uri.startswith("data:"):
            import base64

            binc = base64.b64decode(uri.split(",", 1)[1])
        else:
            binc = (path.parent / uri).read_bytes()
    return doc, binc


def accessor(np, doc, binc, i):
    a = doc["accessors"][i]
    bv = doc["bufferViews"][a["bufferView"]]
    dt = np.dtype(COMP[a["componentType"]])
    nc = NCOMP[a["type"]]
    start = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    stride = bv.get("byteStride") or dt.itemsize * nc
    raw = np.frombuffer(binc, dtype=np.uint8, count=stride * (a["count"] - 1) + dt.itemsize * nc, offset=start)
    if stride == dt.itemsize * nc:
        arr = raw.view(dt).reshape(a["count"], nc)
    else:
        arr = np.lib.stride_tricks.as_strided(raw, (a["count"], dt.itemsize * nc), (stride, 1)).copy().view(dt)
        arr = arr.reshape(a["count"], nc)
    if a.get("normalized") and dt.kind in "ui":
        arr = arr.astype(np.float64) / np.iinfo(dt).max
    return arr if nc > 1 else arr.reshape(-1)


def gltf_meta(doc):
    meta = ((doc.get("asset") or {}).get("extras") or {}).get("orb")
    if not isinstance(meta, dict):
        if any(isinstance(((n.get("extras") or {}).get("orb")), dict) and
               {"fixture", "item", "site", "link"} & set(n["extras"]["orb"]) for n in doc.get("nodes") or []):
            return {"generator": "orb (asset.extras.orb removed)"}  # node provenance survived a re-save
        raise NotOrb("no asset.extras.orb")
    return meta


def is_orb_generator(meta):
    return str((meta or {}).get("generator") or "").startswith("orb ") or bool((meta or {}).get("ir"))


def linear_to_srgb(np, c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def gltf_meshes(path: Path, doc, binc, dest: Path, rep, shown: Path | None = None):
    """Every glTF mesh tagged extras.orb.file -> that OBJ (+ MTL + textures) under dest (progress
    names it under `shown`, where it will end up)."""
    import numpy as np

    written = []
    for mesh in doc.get("meshes", []):
        f = ((mesh.get("extras") or {}).get("orb") or {}).get("file")
        if not f or (dest / _safe_rel(f)) in written:
            continue
        dst = dest / _safe_rel(f)
        dst.parent.mkdir(parents=True, exist_ok=True)
        prims = mesh["primitives"]
        at = prims[0]["attributes"]
        P = accessor(np, doc, binc, at["POSITION"]).astype(np.float64)
        T = accessor(np, doc, binc, at["TEXCOORD_0"]).astype(np.float64) if "TEXCOORD_0" in at else None
        C = None
        if "COLOR_0" in at:
            C = accessor(np, doc, binc, at["COLOR_0"]).astype(np.float64)[:, :3]
            C = linear_to_srgb(np, C)
        lines = [f"# orb import: {f} from {path.name}\n"]
        mtl_lines, textures, mnames = [], {}, {}
        if T is not None:
            lines.append(f"mtllib {dst.stem}.mtl\n")
        if C is not None:
            lines.append(_rows("v %.7g %.7g %.7g %.4g %.4g %.4g\n", np.hstack([P, C]).reshape(-1).tolist(), len(P)))
        else:
            lines.append(_rows("v %.7g %.7g %.7g\n", P.reshape(-1).tolist(), len(P)))
        if T is not None:
            lines.append(_rows("vt %.7g %.7g\n", np.stack([T[:, 0], 1.0 - T[:, 1]], 1).reshape(-1).tolist(), len(T)))
        cur_obj = cur_mtl = None
        for k, pr in enumerate(prims):
            idx = accessor(np, doc, binc, pr["indices"]).astype(np.int64).reshape(-1, 3) + 1
            obj_name = ((pr.get("extras") or {}).get("orb") or {}).get("object")
            if obj_name and obj_name != cur_obj:  # the OBJ's `o` group (MuJoCo splits a scan by it)
                lines.append(f"o {obj_name}\n")
                cur_obj = obj_name
            if T is not None:
                mi = pr.get("material")
                if mi not in mnames:  # one MTL entry per glTF material, however many primitives use it
                    mnames[mi] = mname = f"m{len(mnames)}"
                    mat = doc["materials"][mi] if mi is not None else {}
                    tex = ((mat.get("pbrMetallicRoughness") or {}).get("baseColorTexture") or {}).get("index")
                    mtl_lines += ["", f"newmtl {mname}", "Kd 1 1 1", "Ks 0 0 0", "d 1", "illum 1"]
                    if tex is not None:
                        img = doc["images"][doc["textures"][tex]["source"]]
                        name = Path(img.get("name") or "").name or f"{dst.stem}_tex{len(mnames) - 1}.png"
                        if name not in textures:
                            bv = doc["bufferViews"][img["bufferView"]]
                            o = bv.get("byteOffset", 0)
                            (dst.parent / name).write_bytes(binc[o:o + bv["byteLength"]])
                            textures[name] = dst.parent / name
                        mtl_lines.append(f"map_Kd {name}")
                if mnames[mi] != cur_mtl:
                    lines.append(f"usemtl {mnames[mi]}\n")
                    cur_mtl = mnames[mi]
                lines.append(_rows("f %d/%d %d/%d %d/%d\n", np.repeat(idx, 2, axis=1).reshape(-1).tolist(), len(idx)))
            else:
                lines.append(_rows("f %d %d %d\n", idx.reshape(-1).tolist(), len(idx)))
        dst.write_text("".join(lines))
        written.append(dst)
        if mtl_lines:
            (dst.parent / f"{dst.stem}.mtl").write_text("# orb import\n" + "\n".join(mtl_lines) + "\n")
            written.append(dst.parent / f"{dst.stem}.mtl")
        written += list(textures.values())
        rep.progress("meshes", f"  {_safe_rel(f).as_posix()}: {len(P):,} vertices", files=[(shown or dest) / _safe_rel(f)])
    return written


# --------------------------------------------------------------------------- USD


def usd_meta(stage):
    data = stage.GetRootLayer().customLayerData or {}
    meta = data.get("orb")
    if not meta:
        for prim in stage.Traverse():  # prim provenance survived (customData orb, orb:item)
            od = prim.GetCustomData().get("orb")
            if (isinstance(od, dict) and {"fixture", "item", "site"} & set(od)) or prim.HasAttribute("orb:item"):
                return {"generator": "orb (customLayerData orb removed)"}
        raise NotOrb("no customLayerData orb")
    return dict(meta)


def usd_meshes(path: Path, stage, dest: Path, rep, shown: Path | None = None):
    from pxr import Ar, UsdGeom, UsdShade

    written = []
    for prim in stage.Traverse():
        attr = prim.GetAttribute("orb:file")
        if not attr or not attr.HasValue() or not prim.IsA(UsdGeom.Mesh):  # texture shaders carry orb:file too
            continue
        f = attr.Get()
        dst = dest / _safe_rel(f)
        if dst in written:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        mesh = UsdGeom.Mesh(prim)
        P = mesh.GetPointsAttr().Get()
        counts = list(mesh.GetFaceVertexCountsAttr().Get())
        fvi = list(mesh.GetFaceVertexIndicesAttr().Get())
        pv = UsdGeom.PrimvarsAPI(mesh)
        st = pv.GetPrimvar("st")
        has_st = st and st.HasValue()
        dc = pv.GetPrimvar("displayColor")
        C = dc.Get() if dc and dc.HasValue() and dc.GetInterpolation() == "vertex" else None
        lines = [f"# orb import: {f} from {path.name}\n"]
        mtl, textures = [], {}
        subsets = UsdGeom.Subset.GetGeomSubsets(mesh, "face", "materialBind")
        if has_st:
            lines.append(f"mtllib {dst.stem}.mtl\n")
        if C is not None and len(C) == len(P):
            lines.append(_rows("v %.7g %.7g %.7g %.4g %.4g %.4g\n", [x for p, c in zip(_flat(P, 3), _flat(C, 3))
                                                                     for x in (*p, *c)], len(P)))
        else:
            lines.append(_rows("v %.7g %.7g %.7g\n", _flat(P), len(P)))
        sti = None
        if has_st:
            vals = st.Get()
            sti = list(st.GetIndices()) if st.IsIndexed() else list(range(len(vals)))
            lines.append(_rows("vt %.7g %.7g\n", _flat(vals), len(vals)))
        starts, k = [], 0
        for n in counts:
            starts.append(k)
            k += n

        def face(fi):
            s, n = starts[fi], counts[fi]
            if sti is not None:
                return "f " + " ".join(f"{fvi[s + j] + 1}/{sti[s + j] + 1}" for j in range(n)) + "\n"
            return "f " + " ".join(str(fvi[s + j] + 1) for j in range(n)) + "\n"

        def face_lines(a, b):
            """Faces a..b-1 as OBJ lines: in bulk when they are all triangles, else one by one."""
            if all(counts[fi] == 3 for fi in range(a, b)):
                s0, s1 = starts[a], starts[b - 1] + 3
                v = [i + 1 for i in fvi[s0:s1]]
                if sti is None:
                    return _rows("f %d %d %d\n", v, b - a)
                t = [i + 1 for i in sti[s0:s1]]
                return _rows("f %d/%d %d/%d %d/%d\n", [x for pair in zip(v, t) for x in pair], b - a)
            return "".join(face(fi) for fi in range(a, b))

        def texture_of(target):
            """-> (asset path, original file name) of the bound UsdUVTexture."""
            mat, _ = UsdShade.MaterialBindingAPI(target).ComputeBoundMaterial()
            if not mat:
                return None, None
            for p in mat.GetPrim().GetChildren():
                sh = UsdShade.Shader(p)
                if sh and sh.GetIdAttr().Get() == "UsdUVTexture":
                    a = sh.GetInput("file").Get()
                    orig = p.GetAttribute("orb:file")
                    if a is not None:
                        return a, Path(orig.Get()).name if orig and orig.HasValue() else Path(a.path).name
            return None, None

        groups = [(s.GetPrim(), list(s.GetIndicesAttr().Get())) for s in subsets] or [(prim, range(len(counts)))]
        mat_of = [0] * len(counts)  # face -> material group
        for gi, (_, faces) in enumerate(groups):
            for fi in faces:
                mat_of[fi] = gi
        obj_of = [None] * len(counts)  # face -> the OBJ's `o` group (GeomSubsets of family "object")
        for s in UsdGeom.Subset.GetGeomSubsets(mesh, "face", OBJECT_FAMILY):
            tag = s.GetPrim().GetAttribute("orb:object")
            label = tag.Get() if tag and tag.HasValue() else s.GetPrim().GetName()
            for fi in s.GetIndicesAttr().Get():
                obj_of[fi] = label
        names = {}
        for gi, (target, _) in enumerate(groups):
            if has_st:
                names[gi] = name = f"m{gi}"
                mtl += ["", f"newmtl {name}", "Kd 1 1 1", "Ks 0 0 0", "d 1", "illum 1"]
                a, tname = texture_of(target)
                if a is not None and a.resolvedPath:
                    if tname not in textures:
                        buf = Ar.GetResolver().OpenAsset(Ar.ResolvedPath(a.resolvedPath))
                        (dst.parent / tname).write_bytes(bytes(buf.GetBuffer()))
                        textures[tname] = dst.parent / tname
                    mtl.append(f"map_Kd {tname}")
        cur_obj = cur_mtl = None
        run = 0  # first face not written yet
        for fi in range(len(counts)):  # in the export's (= the OBJ's) face order
            new_obj = obj_of[fi] is not None and obj_of[fi] != cur_obj
            new_mtl = has_st and names[mat_of[fi]] != cur_mtl
            if not (new_obj or new_mtl):
                continue
            if fi > run:
                lines.append(face_lines(run, fi))
                run = fi
            if new_obj:
                lines.append(f"o {obj_of[fi]}\n")
                cur_obj = obj_of[fi]
            if new_mtl:
                cur_mtl = names[mat_of[fi]]
                lines.append(f"usemtl {cur_mtl}\n")
        if len(counts) > run:
            lines.append(face_lines(run, len(counts)))
        dst.write_text("".join(lines))
        written.append(dst)
        if mtl:
            (dst.parent / f"{dst.stem}.mtl").write_text("# orb import\n" + "\n".join(mtl) + "\n")
            written.append(dst.parent / f"{dst.stem}.mtl")
        written += list(textures.values())
        rep.progress("meshes", f"  {_safe_rel(f).as_posix()}: {len(P):,} vertices", files=[(shown or dest) / _safe_rel(f)])
    return written


def _flat(arr, width=None):
    """A Vt array of vectors as a flat list of Python floats (rows of `width` when given)."""
    try:
        import numpy as np
        a = np.asarray(arr, dtype=np.float64) if len(arr) else np.zeros((0, width or 1))
        flat = a.reshape(-1).tolist()
    except ImportError:  # pragma: no cover
        flat = [float(x) for v in arr for x in v]
    if width is None:
        return flat
    return [flat[i:i + width] for i in range(0, len(flat), width)]


def _rows(fmt, values, n):
    """`fmt` (one line) n times, filled from the flat `values`: one % for the lot."""
    return (fmt * n) % tuple(values) if n else ""


# --------------------------------------------------------------------------- import


def _plain(v):
    """pxr VtDictionary values -> plain Python."""
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


def read_meta(path: Path):
    """-> (kind, meta, handle) or raises NotOrb."""
    ext = path.suffix.lower()
    if ext in GLTF_EXT:
        doc, binc = read_gltf(path)
        return "gltf", gltf_meta(doc), (doc, binc)
    if ext in USD_EXT:
        try:
            from pxr import Usd
        except ImportError:
            raise OrbError("importing USD needs the separately installed usd-core package (not included here)") from None
        try:
            stage = Usd.Stage.Open(str(path))
        except Exception:  # pxr.Tf.ErrorException ("Failed to open layer"): a truncated or foreign file
            raise OrbError(f"cannot open {path.name} as USD (not a USD file, or cut short); "
                           "is the download or copy complete?") from None
        if stage is None:
            raise OrbError(f"cannot open {path} as USD")
        return "usd", _plain(usd_meta(stage)), stage
    raise NotOrb(f"{ext} is not an Orb export format")


def import_export(path, name, into=None, force=False, rep=None, decompile=False):
    """An Orb-made GLB/USD -> OrbLang project. Returns a report dict.

    Everything is written to a scratch directory first and compiled there; only a project that
    compiles is moved into place, so a failed import leaves nothing behind."""
    import shutil
    import tempfile

    from .config import project_dest, slugify

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise OrbError(f"no such file: {path}")
    kind, meta, handle = read_meta(path)
    dest = Path(into).expanduser() if into else project_dest(slugify(name))
    sources = meta.get("sources") or {}
    world = json.loads(meta["world"]) if isinstance(meta.get("world"), str) else meta.get("world")
    moved = False
    roots = meta.get("deps") if isinstance(meta.get("deps"), dict) else {}
    _DEP_ROOTS.set({str(k): str(v) for k, v in roots.items()})
    if world:
        world, moved = packaged(world)
    # sources that import other projects (import "../robots") cannot compile without them: rebuild
    # from the world graph, which holds everything the export placed, with their files under deps/
    linked = bool(world) and (moved or any(IMPORT_RE.search(t) for t in sources.values()))
    if not sources and not world:
        if is_orb_generator(meta):
            raise OrbError(f"cannot import {path.name}: it is an Orb export ({meta.get('generator') or 'orb'}) without "
                           "its OrbLang sources or world graph (made with embed_sources = false), so there is "
                           "nothing to rebuild the project from; re-export it with embed_sources = true (the "
                           "default), or import the project's own .orb files")
        raise NotOrb("the Orb provenance has neither sources nor a world graph")
    if dest.exists() and any(p.is_file() and not p.name.startswith(".") for p in dest.glob("*.orb")) and not force:
        raise OrbError(f"{dest} already holds OrbLang files; pick another --into or pass --force")
    if rep is not None:
        rep.stage("sources", f"orb import: {path.name} ({kind}, {meta.get('generator') or 'orb'}) -> {dest}/",
                  kind=kind, dest=str(dest))
    with tempfile.TemporaryDirectory(prefix="orb-import-") as td:
        stage = Path(td)
        if sources and not ((decompile or linked) and world):
            mode = "sources"
            for fn, text in sources.items():
                if Path(fn).name != fn or not fn.endswith(".orb"):
                    raise OrbError(f"refusing to write source {fn!r} outside the project")
                (stage / fn).write_text(text, encoding="utf-8")
                rep.progress("sources", f"  {fn} ({len(text.splitlines())} lines, verbatim)", files=[dest / fn])
        else:
            from .decompile import to_orb

            mode = "decompiled"
            if linked and not decompile and rep is not None:
                rep.progress("sources", "  its sources import other projects: rebuilt from the world graph, "
                             f"their files under {DEPS}/")
            sources = to_orb(world, name)
            for fn, text in sources.items():
                (stage / fn).write_text(text, encoding="utf-8")
                rep.progress("sources", f"  {fn} ({len(text.splitlines())} lines, rebuilt from the world graph)",
                             files=[dest / fn])
        rep.stage("meshes", "rebuilding mesh files from the export's geometry")
        if kind == "gltf":
            meshes = gltf_meshes(path, *handle, stage, rep, shown=dest)
        else:
            meshes = usd_meshes(path, handle, stage, rep, shown=dest)
        licenses = []
        for fn, text in (meta.get("licenses") or {}).items():  # meshes/**/LICENSE the export carried
            rel = _safe_rel(fn)
            (stage / rel).parent.mkdir(parents=True, exist_ok=True)
            (stage / rel).write_text(text, encoding="utf-8")
            licenses.append(stage / rel)
        rep.stage("verify", "compiling the imported project")
        ir = compile_world(load(stage), stage)
        written = [stage / f for f in sources] + list(meshes) + licenses
        dest.mkdir(parents=True, exist_ok=True)
        files = []
        for f in dict.fromkeys(written):
            rel = f.relative_to(stage)
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(dest / rel))
            files.append(dest / rel)
    report = {"source": str(path), "kind": kind, "mode": mode, "dest": str(dest), "generator": meta.get("generator"),
              "project": meta.get("project"), "files": [str(f) for f in files],
              "licenses": [str(dest / f.relative_to(stage)) for f in licenses],
              "counts": {k: len(ir[k]) for k in ("parts", "assemblies", "sites", "items")},
              "fixtures": sum(len(s["fixtures"]) for s in ir["sites"].values())}
    if world:
        from .decompile import same_world

        # a decompiled project is one file: which files the original included is not the world's
        drop = ("root", "warnings") + (("includes",) if mode == "decompiled" else ())
        diffs = same_world({k: v for k, v in world.items() if k not in drop},
                           {k: v for k, v in ir.items() if k not in drop})
        report["world_matches"] = not diffs
        report["differences"] = diffs[:20]
    return report
