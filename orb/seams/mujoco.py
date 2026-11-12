"""MuJoCo seam: world graph -> one MJCF scene (scene.xml + assets/).

In a site with a scan mesh (`kind = "scan_mesh"`), the capture-derived boxes (`source.capture`)
still collide but go to geom group 3 with alpha 0, so the default viewer shows the scan;
`show_capture_boxes = true` draws them.

MuJoCo's OBJ loader keeps only the first object of a file and one material per geom, so an OBJ
with several objects (`o`/`g`) or materials (`usemtl`) is split into one asset OBJ per
object/material group, each its own <mesh> and geom with its own texture or colour. MuJoCo reads
only PNG textures: a JPEG map_Kd (orthoimagery, photo textures) is converted to a PNG in assets/,
at most JPEG_MAX_PX across. MuJoCo has no vertex colours: a vertex-coloured mesh is drawn in its
material's Kd (`orb geo` trees and facades), else a neutral gray.

Geoms of static bodies (fixtures, placed parts) and visual-only geoms of links that state their
inertia are massless (`mass="0"`): MuJoCo still computes a body's inertia from its meshes, and a
flat scan piece (a ceiling quad, a RoomPlan wall face) fails its `A + B >= C` check otherwise.
Door and window visuals stand 5 mm proud of both wall faces and 5 mm in from their edges, so they
never share a plane with the wall (or scanned wall) they sit in. With a floor fixture, the ground
plane sits 1 cm below the floor, so the floor's top face doesn't flicker against it.

Motion (v0.3, `MOTION = "playback"`): the seam's timeline (`seam "mujoco" { timeline = "..." }`,
default the first) is played by `orb.mjplay` from the sidecar `motion.json`, with physics on:

* kinematic items are PRESCRIBED free bodies: a free root joint (and their own hinge/slide joints)
  with a huge armature and `gravcomp="1"`, so nothing but the player moves them. Every control
  tick (10 ms) the player writes the evaluator's pose into qpos and the finite-difference velocity
  to the next tick into qvel; MuJoCo integrates exactly there. Unlike a mocap body they carry their
  real velocity into contacts (a box on a moving bed rides along by friction) while staying on the
  evaluator's pose. Their collision geoms are `contype=2 conaffinity=0`: they never collide with
  the static world or each other, only with dynamic things (whose geoms take `conaffinity=3`).
* dynamic payloads are free bodies with the IR mass/inertia, plus one `weld` equality per attach
  record (`orb:attach:<k>`, carrier link -> payload, inactive): the player activates it at t0 with
  the attach offset as relpose (placing the payload there first) and deactivates it at t1, when
  gravity and contact take over.
* hidden items: the player sets their geoms' contype/conaffinity to 0 and alpha to 0 (a free
  payload is also held where it was hidden), restoring them when shown.
* a joint's parent and child links never collide: one `<contact><exclude>` per jointed link pair
  of an articulated item (MuJoCo's own parent filter misses a root welded to the world).
* mimic joints (`mimic = "<leader>"`): no actuator; a joint equality `orb:mimic:<joint>` holds
  joint1 = offset + multiplier * joint2 (MuJoCo's polycoef). Kinematic items: the player
  prescribes them from the evaluator, which derives them.

Arena (`<size memory>`): MuJoCo's broadphase takes an int per body pair from its arena (4 bytes x
nbody^2: 3000 bodies need ~35 MiB), past the 13-16 MiB MuJoCo gives a world with few dofs, so a site of
thousands of units fails `mj_stackAlloc` on its first step. The seam sizes the arena from the
compiled model (arena_mib: bodies, geoms, mesh vertices and faces, with headroom) and writes it only
when that exceeds what MuJoCo would allocate itself, so small models keep MuJoCo's default.
`seam "mujoco" { memory = "1G" }` sets it by hand (`"auto"`: the estimate).

Native assets (`native "mujoco" { mjcf body mesh at rpy scale replace }` on an item, part,
assembly, link, fixture or site): `mjcf` attaches that MJCF body (copied with every file it
uses into native/, joints dropped: it rides rigidly on its owner) through MuJoCo's <attach>;
`mesh` adds that mesh as a geom. `replace = "visual"` (default) draws the native instead of the
universal visuals (universal collisions kept, hidden; the native is visual only), `"all"` keeps
only the native (it collides), `"none"` overlays it.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from ..lang import OrbError
from ..world import axis_angle, for_seam, hidden_capture_boxes, qmul, qrot, rpy_quat, with_mimics
from .meshio import first_kd, first_texture, load_obj, obj_stats
from .scene import INSTANCE_MIN, ground_drop, instance_groups, opening_box_size

PROCESS_FIDELITY = ()  # MuJoCo has no process timeline; processes are reported as skipped
MOTION = "playback"  # v0.3 timelines: played with physics by orb.mjplay (orb run mujoco --play / --video)
SCAN_RGBA = [0.78, 0.77, 0.74, 1]  # MuJoCo has no vertex colors: scans render a neutral gray
MIN_VERTICES = 4  # MuJoCo refuses smaller meshes ("at least 4 vertices required")
JPEG_MAX_PX = 2048  # JPEG textures are converted to PNG (MuJoCo reads only PNG), at most this size
SIDECAR = "motion.json"  # what orb.mjplay reads next to scene.xml
KIN_ARMATURE = 1e6  # kinematic dofs: so heavy that contacts and welds cannot move them within a tick
KIN, DYN = "kin", "dyn"  # contact classes (see the module docstring)
# MuJoCo's near/far planes are fractions of the model's extent (default znear 1%: 19 m on a 1.2 km
# terrain, so close cameras clipped): the seam writes NEAR metres as that fraction, zfar as default
NEAR = 0.05
FAR = 50.0


def f(xs):
    return " ".join(f"{x:.6g}" for x in (xs if isinstance(xs, (list, tuple)) else [xs]))


def near_fraction(extent):
    """<visual><map znear>: NEAR metres as a fraction of the model extent (at most MuJoCo's 1%)."""
    return min(0.01, max(1e-6, NEAR / max(float(extent), 1e-9)))


def compiled(scene):
    """The MjModel of a written scene.xml, or None without mujoco (or when MuJoCo refuses it: that is
    reported where it is loaded)."""
    try:
        import mujoco
    except ImportError:
        return None
    try:
        return mujoco.MjModel.from_xml_path(str(scene))
    except Exception:  # noqa: BLE001
        return None


def model_extent(scene):
    """MuJoCo's own stat.extent of a written scene.xml (the compiler's), or None without mujoco."""
    m = compiled(scene)
    return None if m is None else float(m.stat.extent)


ARENA_FLOOR_MIB = 16  # the most MuJoCo's own default arena gives a world with few dofs (13-16 MiB)
ARENA_HEADROOM = 1.5
ARENA_BASE_MIB = 4


def arena_mib(nbody, ngeom, nmeshvert=0, nmeshface=0):
    """MiB of arena a model needs to load and step, with headroom: the broadphase's body-pair buffer
    (4 bytes per pair of bodies: measured 34.7 MiB peak for 3006 bodies, 96 MiB for 5006), contacts
    and constraint rows (2 KiB per geom: a few contacts each) and collision scratch (16 B per mesh
    vertex and face; mesh size barely moves the peak, measured), times ARENA_HEADROOM plus
    ARENA_BASE_MIB."""
    need = 4 * nbody * nbody + 2048 * ngeom + 16 * (nmeshvert + nmeshface)
    return math.ceil(need * ARENA_HEADROOM / 2 ** 20) + ARENA_BASE_MIB


def memory_setting(cfg):
    """`seam "mujoco" { memory = ... }` -> MuJoCo's <size memory> text, or None for the automatic size
    (absent or "auto"): bytes as a number, or a whole number with K, M, G or T ("512M", "1G")."""
    v = cfg.get("memory")
    if v is None or (isinstance(v, str) and v.strip().lower() == "auto"):
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 and float(v).is_integer():
        return str(int(v))
    if isinstance(v, str) and re.fullmatch(r"\s*[1-9]\d*\s*[KMGTkmgt]?\s*", v):
        return re.sub(r"\s+", "", v).upper()
    raise OrbError(f'seam "mujoco" memory must be a size such as "512M" or "1G" (a whole number of bytes, '
                   f'or with K, M, G or T), or "auto", not {v!r}')


def tree_counts(m, out):
    """(nbody, ngeom, nmeshvert, nmeshface) of an MJCF tree, roughly (without mujoco to compile it):
    natives' bodies are left out, an OBJ's faces counted as twice its vertices."""
    from .meshio import obj_stats

    verts = 0
    for me in m.iter("mesh"):
        p = out / "assets" / (me.get("file") or "")
        if p.suffix.lower() == ".obj" and p.is_file():
            verts += obj_stats(p)[1]
    return 1 + sum(1 for _ in m.iter("body")), sum(1 for _ in m.iter("geom")), verts, 2 * verts


def rough_extent(ir):
    """The extent MuJoCo would compute (twice the bounding radius), roughly: site floors and items."""
    lo, hi = rough_box(ir)
    return max(1.0, sum((b - a) ** 2 for a, b in zip(lo, hi)) ** 0.5)


def rough_box(ir):
    """(lo, hi) of the site floors and item positions; a 1 m box at the origin when there are none."""
    pts = []
    for s in ir["sites"].values():
        w, d = s.get("floor") or [20, 20]
        x, y, z = s.get("pos") or [0.0, 0.0, 0.0]
        pts += [[x - w / 2, y - d / 2, z], [x + w / 2, y + d / 2, z]]
    pts += [list(it["pos"]) for it in ir["items"].values() if it.get("pos")]
    if not pts:
        return [-0.5, -0.5, 0.0], [0.5, 0.5, 1.0]
    return [min(p[k] for p in pts) for k in range(3)], [max(p[k] for p in pts) for k in range(3)]


def _rgb(c, key):
    """A seam colour setting ("#rrggbb" or [r, g, b]) -> [r, g, b] in 0..1."""
    if isinstance(c, str) and c.startswith("#") and len(c) == 7:
        return [int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    if isinstance(c, (list, tuple)) and len(c) in (3, 4) and all(isinstance(x, (int, float)) for x in c):
        return [float(x) for x in c[:3]]
    raise OrbError(f'seam "mujoco" {key} must be "#rrggbb" or [r, g, b], not {c!r}')


OUTDOOR = 150.0  # metres of extent: past this a world is an outdoor site (default haze, a plain ground)
GROUND = [0.56, 0.55, 0.52]  # the default ground: a warm neutral grey, not a white glare
SKY = [[0.2, 0.4, 0.72], [0.78, 0.84, 0.92]]  # deep blue overhead to a pale horizon
SUN = [-0.35, 0.45, -0.82]  # a low-ish sun from the south-east: shadows fall north-west, faces get shape


def site_color(site, sname):
    """A site's `color` (its ground: a name, "#rrggbb" or [r, g, b]) -> [r, g, b], or None."""
    from ..world import COLORS

    c = site.get("color")
    if c is None:
        return None
    if isinstance(c, str) and c in COLORS:
        return list(COLORS[c][:3])
    try:
        return _rgb(c, "color")
    except OrbError:
        raise OrbError(f"site {sname}: color must be a name ({', '.join(COLORS)}), \"#rrggbb\" or [r, g, b], "
                       f"not {c!r}") from None


def look(cfg, ir=None):
    """The render look from the seam block (every key optional; without them the defaults below):
    `sky = [top, bottom]` skybox gradient, `ground_color` (a plain ground, faint 2 m checker),
    `ground_size = [w, d] m` (the drawn ground plane; default the site floor), `sun = [x, y, z]`
    (the directional light's direction), `sun_at = [x, y, z] m` (its position: the shadow box centre),
    `sun_color`, `shadow_size` (shadow map pixels), `shadow_clip` (metres the shadow box reaches from
    sun_at), `headlight` and `ambient` (the headlight's diffuse and ambient, 0..1), `samples`
    (offscreen multisampling), `haze = [start, end] m` (distance fog in the sky's bottom colour, or
    `haze_color`; the model carries <numeric name="orb:haze"> and `--video` turns MuJoCo's fog on).
    The defaults are a daylight look: a blue sky fading to a pale horizon, a low warm sun from the
    south-west casting soft-edged 4k shadows over the world's middle, a dim headlight (no glare on
    the ground), and for a world wider than OUTDOOR metres a light haze from half its extent."""
    lo, hi = rough_box(ir) if ir else ([-0.5, -0.5, 0.0], [0.5, 0.5, 1.0])
    extent = rough_extent(ir) if ir else 1.0
    mid = [(a + b) / 2 for a, b in zip(lo, hi)]
    n = math.sqrt(sum(c * c for c in SUN))
    back = 0.5 * extent + 10.0  # the sun stands back along its own direction: the shadow box sees the whole world
    lk = {"sky": [list(c) for c in SKY], "ground": None, "ground_size": None,
          "sun": list(SUN), "sun_at": [m - c / n * back for m, c in zip(mid, SUN)], "sun_color": [0.86, 0.83, 0.77],
          "shadow_size": 4096.0, "shadow_clip": None, "headlight": 0.32, "ambient": 0.26, "samples": None,
          "haze": [0.2 * extent, 1.6 * extent] if extent > OUTDOOR else None, "haze_color": None}
    if cfg.get("sky") is not None:
        sky = cfg["sky"]
        if not isinstance(sky, (list, tuple)) or len(sky) != 2:
            raise OrbError('seam "mujoco" sky must be [top, bottom] colours')
        lk["sky"] = [_rgb(sky[0], "sky"), _rgb(sky[1], "sky")]
    if cfg.get("ground_color") is not None:
        lk["ground"] = _rgb(cfg["ground_color"], "ground_color")
    if cfg.get("sun_color") is not None:
        lk["sun_color"] = _rgb(cfg["sun_color"], "sun_color")
    lk["haze_color"] = _rgb(cfg["haze_color"], "haze_color") if cfg.get("haze_color") is not None else \
        (lk["sky"][1] if cfg.get("sky") is not None else [0.62, 0.7, 0.82])  # the default sky's horizon
    if cfg.get("sun") is not None and cfg.get("sun_at") is None:  # a sun turned by hand keeps standing back
        d = [float(x) for x in cfg["sun"]] if isinstance(cfg["sun"], (list, tuple)) and len(cfg["sun"]) == 3 else SUN
        dn = math.sqrt(sum(c * c for c in d)) or 1.0
        lk["sun_at"] = [m - c / dn * back for m, c in zip(mid, d)]
    for k, n in (("ground_size", 2), ("sun", 3), ("sun_at", 3), ("haze", 2)):
        v = cfg.get(k)
        if v is not None:
            if not isinstance(v, (list, tuple)) or len(v) != n:
                raise OrbError(f'seam "mujoco" {k} must be a list of {n} numbers')
            lk[k] = [float(x) for x in v]
    for k in ("shadow_size", "shadow_clip", "headlight", "ambient", "samples"):
        if cfg.get(k) is not None:
            lk[k] = float(cfg[k])
    return lk


def camera_xyaxes(quat):
    """MuJoCo camera axes for an Orb sensor: Orb sensors look along their +X with +Z up; a MuJoCo
    camera looks along its -Z with +Y up, so its x axis is the sensor's -Y and its y axis the
    sensor's +Z, both turned by the sensor's rotation."""
    x = qrot(quat, [0.0, -1.0, 0.0])
    y = qrot(quat, [0.0, 0.0, 1.0])
    return f([round(v, 9) + 0.0 for v in x + y])


def qinv(q):
    return [q[0], -q[1], -q[2], -q[3]]


def compose(a, b):
    """Pose a ∘ b, poses as (pos, quat)."""
    return [x + y for x, y in zip(a[0], qrot(a[1], b[0]))], qmul(a[1], b[1])


def relative(a, b):
    """The pose of b in a's frame: a⁻¹ ∘ b."""
    qi = qinv(a[1])
    return qrot(qi, [y - x for x, y in zip(a[0], b[0])]), qmul(qi, b[1])


def native_spec(block):
    """The MuJoCo native on an IR block: for_seam leaves {"mujoco": {...}} (older IRs may carry the
    entry itself); None without one."""
    nat = block.get("native") if isinstance(block, dict) else None
    if not isinstance(nat, dict):
        return None
    if isinstance(nat.get("mujoco"), dict):
        return nat["mujoco"]
    return nat if ("mjcf" in nat or "mesh" in nat) else None


def native_pose(spec):
    """(pos, quat, uniform scale) of a native in its owner's frame."""
    pos = spec.get("pos", spec.get("at")) or [0.0, 0.0, 0.0]
    if spec.get("quat"):
        quat = list(spec["quat"])
    elif spec.get("rpy"):
        quat = rpy_quat(*spec["rpy"])
    elif spec.get("yaw") is not None:
        quat = rpy_quat(0.0, 0.0, spec["yaw"])
    else:
        quat = [1.0, 0.0, 0.0, 0.0]
    s = spec.get("scale", 1.0)
    if isinstance(s, (list, tuple)):
        if max(s) - min(s) > 1e-9:
            raise OrbError(f"native \"mujoco\" scale must be uniform in MuJoCo, got {s}")
        s = s[0]
    return [float(x) for x in pos], [float(x) for x in quat], float(s)


def replace_of(spec):
    how = (spec or {}).get("replace", "visual")
    if how not in ("visual", "all", "none"):
        raise OrbError(f"native \"mujoco\" replace is visual | all | none, got {how!r}")
    return how


def motion_plan(ir, cfg):
    """(Motion, timeline dict, {item: body}) for the seam's timeline (`timeline` in the seam block,
    default the first); None without timelines. Bodies as the evaluator settles them (defaults,
    and physics = false making every body kinematic)."""
    if not ir.get("timelines"):
        return None
    from ..motion import Motion

    name = cfg.get("timeline")
    m = Motion(ir, name if isinstance(name, str) and name else None)
    return m, m.tl, {k: m.body(k) for k in m.items() if k in ir["items"]}


SCALE_STEP = 1.05  # instanced fixture meshes: scales snap to powers of this (5 % buckets)


def scale_bucket(s, step=SCALE_STEP):
    """s snapped to the nearest power of `step` (sign kept, 0 stays 0): at most half a step off."""
    if s == 0:
        return 0.0
    return math.copysign(float(f"{step ** round(math.log(abs(s)) / math.log(step)):.6g}"), s)


SCALE_ASSETS = 32  # an instanced file drawn at more distinct scales than this gets bucketed scales


def shared_files(ir, least=INSTANCE_MIN, budget=SCALE_ASSETS):
    """Mesh files some fixture draws as `least` or more visuals (geo vehicles, trees) at more than
    `budget` distinct scales: their scales are bucketed (scale_bucket), the rest stay exact."""
    scales = {}
    for site in ir["sites"].values():
        for fx in site["fixtures"].values():
            for ks in instance_groups(fx["geoms"], least).values():
                for k in ks:
                    g = fx["geoms"][k]
                    scales.setdefault(g["file"], set()).add(tuple(g["scale"]))
    return {f for f, ss in scales.items() if len(ss) > budget}


def emit(ir, out: Path, cfg):
    ir = for_seam(ir, "mujoco")
    root = Path(ir["root"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "assets").mkdir(exist_ok=True)
    shutil.rmtree(out / "native", ignore_errors=True)  # native MJCF copies: written afresh below
    plan = motion_plan(ir, cfg)
    motion, tl, bodies = plan if plan else (None, None, {})
    moving = any(v == "kinematic" for v in bodies.values())  # dynamic geoms then also accept kinematic contacts
    timestep = float(cfg.get("timestep", 0.002))
    m = ET.Element("mujoco", model="orb")
    ET.SubElement(m, "compiler", angle="radian", meshdir="assets", autolimits="true")
    ET.SubElement(m, "option", timestep=f(timestep), cone="elliptic", impratio="100")
    lk = look(cfg, ir)
    vis = ET.SubElement(m, "visual")
    ET.SubElement(vis, "headlight", diffuse=f([lk["headlight"]] * 3), ambient=f([lk["ambient"]] * 3), specular="0 0 0")
    ET.SubElement(vis, "global", azimuth="140", elevation="-25", offwidth="1920", offheight="1080")
    clip = ET.SubElement(vis, "map", znear=f(near_fraction(rough_extent(ir))), zfar=f(FAR))  # exact below
    if lk["shadow_size"] or lk["samples"]:
        ET.SubElement(vis, "quality", **({"shadowsize": str(int(lk["shadow_size"]))} if lk["shadow_size"] else {}),
                      **({"offsamples": str(int(lk["samples"]))} if lk["samples"] else {}))
    if lk["haze"]:  # fractions of the extent, set exactly below; the marker turns the fog on in --video
        ET.SubElement(vis, "rgba", fog=f(lk["haze_color"] + [1.0]))
        ET.SubElement(ET.SubElement(m, "custom"), "numeric", name="orb:haze", data="1")
    asset = ET.SubElement(m, "asset")
    ET.SubElement(asset, "texture", type="skybox", builtin="gradient", rgb1=f(lk["sky"][0]), rgb2=f(lk["sky"][1]),
                  width="512", height="3072")
    outdoor = rough_extent(ir) > OUTDOOR

    def plain_ground(name, g):
        """A plain ground with a faint 2 m checker (scale without a CAD grid's glare or moire)."""
        tex = "grid" if name == "floor" else f"grid_{name}"
        ET.SubElement(asset, "texture", type="2d", name=tex, builtin="checker", rgb1=f(g),
                      rgb2=f([c * 0.975 for c in g]), width="64", height="64")
        ET.SubElement(asset, "material", name=name, texture=tex, texuniform="true", texrepeat="0.25 0.25",
                      reflectance="0", specular="0.05")

    if lk["ground"] or outdoor:  # the seam's ground_color, else (a big outdoor world) the neutral ground
        plain_ground("floor", lk["ground"] or GROUND)
    else:  # a room-sized world: a 1 m grid on the neutral ground, for scale
        g = GROUND
        ET.SubElement(asset, "texture", type="2d", name="grid", builtin="checker", mark="edge",
                      rgb1=f([c * 1.08 for c in g]), rgb2=f(g), markrgb=f([c * 0.85 for c in g]),
                      width="300", height="300")
        ET.SubElement(asset, "material", name="floor", texture="grid", texuniform="true", texrepeat="2 2",
                      reflectance="0.03", specular="0.05")
    floor_mats = {}
    for sname, site in ir["sites"].items():  # a site's own `color` paints its ground (the seam's ground_color wins)
        c = None if lk["ground"] else site_color(site, sname)
        if c is not None:
            key = "floor_" + "_".join(str(round(x * 255)) for x in c)
            if key not in floor_mats.values():
                plain_ground(key, c)
            floor_mats[sname] = key
    world = ET.SubElement(m, "worldbody")
    ET.SubElement(world, "light", pos=f(lk["sun_at"]), dir=f(lk["sun"]), directional="true", castshadow="true",
                  **({"diffuse": f(lk["sun_color"])} if lk["sun_color"] else {}))
    meshes, mats, files, textures, parts, names = {}, {}, {}, {}, {}, set()
    natives = []  # files written for native assets

    def material(rgba):
        key = "c_" + "_".join(f"{round(c * 255)}" for c in rgba)
        if key not in mats:
            mats[key] = ET.SubElement(asset, "material", name=key, rgba=f(rgba), specular="0.2", shininess="0.3")
        return key

    def asset_name(src, stem=None, suffix=None):
        """Asset file name for src (a path, or a key for a generated file), unique across
        directories (scans are all called scan.obj)."""
        stem, suffix = stem or src.stem, suffix or src.suffix
        name, k = f"{stem}{suffix}", 1
        while name in files and files[name] != src:
            name, k = f"{stem}_{k}{suffix}", k + 1
        files[name] = src
        return name

    def unique(name):
        base, k = name, 1
        while name in names:
            name, k = f"{base}_{k}", k + 1
        names.add(name)
        return name

    def texture(tex):
        """A PNG map_Kd -> MuJoCo texture + material (one per image file); a JPEG (MuJoCo reads only
        PNG) is converted, at most JPEG_MAX_PX across (orthoimagery tiles, photo textures); else None."""
        if tex is None or tex.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            return None
        if tex not in textures:
            if tex.suffix.lower() == ".png":
                fn = asset_name(tex)
                shutil.copy(tex, out / "assets" / fn)
            else:
                try:
                    from PIL import Image
                except ImportError:  # no pillow: the flat colour, as before
                    return None
                fn = asset_name(tex, suffix=".png")
                im = Image.open(tex).convert("RGB")
                if max(im.size) > JPEG_MAX_PX:
                    k = JPEG_MAX_PX / max(im.size)
                    im = im.resize((max(1, round(im.size[0] * k)), max(1, round(im.size[1] * k))), Image.LANCZOS)
                im.save(out / "assets" / fn)
            name = unique(f"tex_{Path(fn).stem}")
            ET.SubElement(asset, "texture", type="2d", name=name, file=f"assets/{fn}")  # meshdir only covers meshes
            ET.SubElement(asset, "material", name=name, texture=name, specular="0.1", rgba="1 1 1 1")
            textures[tex] = name
        return textures[tex]

    def split(src):
        """[(asset file, texture path | None, Kd | None)] for a mesh file: the file itself, or one
        OBJ per object/material group when MuJoCo would drop all but the first."""
        if src not in parts:
            if not src.is_file():
                raise OrbError(f"mesh file not found: {src}")
            n_parts, n_verts = obj_stats(src) if src.suffix.lower() == ".obj" else (1, MIN_VERTICES)
            if n_parts <= 1 and n_verts >= MIN_VERTICES:
                fn = asset_name(src)
                shutil.copy(src, out / "assets" / fn)
                obj = src.suffix.lower() == ".obj"
                parts[src] = [(fn, first_texture(src) if obj else None, first_kd(src) if obj else None)]
            else:
                o = load_obj(src)
                groups = o.parts()
                if len(groups) == 1 and n_verts >= MIN_VERTICES:
                    fn = asset_name(src)
                    shutil.copy(src, out / "assets" / fn)
                    parts[src] = [(fn, o.texture(groups[0]["slot"]) if o.tri_t is not None else None,
                                   o.kd(groups[0]["slot"]))]
                else:
                    parts[src] = []
                    for k, grp in enumerate(groups):
                        fn = asset_name((src, k), stem=f"{src.stem}_part{k}", suffix=".obj")
                        tex = o.texture(grp["slot"])
                        o.write_part(grp["tris"], out / "assets" / fn, with_uv=tex is not None,
                                     min_vertices=MIN_VERTICES)
                        parts[src].append((fn, tex, o.kd(grp["slot"])))
        return parts[src]

    shared = shared_files(ir)

    def mesh(g, scan=False):
        """-> [(mesh name, texture path | None, Kd | None)], one per part of the geom's file. A fixture
        visual whose file is instanced at many scales (shared_files) has its scale bucketed
        (scale_bucket): MuJoCo scales a mesh asset, so 2000 trees of distinct sizes need not be
        2000 assets."""
        if scan and g["file"] in shared and g["scale"] != [1, 1, 1]:
            g = {**g, "scale": [scale_bucket(s) for s in g["scale"]]}
        key = (g["file"], tuple(g["scale"]))
        if key not in meshes:
            src = (root / g["file"]).resolve()
            meshes[key] = []
            pieces = split(src)
            for fn, tex, kd in pieces:
                name = unique(Path(fn).stem + ("" if g["scale"] == [1, 1, 1] else "_scaled"))
                a = {"name": name, "file": fn, "scale": f(g["scale"])}
                if scan or (len(pieces) > 1 and g["role"] == "visual"):  # open surfaces have no volume
                    a["inertia"] = "shell"
                ET.SubElement(asset, "mesh", a)
                meshes[key].append((name, tex, kd))
        return meshes[key]

    def geom(parent, g, hide_collision, name=None, fixture=False, invisible=False, massless=False, kind=None,
             con=None):
        a = {"pos": f(g["pos"]), "quat": f(g["quat"])}
        if name:
            a["name"] = name
        if massless:  # static body, or a link whose inertia is given: geoms add no mass or inertia
            a["mass"] = "0"
        shape = g["shape"]
        if shape == "box":
            size = opening_box_size(g, kind) if fixture else g["size"]
            a.update(type="box", size=f([s / 2 for s in size]))
        elif shape == "sphere":
            a.update(type="sphere", size=f(g["radius"]))
        elif shape in ("cylinder", "capsule"):
            a.update(type=shape, size=f([g["radius"], g["length"] / 2]))
        scan = fixture and shape == "mesh" and g["role"] == "visual"
        if shape != "mesh":
            style(a, g, hide_collision, scan, invisible, con=con)
            ET.SubElement(parent, "geom", a)
            return
        for k, (mname, tex, kd) in enumerate(mesh(g, scan=scan)):  # one geom per part of the file
            b = {**a, "type": "mesh", "mesh": mname}
            if name and k:
                b["name"] = f"{name}/{k}"
            style(b, g, hide_collision, scan, invisible, tex, kd, con=con)
            ET.SubElement(parent, "geom", b)

    def style(a, g, hide_collision, scan, invisible, tex=None, kd=None, con=None):
        if scan:  # visual-only scanned surface: never collides, shown in group 1
            look = texture(tex) or material(g.get("color") or (list(kd) + [1] if kd else SCAN_RGBA))
            a.update(material=look, contype="0", conaffinity="0", group="1")
        elif invisible and g["role"] != "visual":  # collides, never drawn (a material would set alpha)
            a["rgba"] = f((g.get("color") or [0.5, 0.5, 0.5, 1])[:3] + [0])
        elif g.get("color"):  # an explicit colour wins
            a["material"] = material(g["color"])
        elif tex is not None or kd:  # else the mesh's own .mtl: its PNG map_Kd, or its Kd per material
            look = texture(tex) or (material(list(kd)[:3] + [1]) if kd else None)
            if look:
                a["material"] = look
        if scan:
            pass
        elif g["role"] == "visual":
            a.update(contype="0", conaffinity="0", group="2")
        else:
            if hide_collision or invisible:
                a["group"] = "3"
            if con == KIN:  # prescribed: touches dynamic things only
                a.update(contype="2", conaffinity="0")
            elif con == DYN and moving:  # simulated: also touched by the kinematic items
                a["conaffinity"] = "3"
            if "friction" in g:
                a.update(friction=f([g["friction"], 0.02, 0.01]), condim="6", priority="1")

    def custom(name, data):
        """Provenance as <custom><text>: MuJoCo keeps it in the model (mj_name2id(..., mjOBJ_TEXT))."""
        cu = m.find("custom")
        if cu is None:
            cu = ET.SubElement(m, "custom")
        taken = {t.get("name") for t in cu}
        base, k = name, 1
        while name in taken:
            name, k = f"{base}_{k}", k + 1
        ET.SubElement(cu, "text", name=name, data=json.dumps(data, sort_keys=True, separators=(",", ":")))

    def native(parent, spec, owner, frame=None, con=None):
        """Draw `spec` (a native "mujoco" entry) in `parent`; frame = the owner's frame in the parent
        body's frame (default: the same frame)."""
        how = replace_of(spec)
        pos, quat, scale = native_pose(spec)
        if frame is not None:
            pos, quat = compose(frame, (pos, quat))
        visual = how != "all"
        if spec.get("mesh"):
            g = {"role": "visual" if visual else "collision", "shape": "mesh", "file": spec["mesh"],
                 "scale": [scale] * 3, "pos": pos, "quat": quat, "color": spec.get("color")}
            geom(parent, g, hide_collision=False, massless=True, con=con)
        if spec.get("mjcf"):
            slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", owner).strip("_") or "native"
            fn = native_mjcf(root / spec["mjcf"], out / "native", unique(f"native_{slug}"), spec.get("body"),
                             visual, scale, con, moving)
            natives.append(fn)
            mname = fn.stem
            ET.SubElement(asset, "model", name=mname, file=f"native/{fn.name}")
            fr = ET.SubElement(parent, "frame", pos=f(pos), quat=f(quat))
            ET.SubElement(fr, "attach", model=mname, body=NATIVE_ROOT, prefix=f"{owner}/native/")

    for slug, cap in (ir.get("other", {}).get("capture") or {}).items():
        custom(f"orb:capture:{slug}", cap)

    # site: floor, zones, fixtures
    if outdoor and ir["sites"] and not lk["ground_size"]:  # the ground runs on to the hazy horizon, drawn only
        z = min(st["pos"][2] - ground_drop(st) for st in ir["sites"].values()) - 0.05
        lo, hi = rough_box(ir)
        ET.SubElement(world, "geom", name="orb:horizon", type="plane", size="0 0 1",
                      pos=f([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, z]), material="floor",
                      contype="0", conaffinity="0", group="2")
    for sname, site in ir["sites"].items():
        w, d = lk["ground_size"] or site["floor"] or [20, 20]
        ground = [site["pos"][0], site["pos"][1], site["pos"][2] - ground_drop(site)]
        sq = list(site.get("quat") or [1.0, 0.0, 0.0, 0.0])  # a site an output turned
        turned = {} if sq == [1.0, 0.0, 0.0, 0.0] else {"quat": f(sq)}
        ET.SubElement(world, "geom", name=f"{sname}/floor", type="plane", pos=f(ground),
                      size=f([w / 2, d / 2, 0.05]), material=floor_mats.get(sname, "floor"), **turned)
        for zname, z in site["zones"].items():
            if z["size"]:
                rgba = (z["color"] or [0.5, 0.5, 0.5, 1])[:3] + [0.35]
                ET.SubElement(world, "geom", name=f"{sname}/{zname}", type="box", contype="0", conaffinity="0",
                              pos=f([z["pos"][0], z["pos"][1], z["pos"][2] + 0.002]),
                              size=f([z["size"][0] / 2, z["size"][1] / 2, 0.002]), rgba=f(rgba), **turned)
        if native_spec(site):
            native(world, native_spec(site), sname, frame=(site["pos"], sq))
        hidden = hidden_capture_boxes(site, cfg)
        for fname, fx in site["fixtures"].items():
            body = ET.SubElement(world, "body", name=f"{sname}/{fname}", pos=f(fx["pos"]), quat=f(fx["quat"]))
            kind = (fx.get("attrs") or {}).get("kind")
            spec = native_spec(fx)
            how = replace_of(spec) if spec else "none"
            for g in fx["geoms"]:
                if how == "all" or (how == "visual" and g["role"] == "visual"):
                    continue
                geom(body, g, hide_collision=how == "visual", fixture=True, invisible=fname in hidden,
                     massless=True, kind=kind)
            if spec:
                native(body, spec, f"{sname}/{fname}")
            if fx.get("attrs"):
                custom(f"orb:{fname}", {**fx["attrs"], "fixture": fname, "site": sname})

    # items: assemblies become articulated bodies, parts become static props (or, in a timeline,
    # kinematic / dynamic bodies)
    actuators = ET.Element("actuator")
    qpos, ctrl = [], []
    side_items = {}
    excludes = []  # (parent body, child body) of every joint: no contact between them
    mimic_eqs, made = [], set()  # (mimic joint, leader joint, multiplier, offset), MJCF names; hinges/slides made

    def inertial(body, i):
        ET.SubElement(body, "inertial", pos=f(i["com"]), mass=f(i["mass"]), fullinertia=f(i["inertia"]))

    def keep(g, how):
        return not (how == "all" or (how == "visual" and g["role"] == "visual"))

    for iname, it in ir["items"].items():
        role = bodies.get(iname)
        if (role is None and plan is None and it["kind"] == "part" and it.get("body") == "dynamic"
                and it.get("location") is not None):  # (an item in inventory or installed stays out)
            role = "dynamic"  # no timeline: a part declared dynamic still falls, slides and is pushed
        if it.get("location") is None and role is None:  # installed inside something else, or not placed
            continue
        if iname in bodies:  # where the timeline has it at t = 0
            pose0 = motion.pose(iname, 0.0)
            pose0 = ([float(x) for x in pose0[0]], [float(x) for x in pose0[1]])
        else:
            pose0 = (it["pos"], it["quat"])
        spec = native_spec(it)
        if it["kind"] == "part":
            part = ir["parts"][it["of"]]
            spec = spec or native_spec(part)
            how = replace_of(spec) if spec else "none"
            body = ET.SubElement(world, "body", name=iname, pos=f(pose0[0]), quat=f(pose0[1]))
            con = None
            if role == "kinematic":
                body.set("gravcomp", "1")
                ET.SubElement(body, "joint", name=f"{iname}/free", type="free", armature=f(KIN_ARMATURE), damping="0")
                inertial(body, part.get("inertial") or {"com": [0, 0, 0], "mass": 1.0, "inertia": [1, 1, 1, 0, 0, 0]})
                qpos.extend(list(pose0[0]) + list(pose0[1]))
                con = KIN
            elif role == "dynamic":
                ET.SubElement(body, "freejoint", name=f"{iname}/free")
                qpos.extend(list(pose0[0]) + list(pose0[1]))
                con = DYN
            kept = [g for g in part["geoms"] if keep(g, how)]
            has_mass = role == "dynamic" and not part.get("inertial") and any(g["role"] != "visual" for g in kept)
            if role == "dynamic" and part.get("inertial"):
                inertial(body, part["inertial"])
            elif role == "dynamic" and not has_mass:  # nothing left to weigh: a token inertia
                inertial(body, {"com": [0, 0, 0], "mass": 1.0, "inertia": [0.1, 0.1, 0.1, 0, 0, 0]})
            drawn = how != "none" or any(g["role"] == "visual" for g in kept)  # else its collision geoms show
            for g in kept:
                geom(body, g, hide_collision=drawn, massless=not (has_mass and g["role"] != "visual"), con=con)
            if spec:
                native(body, spec, iname, con=con)
            if role:
                side_items[iname] = {"body": role, "kind": "part", "root": iname,
                                     "free": f"{iname}/free" if role in ("kinematic", "dynamic") else None,
                                     "joints": {}, "links": {}, "root_offset": None}
            continue
        asm = ir["assemblies"][it["of"]]
        spec = spec or native_spec(asm)
        whole = replace_of(spec) if spec else "none"  # an item/assembly native replaces every link's geoms
        kids = {}
        for link in asm["links"]:
            kids.setdefault(link["parent"], []).append(link)
        joints = dict(it.get("joints", {}))
        if role in ("kinematic", "dynamic"):
            joints.update({k: float(v) for k, v in (motion.joints(iname, 0.0) or {}).items()})
        joints = with_mimics(asm, joints)  # mimic joints at their leader's value
        kin = role == "kinematic"
        con = KIN if kin else DYN
        # body = "static" (declared, or the timeline's): the root is welded where the item stands, even
        # when the assembly's root joint is free, and its joints hold their initial values (no freedom:
        # the value is baked into the link pose), unless the timeline has a joints track for it
        static = role == "static" or (role is None and it.get("body") == "static")
        frozen = static and not (tl is not None and iname in (tl.get("joints") or {}))
        info = {"body": role or "default", "kind": "assembly", "root": None, "free": None, "joints": {},
                "links": {}, "root_offset": [asm["links"][0]["pos"], asm["links"][0]["quat"]]}

        def build(parent_el, link, top):
            pos, quat = link["pos"], link["quat"]
            if top:  # the root link's own at/rpy, composed onto the item's pose (as world.frames does)
                pos, quat = compose(pose0, (link["pos"], link["quat"]))
            bname = f"{iname}/{link['name']}"
            body = ET.SubElement(parent_el, "body", name=bname, pos=f(pos), quat=f(quat))
            if not top and not kin and link["joint"]["type"] != "free":  # jointed links never touch (as in PhysX)
                excludes.append((f"{iname}/{link['parent']}", bname))
            info["links"][link["name"]] = bname
            if top:
                info["root"] = bname
            if kin:
                body.set("gravcomp", "1")
            j = link["joint"]
            if link["inertial"]:
                inertial(body, link["inertial"])
            elif kin and (top or j["type"] in ("revolute", "prismatic")):
                inertial(body, {"com": [0, 0, 0], "mass": 1.0, "inertia": [1, 1, 1, 0, 0, 0]})
            free = (j["type"] == "free" and not static) or (top and role in ("kinematic", "dynamic"))
            if frozen and j["type"] in ("revolute", "prismatic"):  # held at its initial value
                q0 = float(joints.get(j["name"], j["initial"]))
                if j["type"] == "revolute":
                    body.set("quat", f(qmul(quat, axis_angle(j["axis"], q0))))
                else:
                    body.set("pos", f([p + q0 * a for p, a in zip(pos, qrot(quat, j["axis"]))]))
            elif free and top:
                jn = f"{iname}/{j['name']}"
                if kin:
                    ET.SubElement(body, "joint", name=jn, type="free", armature=f(KIN_ARMATURE), damping="0")
                else:
                    ET.SubElement(body, "freejoint", name=jn)
                info["free"] = jn
                qpos.extend(list(pos) + list(quat))
            elif j["type"] in ("revolute", "prismatic"):
                jn = f"{iname}/{j['name']}"
                info["joints"][j["name"]] = jn
                a = {"name": jn, "type": "hinge" if j["type"] == "revolute" else "slide", "axis": f(j["axis"])}
                if kin:  # prescribed: no limits, friction or drives to fight the player
                    a.update(armature=f(KIN_ARMATURE), damping="0")
                else:
                    a.update(damping=f(j["damping"]), armature=f(j["armature"]), frictionloss=f(j["friction"]))
                    if j["range"]:
                        a["range"] = f(j["range"])
                ET.SubElement(body, "joint", a)
                made.add(jn)
                q0 = joints.get(j["name"], j["initial"])
                qpos.append(q0)
                act = {"name": jn, "joint": jn}
                if kin:
                    pass  # prescribed by the player (a mimic joint too: the evaluator derives it)
                elif j.get("mimic"):  # no actuator: an equality holds it to its leader (below)
                    mm = j["mimic"]
                    mimic_eqs.append((jn, f"{iname}/{mm['joint']}", mm["multiplier"], mm["offset"]))
                elif j.get("drive"):
                    act.update(kp=f(j["drive"]["kp"]), kv=f(j["drive"]["kd"]))
                    if j["range"]:
                        act["ctrlrange"] = f(j["range"])
                    if j["effort"]:
                        act["forcerange"] = f([-j["effort"], j["effort"]])
                    ET.SubElement(actuators, "position", act)
                    ctrl.append(q0)
                elif j["effort"]:
                    ET.SubElement(actuators, "motor", act, ctrlrange=f([-j["effort"], j["effort"]]))
                    ctrl.append(0.0)
            lspec = native_spec(link) or (native_spec(ir["parts"].get(link["part"]) or {}) if link.get("part") else None)
            how = replace_of(lspec) if lspec else whole
            has_visual = any(g["role"] == "visual" for g in link["geoms"]) or how != "none"
            for g in link["geoms"]:
                if not keep(g, how):
                    continue
                geom(body, g, hide_collision=has_visual,
                     massless=kin or (bool(link["inertial"]) and g["role"] == "visual"), con=con)
            if lspec:
                native(body, lspec, f"{iname}/{link['name']}", con=con)
            if top and spec:  # the item's (or assembly's) native, in the item frame
                native(body, spec, iname, frame=relative((link["pos"], link["quat"]), ([0, 0, 0], [1, 0, 0, 0])),
                       con=con)
            for s in link["sensors"]:
                ET.SubElement(body, "site", name=f"{iname}/{s['name']}", pos=f(s["pos"]), quat=f(s["quat"]),
                              size="0.01")
                if s["type"] == "camera":
                    ET.SubElement(body, "camera", name=f"{iname}/{s['name']}", pos=f(s["pos"]),
                                  xyaxes=camera_xyaxes(s["quat"]), fovy="70")
            for c in kids.get(link["name"], []):
                build(body, c, False)

        build(world, kids[None][0], True)
        if role:
            side_items[iname] = info
        for link in asm["links"]:
            for s in link["sensors"]:
                if s["type"] == "imu":
                    sen = m.find("sensor")
                    if sen is None:
                        sen = ET.SubElement(m, "sensor")
                    site = f"{iname}/{s['name']}"
                    ET.SubElement(sen, "framequat", name=f"{site}/quat", objtype="site", objname=site)
                    ET.SubElement(sen, "gyro", name=f"{site}/gyro", site=site)
                    ET.SubElement(sen, "accelerometer", name=f"{site}/acc", site=site)
    if len(actuators):
        m.append(actuators)
    named_cameras(world, ir)

    # a joint's parent and child links never collide (URDF / PhysX semantics). MuJoCo filters
    # parent-child contacts itself, but not when the parent is welded to the world (a fixed_base
    # root), where touching hulls would jam the joint: say it for every pair
    if excludes:
        contact = ET.SubElement(m, "contact")
        for b1, b2 in excludes:
            ET.SubElement(contact, "exclude", body1=b1, body2=b2)

    # mimic joints: MuJoCo's joint equality q1 = polycoef(q2) = offset + multiplier * q2 (q2 the leader)
    eq = None
    for jn, lead, mult, off in mimic_eqs:
        if lead not in made:  # the leader holds still (a root joint the item's body replaced)
            continue
        if eq is None:
            eq = ET.SubElement(m, "equality")
        ET.SubElement(eq, "joint", name=f"orb:mimic:{jn}", joint1=jn, joint2=lead, polycoef=f([off, mult, 0, 0, 0]))

    # attach records -> welds (inactive; the player switches them)
    welds = []
    if motion is not None and tl.get("physics", True):
        for k, a in enumerate(tl.get("attach") or []):
            item, carrier = side_items.get(a.get("item")), side_items.get(a.get("to"))
            if item is None or carrier is None or item["body"] != "dynamic":
                continue
            cbody = carrier["root"]
            if carrier["kind"] == "assembly":
                cbody = carrier["links"].get(a.get("link")) or carrier["root"]
            # the payload in the carrier link frame (no offset: the link origin, as the evaluator reads it)
            rel = (list(a.get("pos") or [0.0, 0.0, 0.0]), list(a.get("quat") or [1.0, 0.0, 0.0, 0.0]))
            if item["kind"] == "assembly":  # the weld holds the payload's root body, not its item frame
                rel = compose(rel, item["root_offset"])
            if eq is None:
                eq = ET.SubElement(m, "equality")
            name = f"orb:attach:{k}"
            ET.SubElement(eq, "weld", name=name, body1=cbody, body2=item["root"], active="false",
                          relpose=f(list(rel[0]) + list(rel[1])),
                          solref=f([2 * timestep, 1]), solimp="0.95 0.99 0.001")
            welds.append({"name": name, "item": a["item"], "to": a["to"], "link": a.get("link"), "carrier": cbody,
                          "t0": float(a["t0"]), "t1": None if a.get("t1") is None else float(a["t1"]),
                          "relpose": [list(rel[0]), list(rel[1])]})

    memory = memory_setting(cfg)
    if memory:  # set by hand
        m.insert(2, ET.Element("size", memory=memory))
    ET.SubElement(ET.SubElement(m, "keyframe"), "key", name="initial", qpos=f(qpos), **({"ctrl": f(ctrl)} if ctrl else {}))
    ET.indent(m)
    (out / "scene.xml").write_text(ET.tostring(m, encoding="unicode") + "\n")
    model = compiled(out / "scene.xml")
    extent = None if model is None else float(model.stat.extent)
    dirty = False
    if extent and f(near_fraction(extent)) != clip.get("znear"):  # the compiler's extent: znear is exactly NEAR metres
        clip.set("znear", f(near_fraction(extent)))
        dirty = True
    if not memory:  # a site of thousands of units outgrows MuJoCo's default arena: size it (never smaller)
        counts = (model.nbody, model.ngeom, model.nmeshvert, model.nmeshface) if model else tree_counts(m, out)
        need = arena_mib(*counts)
        if need > max(ARENA_FLOOR_MIB, model.narena / 2 ** 20 if model else 0):
            el = ET.Element("size", memory=f"{need}M")
            el.tail = m[1].tail
            m.insert(2, el)
            dirty = True
    if dirty:
        (out / "scene.xml").write_text(ET.tostring(m, encoding="unicode") + "\n")
    if lk["shadow_clip"] or lk["haze"]:  # MuJoCo's shadowclip and fog are fractions of the extent: metres here
        extent = extent or rough_extent(ir)
        if lk["shadow_clip"]:
            clip.set("shadowclip", f(lk["shadow_clip"] / extent))
        if lk["haze"]:
            clip.set("fogstart", f(lk["haze"][0] / extent))
            clip.set("fogend", f(lk["haze"][1] / extent))
        (out / "scene.xml").write_text(ET.tostring(m, encoding="unicode") + "\n")
    written = [out / "scene.xml"] + [out / "assets" / n for n in files] + natives
    side = out / SIDECAR
    if motion is not None:
        side.write_text(json.dumps(sidecar(ir, motion, tl, side_items, welds, timestep), indent=1) + "\n")
        written.append(side)
    elif side.exists():
        side.unlink()
    return written


def named_cameras(world, ir):
    """The project's `camera` blocks as fixed MJCF cameras named `camera/<name>`: the MuJoCo viewer
    lists them and `orb run mujoco` renders each (fov is vertical, as MuJoCo's fovy)."""
    for name, c in (ir.get("cameras") or {}).items():
        eye, aim = c["pos"], c["look_at"]
        fwd = [a - e for a, e in zip(aim, eye)]
        n = math.sqrt(sum(x * x for x in fwd))
        if n < 1e-9:
            continue
        fwd = [x / n for x in fwd]
        right = [fwd[1], -fwd[0], 0.0]  # fwd x z
        if math.hypot(*right[:2]) < 1e-6:  # straight down or up: screen-up is +y
            right = [1.0, 0.0, 0.0]
        rn = math.sqrt(sum(x * x for x in right))
        right = [x / rn for x in right]
        up = [right[1] * fwd[2] - right[2] * fwd[1], right[2] * fwd[0] - right[0] * fwd[2],
              right[0] * fwd[1] - right[1] * fwd[0]]
        ET.SubElement(world, "camera", name=f"camera/{name}", mode="fixed", pos=f(eye), xyaxes=f(right + up),
                      fovy=f([math.degrees(c.get("fov") or math.radians(45))]))


def sidecar(ir, motion, tl, items, welds, timestep):
    """What orb.mjplay needs without the project: the timeline (normalized, plus the IR the
    evaluator reads, minus geometry) and where each item lives in the MuJoCo model."""
    keep = ("orb", "root", "timelines", "cameras", "shots", "items", "seams")
    lean = {k: ir[k] for k in keep if k in ir}
    lean["parts"] = {k: {kk: vv for kk, vv in p.items() if kk != "geoms"} for k, p in ir.get("parts", {}).items()}
    lean["assemblies"] = {k: {**a, "links": [{kk: vv for kk, vv in ln.items() if kk != "geoms"} for ln in a["links"]]}
                          for k, a in ir.get("assemblies", {}).items()}
    lean["sites"] = {k: {kk: vv for kk, vv in s.items() if kk != "fixtures"} for k, s in ir.get("sites", {}).items()}
    return {"orb_motion": 1, "timeline": motion.name, "duration": float(tl["duration"]),
            "loop": bool(tl.get("loop", True)), "physics": bool(tl.get("physics", True)), "timestep": timestep,
            "items": items, "welds": welds, "ir": lean}


# --------------------------------------------------------------------------- native MJCF

NATIVE_ROOT = "orb_native"  # the body a copied native MJCF's chosen subtree is wrapped in
ASSET_DIRS = {"mesh": "meshdir", "skin": "meshdir", "texture": "texturedir", "hfield": "assetdir"}
DROP_SECTIONS = ("actuator", "sensor", "tendon", "equality", "contact", "keyframe", "custom", "extension")


def native_mjcf(src: Path, dest: Path, name, body, visual, scale, con, moving):
    """Copy the native MJCF `src` to dest/<name>.xml (+ dest/<name>/ for every file it uses) as a
    model whose world holds one body, NATIVE_ROOT, wrapping `body` (default: everything in its
    world): joints dropped (it rides rigidly on its owner), masses zero (the owner's IR inertia
    counts), uniformly scaled, visual only when `visual` (else colliding like its owner)."""
    src = Path(src).resolve()
    if not src.is_file():
        raise OrbError(f"native \"mujoco\" mjcf not found: {src}")
    try:
        tree = ET.parse(src)
    except ET.ParseError as e:
        raise OrbError(f"native \"mujoco\" mjcf {src}: {e}") from None
    top = tree.getroot()
    _inline_includes(top, src.parent)
    comp = top.find("compiler")
    dirs = {k: (comp.get(k) if comp is not None else None) for k in ("meshdir", "texturedir", "assetdir")}
    files_dir = dest / name
    files_dir.mkdir(parents=True, exist_ok=True)
    taken = set()
    for el in top.iter():
        if el.get("file") is None or el.tag in ("include",):
            continue
        sub = dirs.get(ASSET_DIRS.get(el.tag, "assetdir")) or dirs.get("assetdir") or ""
        p = Path(el.get("file"))
        path = p if p.is_absolute() else (src.parent / sub / p)
        if not path.is_file():
            path = src.parent / p
        if not path.is_file():
            raise OrbError(f"native \"mujoco\" mjcf {src}: {el.tag} file {el.get('file')} not found")
        fn, k = path.name, 1
        while fn in taken:
            fn, k = f"{path.stem}_{k}{path.suffix}", k + 1
        taken.add(fn)
        shutil.copy(path, files_dir / fn)
        el.set("file", f"{name}/{fn}")
        if el.tag == "mesh" and scale != 1.0:
            s = [float(x) for x in (el.get("scale") or "1 1 1").split()]
            el.set("scale", f([x * scale for x in s]))
    if comp is not None:
        for k in ("meshdir", "texturedir", "assetdir"):
            comp.attrib.pop(k, None)
    for sec in DROP_SECTIONS:
        for el in top.findall(sec):
            top.remove(el)
    wb = top.find("worldbody")
    if wb is None:
        raise OrbError(f"native \"mujoco\" mjcf {src} has no <worldbody>")
    if body:
        found = next((b for b in wb.iter("body") if b.get("name") == body), None)
        if found is None:
            have = [b.get("name") for b in wb.iter("body") if b.get("name")]
            raise OrbError(f"native \"mujoco\" mjcf {src} has no body {body!r} (has: {', '.join(have) or 'none'})")
        chosen = [found]
        found.attrib.pop("pos", None)  # the native's own at/rpy place it, not its pose in its source world
        for k in ("quat", "euler", "axisangle", "xyaxes", "zaxis"):
            found.attrib.pop(k, None)
    else:
        chosen = [c for c in wb if c.tag in ("body", "geom", "frame", "site", "camera", "replicate")]
    for c in list(wb):
        wb.remove(c)
    wrap = ET.SubElement(wb, "body", name=NATIVE_ROOT)
    for c in chosen:
        wrap.append(c)
    for el in list(wrap.iter()):
        for kid in list(el):
            if kid.tag in ("joint", "freejoint", "inertial"):
                el.remove(kid)
        if scale != 1.0 and el is not wrap:
            for k in ("pos", "fromto"):
                if el.get(k):
                    el.set(k, f([float(x) * scale for x in el.get(k).split()]))
            if el.tag == "geom" and el.get("size") and el.get("type", "sphere") != "mesh":
                el.set("size", f([float(x) * scale for x in el.get("size").split()]))
        if el.tag == "geom":
            el.set("mass", "0")
            el.attrib.pop("density", None)
            if visual:
                el.set("contype", "0")
                el.set("conaffinity", "0")
                el.set("group", "2")
            elif con == KIN:
                el.set("contype", "2")
                el.set("conaffinity", "0")
            elif con == DYN and moving:
                el.set("conaffinity", "3")
    dest.mkdir(parents=True, exist_ok=True)
    fn = dest / f"{name}.xml"
    ET.indent(top)
    fn.write_text(ET.tostring(top, encoding="unicode") + "\n")
    return fn


def _inline_includes(el, base: Path, depth=0):
    """Replace every <include file="..."/> with the children of that file's <mujoco>."""
    if depth > 8:
        raise OrbError("native \"mujoco\" mjcf: <include> nested too deep")
    for i, kid in enumerate(list(el)):
        if kid.tag == "include" and kid.get("file"):
            path = (base / kid.get("file")).resolve()
            if not path.is_file():
                raise OrbError(f"native \"mujoco\" mjcf: include {path} not found")
            inc = ET.parse(path).getroot()
            _inline_includes(inc, path.parent, depth + 1)
            idx = list(el).index(kid)
            el.remove(kid)
            for j, c in enumerate(list(inc)):
                el.insert(idx + j, c)
        else:
            _inline_includes(kid, base, depth + 1)
