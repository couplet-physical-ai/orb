"""Timelines: what moves, carries, appears and loops (contract v0.3, section 4).

`Motion(ir, timeline)` evaluates one normalized timeline of the world graph (the IR the core
compiles every bit of OrbLang sugar into: keys, attach intervals, visibility steps, channels,
phases, events, bodies). Pure Python; every pose is in the world frame, SI units, quaternions
[w, x, y, z].

    m = Motion(ir)                          # the first timeline (or Motion(ir, "assembly_line"))
    m.pose("crate-payload", 80.0)                # -> (pos, quat), follows its carrier while attached
    m.bake(30)                              # {item: [(t, pos, quat), ...]} for playback seams
    m.to_json(30)                           # what `orb motion --json --bake 30` prints

Semantics: before the first key an item holds its first key (or its static pose without a
track); after the last it holds the last. Position lerps, rotation slerps (shortest path); a
key's `ease` shapes the segment that ends at it (linear | smooth = smoothstep | step = jump at
the key). An attached item rides its carrier's link frame (carrier pose, forward kinematics at
the carrier's joint values, then the attach offset) on [t0, t1); once detached it holds the
pose it had at the detach (the nominal pose: physics seams let it fall from there). A looping
timeline wraps at `duration`.
"""
from __future__ import annotations

import bisect
import math

from .lang import OrbError
from .world import frames, mimics, qmul, qrot, with_mimics

EASES = ("linear", "smooth", "step")
BODIES = ("static", "kinematic", "dynamic")
DEFAULT_FOV = math.radians(60)
IDENTITY = [1.0, 0.0, 0.0, 0.0]
MAX_CHAIN = 32  # carriers of carriers of ...: deeper than this is a cycle
# where bake() checks a segment between two key times for linearity (irrational-ish fractions:
# a joint turning a whole number of turns cannot hide from all of them)
SEGMENT_PROBES = (0.1180, 0.25, 0.3820, 0.5, 0.6180, 0.75, 0.8820)


# --------------------------------------------------------------------------- small math


def lerp(a, b, u):
    return [x + (y - x) * u for x, y in zip(a, b)]


def qnorm(q):
    n = math.sqrt(sum(x * x for x in q)) or 1.0
    return [x / n for x in q]


def qconj(q):
    return [q[0], -q[1], -q[2], -q[3]]


def slerp(a, b, u):
    """Shortest-path spherical interpolation."""
    d = sum(x * y for x, y in zip(a, b))
    if d < 0:
        b, d = [-x for x in b], -d
    if d > 0.9995:
        return qnorm(lerp(a, b, u))
    th = math.acos(min(1.0, d))
    s = math.sin(th)
    wa, wb = math.sin((1 - u) * th) / s, math.sin(u * th) / s
    return [wa * x + wb * y for x, y in zip(a, b)]


def qangle(a, b):
    """Angle (rad) between two orientations."""
    d = abs(sum(x * y for x, y in zip(a, b)))
    return 2 * math.acos(min(1.0, d))


def yaw_of(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def compose(a, b):
    """Pose a ∘ b: b expressed in a's frame -> world."""
    (pa, qa), (pb, qb) = a, b
    return [x + y for x, y in zip(pa, qrot(qa, pb))], qnorm(qmul(qa, qb))


def inverse(p):
    pos, q = p
    qi = qconj(q)
    return [-x for x in qrot(qi, pos)], qi


def ease_u(u, ease):
    if ease == "smooth":
        return u * u * (3 - 2 * u)
    if ease == "step":
        return 0.0
    return u


def segment(times, t):
    """(i, j, u) for t on a sorted key-time list: keys i..j with raw fraction u (i == j: hold)."""
    n = len(times)
    k = bisect.bisect_right(times, t) - 1
    if k < 0:
        return 0, 0, 0.0
    if k >= n - 1:
        return n - 1, n - 1, 0.0
    t0, t1 = times[k], times[k + 1]
    return k, k + 1, (t - t0) / (t1 - t0) if t1 > t0 else 1.0


def timeline_name(ir, seam=None):
    """The timeline a seam plays: `seam "x" { timeline = "..." }`, else the first; None if the
    project has none."""
    tls = ir.get("timelines") or {}
    want = ((ir.get("seams") or {}).get(seam) or {}).get("timeline") if seam else None
    if isinstance(want, str):
        want = want.removeprefix("timeline.")
        return want if want in tls else None
    return next(iter(tls), None)


# --------------------------------------------------------------------------- the evaluator


class Motion:
    def __init__(self, ir, timeline=None):
        tls = ir.get("timelines") or {}
        if not tls:
            raise OrbError("this project has no timeline")
        if timeline is None:
            timeline = next(iter(tls))
        timeline = str(timeline).removeprefix("timeline.")
        if timeline not in tls:
            raise OrbError(f"no timeline {timeline!r} (have: {', '.join(tls)})")
        tl = tls[timeline]
        self.ir, self.name, self.tl = ir, timeline, tl
        self.duration = float(tl["duration"])
        self.loop = bool(tl.get("loop", True))
        self.physics = bool(tl.get("physics", True))
        self.label = tl.get("label") or timeline
        self.items_ir = ir.get("items") or {}
        self.assemblies = ir.get("assemblies") or {}
        self._tracks = {k: sorted(v, key=lambda e: e["t"]) for k, v in (tl.get("tracks") or {}).items() if v}
        self._track_t = {k: [e["t"] for e in v] for k, v in self._tracks.items()}
        self._joint_keys = {}  # item -> joint -> ([t], [value], [ease])
        for item, keys in (tl.get("joints") or {}).items():
            per = {}
            for k in sorted(keys, key=lambda e: e["t"]):
                for j, v in (k.get("values") or {}).items():
                    ts, vs, es = per.setdefault(j, ([], [], []))
                    ts.append(float(k["t"]))
                    vs.append(float(v))
                    es.append(k.get("ease", "linear"))
            if per:
                self._joint_keys[item] = per
        self.attach_records = sorted(tl.get("attach") or [], key=lambda r: (r["t0"], r["item"]))
        self._attach = {}
        for r in self.attach_records:
            self._attach.setdefault(r["item"], []).append(r)
        self._visible = {k: sorted(v, key=lambda e: e[0]) for k, v in (tl.get("visible") or {}).items() if v}
        self._channels = {k: sorted(v, key=lambda e: e[0]) for k, v in (tl.get("channels") or {}).items() if v}
        self.phases = sorted(tl.get("phases") or [], key=lambda p: p["t0"])
        self.events = sorted(tl.get("events") or [], key=lambda e: e["t"])
        self.bodies = dict(tl.get("bodies") or {})
        self._fk = {}

    # ---------------------------------------------------------------- time
    def wrap(self, t):
        t = float(t)
        if self.loop and self.duration > 0:
            return t % self.duration
        return min(max(t, 0.0), self.duration)

    def clamp(self, t):
        return min(max(float(t), 0.0), self.duration)

    # ---------------------------------------------------------------- what is touched
    def items(self):
        out = dict.fromkeys(self.bodies)
        for src in (self._tracks, self._joint_keys, self._visible):
            out.update(dict.fromkeys(src))
        for r in self.attach_records:
            out.update(dict.fromkeys((r["item"], r["to"])))
        return list(out)

    def body(self, item):
        b = self.bodies.get(item)
        if b is None:
            b = "kinematic" if item in self._tracks or item in self._joint_keys else \
                ("dynamic" if item in self._attach else "static")
        return "kinematic" if b == "dynamic" and not self.physics else b

    # ---------------------------------------------------------------- poses
    def static_pose(self, item):
        it = self.items_ir.get(item) or {}
        return list(it.get("pos") or [0.0, 0.0, 0.0]), list(it.get("quat") or IDENTITY)

    def pose(self, item, t):
        """World pose (pos, quat) of `item` at timeline time t (wrapped when looping)."""
        return self.pose_at(item, self.wrap(t))

    def pose_at(self, item, t, depth=0):
        """Like pose(), at an already wrapped/clamped t (t == duration is the end state)."""
        if depth > MAX_CHAIN:
            raise OrbError(f"timeline {self.name}: attach cycle through {item!r}")
        rec = self._active(item, t)
        if rec is not None:
            return self._carried(rec, t, depth)
        if item in self._tracks:
            return self._track_pose(item, t)
        last = self._last_detached(item, t)
        if last is not None:
            return self._carried(last, last["t1"], depth)
        return self.static_pose(item)

    def _carried(self, rec, t, depth):
        frame = self.link_pose(rec["to"], rec.get("link"), t, depth + 1)
        return compose(frame, (rec.get("pos") or [0.0, 0.0, 0.0], rec.get("quat") or IDENTITY))

    def link_pose(self, item, link, t, depth=0):
        """World pose of `item`'s link `link` (its own frame when link is None or it is a part)."""
        base = self.pose_at(item, t, depth)
        if not link:
            return base
        it = self.items_ir.get(item) or {}
        asm = self.assemblies.get(it.get("of")) if it.get("kind") == "assembly" else None
        if asm is None:
            return base
        key = (item, t)
        fk = self._fk.get(key)
        if fk is None:
            if len(self._fk) > 4096:
                self._fk.clear()
            fk = self._fk[key] = frames(asm, self.joints_at(item, t))
        if link not in fk:
            raise OrbError(f"timeline {self.name}: {item} ({it.get('of')}) has no link {link!r}")
        return compose(base, fk[link])

    def _track_pose(self, item, t):
        keys = self._tracks[item]
        i, j, u = segment(self._track_t[item], t)
        a = keys[i]
        if i == j:
            return list(a["pos"]), list(a["quat"])
        b = keys[j]
        u = ease_u(u, b.get("ease", "linear"))
        return lerp(a["pos"], b["pos"], u), slerp(a["quat"], b["quat"], u)

    # ---------------------------------------------------------------- joints
    def joints(self, item, t):
        """Every movable joint of an assembly item at time t: initial values, then the item's
        static `joints`, then the timeline's joint track."""
        return self.joints_at(item, self.wrap(t))

    def joints_at(self, item, t):
        it = self.items_ir.get(item) or {}
        asm = self.assemblies.get(it.get("of")) if it.get("kind") == "assembly" else None
        out = {}
        if asm:
            for link in asm["links"]:
                j = link["joint"]
                if j["type"] in ("revolute", "prismatic"):
                    out[j["name"]] = float(j.get("initial", 0.0))
        out.update({k: float(v) for k, v in (it.get("joints") or {}).items()})
        for j, (ts, vs, es) in (self._joint_keys.get(item) or {}).items():
            i, k, u = segment(ts, t)
            out[j] = vs[i] if i == k else vs[i] + (vs[k] - vs[i]) * ease_u(u, es[k])
        return with_mimics(asm, out) if asm else out

    # ---------------------------------------------------------------- attach / visibility / channels
    def _active(self, item, t):
        for r in self._attach.get(item, ()):
            if r["t0"] <= t and (r.get("t1") is None or t < r["t1"]):
                return r
        return None

    def _last_detached(self, item, t):
        last = None
        for r in self._attach.get(item, ()):
            if r.get("t1") is not None and r["t1"] <= t:
                last = r
        return last

    def attached(self, item, t):
        """The attach record active at t ({item, to, link, t0, t1, pos, quat}), or None."""
        r = self._active(item, self.wrap(t))
        return dict(r) if r else None

    def visible(self, item, t):
        steps = self._visible.get(item)
        if not steps:
            return True
        t = self.wrap(t)
        k = bisect.bisect_right([s[0] for s in steps], t) - 1
        return (not bool(steps[0][1])) if k < 0 else bool(steps[k][1])

    def visibility(self, item):
        """[[t, bool], ...] from t = 0: the item's visibility steps with its initial state first."""
        steps = self._visible.get(item)
        if not steps:
            return [[0.0, True]]
        out = [[float(s[0]), bool(s[1])] for s in steps]
        if out[0][0] > 0:
            out.insert(0, [0.0, not out[0][1]])
        return out

    def channel(self, name, t):
        steps = self._channels.get(name)
        if not steps:
            return None
        k = bisect.bisect_right([s[0] for s in steps], self.wrap(t)) - 1
        return None if k < 0 else float(steps[k][1])

    def channels(self):
        return list(self._channels)

    def phase(self, t):
        t = self.wrap(t)
        for p in self.phases:
            t1 = p.get("t1", self.duration)
            if p["t0"] <= t < t1 or (t == self.duration and t1 == self.duration and p["t0"] <= t):
                return p
        return None

    # ---------------------------------------------------------------- sampling
    def key_times(self, item, _depth=0):
        """Every time something about `item` changes: its keys, joint keys, attach/detach and
        visibility steps, and while it rides a carrier the carrier's key times."""
        ts = set(self._track_t.get(item, ()))
        for ts_, _, _ in (self._joint_keys.get(item) or {}).values():
            ts.update(ts_)
        ts.update(s[0] for s in self._visible.get(item, ()))
        for r in self._attach.get(item, ()):
            t1 = self.duration if r.get("t1") is None else r["t1"]
            ts.update((r["t0"], t1))
            if _depth < MAX_CHAIN:
                ts.update(x for x in self.key_times(r["to"], _depth + 1) if r["t0"] <= x <= t1)
        return sorted(float(x) for x in ts if 0 <= x <= self.duration)

    def _grid(self, fps, item):
        n = max(1, int(math.ceil(self.duration * fps - 1e-9)))
        ts = {min(i / fps, self.duration) for i in range(n + 1)}
        ts.update(self.key_times(item))
        ts.add(self.duration)
        return sorted(ts)

    def _sampled(self, keys, fps, at, err, tol):
        """(t, *at(t)) samples over [0, duration] for thin(): every key time, and between two key
        times (one eased segment of every channel) the `fps` grid only where linear interpolation
        of the two ends misses a probe by tol / 2. A hold or a linear move costs a few probes,
        not duration * fps samples (a 20-minute timeline is ~30k grid points per item)."""
        keys = sorted({0.0, self.duration, *(float(k) for k in keys if 0 <= k <= self.duration)})
        out = [(keys[0], *at(keys[0]))]
        for a, b in zip(keys, keys[1:]):
            sb = (b, *at(b))
            lo, hi = int(math.floor(a * fps + 1e-9)) + 1, int(math.ceil(b * fps - 1e-9)) - 1
            grid = [g for g in (k / fps for k in range(lo, hi + 1)) if a < g < b]
            if grid:
                sa = out[-1]
                probes = (a + (b - a) * f for f in SEGMENT_PROBES)
                if any(err(sa, sb, (t, *at(t))) > tol / 2 for t in probes):
                    out += [(t, *at(t)) for t in grid]
            out.append(sb)
        return out

    def bake(self, fps, tol=1e-4):
        """{item: [(t, pos, quat), ...]}: every touched item sampled at `fps` (plus its key
        times) over [0, duration], samples that linear interpolation reproduces within `tol`
        (metres and radians) thinned out. Nominal poses: what physics-less playback shows."""
        out = {}
        for item in self.items():
            samples = self._sampled(self.key_times(item), fps, lambda t, i=item: self.pose_at(i, t), _pose_err, tol)
            out[item] = thin(samples, _pose_err, tol)
        return out

    def bake_joints(self, fps, tol=1e-5):
        """{item: [(t, {joint: value}), ...]} for the items with a joint track (their tracked
        joints), thinned like bake()."""
        out = {}
        for item, per in self._joint_keys.items():
            names = list(per)
            it = self.items_ir.get(item) or {}
            asm = self.assemblies.get(it.get("of")) if it.get("kind") == "assembly" else None
            if asm:  # a tracked leader moves its mimic joints too
                names += [n for n, m in mimics(asm).items() if m["joint"] in per and n not in per]

            def at(t, item=item, names=names):
                v = self.joints_at(item, t)
                return ({j: v[j] for j in names},)

            samples = self._sampled(self.key_times(item), fps, at, _joint_err, tol)
            out[item] = thin(samples, _joint_err, tol)
        return out

    # ---------------------------------------------------------------- cameras
    def shots(self):
        return [s for s in self.ir.get("shots") or [] if s.get("timeline") in (None, self.name)]

    def shot_camera(self, shot, t):
        """(pos, look_at, fov) of a shot at timeline time t; follow shots track their item."""
        if isinstance(shot, str):
            found = [s for s in self.ir.get("shots") or [] if s.get("name") == shot]
            if not found:
                raise OrbError(f"no shot {shot!r}")
            shot = found[0]
        cam = (self.ir.get("cameras") or {}).get(shot.get("camera")) if shot.get("camera") else None
        fov = shot.get("fov") or (cam or {}).get("fov") or DEFAULT_FOV
        if shot.get("follow"):
            p, q = self.pose(shot["follow"], t)
            off = shot.get("offset")
            if off is not None:
                if shot.get("relative"):
                    yw = yaw_of(q)
                    c, s = math.cos(yw), math.sin(yw)
                    off = [c * off[0] - s * off[1], s * off[0] + c * off[1], off[2]]
                return [a + b for a, b in zip(p, off)], list(p), fov
            if cam is not None:
                return list(cam["pos"]), list(p), fov
            if shot.get("pos") is not None:
                return list(shot["pos"]), list(p), fov
            return [p[0] - 4.0, p[1] - 4.0, p[2] + 3.0], list(p), fov
        if cam is not None:
            return list(cam["pos"]), list(cam["look_at"]), fov
        return list(shot["pos"]), list(shot.get("look_at") or [0.0, 0.0, 0.0]), fov

    def camera_at(self, shot, t, shots=None, _ref=None):
        """(eye, look, fov) of a shot at t with its camera move: shot_camera, then `lag` (s: a
        follow camera trails its item, averaged over that window), `offset_to` (a follow shot's
        offset at the end), `move_to` (the eye at the end), `look_to`, `look_offset` (added to the
        look-at point), `fov_to`, and `orbit` (rad: the eye turns that far about the vertical
        through the look-at point over the shot, counter-clockwise seen from above). `ease =
        "smooth"` eases every move (default linear). Then `blend` (s): for that long after the shot
        takes over (its t0, or when an overlapping shot it was under ends) the camera eases
        (smoothstep) from the shot it takes over from (blend_cameras), that shot evaluated at the same
        t (its move held at its end, a follow camera still following). `shots`: the shots the
        player cuts between (default this timeline's; a --shot plate passes just its own). Every
        player and video uses this."""
        if isinstance(shot, str):
            shot = next((s for s in self.ir.get("shots") or [] if s.get("name") == shot), None) or \
                self.shot_camera(shot, t)
        cam = self._move(shot, t)
        blend = float(shot.get("blend") or 0.0)
        if blend <= 0:
            return cam
        shots = self.shots() if shots is None else shots
        if not any(s is shot or s.get("name") == shot.get("name") for s in shots):
            return cam
        entry, prev, prev_ref = self._entry(shot, t if _ref is None else min(t, _ref), shots)
        if prev is None or not entry - 0.01 <= t < entry + blend:  # (a player picking shots half a step early)
            return cam
        u = min(max((t - entry) / blend, 0.0), 1.0)
        return blend_cameras(self.camera_at(prev, t, shots, prev_ref), cam, u * u * (3 - 2 * u))

    def active_shot(self, t, shots=None):
        """The shot covering t, or None; where shots overlap, the one that started last (a close-up
        inside a long master shot wins while it runs). Every player cuts by this."""
        return active_shot(self.shots() if shots is None else shots, t)

    def _entry(self, shot, t, shots):
        """(entry, previous shot, its reference time) of the run of `shot` that covers (or last
        covered) t: entry is when it took over, the previous shot the one it took over from (across a
        gap: the one active when the gap began), None at the very first shot."""
        eps = 1e-6
        t0 = float(shot.get("t0") or 0.0)
        same = lambda s: s is not None and (s is shot or s.get("name") == shot.get("name"))
        marks = {t0} | {float(x) for s in shots for x in (s.get("t0"), s.get("t1"))
                        if x is not None and t0 < float(x) <= t + 1e-9}
        entry = t0
        for x in sorted(marks, reverse=True):
            if same(self.active_shot(x, shots)) and not same(self.active_shot(x - eps, shots)):
                entry = x
                break
        prev = self.active_shot(entry - eps, shots)
        if prev is not None and not same(prev):
            return entry, prev, entry - eps
        ends = [float(s["t1"]) for s in shots if s.get("t1") is not None and float(s["t1"]) <= entry + 1e-9
                and not same(s)]
        if not ends:
            return entry, None, None
        gap = max(ends)
        prev = self.active_shot(gap - eps, shots)
        return (entry, prev, gap - eps) if prev is not None and not same(prev) else (entry, None, None)

    def view(self, t, shots=None):
        """(shot, (eye, look, fov)) a player shows at t, or (None, None) for its own overview: the
        active shot with its move and blend; in a gap before a shot that blends, the shot before the
        gap held (its move at its end, a follow camera still following), so the next one blends from
        it instead of cutting to the overview and back."""
        shots = self.shots() if shots is None else shots
        shot = self.active_shot(t, shots)
        if shot is not None:
            return shot, self.camera_at(shot, t, shots)
        nxt = min((s for s in shots if float(s.get("t0") or 0.0) > t), key=lambda s: float(s.get("t0") or 0.0),
                  default=None)
        if nxt is None or not float(nxt.get("blend") or 0.0) > 0:
            return None, None
        ends = [float(s["t1"]) for s in shots if s.get("t1") is not None and float(s["t1"]) <= t]
        prev = self.active_shot(max(ends) - 1e-6, shots) if ends else None
        if prev is None:
            return None, None
        return prev, self.camera_at(prev, t, shots, max(ends) - 1e-6)

    def _move(self, shot, t):
        """camera_at without the blend: the shot's own camera and move."""
        t0, t1 = float(shot.get("t0") or 0.0), float(shot.get("t1") or t)
        u = min(max((t - t0) / max(t1 - t0, 1e-9), 0.0), 1.0)
        if shot.get("ease") == "smooth":
            u = u * u * (3 - 2 * u)

        def lerp(a, b):
            return [x + (y - x) * u for x, y in zip(a, b)] if b is not None else list(a)

        s = dict(shot)
        if s.get("follow") and s.get("offset") is not None and s.get("offset_to") is not None:
            s["offset"] = lerp(s["offset"], s["offset_to"])
        lag = float(shot.get("lag") or 0.0)
        ts = [t - lag * k / 8.0 for k in range(9)] if lag > 0 and s.get("follow") else [t]
        cams = [self.shot_camera(s, max(x, 0.0)) for x in ts]
        eye = [sum(c[0][i] for c in cams) / len(cams) for i in range(3)]
        look = [sum(c[1][i] for c in cams) / len(cams) for i in range(3)]
        fov = cams[0][2]
        if shot.get("move_to") is not None:
            eye = lerp(eye, shot["move_to"])
        if shot.get("look_to") is not None:
            look = lerp(look, shot["look_to"])
        if shot.get("look_offset") is not None:
            look = [a + b for a, b in zip(look, shot["look_offset"])]
        if shot.get("fov_to") is not None:
            fov = fov + (float(shot["fov_to"]) - fov) * u
        if shot.get("orbit"):
            a = float(shot["orbit"]) * u
            c, sn = math.cos(a), math.sin(a)
            dx, dy = eye[0] - look[0], eye[1] - look[1]
            eye = [look[0] + c * dx - sn * dy, look[1] + sn * dx + c * dy, eye[2]]
        return eye, look, fov

    # ---------------------------------------------------------------- state / json
    def state(self, t):
        """Every touched item at time t: pose, attached to, visible, joints (`orb motion --at`)."""
        tw = self.wrap(t)
        out = {}
        for item in self.items():
            pos, quat = self.pose_at(item, tw)
            r = self._active(item, tw)
            st = {"body": self.body(item), "pos": pos, "quat": quat, "visible": self.visible(item, tw),
                  "attached": None if r is None else {"to": r["to"], "link": r.get("link")}}
            if (self.items_ir.get(item) or {}).get("kind") == "assembly":
                st["joints"] = self.joints_at(item, tw)
            out[item] = st
        return out

    def to_json(self, fps=None):
        """What `orb motion --json [--bake fps]` prints (contract 4e)."""
        r6 = lambda v: [round(x, 6) for x in v]
        out = {"timeline": self.name, "label": self.label, "duration": self.duration, "loop": self.loop,
               "physics": self.physics, "fps": fps, "phases": self.phases, "events": self.events,
               "shots": self.shots(), "cameras": self.ir.get("cameras") or {},
               "channels": {k: [[float(a), float(b)] for a, b in v] for k, v in self._channels.items()},
               "items": {}}
        poses = self.bake(fps) if fps else {}
        joints = self.bake_joints(fps) if fps else {}
        for item in self.items():
            it = {"body": self.body(item), "visible": self.visibility(item),
                  "attach": [dict(r) for r in self._attach.get(item, ())], "key_times": self.key_times(item)}
            if fps:
                s = poses[item]
                it.update(t=[round(x[0], 6) for x in s], pos=[r6(x[1]) for x in s], quat=[r6(x[2]) for x in s])
                if item in joints:
                    js = joints[item]
                    names = list(js[0][1]) if js else []
                    it["joints"] = {"t": [round(x[0], 6) for x in js],
                                    "values": {j: [round(x[1][j], 6) for x in js] for j in names}}
            out["items"][item] = it
        return out


# --------------------------------------------------------------------------- shot blends


def active_shot(shots, t):
    """The shot of `shots` covering t, or None; where shots overlap, the one that started last."""
    best = None
    for s in shots:
        t0, t1 = float(s.get("t0") or 0.0), s.get("t1")
        if t0 <= t < (math.inf if t1 is None else float(t1)) and (best is None or t0 > best[0]):
            best = (t0, s)
    return best[1] if best else None


def blend_cameras(a, b, w):
    """(eye, look, fov) a fraction w of the way from camera a to camera b: the look-at point and fov
    linearly, the eye around the look-at point (yaw the short way round, pitch linearly, distance
    geometrically), so a blend swings round the scene instead of cutting through it. A camera on
    its look-at point falls back to a straight line."""
    (ea, la, fa), (eb, lb, fb) = a, b
    look = [x + (y - x) * w for x, y in zip(la, lb)]
    fov = float(fa) + (float(fb) - float(fa)) * w
    da, db = [x - y for x, y in zip(ea, la)], [x - y for x, y in zip(eb, lb)]
    ra, rb = math.hypot(*da), math.hypot(*db)
    if ra < 1e-9 or rb < 1e-9:
        return [x + (y - x) * w for x, y in zip(ea, eb)], look, fov
    ya, yb = math.atan2(da[1], da[0]), math.atan2(db[1], db[0])
    pa, pb = math.asin(max(-1.0, min(1.0, da[2] / ra))), math.asin(max(-1.0, min(1.0, db[2] / rb)))
    dy = (yb - ya + math.pi) % (2 * math.pi) - math.pi
    yaw, pitch, r = ya + dy * w, pa + (pb - pa) * w, ra * (rb / ra) ** w
    eye = [look[0] + r * math.cos(pitch) * math.cos(yaw), look[1] + r * math.cos(pitch) * math.sin(yaw),
           look[2] + r * math.sin(pitch)]
    return eye, look, fov


# --------------------------------------------------------------------------- thinning


def _pose_err(a, b, s):
    """How far sample s is from the interpolation of a and b at s's time."""
    ta, tb = a[0], b[0]
    u = (s[0] - ta) / (tb - ta) if tb > ta else 0.0
    p = lerp(a[1], b[1], u)
    q = slerp(a[2], b[2], u)
    return max(math.dist(p, s[1]), qangle(q, s[2]))


def _joint_err(a, b, s):
    ta, tb = a[0], b[0]
    u = (s[0] - ta) / (tb - ta) if tb > ta else 0.0
    return max((abs(a[1][j] + (b[1][j] - a[1][j]) * u - s[1][j]) for j in s[1]), default=0.0)


def thin(samples, err, tol):
    """Douglas-Peucker over time: keep the samples linear interpolation cannot reproduce."""
    n = len(samples)
    if n <= 2:
        return list(samples)
    keep = [False] * n
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j - i < 2:
            continue
        worst, at = -1.0, -1
        for k in range(i + 1, j):
            e = err(samples[i], samples[j], samples[k])
            if e > worst:
                worst, at = e, k
        if worst > tol:
            keep[at] = True
            stack += [(i, at), (at, j)]
    return [s for s, k in zip(samples, keep) if k]


# --------------------------------------------------------------------------- checks


def check(ir):
    """Problems with the normalized timelines/cameras/shots of an IR (contract 4e), as messages.
    The compiler already refuses most of these with a source position; this checks any IR,
    hand-written ones included."""
    errs = []
    items, asms = ir.get("items") or {}, ir.get("assemblies") or {}

    def movable(item):
        it = items.get(item) or {}
        asm = asms.get(it.get("of")) if it.get("kind") == "assembly" else None
        return {l["joint"]["name"] for l in asm["links"] if l["joint"]["type"] in ("revolute", "prismatic")} \
            if asm else set()

    def j_asm(item):
        it = items.get(item) or {}
        return it.get("kind") == "assembly" and it.get("of") in asms

    def links(item):
        it = items.get(item) or {}
        asm = asms.get(it.get("of")) if it.get("kind") == "assembly" else None
        return {l["name"] for l in asm["links"]} if asm else set()

    for name, tl in (ir.get("timelines") or {}).items():
        where = f"timeline {name}"
        dur = tl.get("duration")
        if not isinstance(dur, (int, float)) or dur <= 0:
            errs.append(f"{where}: duration must be > 0")
            continue

        def in_range(t, what):
            if not isinstance(t, (int, float)) or not -1e-9 <= t <= dur + 1e-9:
                errs.append(f"{where}: {what} t = {t} is outside [0, {dur:g}] s")

        for item, keys in (tl.get("tracks") or {}).items():
            if item not in items:
                errs.append(f"{where}: track of unknown item {item!r}")
            last = -math.inf
            for k in keys:
                in_range(k.get("t"), f"track {item} key")
                if k.get("t", 0) < last:
                    errs.append(f"{where}: track {item} key times decrease ({last:g} s -> {k['t']:g} s)")
                last = k.get("t", last)
                if k.get("ease", "linear") not in EASES:
                    errs.append(f"{where}: track {item}: ease {k.get('ease')!r} (use {' | '.join(EASES)})")
        for item, keys in (tl.get("joints") or {}).items():
            if item not in items:
                errs.append(f"{where}: joints of unknown item {item!r}")
                continue
            known = movable(item)
            last = -math.inf
            for k in keys:
                in_range(k.get("t"), f"joints {item} key")
                if k.get("t", 0) < last:
                    errs.append(f"{where}: joints {item} key times decrease")
                last = k.get("t", last)
                follows = mimics(asms[items[item]["of"]]) if j_asm(item) else {}
                for j in k.get("values") or {}:
                    if j not in known:
                        errs.append(f"{where}: {item} has no movable joint {j!r}")
                    elif j in follows:
                        errs.append(f"{where}: joints {item}: {j} is a mimic joint (it follows "
                                    f"{follows[j]['joint']}); key {follows[j]['joint']} instead")
        per = {}
        for r in tl.get("attach") or []:
            it, to = r.get("item"), r.get("to")
            if it not in items:
                errs.append(f"{where}: attach of unknown item {it!r}")
            if to not in items:
                errs.append(f"{where}: {it} attached to unknown item {to!r}")
            elif r.get("link") and r["link"] not in links(to):
                errs.append(f"{where}: {it} attached to {to} link {r['link']!r}, which {to} does not have")
            in_range(r.get("t0"), f"attach {it}")
            if r.get("t1") is not None:
                in_range(r["t1"], f"detach {it}")
                if r["t1"] <= r.get("t0", 0):
                    errs.append(f"{where}: {it} detaches ({r['t1']:g} s) before it attaches ({r['t0']:g} s)")
            per.setdefault(it, []).append(r)
        for it, recs in per.items():
            recs = sorted(recs, key=lambda r: r["t0"])
            for a, b in zip(recs, recs[1:]):
                if a.get("t1") is None or a["t1"] > b["t0"]:
                    errs.append(f"{where}: {it} is attached to {a['to']} and {b['to']} at once "
                                f"(from {b['t0']:g} s)")
        for r in tl.get("attach") or []:  # cycles: follow the carriers active at each attach time
            seen, cur, t = {r["item"]}, r.get("to"), r.get("t0", 0)
            while cur is not None:
                if cur in seen:
                    errs.append(f"{where}: attach cycle at {t:g} s: {r['item']} ends up carrying itself")
                    break
                seen.add(cur)
                nxt = [x for x in per.get(cur, ()) if x["t0"] <= t and (x.get("t1") is None or t < x["t1"])]
                cur = nxt[0]["to"] if nxt else None
        for item, steps in (tl.get("visible") or {}).items():
            if item not in items:
                errs.append(f"{where}: show/hide of unknown item {item!r}")
            for s in steps:
                in_range(s[0], f"show/hide {item}")
        for ch, steps in (tl.get("channels") or {}).items():
            for s in steps:
                in_range(s[0], f"set {ch}")
        for p in tl.get("phases") or []:
            in_range(p.get("t0"), f"phase {p.get('name')}")
        for e in tl.get("events") or []:
            in_range(e.get("t"), f"event {e.get('name')}")
        for item, b in (tl.get("bodies") or {}).items():
            if b not in BODIES:
                errs.append(f"{where}: body of {item} is {b!r} (use {' | '.join(BODIES)})")
            if item not in items:
                errs.append(f"{where}: body of unknown item {item!r}")
            elif item in (tl.get("tracks") or {}) and b != "kinematic":
                errs.append(f"{where}: {item} has a track, so its body is kinematic, not {b}")
    cams = ir.get("cameras") or {}
    for s in ir.get("shots") or []:
        where = f"shot {s.get('name')}"
        tl = (ir.get("timelines") or {}).get(s.get("timeline"))
        if s.get("timeline") is not None and tl is None:
            errs.append(f"{where}: unknown timeline {s.get('timeline')!r}")
        if s.get("camera") is not None and s["camera"] not in cams:
            errs.append(f"{where}: unknown camera {s['camera']!r}")
        if s.get("follow") is not None and s["follow"] not in items:
            errs.append(f"{where}: follows unknown item {s['follow']!r}")
        if s.get("camera") is None and s.get("follow") is None and s.get("pos") is None:
            errs.append(f"{where}: needs a camera, a position (at) or an item to follow")
        if tl is not None and not 0 <= s.get("t0", 0) < s.get("t1", 0) <= tl["duration"] + 1e-9:
            errs.append(f"{where}: t = [{s.get('t0')}, {s.get('t1')}] s is not a range inside "
                        f"[0, {tl['duration']:g}] s")
        if s.get("blend") is not None:
            b, span = s["blend"], (s.get("t1") or 0) - (s.get("t0") or 0)
            if not isinstance(b, (int, float)) or not 0 <= b <= span + 1e-9:
                errs.append(f"{where}: blend = {b} s is not a time from 0 s to the shot's length ({span:g} s)")
    return errs
