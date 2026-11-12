"""Pure standard-library camera framing math for Orb scene previews and view builtins."""
import math

APERTURE = 20.955  # mm: the horizontal aperture of a 35 mm-ish camera (USD's default)


def _sub(a, b):
    return [x - y for x, y in zip(a, b)]


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _norm(a):
    n = math.sqrt(_dot(a, a))
    return [x / n for x in a] if n > 1e-12 else list(a)


def percentile(xs, q):
    """Linear-interpolated q-quantile (q in 0..1) of xs."""
    xs = sorted(float(x) for x in xs)
    if not xs:
        raise ValueError("percentile of nothing")
    k = (len(xs) - 1) * min(max(q, 0.0), 1.0)
    i = int(math.floor(k))
    j = min(i + 1, len(xs) - 1)
    return xs[i] + (xs[j] - xs[i]) * (k - i)


def room_box(boxes, ceiling=None, z_cap=3.0, pct=0.9, min_height=0.5):
    """The room to frame: XY union of the fixture boxes [(lo, hi), ...]; Z from the lowest bottom
    to the ceiling height when known, else the pct-quantile of the box tops capped at z_cap."""
    boxes = [(list(lo), list(hi)) for lo, hi in boxes]
    if not boxes:
        return [-2.0, -2.0, 0.0], [2.0, 2.0, 2.0]
    floor = min(lo[2] for lo, _ in boxes)
    if ceiling is not None:
        top = float(ceiling)
    else:
        top = min(percentile([hi[2] for _, hi in boxes], pct), floor + z_cap)
    top = max(top, floor + min_height)
    inside = [b for b in boxes if b[0][2] < top] or boxes  # ignore anything wholly above the ceiling
    lo = [min(b[0][0] for b in inside), min(b[0][1] for b in inside), floor]
    hi = [max(b[1][0] for b in inside), max(b[1][1] for b in inside), top]
    return lo, hi


def box_xy_distance(p, lo, hi):
    """Horizontal distance from point p to the box (0 inside its XY footprint)."""
    dx = max(lo[0] - p[0], 0.0, p[0] - hi[0])
    dy = max(lo[1] - p[1], 0.0, p[1] - hi[1])
    return math.hypot(dx, dy)


def robot_region(boxes, target, radius=4.0):
    """The boxes the cameras frame when there is a robot: those within `radius` (XY) of it,
    clipped to the square `radius` around it (a scan spanning a whole floor counts only near the
    robot). No target or radius <= 0: every box. Nothing near: a 2 x 2 m box around the robot."""
    boxes = [(list(lo), list(hi)) for lo, hi in boxes]
    if target is None or radius is None or radius <= 0:
        return boxes
    x, y = target[0], target[1]
    out = []
    for lo, hi in boxes:
        if box_xy_distance(target, lo, hi) <= radius:
            out.append(([max(lo[0], x - radius), max(lo[1], y - radius), lo[2]],
                        [min(hi[0], x + radius), min(hi[1], y + radius), hi[2]]))
    z = target[2] if len(target) > 2 else 0.0
    return out or [([x - 1.0, y - 1.0, z], [x + 1.0, y + 1.0, z + 0.5])]


def box_corners(lo, hi):
    return [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]


def view_dir(azimuth_deg, elevation_deg):
    """Unit vector from the aim point toward the camera."""
    a, e = math.radians(azimuth_deg), math.radians(elevation_deg)
    return [math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)]


def camera_basis(d, up=(0.0, 0.0, 1.0)):
    """Right and up vectors of a camera looking along -d."""
    f = [-x for x in d]
    r = _cross(f, up)
    if math.sqrt(_dot(r, r)) < 1e-6:
        r = _cross(f, [0.0, 1.0, 0.0])
    r = _norm(r)
    return r, _norm(_cross(r, f))


def fit_distance(points, aim, d, tan_h, tan_v, margin=0.15, min_dist=1.0):
    """Smallest distance along d (unit, aim -> camera) at which every point is inside the frame,
    with the half-FOV tangents shrunk by 1 + margin."""
    r, u = camera_basis(d)
    th, tv = tan_h / (1 + margin), tan_v / (1 + margin)
    dist = min_dist
    for p in points:
        q = _sub(p, aim)
        along = _dot(q, d)
        dist = max(dist, along + abs(_dot(q, r)) / th, along + abs(_dot(q, u)) / tv)
    return dist


def fov_tans(aspect=16 / 9, focal=18.0, aperture=APERTURE):
    tan_h = aperture / 2 / focal
    return tan_h, tan_h / aspect


def ray_box_exit(origin, azimuth_deg, lo, hi):
    """How far a horizontal ray from origin (x, y) travels before leaving the XY box; 0 if outside."""
    x, y = origin[0], origin[1]
    if not (lo[0] <= x <= hi[0] and lo[1] <= y <= hi[1]):
        return 0.0
    a = math.radians(azimuth_deg)
    dx, dy = math.cos(a), math.sin(a)
    t = math.inf
    if dx > 1e-9:
        t = min(t, (hi[0] - x) / dx)
    elif dx < -1e-9:
        t = min(t, (lo[0] - x) / dx)
    if dy > 1e-9:
        t = min(t, (hi[1] - y) / dy)
    elif dy < -1e-9:
        t = min(t, (lo[1] - y) / dy)
    return 0.0 if t is math.inf else max(t, 0.0)


def inside_xy(p, lo, hi):
    return lo[0] <= p[0] <= hi[0] and lo[1] <= p[1] <= hi[1]


OVERVIEW_AZIMUTHS = (-45.0, 225.0, 45.0, 135.0, 0.0, 180.0, 90.0, 270.0)  # diagonals first


def robot_visible_from(target, azimuth, lo, hi, elevation=60.0, clearance=0.4):
    """Whether the near wall (room height) leaves the robot in view from this azimuth: the floor
    strip a wall hides is height / tan(elevation) wide."""
    if target is None or not inside_xy(target, lo, hi):
        return True
    shadow = (hi[2] - lo[2]) / max(math.tan(math.radians(elevation)), 1e-3)
    return ray_box_exit(target, azimuth, lo, hi) >= shadow + clearance


def frame_aim(lo, hi, target=None, aim_height=0.25):
    """Where overview and orbit look: the robot when there is one in the room's footprint (so it
    sits in the middle of the frame), else the room centre; aim_height above the floor."""
    if target is not None and inside_xy(target, lo, hi):
        return [target[0], target[1], lo[2] + aim_height]
    return [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2] + aim_height]


def framing_points(lo, hi, target=None):
    """The floor footprint's corners (walls above it may leave the frame) plus the robot."""
    pts = [[x, y, lo[2]] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])]
    if target is not None:
        pts.append(list(target))
    return pts


def overview_fit(lo, hi, target, azimuth, aspect=16 / 9, focal=18.0, aperture=APERTURE, elevation=60.0,
                 margin=0.15, vertical=True):
    """Distance at which the footprint (and robot) fits the frame from this azimuth; vertical=False
    fits only the horizontal field of view."""
    th, tv = fov_tans(aspect, focal, aperture)
    return fit_distance(framing_points(lo, hi, target), frame_aim(lo, hi, target), view_dir(azimuth, elevation),
                        th, tv if vertical else 1e9, margin=margin)


def best_overview_azimuth(target, lo, hi, aspect=16 / 9, focal=18.0, aperture=APERTURE, elevation=60.0,
                          margin=0.15, choices=OVERVIEW_AZIMUTHS):
    """Of the azimuths from which no near wall hides the robot, the one that frames the room
    tightest (diagonals win ties within 3%); if every side hides it, the most open side."""
    seen = [az for az in choices if robot_visible_from(target, az, lo, hi, elevation)]
    if not seen:
        return max(choices, key=lambda az: ray_box_exit(target, az, lo, hi))
    fits = {az: overview_fit(lo, hi, target, az, aspect, focal, aperture, elevation, margin) for az in seen}
    best = min(fits.values())
    return next(az for az in seen if fits[az] <= best * 1.03)


def overview_pose(lo, hi, target=None, aspect=16 / 9, focal=18.0, aperture=APERTURE,
                  elevation=60.0, margin=0.15, azimuth=None):
    """-> (eye, look_at, up): from `elevation` degrees down onto the robot (else the room centre),
    far enough that the floor footprint and the robot fit both ways with `margin` to spare."""
    aim = frame_aim(lo, hi, target)
    if azimuth is None:
        azimuth = best_overview_azimuth(target, lo, hi, aspect, focal, aperture, elevation, margin)
    d = view_dir(azimuth, elevation)
    dist = overview_fit(lo, hi, target, azimuth, aspect, focal, aperture, elevation, margin)
    return [aim[i] + dist * d[i] for i in range(3)], aim, [0, 0, 1]


def overview_candidates(lo, hi, target=None, aspect=16 / 9, focal=18.0, aperture=APERTURE, elevation=60.0,
                        margin=0.15, steep=75.0, choices=OVERVIEW_AZIMUTHS):
    """Overview cameras to choose from by line of sight: every azimuth at `elevation`, then at
    `steep`. Each has a preference (0..1): the configured elevation first, then no near room wall
    hiding the robot, then the tightest framing. -> [{"azimuth", "elevation", "pose", "dist", "pref"}]"""
    out = []
    for k, el in enumerate([elevation] + ([steep] if steep and steep != elevation else [])):
        for az in choices:
            pose = overview_pose(lo, hi, target, aspect, focal, aperture, el, margin, azimuth=az)
            out.append({"azimuth": az, "elevation": el, "pose": pose, "dist": math.dist(pose[0], pose[1]),
                        "seen": robot_visible_from(target, az, lo, hi, el), "level": k})
    best = min(c["dist"] for c in out)
    for c in out:
        c["pref"] = round(0.5 * (c["level"] == 0) + 0.3 * c["seen"] + 0.2 * best / c["dist"], 4)
    return out


def orbit_distance(lo, hi, target=None, aspect=16 / 9, focal=18.0, aperture=APERTURE,
                   elevation=60.0, margin=0.15, samples=24):
    """One distance for the whole orbit: the footprint fits the horizontal field of view from
    every azimuth (diagonal views may crop the far and near corners vertically)."""
    return max(overview_fit(lo, hi, target, 360.0 * k / samples, aspect, focal, aperture, elevation, margin,
                            vertical=False) for k in range(samples))


def orbit_pose(lo, hi, i, n, aspect=16 / 9, focal=18.0, aperture=APERTURE, elevation=60.0, margin=0.15,
               target=None, start_azimuth=None, dist=None):
    """Frame i of n circling the robot (else the room centre) at `elevation`, starting at the
    overview azimuth."""
    aim = frame_aim(lo, hi, target)
    if dist is None:
        dist = orbit_distance(lo, hi, target, aspect, focal, aperture, elevation, margin)
    if start_azimuth is None:
        start_azimuth = best_overview_azimuth(target, lo, hi, aspect, focal, aperture, elevation, margin)
    d = view_dir(start_azimuth + 360.0 * i / max(n, 1), elevation)
    return [aim[k] + dist * d[k] for k in range(3)], aim, [0, 0, 1]


# --------------------------------------------------------------------------- flyover (the orbit video)


def ease_in_out(t):
    """0..1 -> 0..1 along half a cosine: starts and ends at rest, no jolt anywhere."""
    t = min(max(float(t), 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * t)


def flyover_elevations(lo, hi, elevation=60.0, swing=8.0, shadow=0.3, top=80.0):
    """(start, end) elevation of the flyover, degrees: from `swing` above `elevation` down to `swing`
    below it, but never so low that the room's walls hide more than `shadow` of its short side behind
    them (the floor strip a wall hides is height / tan(elevation) wide), so a small room stays high."""
    h = max(hi[2] - lo[2], 0.0)
    short = max(min(hi[0] - lo[0], hi[1] - lo[1]), 0.5)
    lowest = math.degrees(math.atan2(h, shadow * short))
    start = min(elevation + swing, top)
    return start, min(max(elevation - swing, lowest), start)


def flyover_direction(lo, hi, target, start_azimuth, sweep, elevation):
    """+1 (counter-clockwise) or -1: the way to sweep from start_azimuth. With a robot, toward the
    side from which no near wall hides it at the end (robot_visible_from), then the side with more
    open floor in front of it; else counter-clockwise."""
    if target is None or not inside_xy(target, lo, hi):
        return 1

    def key(sign):
        az = start_azimuth + sign * sweep
        return robot_visible_from(target, az, lo, hi, elevation), ray_box_exit(target, az, lo, hi), sign

    return max((1, -1), key=key)


def flyover_candidates(lo, hi, target, start_azimuth, sweep=90.0, elevation=60.0):
    """Flyover paths to choose from by line of sight, as (start, end) azimuths with a preference
    (0..1): from start_azimuth (the overview's) the way flyover_direction picks, then the other way,
    then the two that pass it halfway. -> [{"azimuths": (a0, a1), "pref"}]"""
    s = flyover_direction(lo, hi, target, start_azimuth, sweep, elevation)
    a, h = start_azimuth, sweep / 2.0
    return [{"azimuths": (a, a + s * sweep), "pref": 1.0}, {"azimuths": (a, a - s * sweep), "pref": 0.8},
            {"azimuths": (a - s * h, a + s * h), "pref": 0.6}, {"azimuths": (a + s * h, a - s * h), "pref": 0.5}]


def flyover_poses(lo, hi, n, target=None, aspect=16 / 9, focal=18.0, aperture=APERTURE, elevation=60.0, margin=0.15,
                  start_azimuth=None, sweep=90.0, swing=8.0, push=0.2, clearance=0.5, azimuths=None):
    """n camera poses (eye, look_at, up) of a slow flyover around the robot (else the room centre): from
    start_azimuth (the overview's) the camera sweeps `sweep` degrees around the aim (or from
    azimuths[0] to azimuths[1]: flyover_candidates), descends from above `elevation` to below it
    (flyover_elevations) and pushes in by `push` of its distance, all on one ease_in_out curve, so it
    starts and ends at rest. The first pose frames the floor footprint, the robot and the tops of the
    far walls (the backdrop; the near ones it looks over) both ways; the push-in then lets the edges
    leave the frame. The aim never moves and the camera never comes lower than `clearance` above
    the room's top."""
    aim = frame_aim(lo, hi, target)
    el0, el1 = flyover_elevations(lo, hi, elevation, swing)
    if azimuths is None:
        if start_azimuth is None:
            start_azimuth = best_overview_azimuth(target, lo, hi, aspect, focal, aperture, elevation, margin)
        sign = flyover_direction(lo, hi, target, start_azimuth, sweep, el1)
        azimuths = (start_azimuth, start_azimuth + sign * sweep)
    a0, a1 = azimuths
    th, tv = fov_tans(aspect, focal, aperture)
    d0 = view_dir(a0, el0)
    far_tops = [[x, y, hi[2]] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) if _dot(_sub([x, y, hi[2]], aim), d0) < 0]
    dist0 = fit_distance(framing_points(lo, hi, target) + far_tops, aim, d0, th, tv, margin=margin)
    out = []
    for i in range(max(int(n), 1)):
        s = ease_in_out(i / (n - 1)) if n > 1 else 0.0
        el = el0 + (el1 - el0) * s
        dist = dist0 * (1.0 - push * s)
        dist = max(dist, (hi[2] + clearance - aim[2]) / max(math.sin(math.radians(el)), 1e-3))
        d = view_dir(a0 + (a1 - a0) * s, el)
        out.append(([aim[k] + dist * d[k] for k in range(3)], list(aim), [0, 0, 1]))
    return out


def still_indices(frames, stills):
    """Which of `frames` video frames to keep as the `stills` orbit_NNNN.png: spread evenly, first and
    last included (all of them when there are no more frames than stills)."""
    frames, stills = max(int(frames), 0), max(int(stills), 0)
    if stills >= frames:
        return list(range(frames))
    if stills == 1:
        return [0]
    return [round(k * (frames - 1) / (stills - 1)) for k in range(stills)]


def ring_candidates(center, lo=None, hi=None, dist=2.2, height=1.3, n=8, start_azimuth=0.0,
                    min_dist=1.8, max_dist=2.5, wall_margin=0.25, floor=0.0, close=None):
    """n camera spots around center (x, y) at `height` above floor, each pulled in (not below
    min_dist) so it stays wall_margin inside the room box when one is given.
    -> [{"eye", "azimuth", "dist", "inside"}]"""
    dist = min(max(dist, min_dist), max_dist)
    out = []
    for k in range(n):
        az = (start_azimuth + 360.0 * k / n) % 360.0
        d, inside = dist, True
        if lo is not None and hi is not None:
            ilo = [lo[0] + wall_margin, lo[1] + wall_margin]
            ihi = [hi[0] - wall_margin, hi[1] - wall_margin]
            room = ray_box_exit(center, az, ilo, ihi)
            inside = room >= min_dist
            d = min(dist, max(room, min_dist))
        a = math.radians(az)
        out.append({"eye": [center[0] + d * math.cos(a), center[1] + d * math.sin(a), floor + height],
                    "azimuth": round(az, 3), "dist": round(d, 4), "inside": inside})
    if close is not None and close < min(dist, min_dist) - 1e-6:  # a second, closer ring for small rooms
        for c in ring_candidates(center, lo, hi, close, height, n, start_azimuth, close, close, wall_margin, floor):
            out.append({**c, "close": True})
    return out


def ring_prefs(cands, prefs, far_bonus=0.2):
    """Preferences for ring_candidates with a closer ring: the closer spots rank after the
    configured ones when the line of sight and the picture are as clear."""
    return [p * (1 - far_bonus) + (0.0 if c.get("close") else far_bonus) for c, p in zip(cands, prefs)]


def pick_candidate(scores, prefs=None, inside=None):
    """Index of the best candidate: clearest line of sight (score 0..1, 1 = clear) first, then
    inside the room, then the preference bias (0..1). Scores within 0.05 count as equal."""
    best, best_key = 0, None
    for i, s in enumerate(scores):
        key = (round(float(s) / 0.05), bool(inside[i]) if inside else True, prefs[i] if prefs else 0.0)
        if best_key is None or key > best_key:
            best, best_key = i, key
    return best


def toward_prefs(center, cands, point):
    """0..1 per candidate: 1 when the camera sits on the side of center facing point."""
    to = [point[0] - center[0], point[1] - center[1]]
    n = math.hypot(*to)
    out = []
    for c in cands:
        v = [c["eye"][0] - center[0], c["eye"][1] - center[1]]
        m = math.hypot(*v)
        out.append(0.5 if n < 1e-6 or m < 1e-6 else (1 + (to[0] * v[0] + to[1] * v[1]) / (n * m)) / 2)
    return out


def hero_prefs(cands, yaw_deg):
    """Prefer the robot's front quarters (yaw +-45 deg), then the sides, then the rear."""
    out = []
    for c in cands:
        off = abs((c["azimuth"] - yaw_deg + 180.0) % 360.0 - 180.0)
        out.append(1.0 - abs(off - 45.0) / 135.0)
    return out


def room_light_positions(lo, hi, below=0.3, cell=5.0, cap=3):
    """One light per ~cell x cell metres (at most cap x cap), `below` under the ceiling."""
    sx, sy = hi[0] - lo[0], hi[1] - lo[1]
    nx = min(max(int(math.ceil(sx / cell)), 1), cap)
    ny = min(max(int(math.ceil(sy / cell)), 1), cap)
    z = max(hi[2] - below, lo[2] + 0.5)
    return [[lo[0] + sx * (i + 0.5) / nx, lo[1] + sy * (j + 0.5) / ny, z] for i in range(nx) for j in range(ny)]


def top_pose(lo, hi, aspect=16 / 9, focal=18.0, aperture=APERTURE, margin=0.15, fit_top=False):
    """Straight down onto the floor footprint, its long side across the frame; fit_top: so the
    footprint fits at the room's top (hi z) too, walls and all, not only at the floor."""
    c = [(a + b) / 2 for a, b in zip(lo, hi)]
    sx, sy = (max(b - a, 0.5) for a, b in zip(lo[:2], hi[:2]))
    tan_h, tan_v = fov_tans(aspect, focal, aperture)
    h_x = max(sx / 2 / tan_h, sy / 2 / tan_v)  # up = +y: x runs across the frame
    h_y = max(sy / 2 / tan_h, sx / 2 / tan_v)  # up = -x: y runs across the frame
    h, up = (h_x, [0, 1, 0]) if h_x <= h_y else (h_y, [-1, 0, 0])
    z = max((hi[2] if fit_top else lo[2]) + (1 + margin) * h, hi[2] + 0.5)
    return [c[0], c[1], z], [c[0], c[1], lo[2]], up


def camera_poses(lo, hi, target=None, aspect=16 / 9, focal=18.0, aperture=APERTURE, elevation=60.0,
                 margin=0.15, yaw=0.0, eye_height=1.3, eye_dist=2.2, hero_dist=2.5, hero_height=1.6):
    """-> {name: (eye, look_at, up)} for overview, top, eye and hero, framing the room box (lo, hi).
    target is the robot (x, y, feet z); eye and hero use their geometric defaults here (eye from
    the room-centre side, hero at the robot's front-left quarter); main() refines them by ray tests."""
    c = [(a + b) / 2 for a, b in zip(lo, hi)]
    floor = lo[2]
    overview = overview_pose(lo, hi, target, aspect, focal, aperture, elevation, margin)
    top = top_pose(lo, hi, aspect, focal, aperture, margin)
    center = list(target[:2]) if target else c[:2]
    gz = target[2] if target else floor
    look = [center[0], center[1], gz + 0.25]
    eyes = ring_candidates(center, lo, hi, eye_dist, eye_height, floor=gz)
    i = pick_candidate([1.0] * len(eyes), toward_prefs(center, eyes, c), [e["inside"] for e in eyes])
    heroes = ring_candidates(center, lo, hi, hero_dist, hero_height, start_azimuth=yaw + 45.0,
                             min_dist=1.8, max_dist=hero_dist, floor=gz)
    j = pick_candidate([1.0] * len(heroes), hero_prefs(heroes, yaw), [e["inside"] for e in heroes])
    return {"overview": overview, "top": top, "eye": (eyes[i]["eye"], look, [0, 0, 1]),
            "hero": (heroes[j]["eye"], look, [0, 0, 1])}


def segment_clearance(p0, p1, tris):
    """Fraction (0..1) of the segment p0 -> p1 travelled before the first triangle hit; 1.0 if
    none. tris: (N, 3, 3) array-like. Moller-Trumbore, vectorised with numpy."""
    import numpy as np
    tris = np.asarray(tris, dtype=np.float64).reshape(-1, 3, 3)
    if len(tris) == 0:
        return 1.0
    o = np.asarray(p0, dtype=np.float64)
    dvec = np.asarray(p1, dtype=np.float64) - o
    v0, e1, e2 = tris[:, 0], tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    pv = np.cross(dvec, e2)
    det = np.einsum("ij,ij->i", e1, pv)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    tv = o - v0
    u = np.einsum("ij,ij->i", tv, pv) * inv
    qv = np.cross(tv, e1)
    v = (qv @ dvec) * inv
    t = np.einsum("ij,ij->i", e2, qv) * inv
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-6) & (t <= 1.0)
    return float(t[hit].min()) if hit.any() else 1.0


def _first_hits(o, dirs, tris, t_max):
    """Distance along each unit ray (o + t * dirs[k]) to the first triangle within t_max; inf if none.
    Moller-Trumbore, vectorised over triangles, one ray at a time (tris already culled)."""
    import numpy as np
    out = np.full(len(dirs), np.inf)
    if not len(tris):
        return out
    v0, e1, e2 = tris[:, 0], tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    tv = o - v0
    qv = np.cross(tv, e1)
    t_num = np.einsum("ij,ij->i", e2, qv)
    for k, dvec in enumerate(dirs):
        pv = np.cross(dvec, e2)
        det = np.einsum("ij,ij->i", e1, pv)
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        u = np.einsum("ij,ij->i", tv, pv) * inv
        v = (qv @ dvec) * inv
        t = t_num * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-6) & (t <= t_max)
        if hit.any():
            out[k] = float(t[hit].min())
    return out


def view_clearance(eye, look, tris, aspect=16 / 9, focal=18.0, aperture=APERTURE, floor=None, nx=9, ny=5,
                   span=0.9, near=0.85, floor_tol=0.12):
    """How much of a camera's picture is free of foreground (0..1, 1 = clear): of an nx x ny grid of
    rays across the central `span` of the frame (eye looking at look, Z up), the fraction that hit
    no triangle closer than `near` x the distance to look (whatever they meet further on is level
    with or behind the robot: background). Hits on the floor (z below floor + floor_tol) don't
    count: the floor in front of the robot is meant to be seen. segment_clearance checks the line
    to the robot only; a wall beside it, just in front of the lens, can still fill half the frame,
    and a camera behind a wall of a small room sees nothing but the wall. tris: (N, 3, 3)."""
    import numpy as np
    tris = np.asarray(tris, dtype=np.float64).reshape(-1, 3, 3)
    o = np.asarray(eye, dtype=np.float64)
    fwd = np.asarray(look, dtype=np.float64) - o
    dist = float(np.linalg.norm(fwd))
    if len(tris) == 0 or dist < 1e-6:
        return 1.0
    d = [-x / dist for x in fwd]  # camera_basis takes the direction aim -> camera
    r, u = (np.asarray(v) for v in camera_basis(d))
    f = fwd / dist
    th, tv = fov_tans(aspect, focal, aperture)
    xs = np.linspace(-1, 1, nx) * th * span if nx > 1 else np.zeros(1)
    ys = np.linspace(-1, 1, ny) * tv * span if ny > 1 else np.zeros(1)
    dirs = np.array([f + x * r + y * u for y in ys for x in xs])
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    t_max = near * dist
    # only triangles that can be hit: inside the box around the rays' first t_max metres
    ends = o + dirs * t_max
    lo = np.minimum(ends.min(0), o) - 1e-6
    hi = np.maximum(ends.max(0), o) + 1e-6
    keep = ((tris.max(1) >= lo) & (tris.min(1) <= hi)).all(1)
    if floor is not None:  # wholly on the floor: never counts
        keep &= tris[:, :, 2].max(1) > floor + floor_tol
    t = _first_hits(o, dirs, tris[keep], t_max)
    blocked = np.isfinite(t)
    if floor is not None and blocked.any():  # a hit on a triangle that rises off the floor, but at floor level
        z = o[2] + dirs[:, 2] * np.where(blocked, t, 0.0)
        blocked &= z > floor + floor_tol
    return float(1.0 - blocked.mean())


def view_scores(cands, look, tris, aspect=16 / 9, focal=18.0, aperture=APERTURE, floor=None):
    """Per camera candidate ({"eye": ...}, as ring_candidates): min(line of sight to look,
    view_clearance of its picture). -> (scores, line-of-sight scores, view scores)."""
    los, view = [], []
    for c in cands:
        try:
            los.append(segment_clearance(look, c["eye"], tris))
            view.append(view_clearance(c["eye"], look, tris, aspect, focal, aperture, floor=floor))
        except Exception:  # a degenerate candidate: neutral
            los.append(1.0)
            view.append(1.0)
    return [min(a, b) for a, b in zip(los, view)], los, view


def rnd(xs, k=4):
    return [round(float(x), k) for x in xs]


# --------------------------------------------------------------------------- other renderers' cameras


def focal_for_fovy(fovy_deg, aspect=16 / 9, aperture=APERTURE):
    """The focal length (mm, with `aperture`) whose vertical field of view at `aspect` is fovy_deg:
    lets a renderer that takes a vertical FOV (MuJoCo: 45 deg) use the framing above."""
    tan_v = math.tan(math.radians(fovy_deg) / 2)
    return aperture / 2 / (tan_v * aspect)


def free_camera(eye, aim):
    """(lookat, distance, azimuth, elevation) of a MuJoCo free camera at `eye` looking at `aim`
    (degrees; MuJoCo's azimuth is the direction the camera looks, its elevation is negative when it
    looks down)."""
    v = _sub(aim, eye)
    dist = math.sqrt(_dot(v, v))
    if dist < 1e-9:
        return list(aim), 1e-3, 0.0, 0.0
    f = [x / dist for x in v]
    elevation = math.degrees(math.asin(max(-1.0, min(1.0, f[2]))))
    azimuth = math.degrees(math.atan2(f[1], f[0])) if abs(f[0]) + abs(f[1]) > 1e-9 else 90.0
    return list(aim), dist, azimuth, elevation


# --------------------------------------------------------------------------- a 360 viewpoint (orb run isaac --vr)
# A VR video stands still at one spot and lets the viewer turn their head: that spot should be clear of
# everything by a metre (near-field stereo is uncomfortable) and not on the path of anything that moves.


def point_clearance(p, tris, floor=None, floor_tol=0.05):
    """Distance (m) from point p to the nearest triangle (N, 3, 3); inf if none. Triangles lying wholly
    on the floor (every corner within floor_tol of `floor`, when given) don't count: the floor is
    meant to be there. Exact point-triangle distance, vectorised with numpy."""
    import numpy as np
    tris = np.asarray(tris, dtype=np.float64).reshape(-1, 3, 3)
    if floor is not None and len(tris):
        tris = tris[tris[:, :, 2].max(1) > floor + floor_tol]
    if not len(tris):
        return float("inf")
    p = np.asarray(p, dtype=np.float64)
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    n = np.cross(b - a, c - a)
    nn = np.linalg.norm(n, axis=1)
    ok = nn > 1e-12
    nh = n / np.where(ok, nn, 1.0)[:, None]
    dist_plane = np.einsum("ij,ij->i", p - a, nh)
    q = p - dist_plane[:, None] * nh  # p projected onto each triangle's plane
    inside = ok.copy()
    for u, v in ((a, b), (b, c), (c, a)):  # q on the inner side of all three edges
        inside &= np.einsum("ij,ij->i", np.cross(v - u, q - u), n) >= 0
    best = np.where(inside, np.abs(dist_plane), np.inf)
    for u, v in ((a, b), (b, c), (c, a)):  # else the nearest point is on an edge
        e = v - u
        ee = np.einsum("ij,ij->i", e, e)
        t = np.clip(np.einsum("ij,ij->i", p - u, e) / np.where(ee > 0, ee, 1.0), 0.0, 1.0)
        best = np.minimum(best, np.linalg.norm(p - (u + t[:, None] * e), axis=1))
    return float(best.min())


def avoid_clearance(p, avoid):
    """Horizontal distance (m) from p to the nearest of `avoid` ([x, y, r]: a vertical cylinder of radius r,
    what an item takes up wherever its path goes); inf if none."""
    if avoid is None or not len(avoid):
        return float("inf")
    if len(avoid) < 32:
        return min(math.hypot(p[0] - x, p[1] - y) - r for x, y, r in avoid)
    import numpy as np
    a = np.asarray(avoid, dtype=np.float64).reshape(-1, 3)
    return float((np.hypot(a[:, 0] - p[0], a[:, 1] - p[1]) - a[:, 2]).min())


def box_tris(center, size, quat=(1.0, 0.0, 0.0, 0.0)):
    """The 12 triangles of a box (full sizes) at center, turned by quat (w, x, y, z): [[[x, y, z] * 3] * 12]."""
    w, x, y, z = quat
    R = [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
         [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
         [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]
    h = [s / 2 for s in size]
    corners = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                v = (sx * h[0], sy * h[1], sz * h[2])
                corners.append([center[i] + sum(R[i][j] * v[j] for j in range(3)) for i in range(3)])
    faces = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
             (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    return [[corners[i] for i in f] for f in faces]


def spot_candidates(center, step=0.25, max_r=8.0, n_az=16):
    """Spots at center's height: center itself, then rings `step` apart out to max_r, n_az per ring
    (the first ring's spots start at +X and go counter-clockwise). -> [(ring, [x, y, z])]"""
    out = [(0, list(center))]
    k = 1
    while k * step <= max_r + 1e-9:
        r = k * step
        for j in range(n_az):
            a = 2 * math.pi * j / n_az
            out.append((k, [center[0] + r * math.cos(a), center[1] + r * math.sin(a), center[2]]))
        k += 1
    return out


class _TriGrid:
    """Triangles binned by their XY footprint into square cells `cell` metres wide, so a clearance
    query looks only at the triangles near a point (a scanned room has hundreds of thousands)."""

    def __init__(self, tris, cell):
        import numpy as np
        self.np, self.tris, self.cell = np, tris, float(cell)
        self.lo, self.hi = tris.min(1), tris.max(1)
        self.origin = self.lo[:, :2].min(0)
        i0 = np.floor((self.lo[:, :2] - self.origin) / self.cell).astype(np.int64)
        i1 = np.floor((self.hi[:, :2] - self.origin) / self.cell).astype(np.int64)
        small = ((i1 - i0) < 4).all(1)  # within 4 x 4 cells; a bigger one (a floor, a wall) is checked everywhere
        self.big = np.nonzero(~small)[0]
        self.width = int(i1[:, 1].max()) + 8
        keys, ids = [], []
        for dx in range(4):
            for dy in range(4):
                sel = np.nonzero(small & (i0[:, 0] + dx <= i1[:, 0]) & (i0[:, 1] + dy <= i1[:, 1]))[0]
                keys.append((i0[sel, 0] + dx) * self.width + i0[sel, 1] + dy)
                ids.append(sel)
        keys, ids = np.concatenate(keys), np.concatenate(ids)
        order = np.argsort(keys, kind="stable")
        self.keys, self.ids = keys[order], ids[order]

    def around(self, p):
        """The triangles whose box comes within `cell` of p."""
        np = self.np
        q = np.asarray(p, dtype=np.float64)
        c0 = np.floor((q[:2] - self.cell - self.origin) / self.cell).astype(np.int64)
        c1 = np.floor((q[:2] + self.cell - self.origin) / self.cell).astype(np.int64)
        parts = [self.big]
        for x in range(c0[0], c1[0] + 1):
            if c1[1] < 0 or c0[1] >= self.width:
                continue
            a = np.searchsorted(self.keys, x * self.width + max(c0[1], 0))
            b = np.searchsorted(self.keys, x * self.width + min(c1[1], self.width - 1), side="right")
            parts.append(self.ids[a:b])
        idx = np.unique(np.concatenate(parts))
        if not len(idx):
            return self.tris[:0]
        ok = ((self.lo[idx] <= q + self.cell) & (self.hi[idx] >= q - self.cell)).all(1)
        return self.tris[idx[ok]]


def clear_spot(center, tris=None, min_clear=1.0, floor=None, avoid=(), step=0.25, max_r=8.0, n_az=16):
    """The spot nearest center (same height) at least min_clear from every triangle (point_clearance,
    floor excluded) and every `avoid` cylinder (avoid_clearance); of a ring's spots the clearest, and
    one reachable from center without passing through a triangle (no hopping through a wall) when there
    is one (center inside something: any way out goes through it). Nothing that clear within max_r:
    the clearest spot found. Clearances are measured up to 1.5 x min_clear (enough to compare).
    -> {"at", "clearance", "moved" (m from center), "ok" (clear enough)}"""
    import numpy as np
    tris = np.zeros((0, 3, 3)) if tris is None else np.asarray(tris, dtype=np.float64).reshape(-1, 3, 3)
    cap = 1.5 * max(min_clear, 0.1)
    if len(tris):  # only what can matter: around the search area, off the floor
        reach = max_r + cap
        lo = np.array([center[0] - reach, center[1] - reach, center[2] - cap])
        hi = np.array([center[0] + reach, center[1] + reach, center[2] + cap])
        tris = tris[((tris.max(1) >= lo) & (tris.min(1) <= hi)).all(1)]
        if floor is not None:
            tris = tris[tris[:, :, 2].max(1) > floor + 0.05]
    near = _TriGrid(tris, cap) if len(tris) else None

    def clearance(p):
        c = min(cap, avoid_clearance(p, avoid))
        if near is not None:
            sub = near.around(p)
            if len(sub):
                c = min(c, point_clearance(p, sub))
        return c

    def reachable(p):  # a straight line from center to p through no triangle (near the line)
        q, c = np.asarray(p, dtype=np.float64), np.asarray(center, dtype=np.float64)
        sel = ((near.lo <= np.maximum(q, c)) & (near.hi >= np.minimum(q, c))).all(1)
        return not sel.any() or segment_clearance(list(center), list(p), tris[sel]) >= 1.0

    def result(p, c):
        return {"at": [float(x) for x in p], "clearance": float(c), "moved": float(math.dist(center, p)),
                "ok": c >= min_clear}

    c0 = clearance(center)
    if c0 >= min_clear:
        return result(center, c0)
    rings = {}
    for ring, p in spot_candidates(center, step, max_r, n_az)[1:]:
        rings.setdefault(ring, []).append(p)
    scored, best = [], (c0, list(center))
    for ring in sorted(rings):
        cs = sorted(((clearance(p), p) for p in rings[ring]), key=lambda x: -x[0])
        scored.append(cs)
        best = max(best, cs[0], key=lambda x: x[0])
        for c, p in cs:
            if c < min_clear:
                break
            if not len(tris) or reachable(p):
                return result(p, c)
    for cs in scored:  # nothing reachable in the open: center is inside something; the nearest way out
        if cs[0][0] >= min_clear:
            return result(cs[0][1], cs[0][0])
    return result(best[1], best[0])


def heading_to(eye, target):
    """Degrees (0: +X, 90: +Y, counter-clockwise seen from above) from eye toward target, horizontally;
    None when target is (nearly) straight above or below eye."""
    dx, dy = target[0] - eye[0], target[1] - eye[1]
    if math.hypot(dx, dy) < 1e-6:
        return None
    return math.degrees(math.atan2(dy, dx)) % 360.0
