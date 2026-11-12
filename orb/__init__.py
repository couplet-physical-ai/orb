import os as _os
import sys as _sys


def _headless_gl():
    """MuJoCo picks its OpenGL backend from MUJOCO_GL when it is first imported (default GLFW, which
    needs a display). On a Linux box with no display (a server, CI, ssh), offscreen rendering
    (`MuJoCo offscreen rendering`) then fails; use EGL (or OSMesa) there instead."""
    if not _sys.platform.startswith("linux") or _os.environ.get("MUJOCO_GL"):
        return
    if _os.environ.get("DISPLAY") or _os.environ.get("WAYLAND_DISPLAY"):
        return
    import ctypes

    for backend, libs in (("egl", ("libEGL.so.1", "libEGL.so")), ("osmesa", ("libOSMesa.so.8", "libOSMesa.so"))):
        for lib in libs:
            try:
                ctypes.CDLL(lib)
            except OSError:
                continue
            _os.environ["MUJOCO_GL"] = backend
            if backend == "osmesa":
                _os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
            return


_headless_gl()
