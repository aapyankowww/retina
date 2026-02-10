from __future__ import annotations

import sys
import os
import multiprocessing as mp
from pathlib import Path


def _preload_linux_qt_deps() -> None:
    """Preload xcb dependency that is often missing from system libs."""
    if not sys.platform.startswith("linux"):
        return

    try:
        import ctypes
    except Exception:
        return

    # Fast path: library already resolvable by dynamic linker.
    try:
        ctypes.CDLL("libxcb-xinerama.so.0", mode=ctypes.RTLD_GLOBAL)
        return
    except OSError:
        pass

    candidates: list[Path] = []
    for env_name in ("CONDA_PREFIX",):
        raw = os.environ.get(env_name)
        if not raw:
            continue
        value = Path(raw).expanduser()
        candidates.append(value / "lib" / "libxcb-xinerama.so.0")

    home = Path.home()
    candidates.extend(
        [
            home / "miniconda3" / "lib" / "libxcb-xinerama.so.0",
            home / "anaconda3" / "lib" / "libxcb-xinerama.so.0",
        ]
    )

    seen: set[str] = set()
    for candidate in candidates:
        resolved = str(candidate)
        if resolved in seen:
            continue
        seen.add(resolved)
        if not candidate.is_file():
            continue
        try:
            ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
            return
        except OSError:
            continue

    sys.stderr.write(
        "Qt dependency not found: libxcb-xinerama.so.0. "
        "Install package libxcb-xinerama0 and retry.\n"
    )


def main() -> int:
    mp.freeze_support()
    _preload_linux_qt_deps()
    from ui import create_app

    app, window = create_app()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
