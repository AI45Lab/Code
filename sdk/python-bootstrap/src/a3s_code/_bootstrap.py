"""Lazy loader for the a3s-code native extension.

This module is part of the pure-Python bootstrap published to PyPI under
the `a3s-code` name. On first import it resolves the matching native
wheel for the current interpreter/platform, downloads it from the
project's GitHub Releases, verifies the wheel's sha256 against the
release manifest, extracts the compiled `_native` extension into a
per-user cache, and prepends the cache to `sys.path` so the rest of
`a3s_code/__init__.py` can `from ._native import *` normally.

Override the cache location via `A3S_CODE_CACHE_DIR`. Override the
release source via `A3S_CODE_RELEASES_BASE_URL` (default points at the
GitHub Releases page for `AI45Lab/Code`). Skip the integrity check via
`A3S_CODE_SKIP_HASH_CHECK=1` (not recommended outside of CI).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import sys
import threading
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

# Version is the bootstrap's own version, which equals the matching native
# wheel version on GH Releases. Bumped by the release workflow.
__version__ = "3.4.0"

_DEFAULT_BASE_URL = "https://github.com/AI45Lab/Code/releases/download"
_REQUEST_TIMEOUT_S = 120
_USER_AGENT = f"a3s-code-bootstrap/{__version__}"
_LOAD_LOCK = threading.Lock()
_LOADED = False


class BootstrapError(RuntimeError):
    """Raised when the native extension cannot be located, downloaded, or verified."""


def _base_url() -> str:
    return os.environ.get("A3S_CODE_RELEASES_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")


def _cache_root() -> Path:
    override = os.environ.get("A3S_CODE_CACHE_DIR")
    if override:
        return Path(override).expanduser() / __version__
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "a3s-code" / __version__


def _platform_tag() -> str:
    """Return the wheel platform tag for the current interpreter/platform.

    Mirrors the matrix produced by `.github/workflows/publish-python.yml`.
    Raises `BootstrapError` for unsupported combinations so callers can
    surface a clear install hint.
    """
    sys_plat = sys.platform
    machine = platform.machine().lower()
    if sys_plat == "darwin":
        if machine in ("arm64", "aarch64"):
            return "macosx_11_0_arm64"
    elif sys_plat == "linux":
        if machine in ("x86_64", "amd64"):
            return "manylinux_2_28_x86_64"
    elif sys_plat == "win32":
        if machine in ("amd64", "x86_64"):
            return "win_amd64"
    raise BootstrapError(
        f"a3s-code: no native wheel published for {sys_plat}/{machine}. "
        "Supported platforms: macOS arm64, Linux x86_64 (glibc 2.28+), Windows x86_64."
    )


def _wheel_filename(version: str = __version__) -> str:
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    return f"a3s_code-{version}-{py_tag}-{py_tag}-{_platform_tag()}.whl"


def _release_url(filename: str, version: str = __version__) -> str:
    return f"{_base_url()}/v{version}/{filename}"


def _http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_S) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        raise BootstrapError(f"GET {url} failed: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise BootstrapError(f"GET {url} failed: {exc.reason}") from exc


def _expected_sha256(wheel_name: str, version: str = __version__) -> Optional[str]:
    """Look up the published sha256 for `wheel_name` in the release manifest.

    Returns `None` if the manifest is unreachable — bootstrap will then
    proceed without integrity verification but emit a warning. Override
    with `A3S_CODE_SKIP_HASH_CHECK=1` for hermetic offline mirrors.
    """
    manifest_url = f"{_base_url()}/v{version}/python-native-manifest.json"
    try:
        data = json.loads(_http_get(manifest_url))
    except BootstrapError as exc:
        sys.stderr.write(
            f"a3s-code: warning: manifest fetch failed ({exc}); skipping hash check\n"
        )
        return None
    for asset in data.get("assets", []):
        if asset.get("filename") == wheel_name:
            return asset.get("sha256")
    return None


def _extract_native(wheel_bytes: bytes, target_dir: Path) -> Path:
    """Extract the compiled `_native.*` extension from `wheel_bytes` into
    `target_dir`. Returns the path to the extracted file.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(wheel_bytes)) as zf:
        for name in zf.namelist():
            base = Path(name).name
            # match _native.<abi>.{so,pyd,dylib}
            if base.startswith("_native.") and not base.endswith(".dist-info"):
                out_path = target_dir / base
                with zf.open(name) as src, out_path.open("wb") as dst:
                    dst.write(src.read())
                return out_path
    raise BootstrapError(
        "downloaded wheel did not contain a _native extension; "
        "the release artifact appears to be corrupt"
    )


def _find_cached_native(cache_dir: Path) -> Optional[Path]:
    if not cache_dir.is_dir():
        return None
    for child in cache_dir.iterdir():
        if child.is_file() and child.name.startswith("_native."):
            return child
    return None


def _register_native(native_path: Path) -> None:
    """Load `native_path` as the `a3s_code._native` module.

    `_native` is a compiled extension, not a regular Python file, so use
    `importlib.machinery.ExtensionFileLoader` + the matching spec instead
    of plain `spec_from_file_location` (which works but doesn't set the
    right loader for extensions on all Python versions).
    """
    import importlib.machinery
    import importlib.util

    fullname = "a3s_code._native"
    loader = importlib.machinery.ExtensionFileLoader(fullname, str(native_path))
    spec = importlib.util.spec_from_loader(fullname, loader, origin=str(native_path))
    if spec is None:
        raise BootstrapError(f"failed to build import spec for {native_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[fullname] = module
    spec.loader.exec_module(module)


def ensure_native_loaded(version: str = __version__) -> Path:
    """Idempotently ensure the `_native` extension is registered as
    `a3s_code._native` in `sys.modules`. Returns the cache directory the
    extension was loaded from. Safe across threads — first caller wins.
    """
    global _LOADED
    cache = _cache_root()

    if _LOADED:
        return cache

    with _LOAD_LOCK:
        if _LOADED:
            return cache

        native = _find_cached_native(cache)
        if native is None:
            wheel_name = _wheel_filename(version)
            url = _release_url(wheel_name, version)
            sys.stderr.write(
                f"a3s-code: fetching native wheel {wheel_name} "
                f"from {url} (first import only)...\n"
            )
            wheel_bytes = _http_get(url)

            if os.environ.get("A3S_CODE_SKIP_HASH_CHECK") != "1":
                expected = _expected_sha256(wheel_name, version)
                if expected is not None:
                    actual = hashlib.sha256(wheel_bytes).hexdigest()
                    if actual != expected:
                        raise BootstrapError(
                            f"sha256 mismatch for {wheel_name}: "
                            f"expected {expected}, got {actual}"
                        )

            native = _extract_native(wheel_bytes, cache)

        _register_native(native)
        _LOADED = True
        return cache
