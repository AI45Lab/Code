"""Unit tests for the a3s-code bootstrap loader.

Live network is not required — `_http_get` is monkey-patched to serve
a constructed wheel byte string. There's a separate live integration
test at the bottom guarded by `A3S_CODE_BOOTSTRAP_LIVE=1`.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
import unittest.mock as mock
import zipfile
from pathlib import Path


# Load `_bootstrap` directly from disk so the package `__init__.py`
# (which would call `ensure_native_loaded` and try to hit the network)
# doesn't run during test collection.
_BOOTSTRAP_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "a3s_code" / "_bootstrap.py"
)
_spec = importlib.util.spec_from_file_location("_bootstrap", _BOOTSTRAP_PATH)
_bootstrap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bootstrap)


def _make_wheel(native_blob: bytes = b"fake-extension-blob") -> bytes:
    """Build a minimal in-memory wheel containing _native.something.so."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("a3s_code/__init__.py", "from ._native import *\n")
        zf.writestr("a3s_code/_native.cpython-312-x86_64-linux-gnu.so", native_blob)
        zf.writestr("a3s_code-3.2.1.dist-info/METADATA", "Metadata-Version: 2.1\n")
        zf.writestr("a3s_code-3.2.1.dist-info/WHEEL", "Wheel-Version: 1.0\n")
    return buf.getvalue()


class _FakeVersionInfo:
    """Stand-in for sys.version_info supporting attribute access."""

    def __init__(self, major: int, minor: int):
        self.major = major
        self.minor = minor
        self.micro = 0
        self.releaselevel = "final"
        self.serial = 0


class WheelFilenameTests(unittest.TestCase):
    def _filename_for(self, sys_plat: str, machine: str, py_minor: int) -> str:
        with (
            mock.patch.object(sys, "platform", sys_plat),
            mock.patch.object(_bootstrap.platform, "machine", return_value=machine),
            mock.patch.object(sys, "version_info", _FakeVersionInfo(3, py_minor)),
        ):
            return _bootstrap._wheel_filename(version="3.2.1")

    def test_linux_x86_64_cp312(self):
        self.assertEqual(
            self._filename_for("linux", "x86_64", 12),
            "a3s_code-3.2.1-cp312-cp312-manylinux_2_28_x86_64.whl",
        )

    def test_macos_arm64_cp311(self):
        self.assertEqual(
            self._filename_for("darwin", "arm64", 11),
            "a3s_code-3.2.1-cp311-cp311-macosx_11_0_arm64.whl",
        )

    def test_windows_amd64_cp313(self):
        self.assertEqual(
            self._filename_for("win32", "AMD64", 13),
            "a3s_code-3.2.1-cp313-cp313-win_amd64.whl",
        )

    def test_unsupported_platform_raises(self):
        with self.assertRaises(_bootstrap.BootstrapError) as cm:
            self._filename_for("freebsd", "x86_64", 12)
        self.assertIn("no native wheel published", str(cm.exception))

    def test_unsupported_linux_arch_raises(self):
        with self.assertRaises(_bootstrap.BootstrapError):
            self._filename_for("linux", "ppc64le", 12)


class CacheDirTests(unittest.TestCase):
    def setUp(self):
        self._prev_env = {
            k: os.environ.get(k) for k in ("A3S_CODE_CACHE_DIR", "XDG_CACHE_HOME")
        }
        for k in ("A3S_CODE_CACHE_DIR", "XDG_CACHE_HOME"):
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._prev_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_default_uses_xdg_or_home(self):
        cache = _bootstrap._cache_root()
        self.assertTrue(str(cache).endswith(f"a3s-code/{_bootstrap.__version__}"))

    def test_xdg_cache_home_honored(self):
        os.environ["XDG_CACHE_HOME"] = "/tmp/xdg-test"
        cache = _bootstrap._cache_root()
        self.assertEqual(cache, Path(f"/tmp/xdg-test/a3s-code/{_bootstrap.__version__}"))

    def test_explicit_override_wins(self):
        os.environ["XDG_CACHE_HOME"] = "/tmp/xdg-test"
        os.environ["A3S_CODE_CACHE_DIR"] = "/var/a3s-cache"
        cache = _bootstrap._cache_root()
        self.assertEqual(cache, Path(f"/var/a3s-cache/{_bootstrap.__version__}"))


class ExtractNativeTests(unittest.TestCase):
    def test_extracts_native_extension(self):
        wheel_bytes = _make_wheel(b"native-bytes")
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "a3s_code"
            out = _bootstrap._extract_native(wheel_bytes, target)
            self.assertTrue(out.exists())
            self.assertTrue(out.name.startswith("_native."))
            self.assertEqual(out.read_bytes(), b"native-bytes")

    def test_wheel_without_native_raises(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("a3s_code/__init__.py", "")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(_bootstrap.BootstrapError):
                _bootstrap._extract_native(buf.getvalue(), Path(tmp) / "pkg")


class EnsureNativeLoadedTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="a3s-bootstrap-test-")
        self._prev_cache = os.environ.get("A3S_CODE_CACHE_DIR")
        os.environ["A3S_CODE_CACHE_DIR"] = self._tmp
        # Reset the module-level latch so each test starts clean.
        _bootstrap._LOADED = False

    def tearDown(self):
        if self._prev_cache is None:
            os.environ.pop("A3S_CODE_CACHE_DIR", None)
        else:
            os.environ["A3S_CODE_CACHE_DIR"] = self._prev_cache
        shutil.rmtree(self._tmp, ignore_errors=True)
        _bootstrap._LOADED = False

    def test_downloads_extracts_and_registers_module(self):
        wheel_bytes = _make_wheel()
        expected_sha = hashlib.sha256(wheel_bytes).hexdigest()

        manifest = json.dumps(
            {
                "version": "3.2.1",
                "assets": [
                    {
                        "filename": _bootstrap._wheel_filename("3.2.1"),
                        "sha256": expected_sha,
                    }
                ],
            }
        ).encode()

        def fake_get(url: str) -> bytes:
            if url.endswith("python-native-manifest.json"):
                return manifest
            return wheel_bytes

        # Patch _register_native so the test doesn't actually try to load
        # the fake `_native.*.so` (which is not a real shared object).
        with (
            mock.patch.object(_bootstrap, "_http_get", side_effect=fake_get),
            mock.patch.object(_bootstrap, "_register_native") as register_mock,
        ):
            cache = _bootstrap.ensure_native_loaded("3.2.1")

        # `_cache_root()` keys the cache dir on the module's own __version__,
        # not the version arg passed to `ensure_native_loaded`. Reference it
        # directly so this assertion can't go stale on a version bump (the
        # sibling CacheDirTests follow the same pattern).
        self.assertEqual(cache, Path(self._tmp) / _bootstrap.__version__)
        # Native file extracted at cache root, not under a subdirectory.
        extracted = list(cache.glob("_native.*"))
        self.assertEqual(len(extracted), 1)
        register_mock.assert_called_once_with(extracted[0])

    def test_sha256_mismatch_raises(self):
        wheel_bytes = _make_wheel()
        manifest = json.dumps(
            {
                "version": "3.2.1",
                "assets": [
                    {
                        "filename": _bootstrap._wheel_filename("3.2.1"),
                        "sha256": "0" * 64,
                    }
                ],
            }
        ).encode()

        def fake_get(url: str) -> bytes:
            if url.endswith("python-native-manifest.json"):
                return manifest
            return wheel_bytes

        with (
            mock.patch.object(_bootstrap, "_http_get", side_effect=fake_get),
            mock.patch.object(_bootstrap, "_register_native"),
        ):
            with self.assertRaises(_bootstrap.BootstrapError) as cm:
                _bootstrap.ensure_native_loaded("3.2.1")
        self.assertIn("sha256 mismatch", str(cm.exception))

    def test_skip_hash_check_env(self):
        wheel_bytes = _make_wheel()
        manifest = json.dumps(
            {
                "version": "3.2.1",
                "assets": [
                    {
                        "filename": _bootstrap._wheel_filename("3.2.1"),
                        "sha256": "0" * 64,
                    }
                ],
            }
        ).encode()

        def fake_get(url: str) -> bytes:
            if url.endswith("python-native-manifest.json"):
                return manifest
            return wheel_bytes

        os.environ["A3S_CODE_SKIP_HASH_CHECK"] = "1"
        try:
            with (
                mock.patch.object(_bootstrap, "_http_get", side_effect=fake_get),
                mock.patch.object(_bootstrap, "_register_native"),
            ):
                _bootstrap.ensure_native_loaded("3.2.1")
        finally:
            os.environ.pop("A3S_CODE_SKIP_HASH_CHECK", None)

    def test_idempotent_after_first_call(self):
        wheel_bytes = _make_wheel()
        manifest = json.dumps(
            {
                "version": "3.2.1",
                "assets": [
                    {
                        "filename": _bootstrap._wheel_filename("3.2.1"),
                        "sha256": hashlib.sha256(wheel_bytes).hexdigest(),
                    }
                ],
            }
        ).encode()

        call_count = {"n": 0}

        def fake_get(url: str) -> bytes:
            call_count["n"] += 1
            if url.endswith("python-native-manifest.json"):
                return manifest
            return wheel_bytes

        with (
            mock.patch.object(_bootstrap, "_http_get", side_effect=fake_get),
            mock.patch.object(_bootstrap, "_register_native"),
        ):
            _bootstrap.ensure_native_loaded("3.2.1")
            calls_after_first = call_count["n"]
            _bootstrap.ensure_native_loaded("3.2.1")
            self.assertEqual(call_count["n"], calls_after_first,
                             "second call must not re-download")


@unittest.skipUnless(
    os.environ.get("A3S_CODE_BOOTSTRAP_LIVE") == "1",
    "set A3S_CODE_BOOTSTRAP_LIVE=1 to exercise the live download path against GH Releases",
)
class LiveDownloadTests(unittest.TestCase):
    def test_live_fetch_v3_2_0(self):
        # 3.2.0 has 12 wheels on GH Release — pick whichever matches the
        # current runner's platform/interpreter. Skip if the runner isn't
        # one of the supported triplets.
        try:
            _bootstrap._wheel_filename("3.2.0")
        except _bootstrap.BootstrapError as exc:
            self.skipTest(str(exc))
        tmp = Path(tempfile.mkdtemp(prefix="a3s-live-"))
        try:
            os.environ["A3S_CODE_CACHE_DIR"] = str(tmp)
            _bootstrap._LOADED = False
            with mock.patch.object(_bootstrap, "_register_native"):
                cache = _bootstrap.ensure_native_loaded("3.2.0")
            self.assertTrue(any(cache.glob("_native.*")))
        finally:
            os.environ.pop("A3S_CODE_CACHE_DIR", None)
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
