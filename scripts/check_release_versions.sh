#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$WORKSPACE"

python3 - "${1:-}" <<'PY'
import json
import re
import sys
from pathlib import Path

expected = sys.argv[1].strip()
errors = []


def read(path):
    return Path(path).read_text()


def fail(message):
    errors.append(message)


def first_manifest_version(path):
    match = re.search(r'^version\s*=\s*"([^"]+)"', read(path), re.MULTILINE)
    if not match:
        fail(f"{path}: missing package version")
        return None
    return match.group(1)


def check_equal(label, actual):
    if actual != expected:
        fail(f"{label}: expected {expected}, found {actual!r}")


def check_manifest_version(path):
    actual = first_manifest_version(path)
    if actual is not None:
        check_equal(path, actual)


def check_core_dependency(path):
    text = read(path)
    pattern = r'a3s-code-core\s*=\s*\{[^}]*version\s*=\s*"([^"]+)"'
    match = re.search(pattern, text)
    if not match:
        fail(f"{path}: missing a3s-code-core dependency version")
        return
    check_equal(f"{path} a3s-code-core", match.group(1))


def check_package_json(path):
    data = json.loads(read(path))
    check_equal(f"{path} version", data.get("version"))
    optional = data.get("optionalDependencies") or {}
    for name, value in optional.items():
        if name.startswith("@a3s-lab/code-"):
            check_equal(f"{path} optionalDependency {name}", value)


def check_pyproject(path):
    match = re.search(r'^version\s*=\s*"([^"]+)"', read(path), re.MULTILINE)
    if not match:
        fail(f"{path}: missing project version")
        return
    check_equal(path, match.group(1))


def check_cargo_lock(path):
    text = read(path)
    pattern = r'\[\[package\]\]\s*\nname\s*=\s*"a3s-code-core"\s*\nversion\s*=\s*"([^"]+)"'
    match = re.search(pattern, text)
    if not match:
        fail(f"{path}: missing a3s-code-core package entry")
        return
    check_equal(f"{path} a3s-code-core", match.group(1))


def check_node_lockfile(path):
    data = json.loads(read(path))
    if data.get("name") == "@a3s-lab/code":
        check_equal(f"{path} root version", data.get("version"))

    packages = data.get("packages") or {}
    for key, package in packages.items():
        if not isinstance(package, dict):
            continue
        if package.get("name") == "@a3s-lab/code":
            check_equal(f"{path} package {key or '<root>'}", package.get("version"))
        optional = package.get("optionalDependencies") or {}
        for name, value in optional.items():
            if name.startswith("@a3s-lab/code-"):
                check_equal(f"{path} package {key or '<root>'} optionalDependency {name}", value)


def check_bootstrap_runtime_version(path):
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', read(path), re.MULTILINE)
    if not match:
        fail(f"{path}: missing __version__ literal")
        return
    check_equal(f"{path} __version__", match.group(1))


if not expected:
    expected = first_manifest_version("core/Cargo.toml") or ""

if not re.fullmatch(r'\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?', expected):
    fail(f"release version must be semver-like, found {expected!r}")

check_manifest_version("core/Cargo.toml")
check_manifest_version("sdk/node/Cargo.toml")
check_manifest_version("sdk/python/Cargo.toml")
check_core_dependency("sdk/node/Cargo.toml")
check_core_dependency("sdk/python/Cargo.toml")
check_package_json("sdk/node/package.json")
check_pyproject("sdk/python/pyproject.toml")
check_pyproject("sdk/python-bootstrap/pyproject.toml")
check_bootstrap_runtime_version("sdk/python-bootstrap/src/a3s_code/_bootstrap.py")
check_cargo_lock("Cargo.lock")
check_node_lockfile("sdk/node/package-lock.json")
check_node_lockfile("sdk/node/examples/package-lock.json")

if errors:
    for error in errors:
        print(error, file=sys.stderr)
    sys.exit(1)

print(f"release versions are consistent at {expected}")
PY
