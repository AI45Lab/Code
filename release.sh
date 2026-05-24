#!/usr/bin/env bash
# Release script for a3s-code
# Usage: ./release.sh <version>
# Example: ./release.sh 2.0.0

set -euo pipefail

VERSION=${1:-}

if [ -z "$VERSION" ]; then
    echo "Usage: ./release.sh <version>"
    echo "Example: ./release.sh 2.0.0"
    exit 1
fi

if [[ ! "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?$ ]]; then
    echo "❌ Error: version must look like 2.0.0 or 2.0.0-rc.1"
    exit 1
fi

echo "=========================================="
echo "Releasing a3s-code v${VERSION}"
echo "=========================================="
echo ""

# Check if we're in the code submodule
if [ ! -f "core/Cargo.toml" ]; then
    echo "❌ Error: Must run from crates/code directory"
    exit 1
fi

update_node_lockfile() {
    local file="$1"
    if [ ! -f "$file" ]; then
        return 0
    fi

    python3 - "$VERSION" "$file" <<'PY'
import json
import sys
from pathlib import Path

version = sys.argv[1]
path = Path(sys.argv[2])
data = json.loads(path.read_text())

if data.get("name") == "@a3s-lab/code":
    data["version"] = version

for package in data.get("packages", {}).values():
    if package.get("name") == "@a3s-lab/code":
        package["version"] = version
    optional = package.get("optionalDependencies")
    if isinstance(optional, dict):
        for name in list(optional):
            if name.startswith("@a3s-lab/code-"):
                optional[name] = version

path.write_text(json.dumps(data, indent=2) + "\n")
PY
}

# Check for uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    echo "⚠️  Warning: You have uncommitted changes"
    git status --short
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "Step 1: Update version numbers"
echo "----------------------------------------"

# Update Rust crate versions
echo "  Updating core/Cargo.toml..."
sed -i.bak "s/^version = \".*\"/version = \"${VERSION}\"/" core/Cargo.toml
rm -f core/Cargo.toml.bak

echo "  Updating sdk/node/Cargo.toml..."
sed -i.bak "s/^version = \".*\"/version = \"${VERSION}\"/" sdk/node/Cargo.toml
sed -i.bak "s/a3s-code-core = { version = \"[^\"]*\"/a3s-code-core = { version = \"${VERSION}\"/" sdk/node/Cargo.toml
rm -f sdk/node/Cargo.toml.bak

echo "  Updating sdk/python/Cargo.toml..."
sed -i.bak "s/^version = \".*\"/version = \"${VERSION}\"/" sdk/python/Cargo.toml
sed -i.bak "s/a3s-code-core = { version = \"[^\"]*\"/a3s-code-core = { version = \"${VERSION}\"/" sdk/python/Cargo.toml
rm -f sdk/python/Cargo.toml.bak

# Update Node SDK package.json
echo "  Updating sdk/node/package.json..."
sed -i.bak "s/\"version\": \".*\"/\"version\": \"${VERSION}\"/" sdk/node/package.json
sed -i.bak "s/\"@a3s-lab\\/code-\\([^\"]*\\)\": \"[^\"]*\"/\"@a3s-lab\\/code-\\1\": \"${VERSION}\"/g" sdk/node/package.json
rm -f sdk/node/package.json.bak

# Update Python SDK pyproject.toml
echo "  Updating sdk/python/pyproject.toml..."
sed -i.bak "s/^version = \".*\"/version = \"${VERSION}\"/" sdk/python/pyproject.toml
rm -f sdk/python/pyproject.toml.bak

# Update Python bootstrap shim (pyproject.toml + runtime __version__).
# Must stay in lockstep with core so the bootstrap fetches the matching
# native wheel from GH Releases on first import.
echo "  Updating sdk/python-bootstrap/pyproject.toml..."
sed -i.bak "s/^version = \".*\"/version = \"${VERSION}\"/" sdk/python-bootstrap/pyproject.toml
rm -f sdk/python-bootstrap/pyproject.toml.bak

echo "  Updating sdk/python-bootstrap/src/a3s_code/_bootstrap.py..."
sed -i.bak "s/^__version__ = \".*\"/__version__ = \"${VERSION}\"/" sdk/python-bootstrap/src/a3s_code/_bootstrap.py
rm -f sdk/python-bootstrap/src/a3s_code/_bootstrap.py.bak

echo "  Updating Node package lockfiles..."
update_node_lockfile sdk/node/package-lock.json
update_node_lockfile sdk/node/examples/package-lock.json

echo "✅ Version numbers updated"
echo ""

echo "Step 2: Update Cargo.lock"
echo "----------------------------------------"
cargo check --workspace
cargo check --manifest-path sdk/node/Cargo.toml
cargo check --manifest-path sdk/python/Cargo.toml
echo "✅ Cargo.lock updated"
echo ""

echo "Step 3: Format code"
echo "----------------------------------------"
cargo fmt --all
echo "✅ Code formatted"
echo ""

echo "Step 4: Run tests"
echo "----------------------------------------"
REQUIRE_REAL_PROVIDER=1 scripts/release_preflight.sh
echo "✅ Tests passed"
echo ""

echo "Step 5: Git commit and tag"
echo "----------------------------------------"

# Show changes
echo "Changes to be committed:"
git diff --stat

echo ""
read -p "Commit these changes? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Aborted"
    exit 1
fi

# Commit changes
git add -A
git commit -m "chore: bump version to ${VERSION}

- Update Rust, Node.js, and Python package versions
- Refresh release validation for A3S Code
- Require real-provider ACL env integration before tagging
"

# Create tag
git tag -a "v${VERSION}" -m "Release v${VERSION}

## Tests
- cargo test -p a3s-code-core --lib
- cargo test -p a3s-code-core --tests
- cargo test -p a3s-code-core --features ahp --test test_ahp_idle_with_llm
- git diff --check
- scripts/check_release_versions.sh
- REQUIRE_REAL_PROVIDER=1 scripts/release_preflight.sh
"

echo "✅ Committed and tagged"
echo ""

echo "Step 6: Push to GitHub"
echo "----------------------------------------"
echo "Ready to push:"
echo "  - Commit: $(git log -1 --oneline)"
echo "  - Tag: v${VERSION}"
echo ""
read -p "Push to origin? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Aborted"
    echo ""
    echo "To push manually:"
    echo "  git push origin main"
    echo "  git push origin v${VERSION}"
    exit 1
fi

git push origin main
git push origin "v${VERSION}"

echo "✅ Pushed to GitHub"
echo ""

echo "=========================================="
echo "✅ Release v${VERSION} completed!"
echo "=========================================="
echo ""
echo "GitHub Actions will now:"
echo "  1. Run CI checks"
echo "  2. Publish to crates.io"
echo "  3. Publish Node SDK to npm"
echo "  4. Build Python SDK wheels and attach to GitHub Release (PyPI no longer used)"
echo "  5. Create GitHub Release"
echo ""
echo "Monitor progress at:"
echo "  https://github.com/A3S-Lab/Code/actions"
echo ""
