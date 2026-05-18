#!/usr/bin/env bash
# Integration test runner for SDK confirmation_inheritance feature
# Reads LLM config from .a3s/config.acl and runs tests for both SDKs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
CONFIG_FILE="$PROJECT_ROOT/.a3s/config.acl"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "ERROR: Config file not found: $CONFIG_FILE"
  exit 1
fi

echo "==================================================================="
echo "SDK Integration Tests"
echo "==================================================================="
echo "Config: $CONFIG_FILE"
echo ""

# Export config file path (no secrets here)
export A3S_CONFIG_FILE="$CONFIG_FILE"

# Control test behavior
export A3S_CODE_SDK_REAL_AGENT_SMOKE="${A3S_CODE_SDK_REAL_AGENT_SMOKE:-0}"
export A3S_CODE_SDK_REAL_TIMEOUT_MS="${A3S_CODE_SDK_REAL_TIMEOUT_MS:-180000}"

echo "Test mode: A3S_CODE_SDK_REAL_AGENT_SMOKE=$A3S_CODE_SDK_REAL_AGENT_SMOKE"
echo "  (Set to 1 to enable real LLM calls)"
echo ""

# Test 1: Node SDK
echo "-------------------------------------------------------------------"
echo "Test 1: Node SDK"
echo "-------------------------------------------------------------------"
cd "$SCRIPT_DIR"
if [[ ! -f "index.js" ]]; then
  echo "ERROR: Node SDK not built. Run: npm run build:debug"
  exit 1
fi

npm test
npm run test:helpers
npx tsc --noEmit -p examples/tsconfig.json
node test_confirmation_inheritance.mjs
node examples/basic/test_real_config_env_sdk.mjs
echo ""

# Test 2: Python SDK
echo "-------------------------------------------------------------------"
echo "Test 2: Python SDK"
echo "-------------------------------------------------------------------"
cd "$PROJECT_ROOT/crates/code/sdk/python"

# Check if Python module is built
PYTHON_BIN="python3"
if [[ -f ".venv/bin/python3" ]]; then
  PYTHON_BIN=".venv/bin/python3"
  echo "Using Python from venv: $PYTHON_BIN"
fi

if ! $PYTHON_BIN -c "from a3s_code import Agent, LocalWorkspaceBackend" 2>/dev/null; then
  echo "WARNING: Python SDK not built. Building with maturin develop..."
  if ! command -v maturin &> /dev/null; then
    echo "ERROR: maturin not found. Install with: pip install maturin"
    exit 1
  fi
  maturin develop --quiet
fi

$PYTHON_BIN test_confirmation_inheritance.py
$PYTHON_BIN tests/real_config_env_sdk.py
echo ""

echo "==================================================================="
echo "All SDK integration tests passed ✓"
echo "==================================================================="
