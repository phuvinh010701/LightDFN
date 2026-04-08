#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Build LightDFN WASM module
#
# Prerequisites:
#   curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
#   rustup target add wasm32-unknown-unknown
#
# Output: ../pkg/  (lightdfn_wasm.js + lightdfn_wasm_bg.wasm + .d.ts)
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")"

# ── check tools ───────────────────────────────────────────────────────────────
if ! command -v wasm-pack &>/dev/null; then
    echo "ERROR: wasm-pack not found."
    echo "  Install: curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh"
    exit 1
fi

if ! rustup target list --installed | grep -q wasm32-unknown-unknown; then
    echo "Adding wasm32-unknown-unknown target..."
    rustup target add wasm32-unknown-unknown
fi

# ── build ─────────────────────────────────────────────────────────────────────
# WASM target features and stack size are set in .cargo/config.toml
echo "Building LightDFN WASM (release, SIMD128 + relaxed-simd + bulk-memory)..."
wasm-pack build \
    --target no-modules \
    --release \
    --out-dir ../pkg \
    --out-name lightdfn_wasm

echo ""
echo "Build complete!"
echo "Output files:"
ls -lh ../pkg/lightdfn_wasm*.wasm ../pkg/lightdfn_wasm*.js 2>/dev/null || true