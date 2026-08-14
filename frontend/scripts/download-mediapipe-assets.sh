#!/usr/bin/env bash
# Downloads the MediaPipe Hands model/wasm files so the app can run
# fully offline (no CDN calls at runtime). Run this ONCE while you
# have internet access; after that, no network is needed for hand
# tracking.
#
# Usage: bash scripts/download-mediapipe-assets.sh

set -e
DEST="public/mediapipe/hands"
mkdir -p "$DEST"
BASE_URL="https://cdn.jsdelivr.net/npm/@mediapipe/hands"

FILES=(
  "hands.binarypb"
  "hands_solution_packed_assets.data"
  "hands_solution_packed_assets_loader.js"
  "hands_solution_simd_wasm_bin.js"
  "hands_solution_simd_wasm_bin.wasm"
  "hands_solution_wasm_bin.js"
  "hands_solution_wasm_bin.wasm"
)

for f in "${FILES[@]}"; do
  echo "Downloading $f..."
  curl -fsSL "$BASE_URL/$f" -o "$DEST/$f"
done

echo "Done. MediaPipe Hands assets saved to $DEST"
echo "The app will now load hand tracking fully from disk, no CDN needed."
