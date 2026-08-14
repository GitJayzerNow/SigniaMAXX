# Downloads the MediaPipe Hands model/wasm files so the app can run
# fully offline (no CDN calls at runtime). Run this ONCE while you
# have internet access; after that, no network is needed for hand
# tracking.
#
# Usage (from the frontend/ folder, in PowerShell):
#   .\scripts\download-mediapipe-assets.ps1

$ErrorActionPreference = "Stop"

$dest = "public\mediapipe\hands"
New-Item -ItemType Directory -Force -Path $dest | Out-Null

$baseUrl = "https://cdn.jsdelivr.net/npm/@mediapipe/hands"

$files = @(
    "hands.binarypb",
    "hands_solution_packed_assets.data",
    "hands_solution_packed_assets_loader.js",
    "hands_solution_simd_wasm_bin.js",
    "hands_solution_simd_wasm_bin.wasm",
    "hands_solution_wasm_bin.js",
    "hands_solution_wasm_bin.wasm"
)

foreach ($f in $files) {
    Write-Host "Downloading $f..."
    Invoke-WebRequest -Uri "$baseUrl/$f" -OutFile "$dest\$f"
}

Write-Host "Done. MediaPipe Hands assets saved to $dest"
Write-Host "The app will now load hand tracking fully from disk, no CDN needed."
