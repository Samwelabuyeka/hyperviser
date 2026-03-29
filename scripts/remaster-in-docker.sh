#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-aurora-remaster:ubuntu24}"
TREE_DIR="${TREE_DIR:-distro}"
DISTRO_NAME="${DISTRO_NAME:-Aurora Warp}"
DESKTOP="${DESKTOP:-gnome}"
MODE="${MODE:-auto}"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required for this workflow" >&2
  exit 1
fi

docker build -f "$ROOT_DIR/aurora-distro/Dockerfile.remaster" -t "$IMAGE_NAME" "$ROOT_DIR"

docker run --rm -it \
  --privileged \
  -v "$ROOT_DIR:/workspace/hyperviser" \
  "$IMAGE_NAME" \
  bash -lc "
    cargo run -p aurora-distro -- init-tree --out '$TREE_DIR' --distro-name '$DISTRO_NAME' --desktop '$DESKTOP' &&
    cargo run -p aurora-distro -- scan-system &&
    cargo run -p aurora-distro -- plan-partitions --mode '$MODE' --disk-gb 256 &&
    cargo run -p aurora-distro -- check-tools &&
    cargo run -p aurora-distro -- build-iso --tree '$TREE_DIR' --system-mode '$MODE'
  "
