#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-aurora-remaster:ubuntu24}"
CONTAINER_NAME="${CONTAINER_NAME:-aurora-remaster-build}"
TREE_DIR="${TREE_DIR:-distro}"
DISTRO_NAME="${DISTRO_NAME:-Aurora Warp}"
DESKTOP="${DESKTOP:-gnome}"
MODE="${MODE:-auto}"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required for this workflow" >&2
  exit 1
fi

docker build -f "$ROOT_DIR/aurora-distro/Dockerfile.remaster" -t "$IMAGE_NAME" "$ROOT_DIR"

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
mkdir -p "$ROOT_DIR/$TREE_DIR"

cleanup() {
  local status=$?
  if docker ps -a --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
    mkdir -p "$ROOT_DIR/$TREE_DIR/output" "$ROOT_DIR/$TREE_DIR/build"
    docker cp "$CONTAINER_NAME:/workspace/hyperviser/$TREE_DIR/output/." "$ROOT_DIR/$TREE_DIR/output" >/dev/null 2>&1 || true
    docker cp "$CONTAINER_NAME:/workspace/hyperviser/$TREE_DIR/build/." "$ROOT_DIR/$TREE_DIR/build" >/dev/null 2>&1 || true
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  fi
  return "$status"
}
trap cleanup EXIT

docker run --name "$CONTAINER_NAME" --privileged \
  -v "$ROOT_DIR/$TREE_DIR:/workspace/hyperviser/$TREE_DIR" \
  -t "$IMAGE_NAME" bash -lc "
  export RUSTC=/root/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin/rustc
  export CARGO_NET_RETRY=10
  export CARGO_HTTP_TIMEOUT=120
  export CARGO_REGISTRIES_CRATES_IO_PROTOCOL=sparse
  cargo run -p aurora-distro -- init-tree --out '$TREE_DIR' --distro-name '$DISTRO_NAME' --desktop '$DESKTOP' &&
  cargo run -p aurora-distro -- scan-system &&
  cargo run -p aurora-distro -- plan-partitions --mode '$MODE' --disk-gb 256 &&
  cargo run -p aurora-distro -- check-tools &&
  cargo run -p aurora-distro -- build-iso --tree '$TREE_DIR' --system-mode '$MODE'
"

echo "Copied build artifacts to $ROOT_DIR/$TREE_DIR/output"
