#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-aurora-remaster:ubuntu24}"
TEST_IMAGE_NAME="${TEST_IMAGE_NAME:-aurora-remaster-smoke:latest}"
TREE_DIR="${TREE_DIR:-distro-smoke}"
DISTRO_NAME="${DISTRO_NAME:-Aurora Neon}"
DESKTOP="${DESKTOP:-gnome}"
MODE="${MODE:-auto}"
BUILD_IMAGE="${BUILD_IMAGE:-1}"
CARGO_HOME_IN_CONTAINER="${CARGO_HOME_IN_CONTAINER:-/cargo-cache}"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required for this smoke test" >&2
  exit 1
fi

if [ "$BUILD_IMAGE" = "1" ]; then
  docker build -f "$ROOT_DIR/aurora-distro/Dockerfile.remaster" -t "$IMAGE_NAME" "$ROOT_DIR"
fi

docker build -t "$TEST_IMAGE_NAME" -f - "$ROOT_DIR" <<EOF
FROM $IMAGE_NAME
COPY . /workspace/hyperviser
WORKDIR /workspace/hyperviser
EOF

DOCKER_TTY_ARGS=()
if [ -t 0 ] && [ -t 1 ]; then
  DOCKER_TTY_ARGS=(-it)
fi

docker run --rm "${DOCKER_TTY_ARGS[@]}" \
  --privileged \
  -e CARGO_HOME="$CARGO_HOME_IN_CONTAINER" \
  -e CARGO_REGISTRIES_CRATES_IO_PROTOCOL="${CARGO_REGISTRIES_CRATES_IO_PROTOCOL:-sparse}" \
  -v aurora-cargo-registry:$CARGO_HOME_IN_CONTAINER/registry \
  -v aurora-cargo-git:$CARGO_HOME_IN_CONTAINER/git \
  -v aurora-target-cache:/workspace/hyperviser/target \
  "$TEST_IMAGE_NAME" \
  bash -lc "
    set -euo pipefail
    cd /workspace/hyperviser

    cargo fetch
    cargo test --all --lib
    cargo build --release -p aurora-cli
    ./target/release/aurora version
    ./target/release/aurora detect --format json > /tmp/aurora-detect.json
    ./target/release/aurora status > /tmp/aurora-status.txt

    cargo run -p aurora-distro -- init-tree --out '$TREE_DIR' --distro-name '$DISTRO_NAME' --desktop '$DESKTOP'
    cargo run -p aurora-distro -- scan-system > '$TREE_DIR/build/scan-system.json'
    cargo run -p aurora-distro -- plan-partitions --mode '$MODE' --disk-gb 256 > '$TREE_DIR/build/plan-partitions.json'
    cargo run -p aurora-distro -- check-tools
    cargo run -p aurora-distro -- build-iso --tree '$TREE_DIR' --system-mode '$MODE'

    ISO_PATH=\$(find '$TREE_DIR/output' -maxdepth 1 -name '*.iso' | head -n 1)
    test -n \"\$ISO_PATH\"
    test -f '$TREE_DIR/build/system-profile.json'
    test -f '$TREE_DIR/build/partition-plan.json'

    xorriso -osirrox on -indev \"\$ISO_PATH\" -extract /live/filesystem.squashfs /tmp/filesystem.squashfs >/tmp/xorriso.log 2>&1
    unsquashfs -ll /tmp/filesystem.squashfs > /tmp/filesystem.list
    grep -q 'usr/local/bin/aurora' /tmp/filesystem.list

    echo \"Smoke test passed: runtime built, distro built, ISO created, runtime found inside squashfs.\"
    echo \"ISO: \$ISO_PATH\"
  "
