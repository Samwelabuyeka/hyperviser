#!/usr/bin/env bash
set -euo pipefail

cd /workspace/hyperviser

if [ ! -f Cargo.toml ]; then
  echo "Expected repository to be mounted at /workspace/hyperviser" >&2
  exit 1
fi

exec "$@"
