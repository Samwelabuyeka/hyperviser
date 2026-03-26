#!/bin/bash
#
# AURORA Build Script
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Build type
BUILD_TYPE="${1:-release}"

echo -e "${BLUE}Building AURORA...${NC}"
echo "  Build type: $BUILD_TYPE"
echo ""

# Check Rust installation
if ! command -v cargo &> /dev/null; then
    echo -e "${RED}Cargo not found. Please install Rust.${NC}"
    exit 1
fi

echo "Rust version: $(rustc --version)"
echo "Cargo version: $(cargo --version)"
echo ""

# Build
cd "$(dirname "$0")"

if [ "$BUILD_TYPE" = "release" ]; then
    echo "Building release version..."
    cargo build --release
else
    echo "Building debug version..."
    cargo build
fi

echo ""
echo -e "${GREEN}Build complete!${NC}"
echo ""
echo "Binary location:"
if [ "$BUILD_TYPE" = "release" ]; then
    echo "  target/release/aurora"
else
    echo "  target/debug/aurora"
fi
echo ""
echo "Run with:"
echo "  ./target/$BUILD_TYPE/aurora --help"
