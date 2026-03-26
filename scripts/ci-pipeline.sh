#!/bin/bash
#
# AURORA CI/CD Pipeline
# Comprehensive build, test, and validation pipeline
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
RUSTFLAGS="-C target-cpu=native -C opt-level=3"
export RUSTFLAGS

# Counters
ERRORS=0
WARNINGS=0

# Print functions
print_header() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
    ((ERRORS++))
}

print_warning() {
    echo -e "${YELLOW}!${NC} $1"
    ((WARNINGS++))
}

# Check Rust installation
check_rust() {
    print_header "Checking Rust Installation"
    
    if ! command -v rustc &> /dev/null; then
        print_error "Rust not installed"
        exit 1
    fi
    
    local version=$(rustc --version)
    print_success "Rust: $version"
    
    # Check minimum version
    local min_version="1.75.0"
    local current=$(echo "$version" | cut -d' ' -f2)
    
    if [ "$(printf '%s\n' "$min_version" "$current" | sort -V | head -n1)" != "$min_version" ]; then
        print_error "Rust version $current is older than minimum $min_version"
        exit 1
    fi
    
    print_success "Rust version check passed"
}

# Format check
check_format() {
    print_header "Checking Code Format"
    
    if ! cargo fmt -- --check; then
        print_error "Code formatting issues found"
        print_warning "Run 'cargo fmt' to fix"
        return 1
    fi
    
    print_success "Code format check passed"
}

# Build check
check_build() {
    print_header "Checking Build"
    
    if ! cargo check --all-targets 2>&1; then
        print_error "Build check failed"
        return 1
    fi
    
    print_success "Build check passed"
}

# Clippy check
check_clippy() {
    print_header "Running Clippy"
    
    local clippy_output
    clippy_output=$(cargo clippy --all-targets -- -D warnings 2>&1) || true
    
    if echo "$clippy_output" | grep -q "error"; then
        print_error "Clippy errors found"
        echo "$clippy_output"
        return 1
    fi
    
    if echo "$clippy_output" | grep -q "warning"; then
        print_warning "Clippy warnings found"
        echo "$clippy_output"
    fi
    
    print_success "Clippy check passed"
}

# Check for unsafe code
check_unsafe() {
    print_header "Checking for Unsafe Code"
    
    local unsafe_count
    unsafe_count=$(grep -r "unsafe" --include="*.rs" . 2>/dev/null | grep -v "target/" | grep -v "// SAFETY" | wc -l)
    
    if [ "$unsafe_count" -gt 0 ]; then
        print_warning "Found $unsafe_count unsafe blocks"
        grep -r "unsafe" --include="*.rs" . 2>/dev/null | grep -v "target/" | head -20
    else
        print_success "No unsafe code found"
    fi
}

# Check for TODO/FIXME/placeholder
check_todos() {
    print_header "Checking for TODOs and Placeholders"
    
    local todo_count
    todo_count=$(grep -ri "TODO\|FIXME\|unimplemented!\|panic!\|placeholder" --include="*.rs" . 2>/dev/null | grep -v "target/" | wc -l)
    
    if [ "$todo_count" -gt 0 ]; then
        print_warning "Found $todo_count TODOs/placeholders"
        grep -ri "TODO\|FIXME\|unimplemented!\|panic!\|placeholder" --include="*.rs" . 2>/dev/null | grep -v "target/" | head -20
    else
        print_success "No TODOs or placeholders found"
    fi
}

# Run tests
run_tests() {
    print_header "Running Tests"
    
    if ! cargo test --all 2>&1; then
        print_error "Tests failed"
        return 1
    fi
    
    print_success "All tests passed"
}

# Run benchmarks
run_benchmarks() {
    print_header "Running Benchmarks"
    
    if ! cargo bench 2>&1; then
        print_warning "Benchmarks failed or not available"
        return 0
    fi
    
    print_success "Benchmarks completed"
}

# Build release
build_release() {
    print_header "Building Release"
    
    if ! cargo build --release 2>&1; then
        print_error "Release build failed"
        return 1
    fi
    
    print_success "Release build successful"
    
    # Show binary sizes
    echo ""
    echo "Binary sizes:"
    ls -lh target/release/aurora 2>/dev/null | awk '{print "  " $5, $9}'
    ls -lh target/release/*.so 2>/dev/null | awk '{print "  " $5, $9}'
}

# Documentation check
check_docs() {
    print_header "Checking Documentation"
    
    if ! cargo doc --no-deps 2>&1; then
        print_warning "Documentation build failed"
        return 0
    fi
    
    # Check for missing docs
    local missing_docs
    missing_docs=$(cargo doc --no-deps 2>&1 | grep -c "missing" || true)
    
    if [ "$missing_docs" -gt 0 ]; then
        print_warning "$missing_docs missing documentation items"
    else
        print_success "Documentation check passed"
    fi
}

# Security audit
run_audit() {
    print_header "Running Security Audit"
    
    if command -v cargo-audit &> /dev/null; then
        if ! cargo audit 2>&1; then
            print_warning "Security audit found issues"
        else
            print_success "Security audit passed"
        fi
    else
        print_warning "cargo-audit not installed"
    fi
}

# Check module completeness
check_completeness() {
    print_header "Checking Module Completeness"
    
    local modules=(
        "aurora-core"
        "aurora-profiler"
        "aurora-cpu"
        "aurora-gpu"
        "aurora-orchestrator"
        "aurora-tensor"
        "aurora-memory"
        "aurora-linux"
        "aurora-autotune"
        "aurora-api"
        "aurora-cli"
    )
    
    for module in "${modules[@]}"; do
        if [ -d "$module" ]; then
            # Check for lib.rs or main.rs
            if [ -f "$module/src/lib.rs" ] || [ -f "$module/src/main.rs" ]; then
                print_success "$module: present"
            else
                print_error "$module: missing source files"
            fi
            
            # Check for tests
            if [ -d "$module/src" ] && grep -q "#\[cfg(test)\]" "$module/src/"*.rs 2>/dev/null; then
                print_success "$module: has tests"
            else
                print_warning "$module: missing tests"
            fi
        else
            print_error "$module: directory not found"
        fi
    done
}

# Performance validation
validate_performance() {
    print_header "Validating Performance"
    
    # Check for SIMD usage
    local simd_usage
    simd_usage=$(grep -r "_mm_\|_mm256_\|_mm512_" --include="*.rs" . 2>/dev/null | grep -v "target/" | wc -l)
    
    if [ "$simd_usage" -gt 0 ]; then
        print_success "SIMD intrinsics found: $simd_usage usages"
    else
        print_warning "No SIMD intrinsics found"
    fi
    
    # Check for parallel processing
    local parallel_usage
    parallel_usage=$(grep -r "rayon\|crossbeam\|ThreadPool" --include="*.rs" . 2>/dev/null | grep -v "target/" | wc -l)
    
    if [ "$parallel_usage" -gt 0 ]; then
        print_success "Parallel processing found: $parallel_usage usages"
    else
        print_warning "No parallel processing found"
    fi
}

# Generate report
generate_report() {
    print_header "CI Pipeline Report"
    
    echo ""
    echo "Summary:"
    echo "  Errors:   $ERRORS"
    echo "  Warnings: $WARNINGS"
    echo ""
    
    if [ "$ERRORS" -eq 0 ]; then
        echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
        echo -e "${GREEN}  ALL CHECKS PASSED${NC}"
        echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
        return 0
    else
        echo -e "${RED}═══════════════════════════════════════════════════════════${NC}"
        echo -e "${RED}  PIPELINE FAILED${NC}"
        echo -e "${RED}═══════════════════════════════════════════════════════════${NC}"
        return 1
    fi
}

# Main pipeline
main() {
    echo -e "${BLUE}"
    cat << "EOF"
╔═══════════════════════════════════════════════════════════╗
║              AURORA CI/CD Pipeline                        ║
╚═══════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
    
    # Parse arguments
    local skip_benchmarks=false
    local skip_audit=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-benchmarks)
                skip_benchmarks=true
                shift
                ;;
            --skip-audit)
                skip_audit=true
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --skip-benchmarks  Skip benchmark execution"
                echo "  --skip-audit       Skip security audit"
                echo "  --help             Show this help"
                echo ""
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Run checks
    check_rust
    check_format
    check_build
    check_clippy
    check_unsafe
    check_todos
    run_tests
    
    if [ "$skip_benchmarks" = false ]; then
        run_benchmarks
    fi
    
    build_release
    check_docs
    
    if [ "$skip_audit" = false ]; then
        run_audit
    fi
    
    check_completeness
    validate_performance
    
    generate_report
}

# Run main
main "$@"
