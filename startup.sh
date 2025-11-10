#!/bin/bash


# =============================================================================
# DXNN-OCR Benchmark Environment Setup and Execution Script
# Mirroring PP-OCRv5-Cpp-Baseline methodology for compatibility
# =============================================================================
# 
# Automated Full Pipeline:
#   1. Builds dx_rt if --dx_rt option is provided
#   2. Applies RT optimization with set_env.sh
#   3. Runs sync benchmark
#   4. Runs async benchmark 
#   5. Compares results automatically
#
# Usage: ./startup.sh [--dx_rt <path_to_dx_rt>]

# Parse command line arguments
DX_RT_PATH=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --dx_rt)
            DX_RT_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./startup.sh [--dx_rt <path_to_dx_rt>]"
            exit 1
            ;;
    esac
done

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
LOG_DIR="$PROJECT_ROOT/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create necessary directories
mkdir -p "$LOG_DIR"
mkdir -p "$PROJECT_ROOT/output" "$PROJECT_ROOT/output/json" "$PROJECT_ROOT/output/vis"
mkdir -p "$PROJECT_ROOT/images"

# Logging function - matches PP-OCRv5 format
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/dxnn_benchmark_${TIMESTAMP}.log"
}

# Error handling - exit on critical errors only
set -e
trap 'log "ERROR: Script failed at line $LINENO"' ERR

log "=== Starting DXNN-OCR Benchmark Environment Setup ==="
log "Project Root: $PROJECT_ROOT"
log "Log Directory: $LOG_DIR"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Basic environment checks
log "=== Environment Prerequisites Check ==="

# Check Python
if ! command_exists python3; then
    log "ERROR: python3 is not installed. Please install python3.11+"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "0.0")
if [[ $(echo -e "$PYTHON_VERSION\n3.11" | sort -V | head -n 1) != "3.11" ]]; then
    log "WARNING: Python version is $PYTHON_VERSION, recommended 3.11+"
fi
log "✓ Python: $(python3 --version)"

# Check pip (for venv)
if ! command_exists pip3; then
    log "ERROR: pip3 is not installed. Please install pip3 first."
    exit 1
fi
log "✓ Pip: $(pip3 --version)"

# Check essential tools
for tool in wget unzip; do
    if ! command_exists $tool; then
        log "ERROR: $tool is not installed. Please install: sudo apt install $tool"
        exit 1
    fi
done
log "✓ Essential tools: wget, unzip"

# =============================================================================
# Python Virtual Environment Setup
# =============================================================================

# Define virtual environment path
VENV_DIR="$PROJECT_ROOT/venv"
log "Target virtual environment: $VENV_DIR"

# Check if virtual environment already exists
if [[ -d "$VENV_DIR" ]]; then
    log "✓ Virtual environment already exists at $VENV_DIR"
else
    log "Creating new virtual environment: $VENV_DIR"
    
    # Create virtual environment with Python 3.11 (if available) or default python3
    if command_exists python3.11; then
        python3.11 -m venv "$VENV_DIR"
        log "✓ Using Python 3.11 for virtual environment"
    else
        python3 -m venv "$VENV_DIR"
        log "✓ Using default Python 3 for virtual environment"
    fi
    
    if [[ $? -eq 0 ]]; then
        log "✓ Virtual environment created successfully"
    else
        log "ERROR: Failed to create virtual environment"
        exit 1
    fi
fi

# Activate the virtual environment
log "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Verify environment activation
if [[ -z "$VIRTUAL_ENV" ]]; then
    log "ERROR: Failed to activate virtual environment"
    exit 1
fi
log "✓ Virtual environment activated: $VIRTUAL_ENV"

# Upgrade pip to latest version
log "Upgrading pip to latest version..."
pip install --upgrade pip

# Set library paths for runtime (following PP-OCRv5 pattern)
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:$LD_LIBRARY_PATH"
log "✓ Library paths configured"

# =============================================================================
# DX_RT Build (if specified)
# =============================================================================

if [[ -n "$DX_RT_PATH" ]]; then
    log "=== Building DX_RT ==="
    log "DX_RT Path: $DX_RT_PATH"
    
    # Check if dx_rt path exists
    if [[ ! -d "$DX_RT_PATH" ]]; then
        log "ERROR: DX_RT path does not exist: $DX_RT_PATH"
        exit 1
    fi
    
    # Check if build.sh exists in dx_rt path
    if [[ ! -f "$DX_RT_PATH/build.sh" ]]; then
        log "ERROR: build.sh not found in DX_RT path: $DX_RT_PATH/build.sh"
        exit 1
    fi
    
    # Build dx_rt
    log "Building DX_RT with ./build.sh --clean..."
    cd "$DX_RT_PATH"
    ./build.sh --clean 2>&1 | tee -a "$LOG_DIR/dx_rt_build_${TIMESTAMP}.log"
    BUILD_EXIT_CODE=${PIPESTATUS[0]}
    cd "$PROJECT_ROOT"
    
    if [[ $BUILD_EXIT_CODE -eq 0 ]]; then
        log "✓ DX_RT build completed successfully"
    else
        log "ERROR: DX_RT build failed with exit code $BUILD_EXIT_CODE"
        exit 1
    fi
else
    log "⚠ DX_RT path not specified, skipping DX_RT build"
fi

# =============================================================================
# Dependencies Installation
# =============================================================================

log "=== Installing Dependencies ==="

# Check if requirements.txt exists
if [[ ! -f "$PROJECT_ROOT/requirements.txt" ]]; then
    log "ERROR: requirements.txt not found in project root"
    exit 1
fi

# Install Python dependencies
log "Installing Python packages from requirements.txt..."
pip install -r "$PROJECT_ROOT/requirements.txt" 2>&1 | tee -a "$LOG_DIR/pip_install_${TIMESTAMP}.log"

if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
    log "✓ Python dependencies installed successfully"
else
    log "ERROR: Failed to install Python dependencies"
    exit 1
fi

# Verify essential packages are installed
REQUIRED_PACKAGES=("numpy")
log "Verifying essential packages..."
for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if python -c "import ${pkg//-/_}" 2>/dev/null; then
        log "✓ Package verified: $pkg"
    else
        log "⚠ Warning: Package not found or not importable: $pkg"
    fi
done

# =============================================================================
# Dataset Setup (following C++ Baseline methodology)
# =============================================================================

log "=== Verifying C++ Baseline Dataset ==="

verify_dataset() {
    local dataset_dir="$PROJECT_ROOT/images"
    
    # Check if labels.json exists (copied from C++ baseline)
    if [[ ! -f "$dataset_dir/labels.json" ]]; then
        log "ERROR: labels.json not found. Please ensure you have copied the dataset from PP-OCRv5-Cpp-Baseline"
        log "Expected path: $dataset_dir/labels.json"
        return 1
    fi
    
    log "✓ labels.json found"
    
    # Verify we have image files
    local image_count=$(find "$dataset_dir" -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" | wc -l)
    if [[ $image_count -lt 10 ]]; then
        log "WARNING: Only $image_count images found. Please ensure image files are copied from C++ baseline"
    else
        log "✓ Dataset ready: $image_count images found"
    fi
    
    # Check if we have the correct format by examining labels.json structure
    python3 -c "
import json
try:
    with open('$dataset_dir/labels.json', 'r') as f:
        data = json.load(f)
    print(f'Labels file contains {len(data)} entries')
    # Check if it's the expected format (image names as keys)
    sample_key = list(data.keys())[0] if data else None
    if sample_key and '.png' in sample_key:
        print('✓ Format appears correct (C++ baseline format)')
    else:
        print('⚠ Format may not be correct')
except Exception as e:
    print(f'ERROR reading labels.json: {e}')
    exit(1)
" || return 1
    
    return 0
}

# Verify dataset
if ! verify_dataset; then
    log "ERROR: Dataset verification failed"
    log "Please ensure you have:"
    log "1. Copied labels.json from PP-OCRv5-Cpp-Baseline/images/"
    log "2. Copied image files from PP-OCRv5-Cpp-Baseline/images/"
    exit 1
fi

# =============================================================================
# Environment Verification
# =============================================================================

log "=== Environment Verification ==="

# Verify Python installation in virtual environment
PYTHON_VERSION=$(python --version 2>&1)
PYTHON_PATH=$(which python)
log "✓ Python version: $PYTHON_VERSION"
log "✓ Python path: $PYTHON_PATH"

# =============================================================================
# RT Optimization Configuration
# =============================================================================

log "=== Applying DXRT Optimization Settings ==="

# Apply optimal RT settings for NPU performance
if [[ -f "$PROJECT_ROOT/set_env.sh" ]]; then
    log "Applying DXRT optimization settings..."
    source "$PROJECT_ROOT/set_env.sh" 1 3 1 18 1 4
    log "✓ RT optimization applied (CUSTOM_INTER_OP_THREADS_COUNT=1, CUSTOM_INTRA_OP_THREADS_COUNT=3, etc.)"
else
    log "⚠ Warning: set_env.sh not found, skipping RT optimization"
fi

# =============================================================================
# Benchmark Execution (Automated: Sync + Async + Comparison)
# =============================================================================

log "=== Starting Automated DXNN-OCR Benchmark Pipeline ==="

# Create separate output directories for sync and async results
SYNC_OUTPUT="$PROJECT_ROOT/output_sync"
ASYNC_OUTPUT="$PROJECT_ROOT/output_async"
mkdir -p "$SYNC_OUTPUT" "$ASYNC_OUTPUT"

# Function to run sync benchmark
run_sync_benchmark() {
    log "=== Phase 1: Executing SYNC benchmark (sequential processing) ==="
    local benchmark_cmd="python -u scripts/dxnn_benchmark.py --directory images --ground-truth images/labels.json --output output_sync --runs 3 --save-individual --mode sync"
    log "Sync benchmark command: $benchmark_cmd"
    $benchmark_cmd 2>&1 | tee -a "$LOG_DIR/sync_benchmark_${TIMESTAMP}.log"
    local exit_code=${PIPESTATUS[0]}
    if [[ $exit_code -eq 0 ]]; then
        log "✓ Sync benchmark completed successfully"
        return 0
    else
        log "ERROR: Sync benchmark failed with exit code $exit_code"
        return 1
    fi
}

# Function to run async benchmark
run_async_benchmark() {
    log "=== Phase 2: Executing ASYNC benchmark (parallel processing) ==="
    local benchmark_cmd="python -u scripts/dxnn_benchmark.py --directory images --ground-truth images/labels.json --output output_async --runs 3 --save-individual --mode async"
    log "Async benchmark command: $benchmark_cmd"
    $benchmark_cmd 2>&1 | tee -a "$LOG_DIR/async_benchmark_${TIMESTAMP}.log"
    local exit_code=${PIPESTATUS[0]}
    if [[ $exit_code -eq 0 ]]; then
        log "✓ Async benchmark completed successfully"
        return 0
    else
        log "ERROR: Async benchmark failed with exit code $exit_code"
        return 1
    fi
}

# Function to compare results
compare_results() {
    log "=== Phase 3: Comparing Sync vs Async Performance ==="
    if [[ -f "scripts/compare_sync_async.py" ]]; then
        local compare_cmd="python scripts/compare_sync_async.py output_sync output_async"
        log "Comparison command: $compare_cmd"
        $compare_cmd 2>&1 | tee -a "$LOG_DIR/comparison_${TIMESTAMP}.log"
        local exit_code=${PIPESTATUS[0]}
        if [[ $exit_code -eq 0 ]]; then
            log "✓ Performance comparison completed successfully"
            return 0
        else
            log "ERROR: Performance comparison failed with exit code $exit_code"
            return 1
        fi
    else
        log "⚠ Warning: scripts/compare_sync_async.py not found, skipping comparison"
        return 0
    fi
}

# Execute complete benchmark pipeline
log "Starting automated benchmark pipeline..."

# Phase 1: Sync benchmark
if ! run_sync_benchmark; then
    log "ERROR: Sync benchmark failed, aborting pipeline"
    exit 1
fi

# Phase 2: Async benchmark  
if ! run_async_benchmark; then
    log "ERROR: Async benchmark failed, aborting pipeline"
    exit 1
fi

# Phase 3: Compare results
if ! compare_results; then
    log "ERROR: Results comparison failed"
    exit 1
fi

# =============================================================================
# Cleanup and Summary
# =============================================================================

log "=== Automated Benchmark Pipeline Completion Summary ==="
log "✓ Environment: $VIRTUAL_ENV"
log "✓ Dataset: C++ baseline format (labels.json)"
log "✓ RT Optimization: Applied via set_env.sh"
log "✓ Sync Results: output_sync/"
log "✓ Async Results: output_async/"
log "✓ Performance Comparison: Completed"
log "✓ Logs: $LOG_DIR/"

# Display final performance summary
if [[ -f "output_sync/benchmark_summary.json" && -f "output_async/benchmark_summary.json" ]]; then
    log "=== Final Performance Summary ==="
    python -c "
import json

try:
    with open('output_sync/benchmark_summary.json', 'r') as f:
        sync_results = json.load(f)
    with open('output_async/benchmark_summary.json', 'r') as f:
        async_results = json.load(f)
    
    sync_time = sync_results.get('performance', {}).get('avg_inference_time_ms', 0)
    async_time = async_results.get('performance', {}).get('avg_inference_time_ms', 0)
    
    if sync_time > 0 and async_time > 0:
        speedup = sync_time / async_time
        print(f'✅ SYNC mode: {sync_time:.2f} ms average inference time')
        print(f'✅ ASYNC mode: {async_time:.2f} ms average inference time')
        print(f'🚀 Performance improvement: {speedup:.2f}x speedup with async mode')
    else:
        print('⚠ Could not calculate performance comparison')
        
except Exception as e:
    print(f'Could not display performance summary: {e}')
"
fi

END_TIME=$(date)
log "Automated benchmark pipeline completed at: $END_TIME"
log "Total execution log: $LOG_DIR/dxnn_benchmark_${TIMESTAMP}.log"

echo ""
echo "=========================================================="
echo "DXNN-OCR Automated Benchmark Pipeline Complete! 🎉"
echo "=========================================================="
echo ""
echo "Environment: $VIRTUAL_ENV"
echo "Sync results: output_sync/"
echo "Async results: output_async/"
echo "Log files: $LOG_DIR/"
echo ""
echo "Performance comparison has been automatically generated."
echo "Check the logs above for detailed performance metrics."
echo ""