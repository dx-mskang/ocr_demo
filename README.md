# PP-OCRv5 DEEPX Baseline

[中文 README](README_CN.md)

🚀 PP-OCRv5 DEEPX benchmarking toolchain with NPU acceleration and comprehensive performance evaluation.

## 🎯 Key Features

- **🚀 High Performance**: Async mode delivers **2.5x faster** inference with **3.2x higher** throughput
- **⚡ NPU Acceleration**: DEEPX DX-M1 NPU optimization for maximum performance
- **🔄 Dual Processing Modes**: 
  - Sync mode for stable sequential processing
  - Async mode for maximum parallel performance
- **📊 Comprehensive Benchmarking**: Detailed performance metrics and accuracy evaluation
- **🎨 Rich Visualization**: Automatic generation of OCR result visualizations

## 📈 Performance Results

### Custom Dataset Overview

This project uses a diverse custom Chinese dataset for benchmarking. The dataset consists of various real-world scenarios including street signs, handwritten text, exam papers, textbooks, and newspapers, providing comprehensive coverage of different text recognition challenges with detailed annotations including text content and bounding box coordinates.

**Test Configuration**:
- Dataset: Custom Chinese document dataset (20 images)
- Data Format: PNG images with JSON annotations containing text content
- Model: DXNN-OCR v5 full pipeline (PP-OCRv5 → DEEPX NPU accelerated)
  - Text detection: PP-OCRv5 det → DXNN det_v5 (NPU accelerated)
  - Text classification: PP-OCRv5 cls → DXNN cls_v5 (NPU accelerated)
  - Text recognition: PP-OCRv5 rec → DXNN rec_v5 multi-ratio models (NPU accelerated)
- Hardware configuration:
  - Platform: Rockchip RK3588 IR88MX01 LP4X V10
  - NPU: DEEPX DX-M1 Accelerator Card
    - PCIe: Gen3 X4 interface [01:00:00]
    - Firmware: v2.1.0
  - CPU: ARM Cortex-A55 8-core @ 2.35GHz (8nm process)
  - Memory: 8GB LPDDR4X
  - Operating System: Ubuntu 20.04.6 LTS (Focal)
  - Runtime: DXRT v3.0.0 + RT driver v1.7.1 + PCIe driver v1.4.1

**Benchmark Results**:

#### Sync Mode (Sequential Processing)
| Processing Mode | Average Inference Time (ms) | Average FPS | Average CPS (chars/s) | Average Accuracy (%) | Total Processing Time (s) |
|---|---|---|---|---|---|
| `Sync (Sequential)` | 1325.72 | 1.03 | 284.84 | 93.49 | 27.67 |

#### Async Mode (Parallel Processing)
| Processing Mode | Average Inference Time (ms) | Average FPS | Average CPS (chars/s) | Average Accuracy (%) | Total Processing Time (s) |
|---|---|---|---|---|---|
| `Async (Parallel)` | 523.79 | 1.91 | 908.38 | 92.19 | 11.53 |

**Performance Comparison**:
- **Inference Speed**: Async mode is **2.5x faster** per image (523.79ms vs 1325.72ms)
- **Throughput**: Async mode achieves **3.2x higher** characters per second (908.38 vs 284.84)
- **Overall Processing**: Async mode completes **2.4x faster** for full batch (11.53s vs 27.67s)
- **Accuracy**: Both modes maintain high accuracy (>92%)

- [Detailed Performance Results of PP-OCRv5 on DEEPX NPU](./PP-OCRv5_on_DEEEPX.md)

## � System Requirements

### Hardware Requirements
- **NPU**: DEEPX DX-M1 Accelerator Card
  - PCIe: Gen3 X4 interface
  - Memory: LPDDR5 6000 Mbps, 3.92GiB minimum
  - Board: M.2, Rev 1.5 or higher

### Software Requirements
- **DXRT**: v3.0.0 or higher
  - RT Driver version: v1.7.1 or higher
  - PCIe Driver version: v1.4.1 or higher
  - FW version: v2.1.6 or higher
- **Python Package**: dx-engine v1.1.2 or higher

### Version Verification
```bash
# Check DXRT version and device status
dxrt-cli -s

# Check dx-engine version
pip list | grep dx
```

## �🛠️ Quick Start

### ⚡ Automated Full Pipeline Execution

**One-Step Automated Benchmark Pipeline:**
```bash
git clone https://github.com/Chris-godz/PP-OCRv5-DeepX-Baseline.git
cd PP-OCRv5-DeepX-Baseline
./startup.sh
```

**What `startup.sh` Does Automatically:**

🔧 **Phase 1: Environment Setup**
- Creates and activates Python virtual environment
- Installs all required dependencies from `requirements.txt`
- Verifies dataset and ground truth files

⚡ **Phase 2: RT Optimization** 
- Automatically applies `set_env.sh 1 3 1 18 1 4` for optimal NPU performance
- Configures DXRT environment variables for maximum throughput

🚀 **Phase 3: Sync Benchmark**
- Runs sequential processing benchmark
- Saves results to `output_sync/` directory
- Logs detailed performance metrics

⚡ **Phase 4: Async Benchmark**
- Runs parallel processing benchmark (2.5x faster)
- Saves results to `output_async/` directory
- Captures async-specific performance data

📊 **Phase 5: Automatic Comparison**
- Executes `scripts/compare_sync_async.py`
- Generates side-by-side performance comparison
- Displays speedup metrics and improvement summary

✅ **Final Output:**
- Complete performance comparison with speedup calculations
- Separate result directories for easy analysis
- Comprehensive logs for troubleshooting
- Ready-to-use benchmark reports

**RT Optimization (Recommended):**
```bash
# Apply DXRT optimization settings for maximum performance
source ./set_env.sh 1 3 1 18 1 4

# Explanation of parameters:
# CUSTOM_INTER_OP_THREADS_COUNT=1     # Inter-operation parallelism
# CUSTOM_INTRA_OP_THREADS_COUNT=3     # Intra-operation parallelism  
# DXRT_DYNAMIC_CPU_THREAD=1           # Dynamic CPU thread management
# DXRT_TASK_MAX_LOAD=18               # Maximum task load
# NFH_INPUT_WORKER_THREADS=1          # Input worker threads
# NFH_OUTPUT_WORKER_THREADS=4         # Output worker threads
```

**Advanced Usage Examples:**
```bash
# Step 1: Apply RT optimization (recommended for best performance)
source ./set_env.sh 1 3 1 18 1 4

# Step 2: Run benchmark
# Sync mode (sequential processing)
python scripts/dxnn_benchmark.py \
    -d sampled_dataset/ \
    --mode sync \
    --output results/ \
    --ground-truth sampled_dataset/labels.json \
    --runs 1

# Async mode (parallel processing - 2.5x faster)
python scripts/dxnn_benchmark.py \
    -d sampled_dataset/ \
    --mode async \
    --output async_results/ \
    --ground-truth sampled_dataset/labels.json \
    --runs 1
```

**Interactive GUI Demo:**
```bash
# Launch interactive GUI demo (sync mode)
python demo.py --version v5 --mode sync

# Launch interactive GUI demo (async mode - 2.5x faster)
python demo.py --version v5 --mode async
```

**GUI Features:**
- **🖼️ Visual OCR Interface**: Drag & drop multiple images for instant OCR processing
- **📊 Real-time Performance Metrics**: Live FPS and processing statistics
- **🎯 Accuracy Comparison**: Side-by-side GPU vs NPU accuracy analysis
- **🔍 Result Visualization**: Interactive preview of OCR detection and recognition results
- **⚡ Dual Processing Modes**: Switch between sync and async processing modes

## 📁 Project Structure

```
├── demo.py                 # 🎨 Interactive GUI demo with real-time OCR processing
├── startup.sh              # 🚀 Fully automated benchmark pipeline
│                           # - Environment setup & dependency installation
│                           # - RT optimization (set_env.sh) application
│                           # - Sync benchmark execution → output_sync/
│                           # - Async benchmark execution → output_async/
│                           # - Automatic performance comparison
├── set_env.sh              # 🔧 DXRT optimization settings
├── scripts/
│   ├── dxnn_benchmark.py   # 🔥 Refactored benchmark tool with dual-mode support
│   │                       # - Sync mode: Sequential processing (stable)
│   │                       # - Async mode: Parallel processing (2.5x faster)
│   ├── compare_sync_async.py # 📊 Performance comparison tool
│   ├── calculate_acc.py    # PP-OCRv5 compatible accuracy calculation
│   └── ocr_engine.py       # DXNN NPU engine interface
├── engine/
│   ├── model_files/v5/     # DXNN v5 NPU models (.dxnn format)
│   ├── draw_utils.py       # Visualization utilities
│   ├── utils.py           # Processing utilities
│   └── fonts/             # Chinese fonts (for visualization)
└── images/                 # Custom dataset (20 PNG images + labels.json)
    ├── image_1.png ~ image_20.png  # Test images
    └── labels.json         # Ground truth annotations
```

**Custom Dataset:**
```bash
# Step 1: Apply RT optimization for best performance
source ./set_env.sh 1 3 1 18 1 4

# Step 2: Prepare your own images
mkdir -p images/custom
cp /path/to/your/images/* images/custom/

# Step 3: Run benchmark with sync mode (stable)
python scripts/dxnn_benchmark.py \
    --directory images/custom \
    --mode sync \
    --ground-truth custom_labels.json \
    --output output_custom \
    --runs 3

# Or run benchmark with async mode (2.5x faster)
python scripts/dxnn_benchmark.py \
    --directory images/custom \
    --mode async \
    --ground-truth custom_labels.json \
    --output output_custom_async \
    --runs 3
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This project is forked and developed based on [DEEPX-AI/DXNN-OCR](https://github.com/DEEPX-AI/DXNN-OCR) project
- Thanks to [DEEPX team](https://deepx.ai) for NPU runtime and foundational framework support
- Thanks to the [PaddleOCR team](https://github.com/PaddlePaddle/PaddleOCR) for the excellent OCR framework

## 🏆 Recent Improvements

### Automated Benchmark Pipeline (Nov 2025)
- **🚀 One-Command Execution**: Complete automated pipeline with `./startup.sh`
  - Automatic environment setup and dependency installation
  - RT optimization application (`set_env.sh`) for maximum NPU performance
  - Sequential sync benchmark execution with detailed logging
  - Parallel async benchmark execution (2.5x performance boost)
  - Automatic performance comparison and speedup calculation
- **📊 Comprehensive Results**: Separate output directories (`output_sync/`, `output_async/`) with complete metrics
- **🔧 Zero Configuration**: No manual intervention required - everything automated from start to finish

### Code Refactoring & Performance Optimization (Nov 2025)
- **🔧 Architecture Refactoring**: Complete modular redesign for better maintainability
  - `BenchmarkConfig`: Centralized configuration management
  - `OCRBenchmark`: Core processing logic with helper methods
  - `BenchmarkReporter`: Result processing and visualization
- **⚡ Async Processing**: Introduced parallel processing mode with **2.5x performance boost**
- **📊 Enhanced Reporting**: Comprehensive performance metrics and visualizations
- **🛡️ Robust Error Handling**: Improved stability and error recovery
- **🧪 Extensive Testing**: Validated performance on diverse datasets