# DXNN-OCR PDF API Server Benchmark Report

**Test Configuration**:
- Model: PP-OCR v5 (DEEPX NPU acceleration)
- Total PDFs Tested: 2
- Runs per PDF: 1
- PDF DPI: 150
- Max Pages per PDF: 100
- Success Rate: 100.0%

**Test Results**:
| Filename | Size (MB) | Pages | Inference Time (ms) | FPS | CPS (chars/s) | PPS (pages/s) |
|---|---|---|---|---|---|---|
| `Song_DefectFill_Realistic_Defect_Generation_with_Inpainting_Diffusion_Model_for_Visual_CVPR_2025_paper.pdf` | 6.00 | 10/10 | 12811.47 | 0.08 | **2953.37** | 0.78 |
| `book-rev7.pdf` | 0.99 | 96/96 | 55705.50 | 0.02 | **4095.99** | 1.72 |
| **Average** | - | 106 | **34258.49** | **0.03** | **3882.34** | **1.55** |

**Performance Summary**:
- Average Inference Time: **34258.49 ms** (per-request latency)
- Average FPS: **0.03** (1000/latency)
- Average CPS: **3882.34 chars/s**
- Average PPS: **1.55 pages/s**
- Total Characters Detected: **266006**
- Total Pages Processed: **106**
- Total Processing Time: **68516.97 ms**
- Success Rate: **100.0%** (2/2 PDFs)