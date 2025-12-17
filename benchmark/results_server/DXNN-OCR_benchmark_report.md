# DXNN-OCR Benchmark Report

**Test Configuration**:
- Model: PP-OCR v5 (DEEPX NPU acceleration)
- Total Images Tested: 20
- Success Rate: 100.0%

**Test Results**:
| Filename | Inference Time (ms) | FPS | CPS (chars/s) | Accuracy (%) |
|---|---|---|---|---|
| `image_10.png` | 420.35 | 2.38 | **3011.75** | **99.75** |
| `image_11.png` | 420.35 | 2.38 | **6665.81** | **99.56** |
| `image_12.png` | 420.35 | 2.38 | **5595.28** | **71.80** |
| `image_13.png` | 420.35 | 2.38 | **471.03** | **100.00** |
| `image_14.png` | 420.35 | 2.38 | **5057.64** | **98.99** |
| `image_15.png` | 420.35 | 2.38 | **10916.99** | **99.59** |
| `image_16.png` | 420.35 | 2.38 | **394.91** | **97.92** |
| `image_17.png` | 420.35 | 2.38 | **620.91** | **100.00** |
| `image_18.png` | 420.35 | 2.38 | **4346.34** | **99.66** |
| `image_19.png` | 420.35 | 2.38 | **5048.13** | **97.77** |
| `image_1.png` | 420.35 | 2.38 | **57.09** | **57.14** |
| `image_20.png` | 420.35 | 2.38 | **3761.12** | **98.04** |
| `image_2.png` | 420.35 | 2.38 | **368.74** | **62.00** |
| `image_3.png` | 420.35 | 2.38 | **126.08** | **21.43** |
| `image_4.png` | 420.35 | 2.38 | **325.92** | **44.64** |
| `image_5.png` | 420.35 | 2.38 | **71.37** | **95.24** |
| `image_6.png` | 420.35 | 2.38 | **9306.44** | **97.67** |
| `image_7.png` | 420.35 | 2.38 | **2664.42** | **89.20** |
| `image_8.png` | 420.35 | 2.38 | **2976.06** | **94.15** |
| `image_9.png` | 420.35 | 2.38 | **5652.38** | **96.75** |
| **Average** | **420.35** | **2.38** | **3371.92** | **86.06** |

**Performance Summary**:
- Average Inference Time: **420.35 ms**
- Average FPS: **2.38**
- Average CPS: **3371.92 chars/s**
- Total Characters Detected: **28348**
- Total Processing Time: **8407.08 ms**
- Average Character Accuracy: **86.06%**
- Success Rate: **100.0%** (20/20 images)
