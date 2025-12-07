# DXNN-OCR Benchmark Report

**Test Configuration**:
- Model: PP-OCR v5 (DEEPX NPU acceleration)
- Total Images Tested: 20
- Success Rate: 100.0%

**Test Results**:
| Filename | Inference Time (ms) | FPS | CPS (chars/s) | Accuracy (%) |
|---|---|---|---|---|
| `image_1.png` | 509.99 | 1.96 | **25.49** | **92.86** |
| `image_10.png` | 509.99 | 1.96 | **825.51** | **100.00** |
| `image_11.png` | 509.99 | 1.96 | **1851.02** | **99.45** |
| `image_12.png` | 509.99 | 1.96 | **1425.52** | **80.47** |
| `image_13.png` | 509.99 | 1.96 | **131.38** | **98.48** |
| `image_14.png` | 509.99 | 1.96 | **1396.11** | **99.42** |
| `image_15.png` | 509.99 | 1.96 | **2996.14** | **99.31** |
| `image_16.png` | 509.99 | 1.96 | **107.85** | **95.83** |
| `image_17.png` | 509.99 | 1.96 | **149.02** | **86.75** |
| `image_18.png` | 509.99 | 1.96 | **1203.95** | **96.80** |
| `image_19.png` | 509.99 | 1.96 | **1382.38** | **95.68** |
| `image_2.png` | 509.99 | 1.96 | **105.88** | **70.00** |
| `image_20.png` | 509.99 | 1.96 | **1025.51** | **97.06** |
| `image_3.png` | 509.99 | 1.96 | **13.73** | **50.00** |
| `image_4.png` | 509.99 | 1.96 | **107.85** | **92.86** |
| `image_5.png` | 509.99 | 1.96 | **39.22** | **95.24** |
| `image_6.png` | 509.99 | 1.96 | **2603.98** | **98.23** |
| `image_7.png` | 509.99 | 1.96 | **709.82** | **95.06** |
| `image_8.png` | 509.99 | 1.96 | **870.61** | **98.98** |
| `image_9.png` | 509.99 | 1.96 | **1676.51** | **99.46** |
| **Average** | **509.99** | **1.96** | **932.37** | **92.10** |

**Performance Summary**:
- Average Inference Time: **509.99 ms**
- Average FPS: **1.96**
- Average CPS: **932.37 chars/s**
- Total Characters Detected: **9510**
- Model Initialization Time: **1762.39 ms**
- Total Processing Time: **30809.84 ms**
- Average Character Accuracy: **92.10%**
- Success Rate: **100.0%** (20/20 images)
