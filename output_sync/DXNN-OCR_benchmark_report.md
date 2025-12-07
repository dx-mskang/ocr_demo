# DXNN-OCR Benchmark Report

**Test Configuration**:
- Model: PP-OCR v5 (DEEPX NPU acceleration)
- Total Images Tested: 20
- Success Rate: 100.0%

**Test Results**:
| Filename | Inference Time (ms) | FPS | CPS (chars/s) | Accuracy (%) |
|---|---|---|---|---|
| `image_1.png` | 649.02 | 1.54 | **18.49** | **85.71** |
| `image_10.png` | 1122.65 | 0.89 | **375.01** | **100.00** |
| `image_11.png` | 2548.73 | 0.39 | **370.38** | **99.67** |
| `image_12.png` | 1967.31 | 0.51 | **369.54** | **80.47** |
| `image_13.png` | 719.40 | 1.39 | **91.74** | **100.00** |
| `image_14.png` | 1910.85 | 0.52 | **372.61** | **99.42** |
| `image_15.png` | 3206.46 | 0.31 | **476.54** | **99.25** |
| `image_16.png` | 839.11 | 1.19 | **65.55** | **95.83** |
| `image_17.png` | 417.42 | 2.40 | **182.07** | **86.75** |
| `image_18.png` | 1264.78 | 0.79 | **487.04** | **96.97** |
| `image_19.png` | 1518.14 | 0.66 | **464.38** | **95.68** |
| `image_2.png` | 672.71 | 1.49 | **80.27** | **70.00** |
| `image_20.png` | 1537.01 | 0.65 | **344.83** | **95.88** |
| `image_3.png` | 586.69 | 1.70 | **25.57** | **85.71** |
| `image_4.png` | 627.57 | 1.59 | **86.05** | **91.07** |
| `image_5.png` | 583.75 | 1.71 | **34.26** | **95.24** |
| `image_6.png` | 2369.54 | 0.42 | **560.45** | **98.23** |
| `image_7.png` | 712.29 | 1.40 | **509.63** | **95.68** |
| `image_8.png` | 1359.95 | 0.74 | **326.48** | **98.98** |
| `image_9.png` | 1823.58 | 0.55 | **469.41** | **99.32** |
| **Average** | **1321.85** | **1.04** | **285.51** | **93.49** |

**Performance Summary**:
- Average Inference Time: **1321.85 ms**
- Average FPS: **1.04**
- Average CPS: **285.51 chars/s**
- Total Characters Detected: **9526**
- Model Initialization Time: **1768.75 ms**
- Total Processing Time: **80049.46 ms**
- Average Character Accuracy: **93.49%**
- Success Rate: **100.0%** (20/20 images)
