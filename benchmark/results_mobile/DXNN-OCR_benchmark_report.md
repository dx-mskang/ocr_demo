# DXNN-OCR Benchmark Report

**Test Configuration**:
- Model: PP-OCR v5 (DEEPX NPU acceleration)
- Total Images Tested: 20
- Success Rate: 100.0%

**Test Results**:
| Filename | Inference Time (ms) | FPS | CPS (chars/s) | Accuracy (%) |
|---|---|---|---|---|
| `image_10.png` | 152.37 | 6.56 | **8236.47** | **99.50** |
| `image_11.png` | 152.37 | 6.56 | **17700.21** | **93.83** |
| `image_12.png` | 152.37 | 6.56 | **14543.44** | **71.53** |
| `image_13.png` | 152.37 | 6.56 | **1319.15** | **98.48** |
| `image_14.png` | 152.37 | 6.56 | **13847.77** | **86.85** |
| `image_15.png` | 152.37 | 6.56 | **26638.91** | **79.78** |
| `image_16.png` | 152.37 | 6.56 | **1004.13** | **95.83** |
| `image_17.png` | 152.37 | 6.56 | **1607.92** | **95.18** |
| `image_18.png` | 152.37 | 6.56 | **11931.40** | **99.33** |
| `image_19.png` | 152.37 | 6.56 | **13834.65** | **95.68** |
| `image_1.png` | 152.37 | 6.56 | **137.82** | **42.86** |
| `image_20.png` | 152.37 | 6.56 | **10395.67** | **95.29** |
| `image_2.png` | 152.37 | 6.56 | **1023.82** | **38.00** |
| `image_3.png` | 152.37 | 6.56 | **242.83** | **50.00** |
| `image_4.png` | 152.37 | 6.56 | **892.56** | **44.64** |
| `image_5.png` | 152.37 | 6.56 | **196.89** | **95.24** |
| `image_6.png` | 152.37 | 6.56 | **25424.77** | **96.30** |
| `image_7.png` | 152.37 | 6.56 | **7265.16** | **83.33** |
| `image_8.png` | 152.37 | 6.56 | **8013.33** | **93.13** |
| `image_9.png` | 152.37 | 6.56 | **15363.81** | **95.13** |
| **Average** | **152.37** | **6.56** | **8981.03** | **82.50** |

**Performance Summary**:
- Average Inference Time: **152.37 ms**
- Average FPS: **6.56**
- Average CPS: **8981.03 chars/s**
- Total Characters Detected: **27369**
- Total Processing Time: **3047.42 ms**
- Average Character Accuracy: **82.50%**
- Success Rate: **100.0%** (20/20 images)
