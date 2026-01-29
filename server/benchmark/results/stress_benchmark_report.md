# DXNN-OCR Stress Test Report (C++)

**Test Configuration**:
- Concurrency: High-performance C++ implementation
- Total Images: 20
- Total Requests: 400
- Success Rate: 100.00%

**Throughput Metrics**:
| Metric | Value |
|---|---|
| Benchmark Duration | **118736.09 ms** |
| Successful Requests | 400 |
| **Actual QPS** | **3.37** |
| Average Latency | 4657.55 ms |
| P50 Latency | 3749.62 ms |
| P90 Latency | 9173.18 ms |
| P99 Latency | 12286.85 ms |

**Per-Image Results**:
| Filename | Latency (ms) | FPS | CPS |
|---|---|---|---|
| `image_6.png` | 8191.67 | 0.12 | **477.56** |
| `image_15.png` | 12286.85 | 0.08 | **373.49** |
| `image_18.png` | 7268.33 | 0.14 | **251.36** |
| `image_14.png` | 6335.38 | 0.16 | **335.58** |
| `image_8.png` | 3918.09 | 0.26 | **319.29** |
| `image_19.png` | 4504.47 | 0.22 | **471.09** |
| `image_3.png` | 4735.48 | 0.21 | **11.19** |
| `image_5.png` | 1732.27 | 0.58 | **17.32** |
| `image_9.png` | 3749.62 | 0.27 | **633.66** |
| `image_17.png` | 3087.93 | 0.32 | **84.52** |
| `image_13.png` | 1792.05 | 0.56 | **110.49** |
| `image_10.png` | 3283.88 | 0.30 | **385.52** |
| `image_4.png` | 2756.69 | 0.36 | **49.70** |
| `image_11.png` | 6605.54 | 0.15 | **424.19** |
| `image_12.png` | 9173.18 | 0.11 | **256.40** |
| `image_7.png` | 1815.68 | 0.55 | **616.85** |
| `image_1.png` | 3620.85 | 0.28 | **6.63** |
| `image_16.png` | 1991.84 | 0.50 | **83.34** |
| `image_2.png` | 2969.50 | 0.34 | **52.20** |
| `image_20.png` | 3331.74 | 0.30 | **474.53** |

**Performance Summary**:
- **QPS (Queries Per Second): 3.37**
- Average CPS: 304.32 chars/s
- Total Characters: 28348
