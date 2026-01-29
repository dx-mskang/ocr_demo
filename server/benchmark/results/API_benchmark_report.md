# DXNN-OCR API Server Benchmark Report

Generated at: 2026-01-19 16:33:33

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Model | PP-OCR v5 (DEEPX NPU acceleration) |
| Total Images | 20 |
| Runs per Image | 20 |
| Max In-Flight | 3 |
| Mode | Worker Model (生产者-消费者) |
| Success Rate | 100.0% |

## Performance Summary

| Metric | Value |
|--------|-------|
| **Real QPS (Throughput)** | **3.17** |
| Avg Latency | 944.34 ms |
| Avg FPS | 1.06 |
| Avg CPS | 77.59 chars/s |
| Total Time | 126174.10 ms |
| Total Characters | 9790 |
| Avg Accuracy | 91.55% |

## Latency Distribution

| Percentile | Latency (ms) |
|------------|--------------|
| Min | 295.37 |
| P50 | 851.22 |
| P90 | 1489.26 |
| P99 | 1853.33 |
| Max | 3565.51 |
| Std Dev | 440.95 |

## Request Statistics

| Metric | Value |
|--------|-------|
| Total Requests | 400 |
| Successful | 400 |
| Failed | 0 |
| Timeout | 0 |
| Success Rate | 100.00% |

## Per-Image Results

| Filename | Latency (ms) | FPS | CPS | Accuracy |
|----------|--------------|-----|-----|----------|
| `image_1.png` | 741.35 | 1.35 | 10.79 | 72.73% |
| `image_10.png` | 731.77 | 1.37 | 576.68 | 99.88% |
| `image_11.png` | 1399.82 | 0.71 | 675.80 | 99.37% |
| `image_12.png` | 1344.02 | 0.74 | 632.43 | 83.40% |
| `image_13.png` | 1015.02 | 0.99 | 65.02 | 100.00% |
| `image_14.png` | 1072.18 | 0.93 | 665.93 | 98.18% |
| `image_15.png` | 1620.16 | 0.62 | 946.20 | 99.74% |
| `image_16.png` | 1306.19 | 0.77 | 42.87 | 94.44% |
| `image_17.png` | 719.05 | 1.39 | 120.99 | 100.00% |
| `image_18.png` | 773.52 | 1.29 | 813.17 | 99.84% |
| `image_19.png` | 887.02 | 1.13 | 804.94 | 97.46% |
| `image_2.png` | 673.38 | 1.49 | 78.71 | 70.37% |
| `image_20.png` | 751.76 | 1.33 | 701.02 | 99.05% |
| `image_3.png` | 577.51 | 1.73 | 43.29 | 65.00% |
| `image_4.png` | 444.14 | 2.25 | 159.86 | 70.77% |
| `image_5.png` | 453.81 | 2.20 | 44.07 | 97.56% |
| `image_6.png` | 1427.50 | 0.70 | 935.20 | 97.75% |
| `image_7.png` | 381.65 | 2.62 | 1037.60 | 92.35% |
| `image_8.png` | 1185.35 | 0.84 | 390.60 | 96.37% |
| `image_9.png` | 1381.67 | 0.72 | 633.29 | 96.76% |
| **Average** | **944.34** | **1.06** | **77.59** | **91.55%** |
