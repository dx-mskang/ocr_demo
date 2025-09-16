# DXNN OCR - OCR GUI Application

## 📋 Project Overview

DXNN OCR is an OCR (Optical Character Recognition) GUI application based on DX Runtime. It provides an intuitive user interface using PySide6 and utilizes [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) models to extract text from images.

## 🚀 Installation and Environment Setup

### Prerequisites

For a simple installation, you can just run the `ocr_install.sh` script.

Alternatively, you can manually set up the environment as follows:
```bash
# Python 3.11 or higher
python --version

# Install required packages
pip install -r requirements.txt
```

### Model File Preparation
```
engine/model_files/
├── v4/
│   ├── det.dxnn
│   ├── cls.dxnn
│   └── rec_ratio_*.dxnn
└── v5/
    ├── det_v5.dxnn
    ├── cls_v5.dxnn
    └── rec_v5_ratio_*.dxnn
```

## 🔧 Usage

### Execution Method
```bash
# Run with v4 model
python demo.py --version v4

# Run with v5 model
python demo.py --version v5
```

### How to Use

1. **Image Upload**
   - Click "Upload Images" button or drag and drop files directly
   - Supported formats: PNG, JPG, JPEG, BMP, GIF

2. **Single Image OCR**
   - Click image in thumbnail list
   - Automatic OCR execution and result display

3. **Batch OCR Execution**
   - Click "Run All Inference" button
   - Sequential OCR execution for all uploaded images

4. **Result Verification**
   - **Left Panel**: Inferred text results and processed image preview
   - **Center Panel**: Main image viewer with processing results
   - **Right Panel**: Performance information and file management

## 🏗️ Project Structure

```
DXNNOCR/
├── demo.py                 # Main GUI application
├── engine/
│   ├── paddleocr.py       # PaddleOCR engine implementation (based on https://github.com/PaddlePaddle/PaddleOCR)
│   ├── draw_utils.py      # Image drawing utilities
│   └── model_files/       # DXNN model files
├── dx_engine.py           # Inference engine (external library)
└── README.md             # Project documentation
```

## 📄 License

This project is distributed under the MIT License.

## 📞 Contact

For project inquiries or bug reports, please submit through issues. 
