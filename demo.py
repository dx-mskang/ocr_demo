import sys
import os

import argparse

from dx_engine import InferenceEngine as IE
from dx_engine import InferenceOption as IO

from PySide6.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QListWidget, QListWidgetItem, QFileDialog, QSizePolicy, QGridLayout, QScrollArea, QTextEdit, QTableWidget, QTableWidgetItem
)
from PySide6.QtGui import QPixmap, QIcon, QImage
from PySide6.QtCore import Qt, QSize, QEvent, QMimeData, Signal

import tqdm

import cv2
import numpy as np
import onnxruntime as ort

from engine.paddleocr import PaddleOcr, AsyncPipelineOCR
from engine.draw_utils import draw_with_poly_enhanced, draw_ocr

from collections import deque
import copy

# OCR result structure constants
# OCR results format: [[text, confidence_score], ...]
TEXT_INDEX = 0      # Index for recognized text in OCR result
SCORE_INDEX = 1     # Index for confidence score in OCR result
CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence score to display

preview_image_q = deque(maxlen=20)

class ClickableLabel(QLabel):
    clicked = Signal(int)  # Send index when clicked

    def __init__(self, index, parent=None):
        super().__init__(parent)
        self.index = index

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.index)

class PreviewGridWidget(QWidget):
    imageClicked = Signal(int)  # Signal to pass to external

    def __init__(self, parent=None):
        super().__init__(parent)
        self.grid_layout = QGridLayout()
        self.setLayout(self.grid_layout)
        self.image_labels = []

    def clear(self):
        for label in self.image_labels:
            self.grid_layout.removeWidget(label)
            label.deleteLater()
        self.image_labels.clear()

    def set_images(self, image_q:deque):
        self.clear()

        cols = 2  # Number of desired columns
        for i in range(len(image_q)):
            idx = len(image_q) - i - 1
            label = ClickableLabel(idx)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("border: 1px solid gray; background-color: white;")
            label.clicked.connect(self.imageClicked.emit)

            # Display if image is in numpy array format
            rgb = image_q[idx][0]
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimage = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)

            # Scale to fit label size
            pixmap = pixmap.scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            label.setPixmap(pixmap)

            row = i // cols
            col = i % cols
            self.grid_layout.addWidget(label, row, col)
            self.image_labels.append(label)


class ThumbnailListWidget(QListWidget):
    def __init__(self, image_click_callback, thumbnail_size=60):
        super().__init__()
        self.setViewMode(QListWidget.ViewMode.IconMode)
        # Set smaller fixed square size for thumbnails
        self.setIconSize(QSize(thumbnail_size, thumbnail_size))
        self.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        # Reduce spacing for more compact layout
        self.setSpacing(5)
        self.image_click_callback = image_click_callback
        self.itemClicked.connect(self.handle_item_click)
        self.thumbnail_size = thumbnail_size

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        event.acceptProposedAction()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            added_count = 0
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if os.path.isfile(file_path) and self.is_image(file_path):
                    self.add_thumbnail(file_path)
                    added_count += 1
            if added_count > 0:
                print(f"Added {added_count} images via drag and drop")
        event.acceptProposedAction()

    def is_image(self, path):
        return path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))

    def truncate_filename(self, filename, max_length):
        """
        Truncate filename to fit within thumbnail width
        """
        if len(filename) <= max_length:
            return filename
        
        # Calculate available space for filename (considering padding and ellipsis)
        available_chars = max_length - 3  # Reserve 3 characters for "..."
        
        if available_chars <= 0:
            return "..."
        
        # Split filename and extension
        name, ext = os.path.splitext(filename)
        
        # If extension is too long, truncate it
        if len(ext) > available_chars // 2:
            ext = ext[:available_chars // 2]
        
        # Calculate remaining space for name
        name_chars = available_chars - len(ext)
        
        if name_chars <= 0:
            return "..." + ext
        
        # Truncate name and add ellipsis
        truncated_name = name[:name_chars] + "..."
        return truncated_name + ext

    def add_thumbnail(self, file_path):
        pixmap = QPixmap(file_path)
        if not pixmap.isNull():
            # Create square thumbnail with fixed size
            icon = QIcon(pixmap.scaled(
                self.thumbnail_size, 
                self.thumbnail_size, 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            ))
            
            # Get filename and truncate it to fit thumbnail width
            filename = os.path.basename(file_path)
            # Estimate max characters that can fit in thumbnail width
            # Assuming average character width of 6-8 pixels and some padding
            max_chars = max(8, self.thumbnail_size // 8)
            truncated_filename = self.truncate_filename(filename, max_chars)
            
            # Show truncated filename
            item = QListWidgetItem(icon, truncated_filename)
            item.setData(Qt.ItemDataRole.UserRole, file_path)
            # Store full filename as tooltip for hover display
            item.setToolTip(filename)
            self.addItem(item)

    def handle_item_click(self, item):
        file_path = item.data(Qt.ItemDataRole.UserRole)
        self.image_click_callback(file_path)


class PerformanceInfoWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Performance Information (FPS)")
        title.setStyleSheet("""
            QLabel {
                font-size: 13px;
                color: black;
                padding: 2px;
                border-radius: 2px;
                margin-bottom: 2px;
            }
        """)
        layout.addWidget(title)
        
        # Performance table
        self.performance_table = QTableWidget()
        self.performance_table.setRowCount(4)  # 4 rows including header
        self.performance_table.setColumnCount(3)
        # self.performance_table.setMaximumHeight(100)  # Slightly increased for header
        self.performance_table.setMinimumHeight(80)   # Increased minimum height
        
        # Set table headers
        self.performance_table.setHorizontalHeaderLabels(["", "GPU", "NPU"])
        self.performance_table.setVerticalHeaderLabels(["", "det", "cls", "rec"])
        
        # Hide row and column headers for cleaner look
        self.performance_table.horizontalHeader().setVisible(False)
        self.performance_table.verticalHeader().setVisible(False)
        
        # Set row heights to minimize spacing
        self.performance_table.verticalHeader().setDefaultSectionSize(20)  # Compact row height
        
        # Set table style with minimal padding
        self.performance_table.setStyleSheet("""
            QTableWidget {
                font-family: 'Courier New', monospace;
                font-size: 12px;
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 3px;
                gridline-color: #dee2e6;
                padding: 0px;
                margin: 0px;
            }
            QTableWidget::item {
                padding: 1px 2px;
                border: none;
                margin: 0px;
            }
            QHeaderView::section {
                background-color: #f8f9fa;
                border: none;
                padding: 1px;
                margin: 0px;
            }
        """)
        
        # Set column widths
        self.performance_table.setColumnWidth(0, 40)  # First column for labels
        self.performance_table.setColumnWidth(1, 60)  # GPU column
        self.performance_table.setColumnWidth(2, 200)  # NPU column
        
        # Set initial performance data
        self.update_performance_data()
        
        layout.addWidget(self.performance_table)

        self.setLayout(layout)
    
    def update_performance_data(self, det_gpu=10, det_npu=10.548, cls_gpu=95.3, cls_npu=0.19, rec_gpu=121, rec_npu=0.736, min_rec_npu=0.736):
        """
        Update performance information display
        """
        # Create header row
        header_row = [
            QTableWidgetItem(""),
            QTableWidgetItem("GPU"),
            QTableWidgetItem("NPU")
        ]
        
        # Create table items
        det_row = [
            QTableWidgetItem("det"),
            QTableWidgetItem(f"{det_gpu:.2f}"),
            QTableWidgetItem(f"{1000/(det_npu + 1e-10):.2f}")
        ]
        
        cls_row = [
            QTableWidgetItem("cls"),
            QTableWidgetItem(f"{cls_gpu:.2f}"),
            QTableWidgetItem(f"{1000/(cls_npu + 1e-10):.2f}")
        ]
        
        rec_row = [
            QTableWidgetItem("rec"),
            QTableWidgetItem(f"{rec_gpu:.2f}"),
            QTableWidgetItem(f"{1000/(rec_npu + 1e-10):.2f}(avg), {1000/(min_rec_npu + 1e-10):.2f}(max)")
        ]
        
        # Set items in table
        for col, item in enumerate(header_row):
            self.performance_table.setItem(0, col, item)
        for col, item in enumerate(det_row):
            self.performance_table.setItem(1, col, item)
        for col, item in enumerate(cls_row):
            self.performance_table.setItem(2, col, item)
        for col, item in enumerate(rec_row):
            self.performance_table.setItem(3, col, item)


class EnvironmentInfoWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Environment table
        self.environment_table = QTableWidget()
        self.environment_table.setRowCount(3)  # 3 rows including header
        self.environment_table.setColumnCount(2)
        # self.accuracy_table.setMaximumHeight(100)  # Slightly increased for header
        self.environment_table.setMinimumHeight(80)   # Increased minimum height
        
        # Hide row and column headers for cleaner look
        self.environment_table.horizontalHeader().setVisible(False)
        self.environment_table.verticalHeader().setVisible(False)
        # Set row heights to minimize spacing
        self.environment_table.verticalHeader().setDefaultSectionSize(20)  # Compact row height
        
        # Set table style with minimal padding
        self.environment_table.setStyleSheet("""
            QTableWidget {
                font-family: 'Courier New', monospace;
                font-size: 12px;
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 3px;
                gridline-color: #dee2e6;
                padding: 0px;
                margin: 0px;
            }
            QTableWidget::item {
                padding: 1px 2px;
                border: none;
                margin: 0px;
            }
            QTableWidget::item:first {
                background-color: #e9ecef;
                font-weight: bold;
                border-bottom: 1px solid #dee2e6;
                font-size: 13px;
            }
            QHeaderView::section {
                background-color: #f8f9fa;
                border: none;
                padding: 1px;
                margin: 0px;
            }
        """)
        
        # Set column widths
        self.environment_table.setColumnWidth(0, 100)
        self.environment_table.setColumnWidth(1, 240)
        
        # Set initial accuracy data
        self.set_environment_data()
        
        layout.addWidget(self.environment_table)
        self.setLayout(layout)

    def set_environment_data(self):
        """
        Update environment information display
        """
        # Create merged header row
        header_item = QTableWidgetItem("Environment Information")
        header_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.environment_table.setItem(0, 0, header_item)
        # Merge the first row across all columns
        self.environment_table.setSpan(0, 0, 1, 2)
        
        # Create table items
        gpuInfo_row = [
            QTableWidgetItem("GPU Device"),
            QTableWidgetItem("NVIDIA GeForce RTX 2080 Ti")
        ]
        
        npuInfo_row = [
            QTableWidgetItem("NPU Device"),
            QTableWidgetItem("DEEPX DX-M1")
        ]
        
        # Set items in table (starting from row 1 since row 0 is merged header)

        for col, item in enumerate(gpuInfo_row):
            self.environment_table.setItem(1, col, item)
        for col, item in enumerate(npuInfo_row):
            self.environment_table.setItem(2, col, item)


class AccuracyInfoWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Accuracy Information")
        title.setStyleSheet("""
            QLabel {
                font-size: 13px;
                color: black;
                padding: 0px;
                border-radius: 0px;
                margin-bottom: 0px;
            }
        """)
        layout.addWidget(title)
        
        # Performance table
        self.accuracy_table = QTableWidget()
        self.accuracy_table.setRowCount(4)  # 4 rows including header
        self.accuracy_table.setColumnCount(4)
        # self.accuracy_table.setMaximumHeight(100)  # Slightly increased for header
        self.accuracy_table.setMinimumHeight(80)   # Increased minimum height
        
        # Set table headers
        
        self.accuracy_table.setHorizontalHeaderLabels(["", "GPU", "DX-M1", "GAP"])
        self.accuracy_table.setVerticalHeaderLabels(["", "", "V4", "V5"])
        
        # Hide row and column headers for cleaner look
        self.accuracy_table.horizontalHeader().setVisible(False)
        self.accuracy_table.verticalHeader().setVisible(False)
        
        # Set row heights to minimize spacing
        self.accuracy_table.verticalHeader().setDefaultSectionSize(20)  # Compact row height
        
        # Set table style with minimal padding
        self.accuracy_table.setStyleSheet("""
            QTableWidget {
                font-family: 'Courier New', monospace;
                font-size: 12px;
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 3px;
                gridline-color: #dee2e6;
                padding: 0px;
                margin: 0px;
            }
            QTableWidget::item {
                padding: 1px 2px;
                border: none;
                margin: 0px;
            }
            QTableWidget::item:first {
                background-color: #e9ecef;
                font-weight: bold;
                border-bottom: 1px solid #dee2e6;
                font-size: 13px;
            }
            QHeaderView::section {
                background-color: #f8f9fa;
                border: none;
                padding: 1px;
                margin: 0px;
            }
        """)
        
        # Set column widths
        self.accuracy_table.setColumnWidth(0, 40)  # First column for labels
        self.accuracy_table.setColumnWidth(1, 80)  # GPU column
        self.accuracy_table.setColumnWidth(2, 80)  # NPU column
        self.accuracy_table.setColumnWidth(3, 80)  # GAP column
        
        # Set initial accuracy data
        self.set_accuracy_data()
        
        layout.addWidget(self.accuracy_table)
        self.setLayout(layout)

    def set_accuracy_data(self):
        """
        Update accuracy information display
        """
        # Create merged header row
        header_item = QTableWidgetItem("Accuracy (Error Rate %)")
        header_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.accuracy_table.setItem(0, 0, header_item)
        # Merge the first row across all columns
        self.accuracy_table.setSpan(0, 0, 1, 4)
        
        # Create table items
        header_row = [
            QTableWidgetItem(""),
            QTableWidgetItem("GPU"),
            QTableWidgetItem("DX-M1"),
            QTableWidgetItem("GAP")
        ]
        
        v4_row = [
            QTableWidgetItem("V4"),
            QTableWidgetItem("6.70 %"),
            QTableWidgetItem("7.10 %"),
            QTableWidgetItem("-0.40 %")
        ]
        
        v5_row = [
            QTableWidgetItem("V5"),
            QTableWidgetItem("5.60 %"),
            QTableWidgetItem("6.00 %"),
            QTableWidgetItem("-0.40 %")
        ]
        
        # Set items in table (starting from row 1 since row 0 is merged header)
        for col, item in enumerate(header_row):
            self.accuracy_table.setItem(1, col, item)
        for col, item in enumerate(v4_row):
            self.accuracy_table.setItem(2, col, item)
        for col, item in enumerate(v5_row):
            self.accuracy_table.setItem(3, col, item)


class ImageViewerApp(QWidget):
    def __init__(self, ocr_workers=None, app_version=None, app_mode="sync"):
        super().__init__()
        self.setWindowTitle(f"PySide Image Viewer {app_version}")
        
        # Set window to full screen size
        screen_size = QApplication.primaryScreen().geometry()
        self.setGeometry(100, 100, screen_size.width()/1.2, screen_size.height()/1.2)
        
        # Store OCR workers
        self.ocr_workers = ocr_workers
        self.current_worker_index = 0  # Index for round-robin worker selection
        self.app_version = app_version
        self.app_mode = app_mode
        # Initialize UI first
        self.init_ui()
        
        # Store screen dimensions for reference
        self.max_width = screen_size.width()/2
        self.max_height = screen_size.height()/2
        self.last_result_image = None
        self.last_result_text = None

    def init_ui(self):
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()

        # Swap positions: Text first, then Preview
        info_title = QLabel("Inference Result Text")
        info_title.setStyleSheet("""
            QLabel {
                font-size: 13px;
                color: black;
                padding: 2px;
                border-radius: 2px;
                margin-bottom: 2px;
            }
        """)
        left_layout.addWidget(info_title)

        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse |
            Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        # Set line spacing to increase readability
        self.info_text.setStyleSheet("""
            QTextEdit {
                line-height: 1.2;
                padding: 4px;
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                font-size: 12px;
            }
        """)

        text_scroll_area = QScrollArea()
        text_scroll_area.setWidgetResizable(True)
        text_scroll_area.setWidget(self.info_text)

        left_layout.addWidget(text_scroll_area, stretch=3)

        preview_title = QLabel("Processing Result Preview")
        preview_title.setStyleSheet("""
            QLabel {
                font-size: 13px;
                color: black;
                padding: 2px;
                border-radius: 2px;
                margin-bottom: 2px;
            }
        """)
        left_layout.addWidget(preview_title)

        self.preview_grid = PreviewGridWidget()
        self.preview_grid.imageClicked.connect(self.handle_preview_click)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.preview_grid)
        
        left_layout.addWidget(scroll_area, stretch=2)

        left_container = QWidget()
        left_container.setLayout(left_layout)

        main_layout.addWidget(left_container, stretch=1)

        center_layout = QVBoxLayout()

        main_image_title = QLabel(f"Main Image Viewer ({self.app_version})")
        main_image_title.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: black;
                padding: 2px;
                border-radius: 2px;
                margin-bottom: 3px;
            }
        """)

        center_layout.addWidget(main_image_title)
        
        self.image_label = QLabel("Please select an image")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_label.setStyleSheet("border: 1px solid gray;")
        center_layout.addWidget(self.image_label, stretch=3)
        
        center_container = QWidget()
        center_container.setLayout(center_layout)
        main_layout.addWidget(center_container, stretch=3)

        right_layout = QVBoxLayout()

        # Add Performance Information widget before File Management
        self.performance_widget = PerformanceInfoWidget()
        right_layout.addWidget(self.performance_widget, stretch=2)
        # Add Accuracy Information widget before File Management
        self.envInfo_widget = EnvironmentInfoWidget()
        right_layout.addWidget(self.envInfo_widget, stretch=1)
        # Add Accuracy Information widget before File Management
        self.accuracy_widget = AccuracyInfoWidget()
        right_layout.addWidget(self.accuracy_widget, stretch=2)

        file_title = QLabel("File Management")
        file_title.setStyleSheet("""
            QLabel {
                font-size: 13px;
                color: black;
                padding: 2px;
                border-radius: 2px;
                margin-bottom: 2px;
            }
        """)
        right_layout.addWidget(file_title)

        # Add help text for multiple file selection
        help_text = QLabel("• Drag & drop multiple images here\n• Or use Upload button (Ctrl+Click for multiple)")
        help_text.setStyleSheet("""
            QLabel {
                font-size: 10px;
                color: #666666;
                padding: 2px;
                border-radius: 2px;
                margin-bottom: 4px;
            }
        """)
        right_layout.addWidget(help_text)

        # Use ThumbnailListWidget with smaller thumbnail size for right layout
        self.file_list = ThumbnailListWidget(self.handle_file_list_click, thumbnail_size=50)
        right_layout.addWidget(self.file_list, stretch=8)

        upload_button = QPushButton("Upload Multiple Images")
        upload_button.setToolTip("Click to select multiple image files\n(Use Ctrl+Click or Shift+Click in file dialog for multiple selection)")
        upload_button.clicked.connect(self.upload_images)
        right_layout.addWidget(upload_button)
        
        all_inference_button = QPushButton("Run All Inference")
        all_inference_button.clicked.connect(self.run_imagelist)
        right_layout.addWidget(all_inference_button)

        right_container = QWidget() 
        right_container.setLayout(right_layout)
        main_layout.addWidget(right_container, stretch=1)

        self.setLayout(main_layout)

    def get_next_worker(self):
        """
        Get next worker using round-robin selection
        """
        if not self.ocr_workers:
            return None
        
        worker = self.ocr_workers[self.current_worker_index]
        self.current_worker_index = (self.current_worker_index + 1) % len(self.ocr_workers)
        return worker

    def upload_images(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, 
            "Select Multiple Image Files (Ctrl+Click or Shift+Click for multiple selection)", 
            "", 
            "Images (*.png *.jpg *.jpeg *.bmp *.gif *.tiff *.tif);;All Files (*)"
        )
        if files:
            print(f"Selected {len(files)} image files")
            for file_path in files:
                # Use add_thumbnail method for better visual representation
                self.file_list.add_thumbnail(file_path)
            print(f"Added {len(files)} images to the list")
        else:
            print("No files selected")

    def handle_file_list_click(self, file_path):
        # Updated to handle file_path directly from ThumbnailListWidget
        self.display_image(file_path)
    
    def handle_preview_click(self, index: int):
        image, text, _ = preview_image_q[index]  # Extract as numpy array
        self.last_result_image = image           # Can be displayed directly or overwritten with inference results
        self.last_result_text = text
        self.update_display_from_cached()

    def ocr_run(self, image, file_path):
        worker = self.get_next_worker()
        if worker is None:
            print("No OCR workers available")
            return [], [], []
        try:
            boxes, rotated_crops, rec_results, processed_img = worker(image)
            self.performance_widget.update_performance_data(
                det_npu=worker.detection_time_duration / worker.ocr_run_count,
                cls_npu=worker.classification_time_duration / worker.ori_run_count,
                rec_npu=worker.recognition_time_duration / worker.crops_run_count,
                min_rec_npu=worker.min_recognition_time_duration
            )
        except Exception as e:
            print(f"[DEMO] Error during OCR inference: {e}")
            return [], [], [], image
        return boxes, rotated_crops, rec_results, processed_img

    def ocr_run_batch(self, images, file_paths):
        """
        Batch OCR processing for multiple images
        """
        worker = self.get_next_worker()
        if worker is None:
            print("No OCR workers available")
            return []
        
        try:
            # Async batch processing
            results = worker.process_batch(images, timeout=60.0)
            batch_results = []
            for i, result in enumerate(results):
                boxes = result.get('boxes', [])
                rec_results = result.get('rec_results', [])
                processed_img = result.get('preprocessed_image', images[i])
                rotated_crops = []  # Crops not returned in async mode
                batch_results.append((boxes, rotated_crops, rec_results, processed_img))
            
            self.performance_widget.update_performance_data(
                det_npu=worker.detection_time_duration / worker.ocr_run_count,
                cls_npu=worker.classification_time_duration / worker.ori_run_count,
                rec_npu=worker.recognition_time_duration / worker.crops_run_count,
                min_rec_npu=worker.min_recognition_time_duration
            )
        except Exception as e:
            print(f"[DEMO] Error during batch OCR inference: {e}")
            return [([], [], [], image) for image in images]
        
        return batch_results
    
    def _run_async_batch_processing(self):
        """Process all images in async batch mode"""
        images = []
        file_paths = []
        
        # Collect all images and file paths
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            file_path = item.data(Qt.ItemDataRole.UserRole)
            image = cv2.imread(file_path)
            if image is None:
                print(f"Failed to load image file: {file_path}")
                continue
            images.append(image)
            file_paths.append(file_path)
        
        if not images:
            print("No valid images to process.")
            return
        
        print(f"Processing {len(images)} images in batch mode...")
        # Process all images in batch
        batch_results = self.ocr_run_batch(images, file_paths)
        
        # Extract results for batch processing
        org_images = []
        boxes_list = []
        rotated_crops_list = []
        rec_results_list = []
        
        for boxes, rotated_crops, rec_results, processed_image in batch_results:
            org_images.append(processed_image)
            boxes_list.append(boxes)
            rotated_crops_list.append(rotated_crops)
            rec_results_list.append(rec_results)
        
        # Batch process images with OCR results
        result_images = self.ocr2image_batch(org_images, boxes_list, rotated_crops_list, rec_results_list)
        
        # Add results to preview queue
        for i, result_image in enumerate(result_images):
            preview_image_q.append((copy.deepcopy(result_image), rec_results_list[i], file_paths[i]))
        
        # Update last result with the last processed image
        if result_images:
            self.last_result_image = result_images[-1]
            self.last_result_text = rec_results_list[-1]

    def _run_sync_sequential_processing(self):
        """Process images one by one in sync mode"""
        for i in tqdm.tqdm(range(self.file_list.count())):
            # Get file path from item data instead of text
            item = self.file_list.item(i)
            file_path = item.data(Qt.ItemDataRole.UserRole)
            image = cv2.imread(file_path)
            if image is None:
                self.image_label.setText(f"Failed to load image file: {file_path}")
                return
            boxes, _, rec_results, processed_image = self.ocr_run(image, file_path)
            self.last_result_image = self.ocr2image(processed_image, boxes, boxes, rec_results)
            self.last_result_text = rec_results
            preview_image_q.append((copy.deepcopy(self.last_result_image), rec_results, file_path))

    def run_imagelist(self):
        if self.file_list.count() == 0:
            print("No image files available for inference.")
            return
        
        if self.app_mode == "async":
            self._run_async_batch_processing()
        else:
            self._run_sync_sequential_processing()
        
        self.preview_grid.set_images(preview_image_q)
        
    
    def update_display_from_cached(self):
        if self.last_result_image is None:
            return
        rgb_image = self.last_result_image
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qimage = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        
        target_width = int(self.image_label.width() * 0.8)
        target_height = int(self.image_label.height() * 0.8)
        
        size = QSize(target_width, target_height)

        scaled = pixmap.scaled(
            size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled)
        text_lines = ""
        text_lines += "   recognized text : confidence score\n\n"
        for i, text in enumerate(self.last_result_text):
            # Extract recognized text and confidence score from OCR result
            recognized_text = text['text']
            confidence_score = text['score']
            if confidence_score > CONFIDENCE_THRESHOLD:
                text_lines += f"{i+1}. {recognized_text} : {confidence_score:.2f}\n"
        self.info_text.setText(text_lines)
        
    def set_left_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimage = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)

        scaled_pixmap = pixmap.scaled(
            self.left_image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.left_image_label.setPixmap(scaled_pixmap)

    def ocr2image(self, org_image, boxes:list, rotated_crops:list, rec_results:list):
        from PIL import Image
        image = cv2.cvtColor(org_image, cv2.COLOR_BGR2RGB)
        ret_boxes = [line['bbox'] for line in rec_results]
        # Extract recognized text and confidence scores from OCR results
        txts = [line['text'] for line in rec_results]  # recognized text
        bbox_text_poly_shape_quadruplets = []
        for i in range(len(ret_boxes)):
            bbox_text_poly_shape_quadruplets.append(
                ([np.array(ret_boxes[i]).flatten()], txts[i] if i < len(txts) else "", image.shape, image.shape)
            )
        im_sample = draw_with_poly_enhanced(image, bbox_text_poly_shape_quadruplets)
        return np.array(im_sample)

    def ocr2image_batch(self, org_images:list, boxes_list:list, rotated_crops_list:list, rec_results_list:list):
        """
        Batch processing for ocr2image
        """
        batch_results = []
        
        for i in range(len(org_images)):
            org_image = org_images[i]
            rec_results = rec_results_list[i]
            
            image = cv2.cvtColor(org_image, cv2.COLOR_BGR2RGB)
            ret_boxes = [line['bbox'] for line in rec_results]
            # Extract recognized text from OCR results
            txts = [line['text'] for line in rec_results]  # recognized text
            bbox_text_poly_shape_quadruplets = []
            for j in range(len(ret_boxes)):
                bbox_text_poly_shape_quadruplets.append(
                    ([np.array(ret_boxes[j]).flatten()], txts[j] if j < len(txts) else "", image.shape, image.shape)
                )
            im_sample = draw_with_poly_enhanced(image, bbox_text_poly_shape_quadruplets)
            batch_results.append(np.array(im_sample))
            
        return batch_results

    def display_image(self, file_path):
        image = cv2.imread(file_path)
        if image is None:
            self.image_label.setText(f"Failed to load image file: {file_path}")
            return
        
        if self.app_mode == "async":
            batch_results = self.ocr_run_batch([image], [file_path])
            boxes, crops, rec_results, processed_image = batch_results[0]
        else:
            boxes, crops, rec_results, processed_image = self.ocr_run(image, file_path)
        self.last_result_image = self.ocr2image(processed_image, boxes, crops, rec_results)
        self.last_result_text = rec_results
        preview_image_q.append((copy.deepcopy(self.last_result_image), rec_results, file_path))
        self.update_display_from_cached()
        self.preview_grid.set_images(preview_image_q)
        

    def resizeEvent(self, event):
        if self.image_label.pixmap():
            self.update_display_from_cached()
            self.preview_grid.set_images(preview_image_q)
        super().resizeEvent(event)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', default='v4', choices=['v4', 'v5'])
    parser.add_argument('--mode', default='sync', choices=['sync', 'async'])
    
    args = parser.parse_args()
    
    if args.version == 'v4':
        print("v4 selected. This version is no longer supported. Please use v5.")
        exit(90)
 
    print("v5 selected. Will operate with orientation and distortion correction models and algorithms.")
    print("v5 detection model: Modified to support 640 and 960 resolutions")
    base_model_dirname = "engine/model_files/best"
    det_model_dirname = base_model_dirname
    rec_model_dirname = base_model_dirname
    cls_model_path = f"{base_model_dirname}/textline_ori.dxnn"
    uvdoc_model_path = f"{base_model_dirname}/UVDoc_pruned_p3.dxnn"
    doc_ori_model_path = f"{base_model_dirname}/doc_ori.dxnn"
    
    def make_rec_engines(model_dirname):
        io = IO().set_use_ort(True)
        rec_model_map = {}

        for ratio in [3, 5, 10, 15, 25, 35]:
            model_path = f"{model_dirname}/rec_v5_ratio_{ratio}.dxnn"
            rec_model_map[ratio] = IE(model_path, io)
        return rec_model_map
    
    def make_det_engines(model_dirname):
        det_model_map = {}
        for size in [640, 960]:
            model_path = f"{model_dirname}/det_v5_{size}.dxnn"
            det_model_map[size] = IE(model_path, IO().set_use_ort(True))
        return det_model_map

    rec_models:dict = make_rec_engines(rec_model_dirname)
    det_models:dict = make_det_engines(det_model_dirname)
    cls_model = IE(cls_model_path, IO().set_use_ort(True)) # textline orientation model
    doc_ori_model = IE(doc_ori_model_path, IO().set_use_ort(True))
    doc_unwarping_model = IE(uvdoc_model_path, IO().set_use_ort(True))
    
    ocr_workers = []
    
    if args.mode == "async":
        for ratio in [3, 5, 10, 15, 25, 35]:
            rec_models[f'ratio_{ratio}'] = rec_models[ratio]
        ocr_workers = [AsyncPipelineOCR(
            det_models=det_models, cls_model=cls_model, rec_models=rec_models,
            doc_ori_model=doc_ori_model, doc_unwarping_model=doc_unwarping_model, 
            use_doc_preprocessing=True, use_doc_orientation=True
        )]
    else:
        ocr_workers = [PaddleOcr(det_models, cls_model, rec_models, doc_ori_model, doc_unwarping_model, True) for _ in range(1)]
    
    app = QApplication(sys.argv)
    viewer = ImageViewerApp(ocr_workers=ocr_workers, app_version=args.version, app_mode=args.mode)
    viewer.show()
    sys.exit(app.exec())