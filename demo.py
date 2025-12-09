import sys
import os

import argparse

from dx_engine import InferenceEngine as IE
from dx_engine import InferenceOption as IO

from PySide6.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QListWidget, QFileDialog, QSizePolicy, QGridLayout, QScrollArea, QTextEdit, QTableWidget, QTableWidgetItem, QComboBox
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, QSize, Signal, QTimer, QThread

import tqdm

import cv2
import numpy as np

from engine.paddleocr import PaddleOcr, AsyncPipelineOCR
from engine.draw_utils import make_right_display_image, make_left_display_image

from collections import deque
import queue
import time

# OCR result structure constants
CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence score to display

preview_image_q = deque(maxlen=20)

class CameraThread(QThread):
    """Thread for camera capture"""
    frame_ready = Signal(np.ndarray)
    ocr_frame_ready = Signal(np.ndarray)  # Signal for OCR processing (every 30 frames)
    
    def __init__(self, camera_id=0, ocr_worker=None):
        super().__init__()
        self.camera_id = camera_id
        self.running = False
        self.cap = None
        self.ocr_worker = ocr_worker
    
    def get_preprocessed_image(self, input:np.ndarray):
        return self.ocr_worker.get_preprocessed_image(input)
        
    def run(self):
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"Failed to open camera {self.camera_id}")
            return
        
        # 카메라 최적화 설정
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 크기 최소화
        self.cap.set(cv2.CAP_PROP_FPS, 30)
            
        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                frame = cv2.filter2D(frame, -1, kernel)
                frame = self.get_preprocessed_image(frame)
                self.frame_ready.emit(frame)
                self.ocr_frame_ready.emit(frame)
            
    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.wait()

class OCRProcessThread(QThread):
    """Thread for OCR processing"""
    result_ready = Signal(np.ndarray, list, object, bool, object) # Added object for perf_stats
    
    def __init__(self, ocr_worker):
        super().__init__()
        self.ocr_worker = ocr_worker
        self.running = False
        self.frame_queue = queue.Queue(maxsize=10)
        self.process_interval = 0.01
        self.last_process_time = time.time()

    def run(self):
        self.running = True
        self.last_process_time = time.time()
        
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.05)
                
                if frame is not None:
                    time_elapsed = time.time() - self.last_process_time
                    is_time_to_process = time_elapsed >= self.process_interval
                    # Apply sharpening filter to make frame clearer
                    if is_time_to_process:
                        perf_stats = None
                        if hasattr(self.ocr_worker, '__call__') and callable(self.ocr_worker):
                            boxes, _, rec_results, processed_img, perf_stats = self.ocr_worker(frame)
                        else:
                            results = self.ocr_worker.process_batch([frame], timeout=3.0, pass_preprocessing=True)
                            if results:
                                result = results[0]
                                boxes = result.get('boxes', [])
                                rec_results = result.get('rec_results', [])
                                processed_img = result.get('preprocessed_image', frame)
                                perf_stats = result.get('perf_stats', None)
                            else:
                                boxes, rec_results, processed_img = [], [], frame
                        
                        self.result_ready.emit(processed_img, boxes, rec_results, False, perf_stats)
                        self.last_process_time = time.time()
                    else:
                        self.result_ready.emit(frame, [], [], True, None)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"OCR processing error: {e}")
                
    def process_frame(self, frame):
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
        self.frame_queue.put(frame)
        
    def stop(self):
        self.running = False
        self.wait()

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
        from PySide6.QtWidgets import QListWidgetItem
        from PySide6.QtGui import QIcon
        
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
        
        # Add help text for multiple file selection
        performance_tabel_help_text = QLabel("• CPS : Characters Per Second\n• NPU : DEEPX DX-M1")
        performance_tabel_help_text.setStyleSheet("""
            QLabel {
                font-size: 10px;
                color: #666666;
                padding: 2px;
                border-radius: 2px;
                margin-bottom: 4px;
            }
        """)
        layout.addWidget(performance_tabel_help_text)
        
        # Performance table
        self.performance_table = QTableWidget()
        self.performance_table.setRowCount(6)  # 4 rows including header
        self.performance_table.setColumnCount(3)
        # self.performance_table.setMaximumHeight(100)  # Slightly increased for header
        self.performance_table.setMinimumHeight(80)   # Increased minimum height
        
        # Set table headers
        self.performance_table.setHorizontalHeaderLabels(["", "GPU", "NPU"])
        self.performance_table.setVerticalHeaderLabels(["", "det", "cls", "rec", "e2e", "cps"])
        
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
        self.performance_table.setColumnWidth(2, 280)  # NPU column
        
        # Set initial performance data
        self.update_performance_data()
        
        layout.addWidget(self.performance_table)

        self.setLayout(layout)
    
    def update_performance_data(self, det_gpu=10, det_npu=101.01, cls_gpu=95.3, cls_npu=11.08, rec_gpu=121, rec_npu=13.8, e2e_npu = 300.0, cps = 310, total_chars = 301):
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
            QTableWidgetItem(f"{1000/(rec_npu + 1e-10):.2f} (per crops)")
        ]
        
        e2e_row = [
            QTableWidgetItem("e2e"),
            QTableWidgetItem(" - "),
            QTableWidgetItem(f"{1000/(e2e_npu + 1e-10):.2f} (per frame)")
        ]
        
        cps_row = [
            QTableWidgetItem("cps"),
            QTableWidgetItem(" - "),
            QTableWidgetItem(f"{cps:.2f} chars/sec, total {total_chars}")
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
        for col, item in enumerate(e2e_row):
            self.performance_table.setItem(4, col, item)
        for col, item in enumerate(cps_row):
            self.performance_table.setItem(5, col, item)


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
    def __init__(self, ocr_workers=None, app_version=None, hide_preview=False):
        super().__init__()
        self.setWindowTitle(f"PySide Image Viewer {app_version}")
        
        # Set window to full screen size
        screen_size = QApplication.primaryScreen().geometry()
        # self.setGeometry(100, 100, screen_size.width()/1.2, screen_size.height()/1.2)
        self.setGeometry(100, 100, screen_size.width()/2.3, screen_size.height()/1.6)
        
        # Store OCR workers
        self.ocr_workers = ocr_workers
        self.current_worker_index = 0  # Index for round-robin worker selection
        self.app_version = app_version
        self.app_mode = "async"
        
        # Camera related attributes
        self.camera_thread = None
        self.ocr_thread = None
        self.camera_active = False
        self.last_camera_frame = None
        
        # Initialize UI first
        self.init_ui()
        self.last_result_boxes = None
        self.last_result_text = None
        self.last_processed_img = None
        self.last_scaled_right_image = None

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
        
        # Create horizontal layout for left and right image display
        image_display_layout = QHBoxLayout()
        
        # Left side - Input Image with BBox
        left_viewer_layout = QVBoxLayout()
        left_title = QLabel("Input Image + BBox")
        left_title.setStyleSheet("""
            QLabel {
                font-size: 16px;
                color: black;
                padding: 2px;
                border-radius: 2px;
                margin-bottom: 2px;
                qproperty-alignment: AlignCenter;
            }
        """)
        left_viewer_layout.addWidget(left_title)
        
        self.left_image_label = QLabel("Please select an image")
        self.left_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.left_image_label.setStyleSheet("border: 1px solid gray;")
        left_viewer_layout.addWidget(self.left_image_label)
        
        # Right side - Result Text Overlay
        right_viewer_layout = QVBoxLayout()
        right_title = QLabel("Result Text Overlay")
        right_title.setStyleSheet("""
            QLabel {
                font-size: 16px;
                color: black;
                padding: 2px;
                border-radius: 2px;
                margin-bottom: 2px;
                qproperty-alignment: AlignCenter;
            }
        """)
        right_viewer_layout.addWidget(right_title)
        
        self.right_image_label = QLabel("Please select an image")
        self.right_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.right_image_label.setStyleSheet("border: 1px solid gray;")
        right_viewer_layout.addWidget(self.right_image_label)
        
        image_display_layout.addLayout(left_viewer_layout)
        image_display_layout.addLayout(right_viewer_layout)
        
        center_layout.addLayout(image_display_layout, stretch=3)
        
        center_container = QWidget()
        center_container.setLayout(center_layout)
        main_layout.addWidget(center_container, stretch=3)

        right_layout = QVBoxLayout()

        # Add Performance Information widget before File Management
        self.performance_widget = PerformanceInfoWidget()
        right_layout.addWidget(self.performance_widget, stretch=3)
        # Add Accuracy Information widget before File Management
        self.envInfo_widget = EnvironmentInfoWidget()
        right_layout.addWidget(self.envInfo_widget, stretch=1)
        # Add Accuracy Information widget before File Management
        self.accuracy_widget = AccuracyInfoWidget()
        right_layout.addWidget(self.accuracy_widget, stretch=1)

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
        
        # Camera controls
        camera_label = QLabel("Camera Controls")
        camera_label.setStyleSheet("""
            QLabel {
                font-size: 13px;
                color: black;
                padding: 2px;
                border-radius: 2px;
                margin-top: 10px;
                margin-bottom: 2px;
            }
        """)
        right_layout.addWidget(camera_label)
        
        self.camera_selector = QComboBox()
        self.camera_selector.addItems(["Camera 0", "Camera 1", "Camera 2"])
        right_layout.addWidget(self.camera_selector)
        
        self.camera_button = QPushButton("Start Camera")
        self.camera_button.clicked.connect(self.toggle_camera)
        right_layout.addWidget(self.camera_button)

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
        image, text, _ = preview_image_q[index]
        self.last_result_text = text
        self.last_processed_img = image
        self.update_display_from_cached()

    def ocr_run(self, image, file_path):
        worker = self.get_next_worker()
        if worker is None:
            print("No OCR workers available")
            return [], [], image, None
        try:
            if hasattr(worker, '__call__') and callable(worker):
                # Sync mode
                boxes, _, rec_results, processed_img, perf_stats = worker(image)
            else:
                # Should not happen, but fallback
                boxes, rec_results, processed_img, perf_stats = [], [], image, None
        except Exception as e:
            print(f"[DEMO] Error during OCR inference: {e}")
            return [], [], image, None
        return boxes, rec_results, processed_img, perf_stats

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
            results = worker.process_batch(images, timeout=60.0, pass_preprocessing=False)
            batch_results = []
            for i, result in enumerate(results):
                boxes = result.get('boxes', [])
                rec_results = result.get('rec_results', [])
                processed_img = result.get('preprocessed_image', images[i])
                perf_stats = result.get('perf_stats', None)
                batch_results.append((boxes, rec_results, processed_img, perf_stats))
        except Exception as e:
            print(f"[DEMO] Error during batch OCR inference: {e}")
            return [([], [], image, None) for image in images]
        
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
        rec_results_list = []
        perf_stats_list = []
        
        for boxes, rec_results, processed_image, perf_stats in batch_results:
            org_images.append(processed_image)
            boxes_list.append(boxes)
            rec_results_list.append(rec_results)
            perf_stats_list.append(perf_stats)
        
        # Add results to preview queue
        for i, processed_image in enumerate(org_images):
            preview_image_q.append((processed_image, rec_results_list[i], file_paths[i]))
        
        # Update last result with the last processed image
        if batch_results:
            self.last_result_boxes = boxes_list[-1]
            self.last_result_text = rec_results_list[-1]
            self.last_processed_img = org_images[-1]
            # Update performance widget with last image stats
            if perf_stats_list and perf_stats_list[-1]:
                self.update_performance_widget(perf_stats_list[-1])

    def _run_sync_sequential_processing(self):
        """Process images one by one in sync mode"""
        for i in tqdm.tqdm(range(self.file_list.count())):
            # Get file path from item data instead of text
            item = self.file_list.item(i)
            file_path = item.data(Qt.ItemDataRole.UserRole)
            image = cv2.imread(file_path)
            if image is None:
                self.left_image_label.setText(f"Failed to load image file: {file_path}")
                return
            boxes, rec_results, processed_image, perf_stats = self.ocr_run(image, file_path)
            self.last_result_boxes = boxes
            self.last_result_text = rec_results
            self.last_processed_img = processed_image
            # Update performance widget with last image stats
            if perf_stats:
                self.update_performance_widget(perf_stats)
            preview_image_q.append((processed_image, rec_results, file_path))
    
    def toggle_camera(self):
        if not self.camera_active:
            self.start_camera()
        else:
            self.stop_camera()
            
    def start_camera(self):
        camera_id = self.camera_selector.currentIndex()
        worker = self.get_next_worker()
        
        if worker is None:
            print("No OCR workers available")
            return
        
        # Start camera thread
        self.camera_thread = CameraThread(camera_id, worker)
        self.camera_thread.frame_ready.connect(self.on_camera_frame_display)  # For real-time display
        self.camera_thread.ocr_frame_ready.connect(self.on_camera_frame_ocr)  # For OCR processing
        self.camera_thread.start()
        
        # Start OCR processing thread
        self.ocr_thread = OCRProcessThread(worker)
        self.ocr_thread.result_ready.connect(self.on_ocr_result)
        self.ocr_thread.start()
        
        self.camera_active = True
        self.camera_button.setText("Stop Camera")
        self.camera_selector.setEnabled(False)
        print(f"Camera {camera_id} started")
        
    def stop_camera(self):
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None
            
        if self.ocr_thread:
            self.ocr_thread.stop()
            self.ocr_thread = None
        
        self.camera_active = False
        self.camera_button.setText("Start Camera")
        self.camera_selector.setEnabled(True)
        print("Camera stopped")
        
    def on_camera_frame_display(self, frame):
        """Handle every camera frame for display only"""
        if self.last_result_boxes is not None:
            self.last_processed_img = frame
            self.update_display_from_cached(pass_right_display=True)
            
    def on_camera_frame_ocr(self, frame):
        """Handle frames for OCR processing (every 30 frames)"""
        if self.ocr_thread:
            self.ocr_thread.process_frame(frame)
            
    def on_ocr_result(self, processed_img, boxes, rec_results, pass_result_display=True, perf_stats=None):
        self.last_processed_img = processed_img
        if not pass_result_display:
            self.last_result_boxes = boxes
            self.last_result_text = rec_results
            self.update_display_from_cached()
            # Update performance data if available
            if perf_stats:
                self.update_performance_widget(perf_stats)
        else:
            self.update_display_from_cached(pass_right_display=True)
        
    def closeEvent(self, event):
        if self.camera_thread:
            self.stop_camera()
        super().closeEvent(event)

    def keyPressEvent(self, event):
        """Handle keyboard events"""
        key = event.key()
        if key == Qt.Key.Key_S:
            self.save_current_result_to_preview()
        super().keyPressEvent(event)
    
    def update_performance_widget(self, perf_stats):
        """Update performance widget with stats from OCR worker"""
        if perf_stats is None:
            return
        
        det_time = perf_stats.get('det_time_ms', 0)
        cls_time = perf_stats.get('cls_time_ms', 0)
        rec_time = perf_stats.get('rec_time_ms', 0)
        e2e_time = perf_stats.get('e2e_time_ms', 0)
        cps = perf_stats.get('cps', 0)
        total_chars = perf_stats.get('total_chars', 0)
        
        # Update performance widget (GPU column placeholder, NPU column with actual data)
        self.performance_widget.update_performance_data(
            det_npu=det_time,
            cls_npu=cls_time,
            rec_npu=rec_time,
            e2e_npu=e2e_time,
            cps=cps,
            total_chars=total_chars
        )
    
    def save_current_result_to_preview(self):
        """Save current OCR result to preview queue when space key is pressed"""
        if self.last_processed_img is not None and self.last_result_text is not None:
            # Create result image for preview
            ret_boxes = [line['bbox'] for line in self.last_result_text]
            txts = [line['text'] for line in self.last_result_text]
            
            # Convert to RGB for display
            image_rgb = cv2.cvtColor(self.last_processed_img, cv2.COLOR_BGR2RGB)
            bbox_text_poly_shape_quadruplets = []
            for i in range(len(ret_boxes)):
                bbox_text_poly_shape_quadruplets.append(
                    ([np.array(ret_boxes[i]).flatten()], txts[i] if i < len(txts) else "", image_rgb.shape, image_rgb.shape)
                )
            
            # Add to preview queue with timestamp as file_path identifier
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            preview_image_q.append((self.last_processed_img, self.last_result_text, f"camera_{timestamp}"))
            
            # Update preview grid
            self.preview_grid.set_images(preview_image_q)
            print(f"Saved current result to preview: camera_{timestamp}")
        else:
            print("No OCR result available to save")

    def run_imagelist(self):
        if self.file_list.count() == 0:
            print("No image files available for inference.")
            return
        
        if self.app_mode == "async":
            self._run_async_batch_processing()
        else:
            self._run_sync_sequential_processing()
        
        self.preview_grid.set_images(preview_image_q)
        
    
    def update_display_from_cached(self, pass_right_display=False):
        if self.last_processed_img is None:
            return
        
        # Generate left image (input + bbox)
        ret_boxes = [line['bbox'] for line in self.last_result_text]
        left_image = make_left_display_image(self.last_processed_img, ret_boxes)
        
        if self.last_scaled_right_image is None:
            pass_right_display = False  # Force generate right image first time
        if not pass_right_display:
            # Generate right image (text overlay)
            txts = [line['text'] for line in self.last_result_text]
            right_image = make_right_display_image(self.last_processed_img, ret_boxes, txts)
        
        left_rgb = cv2.cvtColor(np.array(left_image), cv2.COLOR_BGR2RGB)
        h, w, ch = left_rgb.shape
        bytes_per_line = ch * w
        qimage_left = QImage(left_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap_left = QPixmap.fromImage(qimage_left)
        
        target_width = int(self.left_image_label.width() * 0.9)
        target_height = int(self.left_image_label.height() * 0.9)
        size = QSize(target_width, target_height)
        
        scaled_left = pixmap_left.scaled(
            size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.left_image_label.setPixmap(scaled_left)
        
        if not pass_right_display:
            # Display right image using PIL
            right_rgb = cv2.cvtColor(np.array(right_image), cv2.COLOR_BGR2RGB)
            h, w, ch = right_rgb.shape
            bytes_per_line = ch * w
            qimage_right = QImage(right_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap_right = QPixmap.fromImage(qimage_right)
            
            scaled_right = pixmap_right.scaled(
                size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.last_scaled_right_image = scaled_right
        self.right_image_label.setPixmap(self.last_scaled_right_image)
        
        # Update text info
        text_lines = ""
        text_lines += "   recognized text : confidence score\n\n"
        for i, text in enumerate(self.last_result_text):
            # Extract recognized text and confidence score from OCR result
            recognized_text = text['text']
            confidence_score = text['score']
            if confidence_score > CONFIDENCE_THRESHOLD:
                text_lines += f"{i+1}. {recognized_text} : {confidence_score:.2f}\n"
        self.info_text.setText(text_lines)

    def display_image(self, file_path):
        image = cv2.imread(file_path)
        if image is None:
            self.left_image_label.setText(f"Failed to load image file: {file_path}")
            return
        
        perf_stats = None
        if self.app_mode == "async":
            batch_results = self.ocr_run_batch([image], [file_path])
            boxes, rec_results, processed_image, perf_stats = batch_results[0]
        else:
            boxes, rec_results, processed_image, perf_stats = self.ocr_run(image, file_path)
        
        self.last_result_boxes = boxes
        self.last_result_text = rec_results
        self.last_processed_img = processed_image
        
        # Update performance widget
        if perf_stats:
            self.update_performance_widget(perf_stats)
        
        preview_image_q.append((processed_image, rec_results, file_path))
        self.update_display_from_cached()
        self.preview_grid.set_images(preview_image_q)
        

    def resizeEvent(self, event):
        if self.last_processed_img is not None:
            self.update_display_from_cached()
            self.preview_grid.set_images(preview_image_q)
        super().resizeEvent(event)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', default='v4', choices=['v4', 'v5'])
    parser.add_argument('--use-mobile', dest='use_mobile', action='store_true', default=False)
    parser.add_argument('--hide-preview', dest='hide_preview', action='store_true', default=False)
    parser.add_argument('--enable-uvdoc', dest='disable_uvdoc', action='store_false', default=True)
    
    args = parser.parse_args()
    
    if args.version == 'v4':
        print("v4 selected. This version is no longer supported. Please use v5.")
        exit(90)
 
    print("v5 selected. Will operate with orientation and distortion correction models and algorithms.")
    print("v5 detection model: Modified to support 640 and 960 resolutions")
    base_model_dirname = "engine/models/dxnn_optimized"
    if args.use_mobile:
        base_model_dirname = "engine/models/dxnn_mobile_optimized"
    det_model_dirname = base_model_dirname
    rec_model_dirname = base_model_dirname
    rec_dict_dir = f"{base_model_dirname}/ppocrv5_dict.txt"
    cls_model_path = f"{base_model_dirname}/textline_ori.dxnn"
    uvdoc_model_path = f"{base_model_dirname}/UVDoc_pruned_p3.dxnn"
    doc_ori_model_path = f"{base_model_dirname}/doc_ori_fixed.dxnn"
    
    def make_rec_engines(model_dirname):
        io = IO().set_use_ort(True)
        rec_model_map = {}

        for ratio in [3, 5, 10, 15, 25, 35]:
            model_path = f"{model_dirname}/rec_v5_ratio_{ratio}.dxnn"
            if "mobile" in model_path:
                model_path = f"{model_dirname}/rec_mobile_ratio_{ratio}.dxnn"
            rec_model_map[ratio] = IE(model_path, io)
        return rec_model_map
    
    def make_det_engines(model_dirname:str):
        det_model_map = {}
        for size in [640, 960]:
            model_path = f"{model_dirname}/det_v5_{size}.dxnn"
            if "mobile" in model_path:
                model_path = f"{model_dirname}/det_mobile_{size}.dxnn"
            det_model_map[size] = IE(model_path, IO().set_use_ort(True))
        return det_model_map

    rec_models:dict = make_rec_engines(rec_model_dirname)
    det_models:dict = make_det_engines(det_model_dirname)
    cls_model = IE(cls_model_path, IO().set_use_ort(True)) # textline orientation model
    doc_ori_model = IE(doc_ori_model_path, IO().set_use_ort(True))
    doc_unwarping_model = IE(uvdoc_model_path, IO().set_use_ort(True))
    
    ocr_workers = []
    
    for ratio in [3, 5, 10, 15, 25, 35]:
        rec_models[f'ratio_{ratio}'] = rec_models[ratio]
    ocr_workers = [AsyncPipelineOCR(
        det_models=det_models, cls_model=cls_model, rec_models=rec_models, rec_dict_dir=rec_dict_dir,
        doc_ori_model=doc_ori_model, doc_unwarping_model=doc_unwarping_model, 
        use_doc_orientation=not args.disable_uvdoc, use_doc_preprocessing=not args.disable_uvdoc, verbose=False
    )]
    
    app = QApplication(sys.argv)
    viewer = ImageViewerApp(ocr_workers=ocr_workers, app_version=args.version, hide_preview=args.hide_preview)
    viewer.show()
    sys.exit(app.exec())