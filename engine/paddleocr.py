"""
PP-OCRv5 Style OCR Engine Implementation
- Document Preprocessing: Document orientation correction and unwarping 
- Text Detection: Text region detection
- Text Classification: Text rotation angle classification 
- Text Recognition: Text content recognition
- Sync/Async Pipeline: Synchronous/Asynchronous processing support
"""

import os
import sys
import time
import queue
import threading

from dx_engine import InferenceEngine as IE

from .utils import get_rotate_crop_image, filter_tag_det_res
from .utils import det_router, rec_router, split_bbox_for_recognition, merge_recognition_results

from typing import List, Tuple, Dict, Optional, Any
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import json
from datetime import datetime

engine_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(engine_path)

from baidu import det_preprocess, cls_preprocess, rec_preprocess, doc_ori_preprocess, uvdoc_preprocess, resize_align_corners
from models.ocr_postprocess import DetPostProcess, ClsPostProcess, RecLabelDecode, DocOriPostProcess
from preprocessing import PreProcessingCompose

class Node():
    """
    Base node class for OCR pipeline
    Base class for each processing stage (Detection, Classification, Recognition)
    """
    def __init__(self):
        # Base node class - concrete implementation in subclasses
        pass
    
    def prepare_input(self, inp: np.ndarray) -> np.ndarray:
        """
        Prepare input tensor for DXNN model
        NCHW -> NHWC conversion and ensure memory contiguity
        
        Args:
            inp: Input array (numpy array)
            
        Returns:
            Input array prepared for DXNN model
        """
        # Thread-safe version using only numpy (no PyTorch)
        if inp.ndim == 3:
            inp = np.expand_dims(inp, 0)
        # Permute from NCHW to NHWC: (0,2,3,1) means N,C,H,W -> N,H,W,C
        if inp.ndim == 4:
            inp = np.transpose(inp, (0, 2, 3, 1))
        # Ensure contiguous memory layout
        inp = np.ascontiguousarray(inp)
        return inp
    
    def thread_postprocess(self):
        # Thread post-processing method - currently unused
        return 0


class DetectionNode(Node):
    """
    Text Region Detection Node (PP-OCRv5 Detection)
    - Multi-resolution model support (640x640, 960x960)
    - Automatic model selection based on image size
    - DBNet-based text detection
    """
    def __init__(self, models:dict):
        """
        Args:
            models: {resolution: IE_model} dictionary (e.g., {640: model_640, 960: model_960})
        """
        self.det_model_map: dict = models

        # Create preprocessing pipeline for each resolution
        self.det_preprocess_map = {}
        for res in [640, 960]:
            res_det_preprocess = det_preprocess.copy()
            res_det_preprocess[0]['resize']['size'] = [res, res]
            self.det_preprocess_map[res] = PreProcessingCompose(res_det_preprocess)

        # DBNet post-processing configuration
        self.det_postprocess = DetPostProcess(
            thresh=0.3,
            box_thresh=0.6,
            max_candidates=1500,
            unclip_ratio=1.5,
            use_dilation=False,
            score_mode="fast",
            box_type="quad",
        )

        self.router = det_router

        # Debug settings (set to False when not in use)
        self.detection_save_dir = False
        self.counter = 0
        if self.detection_save_dir:
            os.makedirs(self.detection_save_dir, exist_ok=True)
        self.det_run_count = 0
    
    def __call__(self, img):
        """
        Perform text region detection on image
        
        Args:
            img: Input image (numpy array)
            
        Returns:
            tuple: (detected_boxes, execution_count)
        """
        h, w = img.shape[:2]
        # Model selection based on image size
        mapped_res = self.router(w, h)
        res_preprocess = self.det_preprocess_map[mapped_res]
        res_model = self.det_model_map[mapped_res]

        # Preprocessing and inference
        model_input = res_preprocess(img)
        padding_info = res_preprocess.get_last_padding_info()
        model_input = self.prepare_input(model_input)
        output = res_model.run([model_input])
        det_output = self.postprocess(output[0], img.shape, padding_info)
        self.det_run_count += 1
        
        # Save debug images
        if self.detection_save_dir:
            import os
            save_path = os.path.join(self.detection_save_dir, f"det_{self.counter}.jpg")
            self.counter += 1
            self.draw_detection_boxes(img, det_output, save_path)
        
        return det_output, self.det_run_count
    
    def postprocess(self, outputs, image_shape, padding_info=None):
        dt_boxes = self.det_postprocess(outputs, image_shape, padding_info)
        dt_boxes = dt_boxes[0]["points"]
        dt_boxes = filter_tag_det_res(dt_boxes, image_shape)
        return dt_boxes
    
    def draw_detection_boxes(self, image, dt_boxes, save_path=None):
        """Draw detection boxes on image and optionally save it.
        
        Args:
            image: Original image (numpy array)
            dt_boxes: Detection boxes from postprocess
            save_path: Path to save the image with boxes (optional)
            
        Returns:
            Image with drawn boxes
        """
        import os
        
        if len(dt_boxes) == 0:
            if save_path:
                cv2.imwrite(save_path, image)
            return image
            
        draw_img = image.copy()
        
        for box in dt_boxes:
            box = np.array(box).astype(np.int32)
            
            cv2.polylines(draw_img, [box], True, color=(0, 255, 0), thickness=2)
            
            for point in box:
                cv2.circle(draw_img, tuple(point), 3, (255, 0, 0), -1)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, draw_img)
            print(f"Detection result saved: {save_path}")
            
        return draw_img

class ClassificationNode(Node):
    """
    Text Orientation Classification Node (PP-OCRv5 Classification)
    - Determine if text image is rotated 180 degrees
    - Align detected text regions to correct orientation
    """
    def __init__(self, model:IE):
        """
        Args:
            model: Text orientation classification model (DXNN IE)
        """
        self.model = model
        self.cls_preprocess = PreProcessingCompose(cls_preprocess)
        self.cls_postprocess = ClsPostProcess()
        self.ori_run_count = 0

        # Debug settings (currently unused)
        self.debug_dir = "cls_debug/"
        self.debug_counter = 0
        os.makedirs(self.debug_dir, exist_ok=True)
    
    def __call__(self, det_outputs:List[np.ndarray]):
        """
        Perform orientation classification on detected text images
        
        Args:
            det_outputs: List of detected text images
            
        Returns:
            tuple: (classification_results_list, execution_count)
                classification_results: [[label, confidence], ...] format
        """
        outputs = []
        for det_output in det_outputs:
            cls_input = self.cls_preprocess(det_output)
            cls_input = self.prepare_input(cls_input)
            output = self.model.run([cls_input])
            label, score = self.cls_postprocess(output[0])[0]
            outputs.append([label, score])
            self.ori_run_count += 1
        return outputs, self.ori_run_count

    def save_sample_image(self, image, score):
        """Save image for debugging purposes"""
        filename = f"sample_{self.debug_counter:06d}_{score}.jpg"
        save_path = os.path.join(self.debug_dir, filename)
        
        cv2.imwrite(save_path, image)
        self.debug_counter += 1


class RecognitionNode(Node):
    """
    Text Recognition Node (PP-OCRv5 Recognition)
    - Multi-model support for various aspect ratios of text images
    - Automatic model selection by ratio: ratio_3, ratio_5, ratio_10, ratio_15, ratio_25, ratio_35
    - CRNN-based text recognition
    """
    def __init__(self, models:dict):
        """
        Args:
            models: {ratio_key: IE_model} dictionary
        """
        self.rec_model_map: dict = models
        
        # Create preprocessing pipeline for each ratio
        self.rec_preprocess_map = {}

        ratio_rec_preprocess = rec_preprocess.copy()
        ratio_rec_preprocess[0]['resize']['size'] = [48, 120]
        self.rec_preprocess_map[3] = PreProcessingCompose(ratio_rec_preprocess)
        for i in [5, 10, 15, 25, 35]:
            ratio_rec_preprocess[0]['resize']['size'] = [48, 48 * i]
            self.rec_preprocess_map[i] = PreProcessingCompose(ratio_rec_preprocess)
        
        # Load character dictionary
        txt_file_name = 'model_files/ppocrv5_dict.txt'
        character_dict_path = os.path.join(engine_path, txt_file_name)
        
        self.rec_postprocess = RecLabelDecode(character_dict_path=character_dict_path, use_space_char=True)
        self.router = rec_router
        self.drop = 0.3  # Confidence threshold
        
        self.crops_run_count = 0

        # Debug settings (set to False when not in use)
        self.debug_dir = False
        self.debug_counter = 0
        if self.debug_dir:
            os.makedirs(self.debug_dir, exist_ok=True)

    def split_bbox_for_recognition(self, bbox, rec_image_shape, overlap_ratio=0.1):
        """Split long text boxes into multiple parts"""
        return split_bbox_for_recognition(bbox, rec_image_shape, overlap_ratio)
    
    def merge_recognition_results(self, split_results, overlap_ratio=0.1):
        """Merge split recognition results"""
        return merge_recognition_results(split_results, overlap_ratio)
    
    def classify_ratio(self, width, height):
        """Classify by width/height ratio: ≤3, ≤5, ≤15, >15"""
        ratio = width / height if height > 0 else float('inf')
        
        if ratio <= 4:
            return "ratio_3"
        elif ratio <= 10:
            return "ratio_5"
        elif ratio <= 12.5:
            return "ratio_10"
        elif ratio <= 25:
            return "ratio_15"
        else:
            return "ratio_25"
    
    def classify_height(self, height):
        """Classify by image height: 20 or below vs above"""
        if height <= 10:
            return "width_20"
        else:
            return "width_40"
    
    def save_sample_image(self, image, _width, _height):
        """Save image for debugging purposes"""
        filename = f"rec_{self.debug_counter:06d}.jpg"
        save_path = os.path.join(self.debug_dir, filename)
        
        cv2.imwrite(save_path, image)
        self.debug_counter += 1
        
    def __call__(self, original_image, boxes:List, crops:List[np.ndarray]):
        """
        Perform text content recognition on text images
        
        Args:
            original_image: Original image 
            boxes: List of text box coordinates
            crops: List of cropped text images
            
        Returns:
            tuple: (recognition_results_list, execution_count, min_latency)
        """
        outputs = []
        min_latency = 10000000
        for i in range(len(crops)):
            cropped_img = crops[i]
            # Model selection based on image ratio
            mapped_ratio = self.router(cropped_img.shape[1], cropped_img.shape[0])
            ratio_preprocess = self.rec_preprocess_map[mapped_ratio]
            ratio_model = self.rec_model_map[mapped_ratio]

            # Preprocessing and inference
            inp = ratio_preprocess(cropped_img)
            rec_input = self.prepare_input(inp)

            start_time = time.time()
            output = ratio_model.run([rec_input])
            end_time = time.time()

            if min_latency > end_time - start_time:
                min_latency = end_time - start_time
            res = self.rec_postprocess(output[0])[0]
            self.crops_run_count += 1
            
            # Filter by confidence threshold and save results
            if res[1] > self.drop:
                outputs.append({
                    'bbox_index': i,
                    'bbox': boxes[i],
                    'text': res[0],
                    'score': res[1]
                })
        return outputs, self.crops_run_count, min_latency


class DocumentOrientationNode(Node):
    """
    Document Orientation Correction Node (Document Orientation)
    - Detect document image rotation angle (0°, 90°, 180°, 270°)
    - First stage of PP-OCRv5 Document Preprocessing pipeline
    """
    def __init__(self, model: IE):
        """
        Args:
            model: Document orientation classification model (DXNN IE)
        """
        self.model = model
        self.preprocess = PreProcessingCompose(doc_ori_preprocess)
        self.doc_postprocess = DocOriPostProcess(label_list=['0', '90', '180', '270'])

    def __call__(self, image: np.ndarray) -> Tuple[List[Tuple[int, np.ndarray]], float]:
        """
        Analyze document image orientation and rotate to correct direction
        
        Args:
            image: Input document image
            
        Returns:
            tuple: ((rotation_angle, rotated_image), processing_time)
        """
        engine_latency = 0
        
        preprocessed = self.preprocess(image)
        preprocessed = self.prepare_input(preprocessed)
        start_time = time.time()
        
        # DXNN inference
        output = self.model.run([preprocessed])[0]
        
        engine_latency += time.time() - start_time
        
        angle, rotated_image = self.postprocess(output, image)
        return (angle, rotated_image), engine_latency

    def postprocess(self, output: np.ndarray, original_image: np.ndarray) -> Tuple[int, np.ndarray]:
        """Post-processing: Rotate image by predicted angle"""
        doc_ori_results = self.doc_postprocess(output)[0]  # [(label, score), ...]
        label, _ = doc_ori_results
        rotated_image = self.rotate_image(original_image, label)
        
        return label, rotated_image
    
    def rotate_image(self, image: np.ndarray, angle: int) -> np.ndarray:
        """Rotate image by given angle"""
        _ = image.shape[:2]  # height, width (unused)
        
        if str(angle) == "0":
            return image.copy()
        elif str(angle) == "90":
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif str(angle) == "180":
            return cv2.rotate(image, cv2.ROTATE_180)
        elif str(angle) == "270":
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        else:
            return image.copy()


class DocumentUnwarpingNode(Node):
    """
    Document Unwarping Correction Node (Document Unwarping - UVDoc)
    - Correct warped document images to flat form
    - Second stage of PP-OCRv5 Document Preprocessing pipeline
    - Uses UVDoc algorithm
    """
    def __init__(self, model: IE):
        """
        Args:
            model: Document unwarping correction model (UVDoc DXNN IE)
        """
        self.model = model
        self.preprocess = PreProcessingCompose(uvdoc_preprocess)
        self.target_size = (712, 488)
        self.origin_shape = None
    
    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Correct document image distortion to transform into flat form
        
        Args:
            image: Input document image
            
        Returns:
            tuple: (corrected_image, processing_time)
        """
        engine_latency = 0
        preprocessed = self.preprocess(image)
        preprocessed = self.prepare_input(preprocessed)

        start_time = time.time()
        # DXNN inference
        output = self.model.run([preprocessed])[0]
        
        engine_latency += time.time() - start_time
        unwarped_image = self.postprocess(output, image)
        return unwarped_image, engine_latency

    def postprocess(self, uv_map: np.ndarray, original_image: np.ndarray) -> np.ndarray:
        """Post-processing: Image distortion correction using UV map"""
        orig_h, orig_w = original_image.shape[:2]
        
        if uv_map.ndim == 4:
            uv_map = uv_map.squeeze(0)
        
        # Resize UV map to original image size
        uv_resized = np.zeros((2, orig_h, orig_w), dtype=np.float32)
        for i in range(2):
            uv_resized[i] = cv2.resize(uv_map[i], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        corrected_rgb = self.gridsample_torch(original_image, uv_resized, align_corners=True)
        return corrected_rgb
    
    def gridsample_torch(self, input_image: np.ndarray, uv_map: np.ndarray, align_corners: bool = True) -> np.ndarray:
        """PyTorch GridSample implementation (GPU accelerated)"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        input_tensor = torch.from_numpy(input_image.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        uv_tensor = torch.from_numpy(uv_map).to(device)
        grid = uv_tensor.permute(1, 2, 0).unsqueeze(0)  # [1,H,W,2]
        
        # GridSample
        output_tensor = F.grid_sample(
            input_tensor,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=align_corners
        )
        
        # Transform: [1,C,H,W] -> [H,W,C]
        output = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return output.astype(np.uint8)

class DocumentPreprocessingPipeline:
    """
    PP-OCRv5 Style Document Preprocessing Pipeline
    - Stage 1: Document Orientation (document orientation correction)
    - Stage 2: Document Unwarping (document distortion correction)
    - Preprocessing pipeline for improving OCR accuracy
    """
    def __init__(self, 
                 orientation_model: Optional[IE] = None,
                 unwarping_model: Optional[IE] = None,
                 use_doc_orientation: bool = True,
                 use_doc_unwarping: bool = True):
        """
        Args:
            orientation_model: Document orientation correction model
            unwarping_model: Document distortion correction model  
            use_doc_orientation: Whether to use orientation correction
            use_doc_unwarping: Whether to use distortion correction
        """
        self.use_doc_unwarping = use_doc_unwarping and unwarping_model is not None

        self.orientation_node = DocumentOrientationNode(orientation_model) if use_doc_orientation else None
        self.unwarping_node = DocumentUnwarpingNode(unwarping_model)

    def __call__(self, image: np.ndarray) -> Tuple[List[Dict], float]:
        """
        Execute document preprocessing pipeline
        
        Args:
            image: Input document image
            
        Returns:
            tuple: (processed_image, total_processing_time)
        """
        total_latency = 0
        
        current_image = image.copy()
        
        # Stage 1: Document orientation correction
        if self.orientation_node:
            orientation_results, orientation_latency = self.orientation_node(current_image)
            total_latency += orientation_latency
            
            _, rotated_image = orientation_results
            current_image = rotated_image

        # Stage 2: Document distortion correction
        if self.unwarping_node:
            current_image, unwarping_latency = self.unwarping_node(current_image)
            total_latency += unwarping_latency
            
        return current_image, total_latency
    

class PaddleOcr():
    def __init__(self, 
                 det_model, 
                 cls_model: IE, 
                 rec_models: dict, 
                 doc_ori_model: Optional[IE],
                 doc_unwarping_model: Optional[IE] = None,
                 use_doc_preprocessing: bool = True,
                 use_doc_orientation: bool = True):
        '''
        @brief: PP-OCRv5 style OCR engine (including Document Preprocessing)
        @param:
            det_model: DeepX detection model (single IE or dict of {resolution: IE})
            cls_model: DeepX classification model (cls.dxnn) - also serves as doc_ori
            rec_models: DeepX recognition models dict
            doc_unwarping_model: Document distortion correction model (UVDoc)
            use_doc_preprocessing: Whether to use document preprocessing (unwarping)
            use_doc_orientation: Whether to use document orientation correction (doc_ori)
        @return:
            None
        '''
        # Support both single model and dict of models
        if isinstance(det_model, dict):
            self.det_models = det_model
        else:
            # Wrap single model in dict for compatibility
            self.det_models = {640: det_model}
        
        self.cls_model = cls_model
        self.rec_models: dict = rec_models
        self.use_doc_preprocessing = use_doc_preprocessing
        
        self.detection_node = DetectionNode(self.det_models)
        self.classification_node = ClassificationNode(self.cls_model)
        self.recognition_node = RecognitionNode(self.rec_models)
        
        # Document Preprocessing pipeline (PP-OCRv5 architecture: det + rec + doc_ori + UVDoc)
        if use_doc_orientation or use_doc_preprocessing:
            self.doc_preprocessing = DocumentPreprocessingPipeline(
                orientation_model=doc_ori_model,
                unwarping_model=doc_unwarping_model,
                use_doc_orientation=use_doc_orientation,
                use_doc_unwarping=use_doc_preprocessing
            )
        else:
            self.doc_preprocessing = None
            
        self.cls_thresh = 0.9

        self.detection_time_duration = 0
        self.classification_time_duration = 0
        self.recognition_time_duration = 0
        self.min_recognition_time_duration = 0

        # Data collection setup
        self.data_collection_enabled = False
        self.skip_recognition_for_dataset = False
        self.sample_counter = 0
        self.base_sample_dir = "sampled_dataset/"
        self.doc_preprocessing_time_duration = 0
        
        self.ocr_run_count = 0
        self.ori_run_count = 0
        self.crops_run_count = 0

        # self.enable_data_collection()

    def __call__(self, img, debug_output_path=None):
        """
        Execute PP-OCRv5 style OCR pipeline
        
        Processing order:
        1. Document Preprocessing (orientation correction + distortion correction) - same as PP-OCRv5
        2. Text Detection (text region detection)
        3. Text Classification (text orientation classification)
        4. Text Recognition (text content recognition)
        
        Args:
            img: Input image (numpy array)
            debug_output_path: Debug information save path (optional)
            
        Returns:
            tuple: (box_coordinates, cropped_images, recognition_results, preprocessed_image)
        """
        debug_data = {
            'mode': 'sync',
            'timestamp': datetime.now().isoformat(),
            'detection': {},
            'classification': {},
            'recognition': {}
        }
        
        processed_img = img
        
        # 1. Document Preprocessing (same logic as PP-OCRv5)
        if self.doc_preprocessing is not None:
            processed_img, preprocessing_latency = self.doc_preprocessing(img)
            self.doc_preprocessing_time_duration += preprocessing_latency * 1000
            
        # 2. Text Detection (performed on preprocessed image)
        det_start_time = time.time()
        det_outputs, _ = self.detection_node(processed_img)
        boxes = self.sorted_boxes(det_outputs)
        self.ocr_run_count += 1
        # Convert boxes to (n, 4, 2) format if needed
        boxes = self.convert_boxes_to_quad_format(boxes)
        
        # Save detection results
        debug_data['detection']['count'] = len(boxes)
        debug_data['detection']['boxes'] = [box.tolist() for box in boxes]
        
        crops = [self.rotate_if_vertical(self.get_rotate_crop_image(processed_img, box)) for box in boxes]
        
        # Save crops for debugging if debug_output_path specified
        if debug_output_path:
            crop_dir = os.path.join(os.path.dirname(debug_output_path), 'sync_crops')
            os.makedirs(crop_dir, exist_ok=True)
            for i, crop in enumerate(crops):
                crop_path = os.path.join(crop_dir, f'crop_{i:03d}.jpg')
                cv2.imwrite(crop_path, crop)
        
        # Data collection: Save the cropped images after detection
        if self.data_collection_enabled:
            for crop in crops:
                h, w = crop.shape[:2]
                self.save_sample_image(crop, w, h)
        
        self.detection_time_duration += (time.time() - det_start_time) * 1000
        # If we're collecting dataset and skipping recognition, return early
        if self.data_collection_enabled and self.skip_recognition_for_dataset:
            return [box.tolist() for box in boxes], crops, [], processed_img
        
        boxes = [box.tolist() for box in boxes]
        
        # 3. Text Classification (text orientation classification)
        start_time = time.time()
        cls_results, _ = self.classification_node(crops) # Infer whether cropped image should be rotated using classification model
        self.classification_time_duration += (time.time() - start_time) * 1000
        self.ori_run_count += len(crops)
        
        # Save classification results
        debug_data['classification']['results'] = []
        for i, [label, score] in enumerate(cls_results):
            rotated = "180" in label and score > self.cls_thresh
            debug_data['classification']['results'].append({
                'crop_index': int(i),
                'label': str(label),
                'score': float(score),
                'rotated': bool(rotated)
            })
            if rotated:
                crops[i] = cv2.rotate(crops[i], cv2.ROTATE_180)
        
        # 4. Text Recognition (text content recognition)
        start_time = time.time()
        rec_results, _, min_latency = self.recognition_node(processed_img, boxes, crops)
        self.recognition_time_duration += (time.time() - start_time) * 1000
        self.min_recognition_time_duration = min_latency * 1000
        self.crops_run_count += len(crops)
        
        # Save recognition results
        debug_data['recognition']['count'] = len(rec_results)
        debug_data['recognition']['results'] = []
        for result in rec_results:
            debug_data['recognition']['results'].append({
                'bbox_index': result['bbox_index'],
                'text': result['text'],
                'score': float(result['score'])
            })
        
        # Write debug data to file if path is provided
        if debug_output_path:
            os.makedirs(os.path.dirname(debug_output_path) if os.path.dirname(debug_output_path) else '.', exist_ok=True)
            with open(debug_output_path, 'w', encoding='utf-8') as f:
                json.dump(debug_data, f, indent=2, ensure_ascii=False)
        
        return boxes, crops, rec_results, processed_img

    def rotate_if_vertical(self, crop):
        h, w = crop.shape[:2]
        if h > w * 2:
            return cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return crop

    def classify_ratio(self, width, height):
        """Classify image by W/H ratio into buckets: ≤3, ≤5, ≤15, >15"""
        ratio = width / height if height > 0 else float('inf')
        
        if ratio <= 4:
            return "ratio_3"
        elif ratio <= 10:
            return "ratio_5"
        elif ratio <= 12.5:
            return "ratio_10"
        elif ratio <= 25:
            return "ratio_15"
        else:
            return "ratio_25"
    
    def classify_height(self, height):
        """Classify image by height: ≤20 or >20"""
        if height <= 20:
            return "height_20"
        else:
            return "height_40"
    

    # Data collection methods
    def enable_data_collection(self, skip_recognition=True):
        """Enable data collection for cropped images after detection
        
        Args:
            skip_recognition (bool): If True, skip recognition inference to speed up data collection
        """
        self.data_collection_enabled = True
        self.skip_recognition_for_dataset = skip_recognition
        os.makedirs(self.base_sample_dir, exist_ok=True)
        print(f"Data collection enabled. Samples will be saved to: {self.base_sample_dir}")
        if skip_recognition:
            print("Recognition inference will be skipped for faster data collection.")


    @staticmethod
    def convert_boxes_to_quad_format(boxes):
        """
        Convert boxes from (n*4, 2) format to (n, 4, 2) format
        @param boxes: numpy array of shape (n*4, 2) or list of boxes where n is number of text boxes
        @return: numpy array of shape (n, 4, 2) where each box has 4 corner points
        """
        # Convert list to numpy array if needed
        if isinstance(boxes, list):
            if len(boxes) == 0:
                return np.array([])
            # Check if each element is already a box (4 points)
            if len(boxes[0]) == 4:
                # Already in correct format, just convert to numpy
                return np.array(boxes)
            else:
                # Flatten the list and convert to numpy
                boxes = np.array(boxes)
        
        if len(boxes.shape) == 2 and boxes.shape[1] == 2:
            # Check if the number of points is divisible by 4
            if boxes.shape[0] % 4 != 0:
                raise ValueError(f"Number of points ({boxes.shape[0]}) must be divisible by 4")
            
            # Reshape to (n, 4, 2) where n = boxes.shape[0] // 4
            num_boxes = boxes.shape[0] // 4
            boxes_reshaped = boxes.reshape(num_boxes, 4, 2)
            return boxes_reshaped
        elif len(boxes.shape) == 3 and boxes.shape[1] == 4 and boxes.shape[2] == 2:
            # Already in correct format
            return boxes
        else:
            raise ValueError(f"Unexpected box format: {boxes.shape}")
    
    @staticmethod
    def sorted_boxes(dt_boxes: np.ndarray) -> List[np.ndarray]:
        """
        Sort text boxes from top-to-bottom, left-to-right order
        
        Args:
            dt_boxes: Detected text boxes
            
        Returns:
            List of sorted text boxes
        """
        _boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
        for i in range(len(_boxes)-1):
            for j in range(i, -1, -1):
                    if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and (
                        _boxes[j + 1][0][0] < _boxes[j][0][0]
                    ):
                        tmp = _boxes[j]
                        _boxes[j] = _boxes[j + 1]
                        _boxes[j + 1] = tmp
                    else:
                        break
        return _boxes
    
    @staticmethod
    def get_rotate_crop_image(img, points):
        return get_rotate_crop_image(img, points)


# ==================== Async Pipeline Implementation ====================

class OCRJob:
    """Tracks the processing state of a single image through the OCR pipeline"""
    def __init__(self, job_id: str, image: np.ndarray, preprocessed_image: Optional[np.ndarray] = None):
        self.job_id = job_id
        self.image = image  # Original image
        self.preprocessed_image = preprocessed_image if preprocessed_image is not None else image  # Doc-preprocessed image
        self.state = 'created'
        self.timestamps: Dict[str, float] = {}
        
        # Stage results
        self.det_boxes: Optional[List] = None
        self.crops: Optional[List[np.ndarray]] = None
        self.cls_results: Dict[int, Tuple] = {}  # crop_idx -> (label, score)
        self.rec_results: List[Dict] = []  # List of {'bbox_index', 'bbox', 'text', 'score'}
        self.rec_results_dict: Dict[int, Dict] = {} 
        
        # Completion tracking
        self.cls_completed_count: int = 0
        self.rec_completed_count: int = 0
        self.expected_cls_count: int = 0
        self.expected_rec_count: int = 0
        
        self.error: Optional[str] = None
        
        # Debug data tracking
        self.debug_data = {
            'mode': 'async',
            'job_id': job_id,
            'timestamp': datetime.now().isoformat(),
            'detection': {},
            'classification': {'order': [], 'results': []},
            'recognition': {'order': [], 'results': []},
            'callback_order': []
        }


class AsyncPipelineOCR:
    """Asynchronous OCR pipeline using callbacks"""
    def __init__(self, 
                 det_models: dict,
                 cls_model: IE, 
                 rec_models: dict,
                 doc_ori_model: Optional[IE] = None,
                 doc_unwarping_model: Optional[IE] = None,
                 use_doc_preprocessing: bool = False,
                 use_doc_orientation: bool = False):
        """
        Initialize async OCR pipeline
        
        Args:
            det_models: Dict of detection models (resolution -> model)
            cls_model: Classification model
            rec_models: Dict of recognition models (ratio -> model)
            doc_ori_model: Document orientation model
            doc_unwarping_model: Document unwarping model
            use_doc_preprocessing: Enable document preprocessing
            use_doc_orientation: Enable document orientation correction
        """
        # Store models
        self.det_models = det_models
        self.cls_model = cls_model
        self.rec_models = rec_models
        
        # Create nodes
        self.detection_node = DetectionNode(self.det_models)
        self.classification_node = ClassificationNode(self.cls_model)
        self.recognition_node = RecognitionNode(self.rec_models)
        
        self.ocr_run_count = 0
        self.ori_run_count = 0
        self.crops_run_count = 0
        
        self.detection_time_duration = 0
        self.classification_time_duration = 0
        self.recognition_time_duration = 0
        self.min_recognition_time_duration = 0
        
        # Document preprocessing (runs synchronously before async pipeline)
        if use_doc_orientation or use_doc_preprocessing:
            self.doc_preprocessing = DocumentPreprocessingPipeline(
                orientation_model=doc_ori_model,
                unwarping_model=doc_unwarping_model,
                use_doc_orientation=use_doc_orientation,
                use_doc_unwarping=use_doc_preprocessing
            )
        else:
            self.doc_preprocessing = None
        
        # Debug mode
        self.debug = False
        
        # Async management
        self.lock = threading.Lock()
        self.active_jobs: Dict[str, OCRJob] = {}
        self.completed_queue = queue.Queue()
        
        # Configuration
        self.cls_thresh = 0.9
        
        # Statistics
        self.stats = {
            'det_submitted': 0, 'det_completed': 0,
            'cls_submitted': 0, 'cls_completed': 0,
            'rec_submitted': 0, 'rec_completed': 0
        }
        
        # Register callbacks
        self._register_callbacks()
    
    def _register_callbacks(self):
        """Register callbacks for all models"""
        # Detection callbacks (multiple resolution models)
        for res, model in self.det_models.items():
            model.register_callback(self._on_detection_complete)
        
        # Classification callback
        self.cls_model.register_callback(self._on_classification_complete)
        
        # Recognition callbacks (multiple ratio models)
        for ratio_key, model in self.rec_models.items():
            model.register_callback(self._on_recognition_complete)
    
    def process_batch(self, images: List[np.ndarray], timeout: float = 60.0, debug_output_dir: Optional[str] = None) -> List[Dict]:
        """
        Process a batch of images asynchronously
        
        Args:
            images: List of input images (numpy arrays)
            timeout: Timeout in seconds
            debug_output_dir: Optional directory to save debug JSON files
            
        Returns:
            List of results in original order
        """
        if not images:
            return []
        
        print(f"[AsyncPipeline] Starting batch processing: {len(images)} images")
        
        # 1. Document preprocessing (synchronous, before async pipeline)
        preprocessed_images = []
        doc_preprocessing_time = 0
        for i, img in enumerate(images):
            if self.doc_preprocessing is not None:
                processed_img, latency = self.doc_preprocessing(img)
                doc_preprocessing_time += latency * 1000
                preprocessed_images.append(processed_img)
            else:
                preprocessed_images.append(img)
        
        if self.doc_preprocessing is not None:
            print(f"[AsyncPipeline] Document preprocessing completed: {doc_preprocessing_time:.2f}ms")
        
        # 2. Create jobs
        jobs = []
        for i, (orig_img, prep_img) in enumerate(zip(images, preprocessed_images)):
            job = OCRJob(f'job_{i}', orig_img, prep_img)
            job.timestamps['start'] = time.time()
            if debug_output_dir:
                job.debug_output_dir = debug_output_dir
            jobs.append(job)
            self.active_jobs[job.job_id] = job
        
        # 3. Submit detection for all jobs
        self.job_detection_start_time = time.time()
        self.job_classification_start_time = time.time()
        self.job_recognition_start_time = time.time()
        for job in jobs:
            self._submit_detection(job)
        
        # 4. Wait for all jobs to complete
        completed_jobs = {}
        start_time = time.time()
        
        while len(completed_jobs) < len(jobs):
            elapsed = time.time() - start_time
            if elapsed > timeout:
                print(f"[AsyncPipeline] Timeout after {elapsed:.1f}s, {len(completed_jobs)}/{len(jobs)} completed")
                break
            
            try:
                job = self.completed_queue.get(timeout=0.1)
                completed_jobs[job.job_id] = job
                print(f"[AsyncPipeline] Job completed: {job.job_id} ({len(completed_jobs)}/{len(jobs)})")
            except queue.Empty:
                continue
        
        # 5. Restore original order
        completed_in_order = []
        for original_job in jobs:
            if original_job.job_id in completed_jobs:
                completed_in_order.append(completed_jobs[original_job.job_id])
            else:
                # Job timed out or error
                print(f"[AsyncPipeline] Warning: Job {original_job.job_id} did not complete")
                completed_in_order.append(original_job)
        
        # 6. Save debug data if requested
        if debug_output_dir:
            os.makedirs(debug_output_dir, exist_ok=True)
            for job in completed_in_order:
                debug_file = os.path.join(debug_output_dir, f'{job.job_id}_async.json')
                with open(debug_file, 'w', encoding='utf-8') as f:
                    json.dump(job.debug_data, f, indent=2, ensure_ascii=False)
                print(f"[AsyncPipeline] Saved debug data: {debug_file}")
        
        # 7. Cleanup
        self.active_jobs.clear()
        
        # 8. Update Inference Time Durations
        self.detection_time_duration += (self.job_detection_end_time - self.job_detection_start_time) * 1000 
        self.classification_time_duration += (self.job_classification_end_time - self.job_classification_start_time) * 1000
        self.recognition_time_duration += (self.job_recognition_end_time - self.job_recognition_start_time) * 1000
        
        # 9. Format results
        return self._format_results(completed_in_order)
    
    def _submit_detection(self, job: OCRJob):
        """Submit detection for a job"""
        job.timestamps['det_start'] = time.time()
        job.state = 'detecting'
        
        # Use preprocessed image for detection
        img = job.preprocessed_image
        h, w = img.shape[:2]
        
        # Route to appropriate detection model
        mapped_res = self.detection_node.router(w, h)
        res_preprocess = self.detection_node.det_preprocess_map[mapped_res]
        res_model = self.det_models[mapped_res]
        
        # Preprocess
        model_input = res_preprocess(img)
        padding_info = res_preprocess.get_last_padding_info()
        model_input = self.detection_node.prepare_input(model_input)
        
        # Submit async (record inference start time)
        job.timestamps['det_inference_start'] = time.time()
        inference_time = time.time()
        res_model.run_async([model_input], user_arg=(job.job_id, mapped_res, padding_info, inference_time))
        
        with self.lock:
            self.stats['det_submitted'] += 1
    
    def _on_detection_complete(self, outputs: List[np.ndarray], user_arg: Any) -> int:
        """Callback when detection completes"""
        job_id, _, padding_info, _ = user_arg
        
        try:
            with self.lock:
                job = self.active_jobs.get(job_id)
                if not job:
                    return 0
                
                # Calculate inference time and accumulate
                self.job_detection_end_time = time.time()
                self.stats['det_completed'] += 1
                job.debug_data['callback_order'].append('det_complete')
            
            # Postprocess (outside lock)
            job.timestamps['det_end'] = time.time()
            det_output = self.detection_node.postprocess(outputs[0], job.preprocessed_image.shape, padding_info)
            job.det_boxes = self.sorted_boxes(det_output)
            
            # Convert boxes and create crops
            boxes_quad = self.convert_boxes_to_quad_format(job.det_boxes)
            job.crops = [self.rotate_if_vertical(self.get_rotate_crop_image(job.preprocessed_image, box)) 
                        for box in boxes_quad]
            job.det_boxes = [box.tolist() for box in job.det_boxes]
            
            # Record detection results
            job.debug_data['detection']['count'] = len(job.det_boxes)
            job.debug_data['detection']['boxes'] = job.det_boxes
            
            # Save crops for debugging (if job has debug_output_dir set)
            if hasattr(job, 'debug_output_dir') and job.debug_output_dir:
                crop_dir = os.path.join(job.debug_output_dir, 'async_crops')
                os.makedirs(crop_dir, exist_ok=True)
                for i, crop in enumerate(job.crops):
                    crop_path = os.path.join(crop_dir, f'crop_{i:03d}.jpg')
                    cv2.imwrite(crop_path, crop)
            
            print(f"[AsyncPipeline] {job_id}: Detection complete, {len(job.det_boxes)} boxes found")
            self.ocr_run_count += 1
            
            # Submit classification
            self._submit_classification(job)
            
        except Exception as e:
            print(f"[AsyncPipeline] Error in detection callback for {job_id}: {e}")
            import traceback
            traceback.print_exc()
            job.error = str(e)
            job.state = 'error'
            self.completed_queue.put(job)
        
        return 0
    
    def _submit_classification(self, job: OCRJob):
        """Submit classification for all crops"""
        if not job.crops:
            # No crops, skip to completion
            job.state = 'completed'
            job.timestamps['end'] = time.time()
            self.completed_queue.put(job)
            return
        
        job.timestamps['cls_start'] = time.time()
        job.state = 'classifying'
        job.expected_cls_count = len(job.crops)
        if len(job.crops) == 0:
            self.job_classification_end_time = time.time()
        
        # Submit each crop
        for crop_idx, crop in enumerate(job.crops):
            # Preprocess (CRITICAL: Must match sync version!)
            # Use lock because cls_preprocess may not be thread-safe
            with self.lock:
                # 1. Apply cls_preprocess first
                cls_preprocessed = self.classification_node.cls_preprocess(crop)
                # 2. Then prepare_input for model format
                cls_input = self.classification_node.prepare_input(cls_preprocessed)
                self.stats['cls_submitted'] += 1
            
            inference_start_time = time.time()
            # Record inference start time
            # Submit async (outside lock)
            self.ori_run_count += 1
            self.cls_model.run_async([cls_input], user_arg=(job.job_id, crop_idx, 'cls', inference_start_time))
    
    def _on_classification_complete(self, outputs: List[np.ndarray], user_arg: Any) -> int:
        """Callback when classification completes"""
        job_id, crop_idx, _, inference_start_time = user_arg
        all_done = False
        try:
            # Calculate inference time
            self.job_classification_end_time = time.time()
            
            # Postprocess (outside lock)
            label, score = self.classification_node.cls_postprocess(outputs[0])[0]
            
            # Update job state
            with self.lock:
                job = self.active_jobs.get(job_id)
                if not job:
                    return 0
                
                job.cls_results[crop_idx] = (label, score)
                job.cls_completed_count += 1
                all_done = (job.cls_completed_count >= job.expected_cls_count)
                
                # Record callback order and result
                job.debug_data['callback_order'].append(f'cls_complete_{crop_idx}')
                job.debug_data['classification']['order'].append(int(crop_idx))
                rotated = "180" in label and score > self.cls_thresh
                job.debug_data['classification']['results'].append({
                    'crop_index': int(crop_idx),
                    'label': str(label),
                    'score': float(score),
                    'rotated': bool(rotated)
                })
                
                # CRITICAL: Mark job as 'rotating' to prevent multiple threads from entering rotation block
                if all_done and job.state == 'classifying':
                    job.state = 'rotating'  # Prevent race condition
                else:
                    all_done = False  # Another thread is already handling rotation
                
                self.stats['cls_completed'] += 1
            
            # If all classification done, apply rotation and submit recognition
            if all_done:
                job.timestamps['cls_end'] = time.time()
                print(f"[AsyncPipeline] {job_id}: Classification complete, {job.cls_completed_count} crops")
                
                # Debug: print all classification results
                if hasattr(self, 'debug') and self.debug:
                    print(f"[DEBUG] Classification results for {job_id}: {len(job.cls_results)}/{job.expected_cls_count} completed")
                    for idx in sorted(job.cls_results.keys())[:20]:  # First 20
                        label, score = job.cls_results[idx]
                        print(f"  crop_{idx}: label='{label}', score={score:.3f}, will_rotate={'180' in label and score > self.cls_thresh}")
                
                # Apply rotation (CRITICAL: must be done BEFORE recognition submit)
                # State is already 'rotating', so only one thread will execute this
                rotated_indices = []
                for idx in range(len(job.crops)):
                    if idx in job.cls_results:
                        label, score = job.cls_results[idx]
                        if "180" in label and score > self.cls_thresh:
                            # Rotate and make contiguous copy for NPU
                            rotated = cv2.rotate(job.crops[idx], cv2.ROTATE_180)
                            job.crops[idx] = np.ascontiguousarray(rotated, dtype=np.uint8)
                            rotated_indices.append(idx)
                            if hasattr(self, 'debug') and self.debug:
                                print(f"[AsyncPipeline] {job_id}: Rotated crop_{idx}")
                
                # Submit recognition (now with properly rotated crops)
                self._submit_recognition(job)
        
        except Exception as e:
            print(f"[AsyncPipeline] Error in classification callback for {job_id}: {e}")
            import traceback
            traceback.print_exc()
            with self.lock:
                job = self.active_jobs.get(job_id)
                if job:
                    job.error = str(e)
                    job.state = 'error'
                    self.completed_queue.put(job)
        
        return 0
    
    def _submit_recognition(self, job: OCRJob):
        """Submit recognition for all crops (after all rotations are complete)"""
        if not job.crops:
            job.state = 'completed'
            job.timestamps['end'] = time.time()
            self.completed_queue.put(job)
            return
        
        job.timestamps['rec_start'] = time.time()
        job.state = 'recognizing'
        job.expected_rec_count = len(job.crops)
        
        job.rec_results_dict = {}
        
        # Submit each crop
        for crop_idx in range(len(job.crops)):
            crop = job.crops[crop_idx]
            try:
                # Debug logging
                if hasattr(self, 'debug') and self.debug:
                    # Check if this crop was rotated
                    was_rotated = (crop_idx in job.cls_results and 
                                  "180" in job.cls_results[crop_idx][0] and 
                                  job.cls_results[crop_idx][1] > self.cls_thresh)
                    print(f"[DEBUG] Submitting rec for crop_{crop_idx}, shape={crop.shape}, rotated={was_rotated}")
                
                if not crop.flags['C_CONTIGUOUS']:
                    crop = np.ascontiguousarray(crop, dtype=crop.dtype)
                
                # Route to appropriate recognition model
                mapped_ratio = self.recognition_node.router(crop.shape[1], crop.shape[0])
                ratio_preprocess = self.recognition_node.rec_preprocess_map[mapped_ratio]
                ratio_model = self.rec_models[mapped_ratio]
                
                # Preprocess
                inp = ratio_preprocess(crop)
                rec_input = self.recognition_node.prepare_input(inp)
                
                # Record inference start time
                inference_start_time = time.time()
                # Submit async
                ratio_model.run_async([rec_input], user_arg=(job.job_id, crop_idx, 'rec', mapped_ratio, inference_start_time))
                self.crops_run_count += 1
                with self.lock:
                    self.stats['rec_submitted'] += 1
            except Exception as e:
                print(f"[AsyncPipeline] ERROR submitting rec for {job.job_id} crop_{crop_idx}: {e}")
                import traceback
                traceback.print_exc()
    
    def _on_recognition_complete(self, outputs: List[np.ndarray], user_arg: Any) -> int:
        """Callback when recognition completes"""
        job_id, crop_idx, _, _, inference_start_time = user_arg
        self.job_recognition_end_time = time.time()
        
        try:
            # Calculate inference time
            inference_time = time.time() - inference_start_time
            # Postprocess (outside lock)
            res = self.recognition_node.rec_postprocess(outputs[0])[0]  # (text, score)
            text, score = res
            
            # Debug logging
            if hasattr(self, 'debug') and self.debug:
                print(f"[DEBUG] Rec complete: {job_id}, crop_{crop_idx}, score={score:.3f}, text='{text[:20]}...'")
            
            # Update job state
            with self.lock:
                job = self.active_jobs.get(job_id)
                if not job:
                    if hasattr(self, 'debug') and self.debug:
                        print(f"[DEBUG] Job {job_id} not found in active_jobs!")
                    return 0
                
                # Update minimum recognition time
                min_time_ms = inference_time * 1000
                if self.min_recognition_time_duration == 0 or min_time_ms < self.min_recognition_time_duration:
                    self.min_recognition_time_duration = min_time_ms
                
                # Record callback order
                job.debug_data['callback_order'].append(f'rec_complete_{crop_idx}')
                job.debug_data['recognition']['order'].append(crop_idx)
                
                # Apply drop threshold and store in dictionary (order-independent)
                if score > self.recognition_node.drop:
                    # Store result in dict format indexed by crop_idx
                    result = {
                        'bbox_index': crop_idx,
                        'bbox': job.det_boxes[crop_idx] if crop_idx < len(job.det_boxes) else None,
                        'text': text,
                        'score': score
                    }
                    job.rec_results_dict[crop_idx] = result
                    
                    # Record recognition result
                    job.debug_data['recognition']['results'].append({
                        'bbox_index': crop_idx,
                        'text': text,
                        'score': float(score)
                    })
                
                job.rec_completed_count += 1
                all_done = (job.rec_completed_count >= job.expected_rec_count)
                
                self.stats['rec_completed'] += 1
            
            # If all recognition done, mark job as complete
            if all_done:
                job.timestamps['rec_end'] = time.time()
                job.timestamps['end'] = time.time()
                job.state = 'completed'
                
                job.rec_results = [
                    job.rec_results_dict[idx] 
                    for idx in sorted(job.rec_results_dict.keys())
                ]
                
                print(f"[AsyncPipeline] {job_id}: Recognition complete, {len(job.rec_results)} texts recognized")
                
                self.completed_queue.put(job)
        
        except Exception as e:
            print(f"[AsyncPipeline] Error in recognition callback for {job_id}: {e}")
            import traceback
            traceback.print_exc()
            job.error = str(e)
            job.state = 'error'
            self.completed_queue.put(job)
        
        return 0
    
    def _format_results(self, completed_jobs: List[OCRJob]) -> List[Dict]:
        """Format job results to match sync version output"""
        results = []
        
        for job in completed_jobs:
            if job.error:
                print(f"[AsyncPipeline] Job {job.job_id} had error: {job.error}")
            
            # Calculate timings
            pipeline_total = 0
            if 'start' in job.timestamps and 'end' in job.timestamps:
                pipeline_total = (job.timestamps['end'] - job.timestamps['start']) * 1000
            
            det_time = 0
            if 'det_start' in job.timestamps and 'det_end' in job.timestamps:
                det_time = (job.timestamps['det_end'] - job.timestamps['det_start']) * 1000
            
            cls_time = 0
            if 'cls_start' in job.timestamps and 'cls_end' in job.timestamps:
                cls_time = (job.timestamps['cls_end'] - job.timestamps['cls_start']) * 1000
            
            rec_time = 0
            if 'rec_start' in job.timestamps and 'rec_end' in job.timestamps:
                rec_time = (job.timestamps['rec_end'] - job.timestamps['rec_start']) * 1000
            
            result = {
                'job_id': job.job_id,
                'boxes': job.det_boxes or [],
                'rec_results': job.rec_results,  # Already in dict format
                'preprocessed_image': job.preprocessed_image,
                'total_latency_ms': pipeline_total,
                'det_latency_ms': det_time,
                'cls_latency_ms': cls_time,
                'rec_latency_ms': rec_time,
                'state': job.state,
                'error': job.error
            }
            results.append(result)
        
        return results
    
    @staticmethod
    def convert_boxes_to_quad_format(boxes):
        """Convert boxes from (n*4, 2) format to (n, 4, 2) format"""
        if isinstance(boxes, list):
            if len(boxes) == 0:
                return np.array([])
            if len(boxes[0]) == 4:
                return np.array(boxes)
            else:
                boxes = np.array(boxes)
        
        if len(boxes.shape) == 2 and boxes.shape[1] == 2:
            if boxes.shape[0] % 4 != 0:
                raise ValueError(f"Number of points ({boxes.shape[0]}) must be divisible by 4")
            num_boxes = boxes.shape[0] // 4
            boxes_reshaped = boxes.reshape(num_boxes, 4, 2)
            return boxes_reshaped
        elif len(boxes.shape) == 3 and boxes.shape[1] == 4 and boxes.shape[2] == 2:
            return boxes
        else:
            raise ValueError(f"Unexpected box format: {boxes.shape}")
    
    @staticmethod
    def sorted_boxes(dt_boxes: np.ndarray) -> List[np.ndarray]:
        """Sort text boxes from top-to-bottom, left-to-right order"""
        _boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
        for i in range(len(_boxes)-1):
            for j in range(i, -1, -1):
                if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                    tmp = _boxes[j]
                    _boxes[j] = _boxes[j + 1]
                    _boxes[j + 1] = tmp
                else:
                    break
        return _boxes
    
    @staticmethod
    def rotate_if_vertical(crop):
        """Rotate crop if it's vertical (h > w*2)"""
        h, w = crop.shape[:2]
        if h > w * 2:
            return cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return crop
    
    @staticmethod
    def get_rotate_crop_image(img, points):
        """Get rotated crop from image using points"""
        return get_rotate_crop_image(img, points)
    
    def get_timing_stats(self):
        """Get timing statistics for async pipeline"""
        return {
            'detection_time_ms': self.detection_time_duration,
            'classification_time_ms': self.classification_time_duration,
            'recognition_time_ms': self.recognition_time_duration,
            'min_recognition_time_ms': self.min_recognition_time_duration,
            'total_ocr_runs': self.ocr_run_count,
            'avg_detection_per_run_ms': self.detection_time_duration / max(1, self.ocr_run_count),
            'avg_classification_per_inference_ms': self.classification_time_duration / max(1, self.stats['cls_completed']),
            'avg_recognition_per_inference_ms': self.recognition_time_duration / max(1, self.stats['rec_completed'])
        }
    
    def print_timing_stats(self):
        """Print timing statistics for async pipeline"""
        stats = self.get_timing_stats()
        print("\n=== ASYNC Pipeline Timing Statistics ===")
        print(f"Total OCR runs: {stats['total_ocr_runs']}")
        print(f"Detection total time: {stats['detection_time_ms']:.2f} ms")
        print(f"Classification total time: {stats['classification_time_ms']:.2f} ms")
        print(f"Recognition total time: {stats['recognition_time_ms']:.2f} ms")
        print(f"Min recognition time: {stats['min_recognition_time_ms']:.2f} ms")
        print(f"Avg detection per run: {stats['avg_detection_per_run_ms']:.2f} ms")
        print(f"Avg classification per inference: {stats['avg_classification_per_inference_ms']:.2f} ms")
        print(f"Avg recognition per inference: {stats['avg_recognition_per_inference_ms']:.2f} ms")
        print(f"Stats: {self.stats}")
        print("=========================================\n")

"""
=== PP-OCRv5 DXNN OCR Engine Main Functions and Features Summary ===

Main Classes:
1. Node: OCR pipeline base node class
2. DetectionNode: Text region detection (multi-resolution support: 640x640, 960x960)
3. ClassificationNode: Text orientation classification (180-degree rotation detection)
4. RecognitionNode: Text content recognition (multi-ratio models: ratio_3~35)
5. DocumentOrientationNode: Document orientation correction (0°, 90°, 180°, 270°)
6. DocumentUnwarpingNode: Document distortion correction (UVDoc algorithm)
7. DocumentPreprocessingPipeline: Document preprocessing pipeline
8. PaddleOcr: Main OCR engine (synchronous processing)
9. AsyncPipelineOCR: Asynchronous OCR pipeline (callback-based)

Processing Pipeline:
1. Document Preprocessing (optional)
   - Document Orientation: Document rotation angle correction
   - Document Unwarping: Document distortion correction (UVDoc)
2. Text Detection: DBNet-based text region detection
3. Text Classification: Text image rotation detection
4. Text Recognition: CRNN-based text content recognition

Model Selection Logic:
- Detection: Select 640x640 or 960x960 model based on image size
- Recognition: Automatically select appropriate model based on text aspect ratio
  * ratio_3: ratio ≤ 4
  * ratio_5: 4 < ratio ≤ 10  
  * ratio_10: 10 < ratio ≤ 12.5
  * ratio_15: 12.5 < ratio ≤ 25
  * ratio_25: ratio > 25

Key Features:
- Synchronous/asynchronous processing mode support
- Debug image saving functionality
- Performance statistics and timing analysis
- Batch processing support (asynchronous)
- JSON format debug data output

Usage Examples:
# Synchronous processing
ocr = PaddleOcr(det_models, cls_model, rec_models, doc_ori_model, uvdoc_model)
boxes, crops, results, processed_img = ocr(image)

# Asynchronous batch processing  
async_ocr = AsyncPipelineOCR(det_models, cls_model, rec_models, doc_ori_model, uvdoc_model)
results = async_ocr.process_batch(images, timeout=60.0)
"""
