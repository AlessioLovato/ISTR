#!/usr/bin/env python3
"""
Unified Model Comparison Pipeline

This comprehensive script:
1. Runs inference on YOLO, Mask R-CNN, and ISTR models
2. Generates COCO-format predictions
3. Computes COCO metrics (AP/AR) using pycocotools
4. Computes YOLO-style metrics (Precision/Recall/F1)
5. Computes additional statistics (bbox stats, IoU distributions)
6. Generates comprehensive Markdown and CSV reports
7. Saves annotated images for visualization
8. Provides interactive gallery view

Usage:
    # Full pipeline with inference
    python compare_models_unified.py
    
    # Force rerun inference
    python compare_models_unified.py --force-inference
    
    # Load cached results and show gallery
    python compare_models_unified.py --load-only --show
    
    # Run inference and show gallery
    python compare_models_unified.py --show
"""

import os
import sys
import time
import csv
import json
import argparse
import seaborn as sns
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import subprocess
import shutil

# COCO tools
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from pycocotools.cocoeval import COCOeval

# Detectron2 imports
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.detection_utils import read_image
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator
from detectron2.evaluation.coco_evaluation import instances_to_coco_json

# ISTR imports
try:
    from istr import add_ISTR_config
except ImportError:
    add_ISTR_config = None

# Ultralytics YOLO
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Dataset definitions
    "datasets": {
        "yolo_normal": {
            "path": "../shared/YOLODataset_big_images_rev",
            "format": "yolo",  # 'yolo' or 'coco'
            "description": "Big Images Rev (YOLO format)"
        },
        "yolo_contrast": {
            "path": "../shared/YOLODataset_big_images_rev_contrast",
            "format": "yolo",
            "description": "Big Images Rev Contrast (YOLO format)"
        },
        "big-images-rev": {
            "path": "../shared/coco-big-images-rev",
            "format": "coco",
            "description": "Big Images Rev (COCO format)"
        },
        "big-images-rev-contrast": {
            "path": "../shared/coco-big-images-rev-contrast",
            "format": "coco",
            "description": "Big Images Rev Contrast (COCO format)"
        },
    },
    
    # Model definitions
    "models": {
        "yolo_normal": {
            "name": "YOLOv11n-seg",
            "run_name": "BIR",
            "type": "yolo",  # 'yolo' or 'detectron2'
            "weights": "../shared/n_bir.pt",
            "confidence": 0.5,
            "epoch": "final",
            "evaluate_on": ["big-images-rev", "big-images-rev-contrast"]  # YOLO works with COCO format too for evaluation
        },
        "yolo_contrast": {
            "name": "YOLOv11n-seg",
            "run_name": "BIRC",
            "type": "yolo",
            "weights": "../shared/n_birc.pt",
            "confidence": 0.5,
            "epoch": "final",
            "evaluate_on": ["big-images-rev", "big-images-rev-contrast"]
        },
        "maskrcnn_normal": {
            "name": "Mask R-CNN R50-FPN",
            "run_name": "maskrcnn_R_50_b16_big_images_rev_short",
            "type": "detectron2",
            "config": "detectron2/detectron2/model_zoo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
            "weights": "../shared/models/output_maskrcnn_R_50_b16_big_images_rev_short/model_final.pth",
            "confidence": 0.5,
            "epoch": "final",
            "evaluate_on": ["big-images-rev", "big-images-rev-contrast"]  # only COCO datasets
        },
        "maskrcnn_contrast": {
            "name": "Mask R-CNN R50-FPN",
            "run_name": "maskrcnn_R_50_b16_aug_big_images_rev_contrast",  # Custom run description
            "type": "detectron2",
            "config": "detectron2/detectron2/model_zoo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
            "weights": "../shared/models/output_maskrcnn_R_50_b16_aug_big_images_rev_contrast/model_final.pth",
            "confidence": 0.5,
            "epoch": "final",
            "evaluate_on": ["big-images-rev", "big-images-rev-contrast"]
        },
        "istr_normal": {
            "name": "ISTR-PCA-R50",
            "run_name": "pca_50_big-images-rev",
            "type": "detectron2",
            "config": "detectron2/detectron2/model_zoo/configs/ISTR/ISTR-PCA-R50-3x.yaml",
            "weights": "../shared/models/output_pca_50_big-images-rev/model_0019999.pth",
            "confidence": 0.5,
            "epoch": "19999",
            "evaluate_on": ["big-images-rev", "big-images-rev-contrast"]
        },
        "istr_contrast": {
            "name": "ISTR-PCA-R50",
            "run_name": "pca_50_big-images-rev-contrast",
            "type": "detectron2",
            "config": "detectron2/detectron2/model_zoo/configs/ISTR/ISTR-PCA-R50-3x.yaml",
            "weights": "../shared/models/output_pca_50_big-images-rev-contrast/model_0019999.pth",
            "confidence": 0.5,
            "epoch": "19999",
            "evaluate_on": ["big-images-rev", "big-images-rev-contrast"]
        },
    },
    
    "output_dir": "model_comparison_results",
    "image_format": "png",
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def truncate3(x):
    """Truncate float to 3 decimal places."""
    return float(str(x)[: str(x).find('.') + 4]) if isinstance(x, float) else x


def summarize_dict(d):
    """Convert dict values to truncated floats."""
    return {k: truncate3(float(v)) for k, v in d.items()}


def iou_distribution(iou_list):
    """Compute IoU distribution statistics."""
    if not iou_list:
        return {"median": 0, "p25": 0, "p75": 0, "min": 0, "max": 0}
    arr = np.array(iou_list)
    return {
        "median": truncate3(np.median(arr)),
        "p25": truncate3(np.percentile(arr, 25)),
        "p75": truncate3(np.percentile(arr, 75)),
        "min": truncate3(arr.min()),
        "max": truncate3(arr.max())
    }


# ============================================================================
# DATASET LOADING
# ============================================================================

def read_yolo_annotations(label_path):
    """Read YOLO polygon format annotations."""
    annotations = []
    if not os.path.exists(label_path):
        return annotations
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            
            cls_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            
            points = []
            for i in range(0, len(coords), 2):
                points.append((coords[i], coords[i + 1]))
            
            annotations.append((cls_id, points))
    
    return annotations


def draw_yolo_annotations(image, annotations, class_names):
    """Draw YOLO polygon annotations on image (no labels, just polygons)."""
    result = image.copy()
    h, w = image.shape[:2]
    
    colors = {
        0: (0, 255, 0),      # positive - green
        1: (0, 0, 255),      # negative - red  
        2: (255, 0, 0),      # lines - blue
    }
    
    for cls_id, points in annotations:
        pixel_points = []
        for x, y in points:
            pixel_points.append((int(x * w), int(y * h)))
        
        if len(pixel_points) < 3:
            continue
        
        color = colors.get(cls_id, (0, 255, 0))
        
        # Draw filled polygon with transparency
        overlay = result.copy()
        pts = np.array(pixel_points, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.3, result, 0.7, 0, result)
        
        # Draw polygon outline
        cv2.polylines(result, [pts], isClosed=True, color=color, thickness=2)
    
    return result


def get_test_images(dataset_name):
    """Get list of test images from dataset."""
    dataset_info = CONFIG["datasets"][dataset_name]
    dataset_path = dataset_info["path"]
    dataset_format = dataset_info["format"]
    
    if dataset_format == "coco":
        test_dir = Path(dataset_path) / "test"
        
        # Find all images in directory
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_list = []
        
        for img_path in test_dir.iterdir():
            if img_path.suffix.lower() in image_extensions:
                image_list.append((str(img_path), img_path.name))
        
        return sorted(image_list)
    
    elif dataset_format == "yolo":
        test_dir = Path(dataset_path) / "images" / "test"
        labels_dir = Path(dataset_path) / "labels" / "test"
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_list = []
        
        for img_path in test_dir.iterdir():
            if img_path.suffix.lower() in image_extensions:
                label_path = labels_dir / (img_path.stem + '.txt')
                image_list.append((str(img_path), img_path.name, str(label_path)))
        
        return sorted(image_list)
    
    else:
        raise ValueError(f"Unknown dataset format: {dataset_format}")


def load_datasets():
    """Load all datasets and extract test images."""
    datasets = {}
    
    print("Loading datasets...")
    
    for dataset_name in CONFIG["datasets"]:
        datasets[dataset_name] = get_test_images(dataset_name)
        dataset_info = CONFIG["datasets"][dataset_name]
        print(f"  {dataset_name} ({dataset_info['format']}): {len(datasets[dataset_name])} images")
        if len(datasets[dataset_name]) == 0:
            raise ValueError(f"No images found in {dataset_name}")
    
    return datasets


def register_coco_datasets():
    """Register COCO datasets with detectron2."""
    print("\nRegistering COCO datasets...")
    
    for dataset_name, dataset_info in CONFIG["datasets"].items():
        if dataset_info["format"] == "coco":
            dataset_path = Path(dataset_info["path"])
            register_coco_instances(
                f"comparison_{dataset_name}",
                {"thing_classes": ["positive", "negative", "lines"]},
                str(dataset_path / "test" / "_annotations.coco.json"),
                str(dataset_path / "test")
            )
            print(f"  Registered: comparison_{dataset_name}")


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_yolo_model(model_key):
    """Load YOLO model."""
    if YOLO is None:
        raise ImportError("ultralytics package not found. Install with: pip install ultralytics")
    
    model_info = CONFIG["models"][model_key]
    model_path = model_info["weights"]
    print(f"Loading YOLO model {model_key} from {model_path}...")
    
    model = YOLO(model_path)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    return model


def load_detectron2_model(model_key):
    """Load Detectron2 model (Mask R-CNN, ISTR, or any other detectron2 model)."""
    model_info = CONFIG["models"][model_key]
    cfg = get_cfg()
    
    config_file = model_info["config"]
    
    # Auto-detect ISTR models from config path
    if "ISTR" in config_file and add_ISTR_config is not None:
        add_ISTR_config(cfg)
    
    weights = model_info["weights"]
    
    print(f"Loading {model_info['name']} from {weights}...")
    
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weights
    # Use very low threshold for evaluation - COCO eval handles filtering
    # This matches train_net.py behavior
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0001
    cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.freeze()
    
    model = DefaultTrainer.build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()
    
    # Get metadata from first registered COCO dataset
    for dataset_name, dataset_info in CONFIG["datasets"].items():
        if dataset_info["format"] == "coco":
            metadata = MetadataCatalog.get(f"comparison_{dataset_name}")
            break
    
    return model, cfg, metadata


# ============================================================================
# INFERENCE FUNCTIONS
# ============================================================================

def run_yolo_inference(model, image_path, confidence_threshold, eval_mode=False):
    """Run YOLO inference on a single image.
    
    Args:
        model: YOLO model
        image_path: Path to image
        confidence_threshold: Threshold for visualization
        eval_mode: If True, use very low threshold (0.001) for evaluation
    """
    start_time = time.time()
    
    # Use very low threshold for evaluation to match train_net.py behavior
    conf_thr = 0.001 if eval_mode else confidence_threshold
    results = model(image_path, conf=conf_thr, verbose=False)
    
    inference_time = time.time() - start_time
    
    # Get annotated image (convert BGR to RGB)
    # For visualization, filter by user's threshold
    if eval_mode:
        annotated_img_bgr = results[0].plot(conf=True)
    else:
        annotated_img_bgr = results[0].plot()
    annotated_img = cv2.cvtColor(annotated_img_bgr, cv2.COLOR_BGR2RGB)
    
    num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
    
    return annotated_img, num_detections, inference_time, results[0]


def run_detectron2_inference(model, metadata, image_path, confidence_threshold):
    """Run Detectron2 model inference on a single image.
    
    Note: confidence_threshold is only used for counting displayed detections.
    The model uses cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST set during model loading.
    """
    img = read_image(image_path, format="BGR")
    
    start_time = time.time()
    
    with torch.no_grad():
        height, width = img.shape[:2]
        image_tensor = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
        inputs = {"image": image_tensor, "height": height, "width": width}
        predictions = model([inputs])[0]
    
    inference_time = time.time() - start_time
    
    # Visualize
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.IMAGE)
    
    num_detections = 0
    if "instances" in predictions:
        instances = predictions["instances"].to("cpu")
        
        if hasattr(instances, "scores"):
            keep = instances.scores >= confidence_threshold
            instances = instances[keep]
            num_detections = len(instances)
        
        vis_output = visualizer.draw_instance_predictions(predictions=instances)
        annotated_img = vis_output.get_image()
    else:
        annotated_img = img[:, :, ::-1]
    
    return annotated_img, num_detections, inference_time, predictions


def binary_mask_to_rle(bmask):
    """Convert binary mask to COCO RLE format."""
    if bmask.dtype != np.uint8:
        bmask = bmask.astype(np.uint8)
    rle = maskUtils.encode(np.asfortranarray(bmask))
    if isinstance(rle['counts'], bytes):
        rle['counts'] = rle['counts'].decode('ascii')
    return rle


def yolo_result_to_coco(results_obj, image_id, confidence_thr=None):
    """Convert YOLO results to COCO format.
    
    Note: confidence_thr is ignored for COCO evaluation - all predictions are kept
    and COCO evaluation handles filtering internally.
    """
    out = []
    
    try:
        boxes_data = getattr(results_obj, "boxes", None)
        if boxes_data is not None and hasattr(boxes_data, "data"):
            b = boxes_data.data.cpu().numpy()
            boxes = b[:, :4]
            scores = b[:, 4]
            classes = b[:, 5].astype(int)
        else:
            return out
    except Exception:
        return out
    
    # Try to get masks
    masks = None
    try:
        masks_obj = getattr(results_obj, "masks", None)
        if masks_obj is not None:
            try:
                masks = masks_obj.data.cpu().numpy()
            except Exception:
                try:
                    masks = masks_obj.cpu().numpy()
                except Exception:
                    masks = None
    except Exception:
        pass
    
    if masks is not None and len(masks) > 0:
        for i in range(masks.shape[0]):
            conf = float(scores[i])
            # Don't filter by confidence - COCO evaluation handles this
            mask_bin = masks[i].astype(np.uint8)
            rle = binary_mask_to_rle(mask_bin)
            bbox = maskUtils.toBbox(rle).tolist()
            cat = int(classes[i])
            out.append({
                "image_id": int(image_id),
                "category_id": int(cat),
                "segmentation": rle,
                "bbox": [float(x) for x in bbox],
                "score": float(conf)
            })
    elif boxes is not None:
        for i in range(len(boxes)):
            conf = float(scores[i])
            # Don't filter by confidence - COCO evaluation handles this
            x1, y1, x2, y2 = boxes[i].tolist()
            w = x2 - x1
            h = y2 - y1
            cat = int(classes[i])
            out.append({
                "image_id": int(image_id),
                "category_id": int(cat),
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "score": float(conf)
            })
    
    return out


def detectron2_preds_to_coco(predictions, image_id, confidence_thr=None):
    """Convert detectron2 predictions to COCO format.
    
    Note: confidence_thr is ignored for COCO evaluation - all predictions are kept
    and COCO evaluation handles filtering internally.
    """
    results = []
    if "instances" not in predictions:
        return results
    
    instances = predictions["instances"].to("cpu")
    scores = instances.scores.tolist() if hasattr(instances, "scores") else [1.0] * len(instances)
    classes = instances.pred_classes.tolist() if hasattr(instances, "pred_classes") else [0] * len(instances)
    
    if hasattr(instances, "pred_masks"):
        masks = instances.pred_masks.numpy()
        for i, mask in enumerate(masks):
            # Don't filter by confidence - COCO evaluation handles this
            rle = binary_mask_to_rle(mask)
            bbox = maskUtils.toBbox(rle).tolist()
            cat_id = int(classes[i])
            results.append({
                "image_id": int(image_id),
                "category_id": cat_id,
                "segmentation": rle,
                "score": float(scores[i]),
                "bbox": [float(x) for x in bbox]
            })
    else:
        boxes = instances.pred_boxes.tensor.numpy() if hasattr(instances, "pred_boxes") else None
        if boxes is not None:
            for i, box in enumerate(boxes):
                # Don't filter by confidence - COCO evaluation handles this
                x1, y1, x2, y2 = box.tolist()
                w = x2 - x1
                h = y2 - y1
                cat_id = int(classes[i])
                results.append({
                    "image_id": int(image_id),
                    "category_id": cat_id,
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(scores[i])
                })
    
    return results


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def compute_yolo_metrics_for_pair(gt_boxes, pred_boxes, iou_threshold=0.50, use_masks=True):
    """Compute YOLO-style precision, recall, F1 for a single model/dataset pair.
    
    Args:
        gt_boxes: Ground truth annotations
        pred_boxes: Predicted annotations
        iou_threshold: IoU threshold for matching
        use_masks: If True, use mask IoU; if False, use bbox IoU
    """
    def compute_mask_iou(seg1, seg2):
        """Compute IoU between two segmentation masks (RLE format)."""
        if seg1 is None or seg2 is None:
            return 0.0
        try:
            iou = maskUtils.iou([seg1], [seg2], [0])[0][0]
            return float(iou)
        except Exception as e:
            # Log the error for debugging
            print(f"Warning: Mask IoU computation failed: {e}")
            return 0.0
    
    def compute_bbox_iou(b1, b2):
        """Compute IoU between two bounding boxes (XYWH format)."""
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[0] + b1[2], b2[0] + b2[2])
        y2 = min(b1[1] + b1[3], b2[1] + b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = b1[2] * b1[3] + b2[2] * b2[3] - inter
        return inter / union if union > 0 else 0
    
    # Group predictions and ground truth by image_id and category_id
    preds_by_img_cat = defaultdict(list)
    gts_by_img_cat = defaultdict(list)
    
    for pred in pred_boxes:
        key = (pred['image_id'], pred['category_id'])
        preds_by_img_cat[key].append(pred)
    
    for gt in gt_boxes:
        key = (gt['image_id'], gt['category_id'])
        gts_by_img_cat[key].append(gt)
    
    TP = 0
    FP = 0
    total_gt = len(gt_boxes)
    
    # Process each image-category combination
    for key in preds_by_img_cat.keys():
        preds = preds_by_img_cat[key]
        gts = gts_by_img_cat.get(key, [])
        
        if len(gts) == 0:
            # All predictions are false positives
            FP += len(preds)
            continue
        
        # Track which GTs have been matched in this image-category
        gt_matched = set()
        
        # Sort predictions by confidence (descending)
        preds_sorted = sorted(preds, key=lambda x: x.get('score', 1.0), reverse=True)
        
        for pred in preds_sorted:
            best_iou = 0
            best_gt_idx = None
            
            for gt_idx, gt in enumerate(gts):
                # Skip already matched GTs
                if gt_idx in gt_matched:
                    continue
                
                # Compute IoU
                if use_masks:
                    pred_seg = pred.get('segmentation')
                    gt_seg = gt.get('segmentation')
                    if pred_seg and gt_seg:
                        iou = compute_mask_iou(pred_seg, gt_seg)
                    else:
                        iou = compute_bbox_iou(pred['bbox'], gt['bbox'])
                else:
                    iou = compute_bbox_iou(pred['bbox'], gt['bbox'])
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Check if match is above threshold
            if best_iou >= iou_threshold and best_gt_idx is not None:
                TP += 1
                gt_matched.add(best_gt_idx)
            else:
                FP += 1
    
    FN = total_gt - TP
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision_50": truncate3(precision),
        "recall_50": truncate3(recall),
        "f1_50": truncate3(f1),
        "TP": TP,
        "FP": FP,
        "FN": FN
    }


def compute_all_yolo_metrics(coco_predictions, coco_map):
    """Compute YOLO-style metrics for all model/dataset combinations (both bbox and mask)."""
    yolo_metrics_mask = []
    yolo_metrics_bbox = []
    
    print("\nComputing YOLO-style metrics (Precision/Recall/F1 @ IoU=0.5)...")
    
    for key, pred_path in coco_predictions.items():
        parts = key.split('__')
        if len(parts) != 2:
            continue
        model_key, dataset_name = parts
        
        # Find corresponding GT
        gt_info = None
        for ds_key, info in coco_map.items():
            if dataset_name in ds_key or ds_key in dataset_name:
                gt_info = info
                break
        
        if not gt_info or not pred_path.exists():
            continue
        
        # Load predictions
        with open(pred_path, 'r') as f:
            pred_boxes = json.load(f)
        
        # Load ground truth with segmentation masks
        coco_gt = gt_info['coco']
        gt_boxes = []
        for ann_id in coco_gt.anns:
            ann = coco_gt.anns[ann_id]
            gt_item = {
                'image_id': ann['image_id'],
                'bbox': ann['bbox'],
                'category_id': ann['category_id']
            }
            # Add segmentation if available
            if 'segmentation' in ann:
                # Convert polygon to RLE if needed
                if isinstance(ann['segmentation'], list):
                    # Polygon format - convert to RLE
                    img_info = coco_gt.imgs[ann['image_id']]
                    h, w = img_info['height'], img_info['width']
                    rle = maskUtils.frPyObjects(ann['segmentation'], h, w)
                    if isinstance(rle, list):
                        rle = maskUtils.merge(rle)
                    gt_item['segmentation'] = rle
                else:
                    gt_item['segmentation'] = ann['segmentation']
            gt_boxes.append(gt_item)
        
        # Compute mask metrics
        metrics_mask = compute_yolo_metrics_for_pair(gt_boxes, pred_boxes, iou_threshold=0.50, use_masks=True)
        metrics_mask['model'] = model_key
        metrics_mask['dataset'] = dataset_name
        metrics_mask['type'] = 'mask'
        yolo_metrics_mask.append(metrics_mask)
        
        # Compute bbox metrics
        metrics_bbox = compute_yolo_metrics_for_pair(gt_boxes, pred_boxes, iou_threshold=0.50, use_masks=False)
        metrics_bbox['model'] = model_key
        metrics_bbox['dataset'] = dataset_name
        metrics_bbox['type'] = 'bbox'
        yolo_metrics_bbox.append(metrics_bbox)
        
        print(f"  {model_key} on {dataset_name}:")
        print(f"    Mask - P={metrics_mask['precision_50']:.3f}, R={metrics_mask['recall_50']:.3f}, F1={metrics_mask['f1_50']:.3f}")
        print(f"    BBox - P={metrics_bbox['precision_50']:.3f}, R={metrics_bbox['recall_50']:.3f}, F1={metrics_bbox['f1_50']:.3f}")
    
    return yolo_metrics_mask, yolo_metrics_bbox


def plot_yolo_metrics(yolo_metrics_mask, yolo_metrics_bbox, output_dir):
    """Generate comparison bar plots for YOLO-style metrics (both bbox and mask)."""
    if not yolo_metrics_mask and not yolo_metrics_bbox:
        return []
    
    output_dir = Path(output_dir)
    plot_paths = []
    
    # Create comparison plots for mask metrics
    if yolo_metrics_mask:
        yolo_metrics = yolo_metrics_mask
    
    # Create comparison plot (all models side by side)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Group by dataset for comparison
    dataset_groups = defaultdict(lambda: defaultdict(list))
    for m in yolo_metrics:
        dataset_groups[m['dataset']]['models'].append(m['model'])
        dataset_groups[m['dataset']]['precision'].append(m['precision_50'])
        dataset_groups[m['dataset']]['recall'].append(m['recall_50'])
        dataset_groups[m['dataset']]['f1'].append(m['f1_50'])
    
    metrics_to_plot = ['precision', 'recall', 'f1']
    metric_names = ['Precision@0.5', 'Recall@0.5', 'F1@0.5']
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    for idx, (metric_key, metric_name, color) in enumerate(zip(metrics_to_plot, metric_names, colors)):
        ax = axes[idx]
        
        # Prepare data for grouped bar chart
        datasets = sorted(dataset_groups.keys())
        models_set = sorted(set(m['model'] for m in yolo_metrics))
        
        x = np.arange(len(datasets))
        width = 0.8 / len(models_set)
        
        for i, model in enumerate(models_set):
            values = []
            for dataset in datasets:
                if model in dataset_groups[dataset]['models']:
                    model_idx = dataset_groups[dataset]['models'].index(model)
                    values.append(dataset_groups[dataset][metric_key][model_idx])
                else:
                    values.append(0)
            
            ax.bar(x + i * width - 0.4 + width/2, values, width, 
                   label=model, alpha=0.8)
        
        ax.set_xlabel('Dataset', fontsize=11, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=11, fontweight='bold')
        ax.set_title(f'{metric_name} Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=15, ha='right')
        ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    comparison_path = output_dir / 'yolo_metrics_comparison.png'
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    plot_paths.append(('comparison', comparison_path))
    
    print(f"Generated {len(plot_paths)} YOLO metrics plots")
    return plot_paths


def coco_evaluate(gt_json_path, pred_json_path, iou_type='segm'):
    """Run COCO evaluation using pycocotools.
    
    Args:
        gt_json_path: Path to ground truth COCO JSON
        pred_json_path: Path to predictions COCO JSON
        iou_type: 'bbox' or 'segm' for evaluation type
    """
    try:
        cocoGt = COCO(str(gt_json_path))
        cocoDt = cocoGt.loadRes(str(pred_json_path))
        
        cocoEval = COCOeval(cocoGt, cocoDt, iouType=iou_type)
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        
        stats = cocoEval.stats
        keys = [
            "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
            "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"
        ]
        
        result = summarize_dict(dict(zip(keys, stats)))
        result['iou_type'] = iou_type
        return result
    
    except Exception as e:
        print(f"Error in COCO evaluation ({iou_type}): {e}")
        return None


def coco_evaluate_with_evaluator(dataset_name, pred_json_path, output_dir):
    """Run COCO evaluation using Detectron2's COCOEvaluator (matches train_net.py exactly).
    
    Args:
        dataset_name: Registered dataset name (e.g., 'comparison_big-images-rev')
        pred_json_path: Path to predictions JSON
        output_dir: Output directory for results
        
    Returns:
        dict: Evaluation results with both bbox and segm metrics
    """
    try:
        # Create evaluator (matches train_net.py behavior)
        evaluator = COCOEvaluator(dataset_name, output_dir=str(output_dir))
        evaluator.reset()
        
        # Load predictions
        with open(pred_json_path, 'r') as f:
            predictions_list = json.load(f)
        
        # Group by image_id
        predictions_by_image = defaultdict(list)
        for pred in predictions_list:
            predictions_by_image[pred['image_id']].append(pred)
        
        # Process each image (simulating evaluator.process())
        for image_id, preds in predictions_by_image.items():
            prediction = {
                "image_id": image_id,
                "instances": preds  # COCOEvaluator expects this format
            }
            evaluator._predictions.append(prediction)
        
        # Run evaluation
        results = evaluator.evaluate()
        return results
        
    except Exception as e:
        print(f"Error in COCOEvaluator evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None


def bbox_stats(bboxes):
    """Generate bar plots for YOLO-style metrics."""
    if not yolo_metrics:
        return []
    
    output_dir = Path(output_dir)
    plot_paths = []
    
    # Group by model
    model_groups = defaultdict(list)
    for m in yolo_metrics:
        model_groups[m['model']].append(m)
    
    # Create individual plots for each model
    for model_name, metrics_list in sorted(model_groups.items()):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        datasets = [m['dataset'] for m in metrics_list]
        precision = [m['precision_50'] for m in metrics_list]
        recall = [m['recall_50'] for m in metrics_list]
        f1 = [m['f1_50'] for m in metrics_list]
        
        x = np.arange(len(datasets))
        width = 0.25
        
        ax.bar(x - width, precision, width, label='Precision@0.5', color='#2ecc71', alpha=0.8)
        ax.bar(x, recall, width, label='Recall@0.5', color='#3498db', alpha=0.8)
        ax.bar(x + width, f1, width, label='F1@0.5', color='#e74c3c', alpha=0.8)
        
        ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title(f'{model_name.upper()} - YOLO-Style Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=15, ha='right')
        ax.legend(loc='upper left', framealpha=0.9)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plot_path = output_dir / f'yolo_metrics_{model_name}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        plot_paths.append((model_name, plot_path))
    
    # Create comparison plot (all models side by side)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Group by dataset for comparison
    dataset_groups = defaultdict(lambda: defaultdict(list))
    for m in yolo_metrics:
        dataset_groups[m['dataset']]['models'].append(m['model'])
        dataset_groups[m['dataset']]['precision'].append(m['precision_50'])
        dataset_groups[m['dataset']]['recall'].append(m['recall_50'])
        dataset_groups[m['dataset']]['f1'].append(m['f1_50'])
    
    metrics_to_plot = ['precision', 'recall', 'f1']
    metric_names = ['Precision@0.5', 'Recall@0.5', 'F1@0.5']
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    for idx, (metric_key, metric_name, color) in enumerate(zip(metrics_to_plot, metric_names, colors)):
        ax = axes[idx]
        
        # Prepare data for grouped bar chart
        datasets = sorted(dataset_groups.keys())
        models_set = sorted(set(m['model'] for m in yolo_metrics))
        
        x = np.arange(len(datasets))
        width = 0.8 / len(models_set)
        
        for i, model in enumerate(models_set):
            values = []
            for dataset in datasets:
                if model in dataset_groups[dataset]['models']:
                    model_idx = dataset_groups[dataset]['models'].index(model)
                    values.append(dataset_groups[dataset][metric_key][model_idx])
                else:
                    values.append(0)
            
            ax.bar(x + i * width - 0.4 + width/2, values, width, 
                   label=model, alpha=0.8)
        
        ax.set_xlabel('Dataset', fontsize=11, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=11, fontweight='bold')
        ax.set_title(f'{metric_name} Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=15, ha='right')
        ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    comparison_path = output_dir / 'yolo_metrics_comparison.png'
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    plot_paths.append(('comparison', comparison_path))
    
    print(f"Generated {len(plot_paths)} YOLO metrics plots")
    return plot_paths


def generate_confusion_matrices(yolo_metrics, coco_predictions, coco_map, output_dir):
    """Generate confusion matrices in a 3-column grid layout."""
    output_dir = Path(output_dir)
    confusion_paths = []
    all_matrices = {}  # Store all matrices for grid layout
    
    print("\nGenerating confusion matrices...")
    
    for key, pred_path in coco_predictions.items():
        parts = key.split('__')
        if len(parts) != 2:
            continue
        model_key, dataset_name = parts
        
        # Find GT
        gt_info = None
        for ds_key, info in coco_map.items():
            if dataset_name in ds_key or ds_key in dataset_name:
                gt_info = info
                break
        
        if not gt_info or not pred_path.exists():
            continue
        
        # Load predictions and GT
        with open(pred_path, 'r') as f:
            pred_boxes = json.load(f)
        
        coco_gt = gt_info['coco']
        
        # Build confusion matrix for 3 classes: 0=positive, 1=negative, 2=lines
        # Note: COCO uses category_ids 0, 1, 2 directly (no background class)
        num_classes = 3
        cm = np.zeros((num_classes, num_classes), dtype=int)
        
        # Group predictions by image, filtering invalid category IDs
        pred_by_img = defaultdict(list)
        for pred in pred_boxes:
            cat_id = pred.get('category_id', -1)
            if 0 <= cat_id < num_classes:
                pred_by_img[pred['image_id']].append(pred)
            else:
                # Skip predictions with invalid category IDs
                continue
        
        # Group GT by image
        gt_by_img = defaultdict(list)
        for ann_id in coco_gt.anns:
            ann = coco_gt.anns[ann_id]
            gt_by_img[ann['image_id']].append(ann)
        
        # Compute IoU matching for confusion matrix using masks
        def compute_iou(pred, gt):
            """Compute IoU using masks if available, fallback to bbox."""
            pred_seg = pred.get('segmentation')
            
            # Get GT segmentation
            gt_seg = None
            if 'segmentation' in gt:
                if isinstance(gt['segmentation'], list):
                    # Polygon format - convert to RLE
                    img_info = coco_gt.imgs[gt['image_id']]
                    h, w = img_info['height'], img_info['width']
                    rle = maskUtils.frPyObjects(gt['segmentation'], h, w)
                    if isinstance(rle, list):
                        rle = maskUtils.merge(rle)
                    gt_seg = rle
                else:
                    gt_seg = gt['segmentation']
            
            # Use mask IoU if available
            if pred_seg and gt_seg:
                try:
                    iou = maskUtils.iou([pred_seg], [gt_seg], [0])[0][0]
                    return float(iou)
                except:
                    pass
            
            # Fallback to bbox IoU
            b1, b2 = pred['bbox'], gt['bbox']
            x1 = max(b1[0], b2[0])
            y1 = max(b1[1], b2[1])
            x2 = min(b1[0] + b1[2], b2[0] + b2[2])
            y2 = min(b1[1] + b1[3], b2[1] + b1[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            union = b1[2] * b1[3] + b2[2] * b2[3] - inter
            return inter / union if union > 0 else 0
        
        matched_gt = set()
        
        # Match predictions to GT
        for img_id in pred_by_img:
            preds = pred_by_img[img_id]
            gts = gt_by_img.get(img_id, [])
            
            for pred in preds:
                pred_class = pred['category_id']
                if not (0 <= pred_class < num_classes):
                    continue  # Skip invalid predictions
                    
                best_iou = 0
                best_gt_class = None
                best_gt_idx = None
                
                for idx, gt in enumerate(gts):
                    gt_class = gt['category_id']
                    if not (0 <= gt_class < num_classes):
                        continue  # Skip invalid GT
                        
                    gt_key = (img_id, idx)
                    if gt_key in matched_gt:
                        continue
                    iou = compute_iou(pred, gt)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_class = gt_class
                        best_gt_idx = gt_key
                
                if best_iou >= 0.5 and best_gt_idx is not None and best_gt_class is not None:
                    matched_gt.add(best_gt_idx)
                    cm[best_gt_class, pred_class] += 1
                # False positives are not tracked in confusion matrix
            
            # Unmatched GT (false negatives) - not tracked in current implementation
            # Could add separate tracking if needed
        
        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(6, 5))
        
        class_names = ['Positive', 'Negative', 'Lines']
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names,
               yticklabels=class_names,
               title=f'Confusion Matrix: {model_key} on {dataset_name}',
               ylabel='True Label',
               xlabel='Predicted Label')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        cm_path = output_dir / f'confusion_matrix_{model_key}__{dataset_name}.png'
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        confusion_paths.append((f"{model_key}__{dataset_name}", cm_path))
        
        # Store matrix data for grid layout
        all_matrices[key] = {
            'cm': cm,
            'model': model_key,
            'dataset': dataset_name
        }
    
    # Create grid layout with 3 columns
    if all_matrices:
        print("\nCreating confusion matrix grid layout...")
        
        # Sort matrices by model and dataset
        sorted_keys = sorted(all_matrices.keys())
        n_matrices = len(sorted_keys)
        n_cols = 3
        n_rows = (n_matrices + n_cols - 1) // n_cols  # Ceiling division
        
        # Create figure with subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        
        # Flatten axes array for easier indexing
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        class_names = ['BG', 'Pos', 'Neg', 'Lines']
        
        for idx, key in enumerate(sorted_keys):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            matrix_data = all_matrices[key]
            cm = matrix_data['cm']
            model_key = matrix_data['model']
            dataset_name = matrix_data['dataset']
            
            # Plot confusion matrix
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Set labels and title
            ax.set(xticks=np.arange(cm.shape[1]),
                   yticks=np.arange(cm.shape[0]),
                   xticklabels=class_names,
                   yticklabels=class_names,
                   ylabel='True',
                   xlabel='Predicted')
            
            # Shorter title for compact display
            title = f"{model_key.replace('_', '-').upper()}\n{dataset_name.replace('coco_', '')}"
            ax.set_title(title, fontsize=10, fontweight='bold')
            
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
            plt.setp(ax.get_yticklabels(), fontsize=8)
            
            # Add text annotations
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], 'd'),
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black",
                           fontsize=8)
        
        # Hide unused subplots
        for idx in range(n_matrices, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        grid_path = output_dir / 'confusion_matrices_grid.png'
        plt.savefig(grid_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        confusion_paths.append(('grid', grid_path))
        print(f"Created confusion matrix grid: {grid_path}")
    
    print(f"Generated confusion matrix grid with {len(all_matrices)} matrices")
    return confusion_paths


def generate_pr_curves(coco_predictions, coco_map, output_dir):
    """Generate Precision-Recall and Confidence curves for each model/dataset, plus merged comparisons."""
    output_dir = Path(output_dir)
    curve_paths = []
    all_curves_data = {}  # Store data for merged plots
    
    print("\nGenerating PR curves...")
    
    for key, pred_path in coco_predictions.items():
        parts = key.split('__')
        if len(parts) != 2:
            continue
        model_key, dataset_name = parts
        
        # Find GT
        gt_info = None
        for ds_key, info in coco_map.items():
            if dataset_name in ds_key or ds_key in dataset_name:
                gt_info = info
                break
        
        if not gt_info or not pred_path.exists():
            continue
        
        # Load predictions and GT
        with open(pred_path, 'r') as f:
            pred_boxes = json.load(f)
        
        if not pred_boxes:
            continue
        
        coco_gt = gt_info['coco']
        gt_boxes = []
        for ann_id in coco_gt.anns:
            ann = coco_gt.anns[ann_id]
            gt_boxes.append({
                'image_id': ann['image_id'],
                'bbox': ann['bbox'],
                'category_id': ann['category_id']
            })
        
        # Sort predictions by confidence
        pred_boxes_sorted = sorted(pred_boxes, key=lambda x: x.get('score', 0), reverse=True)
        
        # Compute precision/recall at different confidence thresholds using masks
        def compute_iou(pred, gt):
            """Compute IoU using masks if available, fallback to bbox."""
            pred_seg = pred.get('segmentation')
            gt_seg = gt.get('segmentation')
            
            # Use mask IoU if available
            if pred_seg and gt_seg:
                try:
                    iou = maskUtils.iou([pred_seg], [gt_seg], [0])[0][0]
                    return float(iou)
                except:
                    pass
            
            # Fallback to bbox IoU
            b1, b2 = pred['bbox'], gt['bbox']
            x1 = max(b1[0], b2[0])
            y1 = max(b1[1], b2[1])
            x2 = min(b1[0] + b1[2], b2[0] + b2[2])
            y2 = min(b1[1] + b1[3], b2[1] + b2[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            union = b1[2] * b1[3] + b2[2] * b2[3] - inter
            return inter / union if union > 0 else 0
        
        confidences = []
        precisions = []
        recalls = []
        
        # Compute metrics at different thresholds
        for conf_threshold in np.linspace(0.05, 0.95, 50):
            filtered_preds = [p for p in pred_boxes_sorted if p.get('score', 0) >= conf_threshold]
            
            if not filtered_preds:
                continue
            
            TP = 0
            FP = 0
            gt_used = set()
            
            for pred in filtered_preds:
                best_iou = 0
                best_gt = None
                for i, gt in enumerate(gt_boxes):
                    if gt['image_id'] != pred['image_id']:
                        continue
                    iou = compute_iou(pred, gt)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = i
                
                if best_iou >= 0.5:
                    if best_gt not in gt_used:
                        TP += 1
                        gt_used.add(best_gt)
                    else:
                        FP += 1
                else:
                    FP += 1
            
            FN = len(gt_boxes) - len(gt_used)
            
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            
            confidences.append(conf_threshold)
            precisions.append(precision)
            recalls.append(recall)
        
        if not confidences:
            continue
        
        # Store data for merged plots (skip individual plots)
        all_curves_data[f"{model_key}__{dataset_name}"] = {
            'confidences': confidences,
            'precisions': precisions,
            'recalls': recalls,
            'model': model_key,
            'dataset': dataset_name
        }
    
    # Create merged comparison plots
    if all_curves_data:
        print("\nGenerating merged PR curve comparisons...")
        
        # Define colors for models using seaborn palette
        model_keys = list(CONFIG['models'].keys())
        colors_palette = sns.color_palette("husl", len(model_keys))
        model_colors = {key: colors_palette[i] for i, key in enumerate(model_keys)}
        
        # Get unique dataset names from the curves data
        unique_datasets = sorted(set(data['dataset'] for data in all_curves_data.values()))
        
        # Group by dataset for comparison
        for dataset in unique_datasets:
            fig, axes = plt.subplots(1, 3, figsize=(20, 6))
            
            dataset_curves = {k: v for k, v in all_curves_data.items() if v['dataset'] == dataset}
            
            if not dataset_curves:
                plt.close()
                continue
            
            # Plot PR curves
            for key, data in dataset_curves.items():
                model = data['model']
                color = model_colors.get(model, '#95a5a6')
                label = model.replace('_', ' ').title()
                axes[0].plot(data['recalls'], data['precisions'], 
                           color=color, linewidth=2, label=label, alpha=0.8)
            
            axes[0].set_xlabel('Recall', fontsize=12, fontweight='bold')
            axes[0].set_ylabel('Precision', fontsize=12, fontweight='bold')
            axes[0].set_title('Precision-Recall Curves', fontsize=13, fontweight='bold')
            axes[0].legend(loc='best', framealpha=0.9, fontsize=9)
            axes[0].grid(alpha=0.3)
            axes[0].set_xlim([0, 1])
            axes[0].set_ylim([0, 1])
            
            # Plot Precision-Confidence curves
            for key, data in dataset_curves.items():
                model = data['model']
                color = model_colors.get(model, '#95a5a6')
                label = model.replace('_', ' ').title()
                axes[1].plot(data['confidences'], data['precisions'],
                           color=color, linewidth=2, label=label, alpha=0.8)
            
            axes[1].set_xlabel('Confidence Threshold', fontsize=12, fontweight='bold')
            axes[1].set_ylabel('Precision', fontsize=12, fontweight='bold')
            axes[1].set_title('Precision vs Confidence', fontsize=13, fontweight='bold')
            axes[1].legend(loc='best', framealpha=0.9, fontsize=9)
            axes[1].grid(alpha=0.3)
            axes[1].set_xlim([0, 1])
            axes[1].set_ylim([0, 1])
            
            # Plot Recall-Confidence curves
            for key, data in dataset_curves.items():
                model = data['model']
                color = model_colors.get(model, '#95a5a6')
                label = model.replace('_', ' ').title()
                axes[2].plot(data['confidences'], data['recalls'],
                           color=color, linewidth=2, label=label, alpha=0.8)
            
            axes[2].set_xlabel('Confidence Threshold', fontsize=12, fontweight='bold')
            axes[2].set_ylabel('Recall', fontsize=12, fontweight='bold')
            axes[2].set_title('Recall vs Confidence', fontsize=13, fontweight='bold')
            axes[2].legend(loc='best', framealpha=0.9, fontsize=9)
            axes[2].grid(alpha=0.3)
            axes[2].set_xlim([0, 1])
            axes[2].set_ylim([0, 1])
            
            fig.suptitle(f'Model Comparison - {dataset.replace("_", " ").title()}', 
                        fontsize=15, fontweight='bold', y=1.00)
            plt.tight_layout()
            
            merged_path = output_dir / f'pr_curves_merged_{dataset}.png'
            plt.savefig(merged_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            curve_paths.append((f'merged_{dataset}', merged_path))
    
    print(f"Generated {len(curve_paths)} PR curve sets (including merged comparisons)")
    return curve_paths


# ============================================================================
# RESULTS MANAGEMENT
# ============================================================================

def save_inference_results(results, output_dir):
    """Save all inference results as images."""
    output_dir = Path(output_dir)
    
    for model_key, dataset_results in results.items():
        for dataset_name, image_list in dataset_results.items():
            save_dir = output_dir / model_key / dataset_name
            save_dir.mkdir(parents=True, exist_ok=True)
            
            for img_name, annotated_img in image_list:
                save_path = save_dir / (Path(img_name).stem + '.png')
                cv2.imwrite(str(save_path), annotated_img[:, :, ::-1])
    
    print(f"\nSaved all inference results to {output_dir}")


def load_inference_results(output_dir):
    """Load previously saved inference results."""
    output_dir = Path(output_dir)
    
    if not output_dir.exists():
        return None
    
    results = defaultdict(lambda: defaultdict(list))
    
    for model_dir in output_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_key = model_dir.name
        
        for dataset_dir in model_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            
            dataset_name = dataset_dir.name
            
            for img_path in sorted(dataset_dir.iterdir()):
                if img_path.suffix.lower() == '.png':
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        img_rgb = img[:, :, ::-1]
                        results[model_key][dataset_name].append((img_path.name, img_rgb))
    
    if not results:
        return None
    
    return dict(results)


def save_coco_predictions(predictions, output_path):
    """Save COCO-format predictions to JSON."""
    with open(output_path, 'w') as f:
        json.dump(predictions, f)


# ============================================================================
# MAIN INFERENCE PIPELINE
# ============================================================================

def run_all_inferences(datasets):
    """Run all model/dataset combinations and collect results."""
    results = defaultdict(lambda: defaultdict(list))
    stats = []
    coco_predictions = {}
    
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build COCO image ID mapping for all COCO datasets
    coco_map = {}
    for dataset_name, dataset_info in CONFIG["datasets"].items():
        if dataset_info["format"] == "coco":
            ann_path = Path(dataset_info["path"]) / "test" / "_annotations.coco.json"
            if ann_path.exists():
                coco_obj = COCO(str(ann_path))
                filename_to_id = {v['file_name']: int(k) for k, v in coco_obj.imgs.items()}
                coco_map[dataset_name] = {
                    'coco': coco_obj,
                    'filename_to_id': filename_to_id,
                    'ann_path': ann_path
                }
    
    # -------------------------------------------------------------------------
    # Iterate through all models
    # -------------------------------------------------------------------------
    for model_key, model_info in CONFIG["models"].items():
        print("\n" + "="*80)
        print(f"Running {model_info['name']} (epoch: {model_info['epoch']})")
        print("="*80)
        
        model_type = model_info["type"]
        confidence = model_info["confidence"]
        
        # Load model
        if model_type == "yolo":
            if YOLO is None:
                print(f"Skipping {model_key} (ultralytics not installed)")
                continue
            model = load_yolo_model(model_key)
        elif model_type == "detectron2":
            model, cfg, metadata = load_detectron2_model(model_key)
        else:
            print(f"Unknown model type: {model_type}")
            continue
        
        # Process each dataset this model should evaluate on
        for dataset_name in model_info["evaluate_on"]:
            print(f"\nProcessing {dataset_name} ({CONFIG['datasets'][dataset_name]['description']})...")
            
            if dataset_name not in datasets:
                print(f"Warning: Dataset {dataset_name} not found, skipping")
                continue
            
            images = datasets[dataset_name]
            dataset_format = CONFIG["datasets"][dataset_name]["format"]
            
            preds_list = []
            gt_info = coco_map.get(dataset_name, {})
            filename_to_id = gt_info.get('filename_to_id', {})
            
            # Run inference on all images
            for img_data in tqdm(images, desc=f"{model_key} on {dataset_name}"):
                # Handle both YOLO (3-tuple) and COCO (2-tuple) formats
                if len(img_data) == 3:
                    img_path, img_name, label_path = img_data
                else:
                    img_path, img_name = img_data
                
                # Run inference based on model type
                if model_type == "yolo":
                    # Use eval_mode=True for COCO evaluation (keeps all predictions)
                    eval_mode = (dataset_format == "coco")
                    annotated_img, num_detections, inference_time, res0 = run_yolo_inference(
                        model, img_path, confidence, eval_mode=eval_mode
                    )
                    
                    # Convert to COCO format if evaluating on COCO dataset
                    if dataset_format == "coco":
                        image_id = filename_to_id.get(img_name)
                        if image_id:
                            # Don't filter by confidence - COCO eval handles this
                            coco_preds = yolo_result_to_coco(res0, image_id)
                            preds_list.extend(coco_preds)
                
                elif model_type == "detectron2":
                    annotated_img, num_detections, inference_time, predictions = run_detectron2_inference(
                        model, metadata, img_path, confidence
                    )
                    
                    # Convert to COCO format (detectron2 only works with COCO datasets)
                    image_id = filename_to_id.get(img_name)
                    if image_id:
                        # Don't filter by confidence - COCO eval handles this
                        coco_preds = detectron2_preds_to_coco(predictions, image_id)
                        preds_list.extend(coco_preds)
                
                # Store results
                results[model_key][dataset_name].append((img_name, annotated_img))
                
                stats.append({
                    "model": model_info.get("run_name", model_info["name"]),
                    "dataset": dataset_name,
                    "image": img_name,
                    "num_detections": num_detections,
                    "inference_time_ms": f"{inference_time * 1000:.2f}",
                })
            
            # Save COCO predictions if applicable
            if dataset_format == "coco" or model_type == "yolo":
                pred_json_path = output_dir / f"{model_key}__{dataset_name}_coco_results.json"
                save_coco_predictions(preds_list, pred_json_path)
                if dataset_format == "coco":
                    coco_predictions[f"{model_key}__{dataset_name}"] = pred_json_path
        
        # Cleanup
        del model
        if model_type == "detectron2":
            del cfg, metadata
        torch.cuda.empty_cache()
    
    # Save all results as images
    print("\nSaving inference results...")
    save_inference_results(results, output_dir)
    
    return dict(results), stats, coco_predictions, coco_map


# ============================================================================
# REPORTING
# ============================================================================

def generate_comprehensive_report(stats, coco_predictions, coco_map, output_dir):
    """Generate comprehensive Markdown report with all metrics."""
    output_dir = Path(output_dir)
    
    # Run COCO evaluations
    coco_results = []
    print("\nRunning COCO evaluations...")
    for key, pred_path in coco_predictions.items():
        # Extract model and dataset from key
        parts = key.split('__')
        if len(parts) != 2:
            continue
        model_key, dataset_name = parts
        
        # Find corresponding GT annotation
        gt_info = None
        for ds_key, info in coco_map.items():
            if dataset_name in ds_key or ds_key in dataset_name:
                gt_info = info
                break
        
        if gt_info and pred_path.exists():
            gt_ann_path = gt_info['ann_path']
            
            # Evaluate with bbox
            result_bbox = coco_evaluate(gt_ann_path, pred_path, iou_type='bbox')
            if result_bbox:
                result_bbox['model'] = model_key
                result_bbox['dataset'] = dataset_name
                result_bbox['iou_type'] = 'bbox'
                coco_results.append(result_bbox)
            
            # Evaluate with segmentation
            result_segm = coco_evaluate(gt_ann_path, pred_path, iou_type='segm')
            if result_segm:
                result_segm['model'] = model_key
                result_segm['dataset'] = dataset_name
                result_segm['iou_type'] = 'segm'
                coco_results.append(result_segm)
    
    # Compute YOLO-style metrics (both bbox and mask)
    yolo_metrics_mask, yolo_metrics_bbox = compute_all_yolo_metrics(coco_predictions, coco_map)
    
    # Generate plots
    plot_paths = plot_yolo_metrics(yolo_metrics_mask, yolo_metrics_bbox, output_dir)
    
    # Generate confusion matrices (using mask metrics)
    confusion_paths = generate_confusion_matrices(yolo_metrics_mask, coco_predictions, coco_map, output_dir)
    
    # Generate PR curves
    pr_curve_paths = generate_pr_curves(coco_predictions, coco_map, output_dir)
    
    # Separate bbox and segm results
    bbox_results = [r for r in coco_results if r.get('iou_type') == 'bbox']
    segm_results = [r for r in coco_results if r.get('iou_type') == 'segm']
    
    # Generate Markdown report
    md_path = output_dir / "comprehensive_evaluation_report.md"
    with open(md_path, 'w') as f:
        f.write("# Comprehensive Model Comparison Report\n\n")
        local_str = time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime())
        f.write(f"**Generated:** {local_str}\n\n")
        # If local timezone is not UTC, also include UTC ("legal") time
        try:
            isdst = time.localtime().tm_isdst
            tzname = time.tzname[isdst]
        except Exception:
            tzname = time.tzname[0] if isinstance(time.tzname, (list, tuple)) else time.tzname
        if tzname != 'UTC':
            utc_str = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
            f.write(f"**UTC (legal time):** {utc_str}\n\n")
        
        # Model Information
        f.write("## Models Evaluated\n\n")
        for model_key, model_info in CONFIG['models'].items():
            model_name = model_info.get('name', model_key.upper())
            run_name = model_info.get('run_name', model_name)
            model_type = model_info.get('type', 'unknown')
            epoch = model_info.get('epoch', 'N/A')
            weights = Path(model_info['weights']).name
            datasets_eval = ", ".join(model_info.get('evaluate_on', []))
            
            f.write(f"### {run_name}\n\n")
            f.write(f"- **Model**: {model_name}\n")
            f.write(f"- **Type**: {model_type}\n")
            f.write(f"- **Epoch**: {epoch}\n")
            f.write(f"- **Weights**: {weights}\n")
            f.write(f"- **Evaluated on**: {datasets_eval}\n")
            f.write("\n")
        
        # Table of Contents
        f.write("## Table of Contents\n\n")
        f.write("1. [Models Evaluated](#models-evaluated)\n")
        f.write("2. [COCO-Style Metrics](#coco-style-metrics)\n")
        f.write("3. [YOLO-Style Metrics](#yolo-style-metrics)\n")
        f.write("4. [Inference Statistics](#inference-statistics)\n")
        f.write("5. [Summary](#summary)\n\n")
        
        # COCO-Style Metrics
        f.write("## 2. COCO-Style Metrics\n\n")
        
        # BBox Metrics
        f.write("### BBox Metrics\n\n")
        if bbox_results:
            headers = ["Model", "Dataset", "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large"]
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
            
            for res in bbox_results:
                row = [
                    res.get('model', ''),
                    res.get('dataset', ''),
                    f"{res.get('AP', 0):.3f}",
                    f"{res.get('AP50', 0):.3f}",
                    f"{res.get('AP75', 0):.3f}",
                    f"{res.get('AP_small', 0):.3f}",
                    f"{res.get('AP_medium', 0):.3f}",
                    f"{res.get('AP_large', 0):.3f}",
                ]
                f.write("| " + " | ".join(row) + " |\n")
            f.write("\n")
        else:
            f.write("*No bbox evaluation results available.*\n\n")
        
        # Segmentation Metrics
        f.write("### Segmentation (Mask) Metrics\n\n")
        if segm_results:
            headers = ["Model", "Dataset", "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large"]
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
            
            for res in segm_results:
                row = [
                    res.get('model', ''),
                    res.get('dataset', ''),
                    f"{res.get('AP', 0):.3f}",
                    f"{res.get('AP50', 0):.3f}",
                    f"{res.get('AP75', 0):.3f}",
                    f"{res.get('AP_small', 0):.3f}",
                    f"{res.get('AP_medium', 0):.3f}",
                    f"{res.get('AP_large', 0):.3f}",
                ]
                f.write("| " + " | ".join(row) + " |\n")
            f.write("\n")
        else:
            f.write("*No segmentation evaluation results available.*\n\n")
        
        # YOLO-Style Metrics
        f.write("\n## 3. YOLO-Style Metrics\n\n")
        f.write("*Precision, Recall, and F1 Score computed at IoU threshold = 0.5*\n\n")
        
        if yolo_metrics_mask or yolo_metrics_bbox:
            # Add comparison plots
            comparison_plot_mask = next((p for name, p in plot_paths if name == 'comparison_mask'), None)
            comparison_plot_bbox = next((p for name, p in plot_paths if name == 'comparison_bbox'), None)
            
            if comparison_plot_mask:
                f.write(f"### Overall Comparison (Mask IoU)\n\n")
                f.write(f"![YOLO Metrics Comparison - Mask]({comparison_plot_mask.name})\n\n")
            
            if comparison_plot_bbox:
                f.write(f"### Overall Comparison (BBox IoU)\n\n")
                f.write(f"![YOLO Metrics Comparison - BBox]({comparison_plot_bbox.name})\n\n")
            
            # Mask Metrics Table
            if yolo_metrics_mask:
                f.write("### Mask Metrics Table (IoU@0.5)\n\n")
                headers = ["Model", "Dataset", "Precision", "Recall", "F1", "TP", "FP", "FN"]
                f.write("| " + " | ".join(headers) + " |\n")
                f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
                
                for m in yolo_metrics_mask:
                    row = [
                        m.get('model', ''),
                        m.get('dataset', ''),
                        f"{m.get('precision_50', 0):.3f}",
                        f"{m.get('recall_50', 0):.3f}",
                        f"{m.get('f1_50', 0):.3f}",
                        str(m.get('TP', 0)),
                        str(m.get('FP', 0)),
                        str(m.get('FN', 0)),
                    ]
                    f.write("| " + " | ".join(row) + " |\n")
                f.write("\n")
            
            # BBox Metrics Table
            if yolo_metrics_bbox:
                f.write("### BBox Metrics Table (IoU@0.5)\n\n")
                headers = ["Model", "Dataset", "Precision", "Recall", "F1", "TP", "FP", "FN"]
                f.write("| " + " | ".join(headers) + " |\n")
                f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
                
                for m in yolo_metrics_bbox:
                    row = [
                        m.get('model', ''),
                        m.get('dataset', ''),
                        f"{m.get('precision_50', 0):.3f}",
                        f"{m.get('recall_50', 0):.3f}",
                        f"{m.get('f1_50', 0):.3f}",
                        str(m.get('TP', 0)),
                        str(m.get('FP', 0)),
                        str(m.get('FN', 0)),
                    ]
                    f.write("| " + " | ".join(row) + " |\n")
                f.write("\n")
            
            # Best performers (mask metrics)
            if yolo_metrics_mask:
                f.write("### Top Performers (Mask Metrics)\n\n")
                best_f1 = max(yolo_metrics_mask, key=lambda x: x.get('f1_50', 0))
                best_precision = max(yolo_metrics_mask, key=lambda x: x.get('precision_50', 0))
                best_recall = max(yolo_metrics_mask, key=lambda x: x.get('recall_50', 0))
                
                f.write(f"- **Best F1 Score**: {best_f1['model']} on {best_f1['dataset']} ({best_f1.get('f1_50', 0):.3f})\n")
                f.write(f"- **Best Precision**: {best_precision['model']} on {best_precision['dataset']} ({best_precision.get('precision_50', 0):.3f})\n")
                f.write(f"- **Best Recall**: {best_recall['model']} on {best_recall['dataset']} ({best_recall.get('recall_50', 0):.3f})\n")
        else:
            f.write("*No YOLO-style metrics available.*\n")
        
        # Confusion Matrices
        if confusion_paths:
            f.write("\n### Confusion Matrices\n\n")
            f.write("*All confusion matrices displayed in a 3-column grid layout for easy comparison.*\n\n")
            # Show grid layout
            grid_path = next((p for name, p in confusion_paths if name == 'grid'), None)
            if grid_path:
                f.write(f"![Confusion Matrices Grid]({grid_path.name})\n\n")
        
        # Precision-Recall Curves
        if pr_curve_paths:
            f.write("\n### Precision-Recall Curves (Model Comparisons)\n\n")
            
            # Show only merged comparisons
            merged_curves = [(k, p) for k, p in pr_curve_paths if k.startswith('merged_')]
            if merged_curves:
                for key, pr_path in sorted(merged_curves):
                    dataset = key.replace('merged_', '').replace('_', ' ').title()
                    f.write(f"#### {dataset}\n\n")
                    f.write(f"![PR Curves Comparison]({pr_path.name})\n\n")
        
        # Inference Statistics
        f.write("\n## 4. Inference Statistics\n\n")
        if stats:
            # Group by run_name (which is now stored in stats['model'])
            model_groups = defaultdict(list)
            for s in stats:
                model_groups[s['model']].append(s)
            
            for run_name, model_stats in sorted(model_groups.items()):
                f.write(f"### {run_name}\n\n")
                
                # Compute statistics
                times = [float(s['inference_time_ms']) for s in model_stats]
                detections = [int(s['num_detections']) for s in model_stats]
                
                # Group by dataset
                dataset_stats = defaultdict(lambda: {'times': [], 'detections': []})
                for s in model_stats:
                    dataset_stats[s['dataset']]['times'].append(float(s['inference_time_ms']))
                    dataset_stats[s['dataset']]['detections'].append(int(s['num_detections']))
                
                f.write(f"- **Total Images**: {len(model_stats)}\n")
                f.write(f"- **Avg Inference Time**: {np.mean(times):.2f} ms\n")
                f.write(f"- **Avg Detections**: {np.mean(detections):.1f}\n")
                f.write(f"- **Total Detections**: {sum(detections)}\n\n")
                
                # Per-dataset breakdown
                if len(dataset_stats) > 1:
                    f.write(f"**Per-Dataset Breakdown:**\n\n")
                    for dataset, ds_stats in sorted(dataset_stats.items()):
                        f.write(f"- **{dataset}**: {len(ds_stats['times'])} images, "
                               f"avg time {np.mean(ds_stats['times']):.2f} ms, "
                               f"avg detections {np.mean(ds_stats['detections']):.1f}\n")
                    f.write("\n")
        
        # Summary
        f.write("\n## 5. Summary\n\n")
        if coco_results:
            best_ap = max(coco_results, key=lambda x: x.get('AP', 0))
            f.write(f"**Best Overall AP (COCO)**: {best_ap['model']} on {best_ap['dataset']} "
                   f"({best_ap.get('AP', 0):.3f})\n\n")
        
        if yolo_metrics_mask:
            best_f1 = max(yolo_metrics_mask, key=lambda x: x.get('f1_50', 0))
            f.write(f"**Best Overall F1 (YOLO-style, Mask)**: {best_f1['model']} on {best_f1['dataset']} "
                   f"({best_f1.get('f1_50', 0):.3f})\n\n")
        
        f.write("---\n\n")
        f.write("*Report generated by compare_models_unified.py*\n")
    
    print(f"Comprehensive report saved to: {md_path}")
    
    # Convert to PDF
    # convert_markdown_to_pdf(md_path, output_dir)
    
    # Also save as CSV
    if coco_results:
        csv_path = output_dir / "coco_evaluation_results.csv"
        keys = ["model", "dataset", "iou_type", "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
                "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in coco_results:
                writer.writerow({k: row.get(k, "") for k in keys})
        print(f"COCO results CSV saved to: {csv_path}")
    
    # Save YOLO metrics CSV (both bbox and mask)
    if yolo_metrics_mask or yolo_metrics_bbox:
        yolo_csv_path = output_dir / "yolo_metrics_results.csv"
        yolo_keys = ["model", "dataset", "type", "precision_50", "recall_50", "f1_50", "TP", "FP", "FN"]
        with open(yolo_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=yolo_keys)
            writer.writeheader()
            for row in yolo_metrics_mask + yolo_metrics_bbox:
                writer.writerow({k: row.get(k, "") for k in yolo_keys})
        print(f"YOLO metrics CSV saved to: {yolo_csv_path}")


def convert_markdown_to_pdf(md_path, output_dir):
    """Convert markdown report to PDF using pandoc or wkhtmltopdf."""
    md_path = Path(md_path)
    pdf_path = md_path.with_suffix('.pdf')
    
    print(f"\nConverting Markdown to PDF...")
    
    # Try pandoc first (better quality)
    if shutil.which('pandoc'):
        try:
            cmd = [
                'pandoc',
                str(md_path),
                '-o', str(pdf_path),
                '--pdf-engine=xelatex',
                '-V', 'geometry:margin=1in',
                '--highlight-style=tango'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                print(f"✓ PDF report created: {pdf_path}")
                return pdf_path
            else:
                print(f"Pandoc failed: {result.stderr}")
        except Exception as e:
            print(f"Pandoc conversion failed: {e}")
    
    # Try markdown2 + pdfkit as fallback
    try:
        import markdown2
        import pdfkit
        
        # Convert markdown to HTML
        with open(md_path, 'r') as f:
            md_content = f.read()
        
        html_content = markdown2.markdown(md_content, extras=['tables', 'fenced-code-blocks'])
        
        # Add CSS styling
        styled_html = f"""
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                img {{ max-width: 100%; height: auto; margin: 20px 0; }}
                h1, h2, h3 {{ color: #333; }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        # Convert HTML to PDF
        pdfkit.from_string(styled_html, str(pdf_path))
        print(f"✓ PDF report created: {pdf_path}")
        return pdf_path
        
    except ImportError:
        print("⚠ PDF conversion libraries not found. Install with:")
        print("  apt-get install pandoc texlive-xetex")
        print("  or: pip install markdown2 pdfkit && apt-get install wkhtmltopdf")
        return None
    except Exception as e:
        print(f"⚠ PDF conversion failed: {e}")
        return None


def save_stats_table(stats):
    """Save basic statistics as Markdown and CSV."""
    output_dir = Path(CONFIG["output_dir"])
    
    # CSV
    csv_path = output_dir / "inference_stats.csv"
    with open(csv_path, 'w', newline='') as f:
        if stats:
            writer = csv.DictWriter(f, fieldnames=stats[0].keys())
            writer.writeheader()
            writer.writerows(stats)
    print(f"Inference stats saved to: {csv_path}")
    
    # Markdown
    md_path = output_dir / "inference_stats.md"
    with open(md_path, 'w') as f:
        f.write("# Inference Statistics\n\n")
        
        if stats:
            headers = list(stats[0].keys())
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
            
            for row in stats:
                f.write("| " + " | ".join(str(row[h]) for h in headers) + " |\n")
        
        # Summary
        f.write("\n## Summary\n\n")
        summary = defaultdict(lambda: defaultdict(list))
        for row in stats:
            model = row["model"]
            dataset = row["dataset"]
            summary[model][dataset].append(float(row["inference_time_ms"]))
        
        f.write("| Model | Dataset | Avg Time (ms) | Images |\n")
        f.write("| --- | --- | --- | --- |\n")
        
        for model in sorted(summary.keys()):
            for dataset in sorted(summary[model].keys()):
                times = summary[model][dataset]
                f.write(f"| {model} | {dataset} | {np.mean(times):.2f} | {len(times)} |\n")
    
    print(f"Inference stats MD saved to: {md_path}")


# ============================================================================
# VISUALIZATION (Gallery View)
# ============================================================================

def create_gallery_view(original_img, annotated_img, results, image_name):
    """Create gallery view for comparison."""
    original_rgb = original_img[:, :, ::-1]
    annotated_rgb = annotated_img[:, :, ::-1] if len(annotated_img.shape) == 3 else annotated_img
    
    def find_image(model_key, dataset_key):
        if model_key not in results or dataset_key not in results[model_key]:
            return None
        for img_name, img in results[model_key][dataset_key]:
            if img_name == image_name or Path(img_name).stem == Path(image_name).stem:
                return img
        return None
    
    # Get all model results (12 combinations)
    yolo_nn = find_image("yolo_normal", "yolo_normal")
    yolo_nc = find_image("yolo_normal", "yolo_contrast")
    yolo_cn = find_image("yolo_contrast", "yolo_normal")
    yolo_cc = find_image("yolo_contrast", "yolo_contrast")
    
    maskrcnn_nn = find_image("maskrcnn_normal", "coco_normal")
    maskrcnn_nc = find_image("maskrcnn_normal", "coco_contrast")
    maskrcnn_cn = find_image("maskrcnn_contrast", "coco_normal")
    maskrcnn_cc = find_image("maskrcnn_contrast", "coco_contrast")
    
    istr_nn = find_image("istr_normal", "coco_normal")
    istr_nc = find_image("istr_normal", "coco_contrast")
    istr_cn = find_image("istr_contrast", "coco_normal")
    istr_cc = find_image("istr_contrast", "coco_contrast")
    
    target_height = min(600, original_rgb.shape[0])
    spacing = 8
    
    def resize_to_height(img, h):
        if img is None:
            return np.ones((h, int(h * 1.5), 3), dtype=np.uint8) * 200
        old_h, old_w = img.shape[:2]
        new_w = int(old_w * (h / old_h))
        return cv2.resize(img, (new_w, h))
    
    def add_label(img, text):
        labeled = img.copy()
        label_height = 25
        label_bar = np.ones((label_height, labeled.shape[1], 3), dtype=np.uint8) * 50
        cv2.putText(label_bar, text, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return np.vstack([label_bar, labeled])
    
    # Resize and label all images
    yolo_nn_l = add_label(resize_to_height(yolo_nn, target_height), "YOLO-N on Normal")
    yolo_nc_l = add_label(resize_to_height(yolo_nc, target_height), "YOLO-N on Contrast")
    yolo_cn_l = add_label(resize_to_height(yolo_cn, target_height), "YOLO-C on Normal")
    yolo_cc_l = add_label(resize_to_height(yolo_cc, target_height), "YOLO-C on Contrast")
    
    mask_nn_l = add_label(resize_to_height(maskrcnn_nn, target_height), "Mask-N on Normal")
    mask_nc_l = add_label(resize_to_height(maskrcnn_nc, target_height), "Mask-N on Contrast")
    mask_cn_l = add_label(resize_to_height(maskrcnn_cn, target_height), "Mask-C on Normal")
    mask_cc_l = add_label(resize_to_height(maskrcnn_cc, target_height), "Mask-C on Contrast")
    
    istr_nn_l = add_label(resize_to_height(istr_nn, target_height), "ISTR-N on Normal")
    istr_nc_l = add_label(resize_to_height(istr_nc, target_height), "ISTR-N on Contrast")
    istr_cn_l = add_label(resize_to_height(istr_cn, target_height), "ISTR-C on Normal")
    istr_cc_l = add_label(resize_to_height(istr_cc, target_height), "ISTR-C on Contrast")
    
    # Create spacing
    white_space = np.ones((yolo_nn_l.shape[0], spacing, 3), dtype=np.uint8) * 255
    
    # Create rows
    row1 = np.hstack([yolo_nn_l, white_space, yolo_nc_l, white_space, yolo_cn_l, white_space, yolo_cc_l])
    row2 = np.hstack([mask_nn_l, white_space, mask_nc_l, white_space, mask_cn_l, white_space, mask_cc_l])
    row3 = np.hstack([istr_nn_l, white_space, istr_nc_l, white_space, istr_cn_l, white_space, istr_cc_l])
    
    row_spacing = np.ones((spacing, row1.shape[1], 3), dtype=np.uint8) * 255
    model_results = np.vstack([row1, row_spacing, row2, row_spacing, row3])
    
    # Create left column
    original_resized = resize_to_height(original_rgb, target_height)
    annotated_resized = resize_to_height(annotated_rgb, target_height)
    
    original_labeled = add_label(original_resized, "Original")
    annotated_labeled = add_label(annotated_resized, "Ground Truth")
    
    left_spacing = np.ones((spacing, original_labeled.shape[1], 3), dtype=np.uint8) * 255
    left_column = np.vstack([original_labeled, left_spacing, annotated_labeled])
    
    # Match heights
    if left_column.shape[0] < model_results.shape[0]:
        padding = np.ones((model_results.shape[0] - left_column.shape[0], left_column.shape[1], 3), dtype=np.uint8) * 255
        left_column = np.vstack([left_column, padding])
    elif left_column.shape[0] > model_results.shape[0]:
        padding = np.ones((left_column.shape[0] - model_results.shape[0], model_results.shape[1], 3), dtype=np.uint8) * 255
        model_results = np.vstack([model_results, padding])
    
    # Combine
    column_spacing = np.ones((left_column.shape[0], spacing * 3, 3), dtype=np.uint8) * 255
    gallery = np.hstack([left_column, column_spacing, model_results])
    
    # Add title
    title_height = 50
    title_bar = np.ones((title_height, gallery.shape[1], 3), dtype=np.uint8) * 240
    cv2.putText(title_bar, f"Model Comparison - {image_name}", (20, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    gallery = np.vstack([title_bar, gallery])
    
    return gallery


def display_gallery(datasets, results, metadata=None):
    """Display gallery view for all images."""
    if metadata is None:
        metadata = MetadataCatalog.get("comparison_coco_normal")
    
    first_dataset = datasets["coco_normal"]
    yolo_class_names = {0: 'positive', 1: 'negative', 2: 'lines'}
    
    window_name = "Model Comparison Gallery - Press 'q' to quit, arrow keys to navigate"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    current_idx = 0
    
    while True:
        img_path, img_name = first_dataset[current_idx]
        
        original_img = cv2.imread(img_path)
        
        # Try to load ground truth
        annotated_img = original_img.copy()
        has_gt = False
        
        for yolo_data in datasets.get("yolo_normal", []):
            if len(yolo_data) == 3:
                yolo_img_path, yolo_img_name, label_path = yolo_data
                if Path(img_name).stem == Path(yolo_img_name).stem:
                    if Path(label_path).exists():
                        gt_annotations = read_yolo_annotations(label_path)
                        if gt_annotations:
                            annotated_img = draw_yolo_annotations(original_img.copy(), gt_annotations, yolo_class_names)
                            has_gt = True
                    break
        
        if not has_gt:
            annotated_img = original_img.copy()
            cv2.putText(annotated_img, "No Ground Truth", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
        
        gallery = create_gallery_view(original_img, annotated_img, results, img_name)
        
        cv2.imshow(window_name, gallery[:, :, ::-1])
        
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q') or key == 27:
            break
        elif key == 83 or key == ord('d'):
            current_idx = (current_idx + 1) % len(first_dataset)
        elif key == 81 or key == ord('a'):
            current_idx = (current_idx - 1) % len(first_dataset)
    
    cv2.destroyAllWindows()


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified model comparison pipeline with comprehensive evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--show", action="store_true", help="Display gallery view")
    parser.add_argument("--force-inference", action="store_true", help="Force rerun inference")
    parser.add_argument("--load-only", action="store_true", help="Only load cached results")
    
    args = parser.parse_args()
    
    print("="*80)
    print("UNIFIED MODEL COMPARISON PIPELINE")
    print("="*80)
    
    output_dir = Path(CONFIG["output_dir"])
    results = None
    stats = None
    coco_predictions = None
    coco_map = None
    
    # Try to load cached results
    if not args.force_inference:
        print("\nChecking for cached results...")
        results = load_inference_results(output_dir)
        
        if results:
            print(f"Found cached results in {output_dir}")
        else:
            print("No cached results found")
    
    # Run inference if needed
    if results is None or args.force_inference:
        if args.load_only:
            print("\nError: No cached results and --load-only specified")
            return
        
        print("\nRunning inference...")
        register_coco_datasets()
        datasets = load_datasets()
        
        results, stats, coco_predictions, coco_map = run_all_inferences(datasets)
        
        # Save basic stats
        save_stats_table(stats)
        
        # Generate comprehensive report
        generate_comprehensive_report(stats, coco_predictions, coco_map, output_dir)
    else:
        print("\nUsing cached results")
        datasets = load_datasets()
        register_coco_datasets()
    
    # Display gallery if requested
    if args.show:
        if results is None:
            print("\nError: No results available")
            return
        
        print("\nDisplaying gallery view...")
        print("Controls: Arrow keys or a/d to navigate, q/ESC to quit")
        metadata = MetadataCatalog.get("comparison_coco_normal")
        display_gallery(datasets, results, metadata)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print("  - Annotated images (PNG)")
    print("  - COCO prediction JSONs")
    print("  - Comprehensive evaluation report (MD + CSV)")
    print("  - Inference statistics (MD + CSV)")


if __name__ == "__main__":
    main()
