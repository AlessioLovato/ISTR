#!/usr/bin/env python3
"""
Comprehensive model comparison script.

This script compares YOLO, Mask R-CNN, and ISTR models across multiple datasets:
- YOLO format dataset (normal and contrast)
- COCO format dataset (normal and contrast)

For each combination of model and dataset, it:
1. Runs inference on test split
2. Saves annotated images
3. Collects inference statistics
4. Generates comparison gallery view
5. Exports stats table (Markdown and CSV)

Usage:
    python compare_models.py [--show]
"""

import os
import time
import csv
import json
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm
import torch

# Detectron2 imports
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.detection_utils import read_image
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances

# ISTR imports
from istr import add_ISTR_config


# ============================================================================
# CONFIGURATION - All hardcoded paths and settings
# ============================================================================

CONFIG = {
    # Dataset paths (relative to script location)
    "datasets": {
        "yolo_normal": "../shared/YOLODataset_big_images_rev",
        "yolo_contrast": "../shared/YOLODataset_big_images_rev_contrast",
        "coco_normal": "../shared/coco-big-images-rev",
        "coco_contrast": "../shared/coco-big-images-rev-contrast",
    },
    
    # Model paths and configs
    # Each model type has 'normal' (trained on normal images) and 'contrast' (trained on contrast images) variants
    "models": {
        "yolo": {
            "normal": {
                "weights": "../shared/n_bir.pt",
                "confidence": 0.5,
            },
            "contrast": {
                "weights": "../shared/n_birc.pt",
                "confidence": 0.5,
            },
        },
        "maskrcnn": {
            "config": "detectron2/detectron2/model_zoo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
            "normal": {
                "weights": "../shared/models/output_maskrcnn_R_50_b16_big_images_rev_short/model_final.pth",
                "confidence": 0.5,
            },
            "contrast": {
                "weights": "../shared/models/output_maskrcnn_R_50_b16_aug_big_images_rev_contrast/model_final.pth",
                "confidence": 0.5,
            },
        },
        "istr": {
            "config": "detectron2/detectron2/model_zoo/configs/ISTR/ISTR-PCA-R50-3x.yaml",
            "normal": {
                "weights": "../shared/models/output_pca_50_big-images-rev/model_0009999.pth",
                "confidence": 0.5,
            },
            "contrast": {
                "weights": "../shared/models/output_pca_50_big-images-rev-contrast/model_0009999.pth",
                "confidence": 0.5,
            },
        },
    },
    
    # Output settings
    "output_dir": "model_comparison_results",
    "image_format": "png",
}


# ============================================================================
# DATASET LOADING
# ============================================================================

def read_yolo_annotations(label_path):
    """
    Read YOLO polygon format annotations.
    Returns list of (class_id, [(x1, y1), (x2, y2), ...]) in normalized coords
    """
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
            
            # Parse polygon points
            points = []
            for i in range(0, len(coords), 2):
                points.append((coords[i], coords[i + 1]))
            
            annotations.append((cls_id, points))
    
    return annotations


def draw_yolo_annotations(image, annotations, class_names):
    """
    Draw YOLO polygon annotations on image.
    Annotations are in normalized coordinates (0-1).
    """
    result = image.copy()
    h, w = image.shape[:2]
    
    # Class colors (BGR format for OpenCV)
    colors = {
        0: (0, 255, 0),      # positive - green
        1: (0, 0, 255),      # negative - red  
        2: (255, 0, 0),      # lines - blue
    }
    
    for cls_id, points in annotations:
        # Convert normalized coords to pixel coords
        pixel_points = []
        for x, y in points:
            pixel_points.append((int(x * w), int(y * h)))
        
        if len(pixel_points) < 3:
            continue
        
        # Get class info
        class_name = class_names.get(cls_id, f'class_{cls_id}')
        color = colors.get(cls_id, (0, 255, 0))
        
        # Draw filled polygon with transparency
        overlay = result.copy()
        pts = np.array(pixel_points, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.3, result, 0.7, 0, result)
        
        # Draw polygon outline
        cv2.polylines(result, [pts], isClosed=True, color=color, thickness=2)
    
    return result


def get_test_images(dataset_path, dataset_format="coco"):
    """
    Get list of test images from dataset.
    
    Args:
        dataset_path: Path to dataset folder
        dataset_format: Either "coco" or "yolo"
    
    Returns:
        List of (image_path, image_name) tuples
    """
    if dataset_format == "coco":
        test_dir = Path(dataset_path) / "test"
        
        # First, try to load from COCO annotations
        annotations_file = test_dir / "_annotations.coco.json"
        annotated_images = set()
        
        if annotations_file.exists():
            with open(annotations_file, 'r') as f:
                coco_data = json.load(f)
            
            for img_info in coco_data['images']:
                annotated_images.add(img_info['file_name'])
        
        # Now find all images in directory (including unannotated ones)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_list = []
        
        for img_path in test_dir.iterdir():
            if img_path.suffix.lower() in image_extensions:
                image_list.append((str(img_path), img_path.name))
        
        return sorted(image_list)
    
    elif dataset_format == "yolo":
        # YOLO datasets have images in images/test/ subdirectory
        test_dir = Path(dataset_path) / "images" / "test"
        labels_dir = Path(dataset_path) / "labels" / "test"
        
        # List all images in test directory
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_list = []
        
        for img_path in test_dir.iterdir():
            if img_path.suffix.lower() in image_extensions:
                # Find corresponding label file
                label_path = labels_dir / (img_path.stem + '.txt')
                image_list.append((str(img_path), img_path.name, str(label_path)))
        
        return sorted(image_list)
    
    else:
        raise ValueError(f"Unknown dataset format: {dataset_format}")


def load_datasets():
    """Load all datasets and extract test images."""
    datasets = {}
    
    print("Loading datasets...")
    
    # YOLO datasets
    datasets["yolo_normal"] = get_test_images(CONFIG["datasets"]["yolo_normal"], "yolo")
    datasets["yolo_contrast"] = get_test_images(CONFIG["datasets"]["yolo_contrast"], "yolo")
    
    # Register COCO datasets with detectron2 for proper metadata
    print("\nRegistering COCO datasets with detectron2...")
    
    # Register coco_normal with proper class names
    coco_normal_path = Path(CONFIG["datasets"]["coco_normal"])
    register_coco_instances(
        "comparison_coco_normal",
        {"thing_classes": ["positive", "negative", "lines"]},
        str(coco_normal_path / "test" / "_annotations.coco.json"),
        str(coco_normal_path / "test")
    )
    
    # Register coco_contrast with proper class names
    coco_contrast_path = Path(CONFIG["datasets"]["coco_contrast"])
    register_coco_instances(
        "comparison_coco_contrast",
        {"thing_classes": ["positive", "negative", "lines"]},
        str(coco_contrast_path / "test" / "_annotations.coco.json"),
        str(coco_contrast_path / "test")
    )
    
    # COCO datasets
    datasets["coco_normal"] = get_test_images(CONFIG["datasets"]["coco_normal"], "coco")
    datasets["coco_contrast"] = get_test_images(CONFIG["datasets"]["coco_contrast"], "coco")
    
    # Verify all datasets have images
    for name, images in datasets.items():
        print(f"  {name}: {len(images)} images")
        if len(images) == 0:
            raise ValueError(f"No images found in {name}")
    
    return datasets


# ============================================================================
# MODEL INFERENCE - YOLO
# ============================================================================

def load_yolo_model(variant="normal"):
    """Load YOLO model.
    
    Args:
        variant: Either "normal" or "contrast" for model trained on that dataset type
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics package not found. Install with: pip install ultralytics")
    
    model_path = CONFIG["models"]["yolo"][variant]["weights"]
    print(f"Loading YOLO model ({variant}) from {model_path}...")
    
    model = YOLO(model_path)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    return model


def run_yolo_inference(model, image_path, confidence_threshold):
    """
    Run YOLO inference on a single image.
    
    Returns:
        (annotated_image, num_detections, inference_time)
    """
    start_time = time.time()
    
    # Run inference
    results = model(image_path, conf=confidence_threshold, verbose=False)
    
    inference_time = time.time() - start_time
    
    # Get annotated image (YOLO plot() returns BGR, convert to RGB)
    annotated_img_bgr = results[0].plot()
    annotated_img = cv2.cvtColor(annotated_img_bgr, cv2.COLOR_BGR2RGB)
    
    # Count detections
    num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
    
    return annotated_img, num_detections, inference_time


# ============================================================================
# MODEL INFERENCE - Detectron2 (Mask R-CNN and ISTR)
# ============================================================================

def load_detectron2_model(model_name, variant="normal"):
    """
    Load Mask R-CNN or ISTR model.
    
    Args:
        model_name: Either "maskrcnn" or "istr"
        variant: Either "normal" or "contrast" (model training variant)
    
    Returns:
        (model, cfg, metadata)
    """
    cfg = get_cfg()
    
    if model_name == "istr":
        add_ISTR_config(cfg)
    
    config_file = CONFIG["models"][model_name]["config"]
    weights = CONFIG["models"][model_name][variant]["weights"]
    
    print(f"Loading {model_name.upper()} model ({variant}) from {weights}...")
    
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIG["models"][model_name][variant]["confidence"]
    cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.freeze()
    
    # Build model
    model = DefaultTrainer.build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()
    
    # Get metadata from registered custom dataset
    # Use the normal dataset metadata (labels should be same for normal and contrast)
    metadata = MetadataCatalog.get("comparison_coco_normal")
    
    return model, cfg, metadata


def run_detectron2_inference(model, metadata, image_path, confidence_threshold):
    """
    Run Detectron2 model inference on a single image.
    
    Returns:
        (annotated_image, num_detections, inference_time)
    """
    # Read image
    img = read_image(image_path, format="BGR")
    
    start_time = time.time()
    
    # Run inference
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
        
        # Filter by confidence
        if hasattr(instances, "scores"):
            keep = instances.scores >= confidence_threshold
            instances = instances[keep]
            num_detections = len(instances)
        
        vis_output = visualizer.draw_instance_predictions(predictions=instances)
        annotated_img = vis_output.get_image()
    else:
        annotated_img = img[:, :, ::-1]
    
    return annotated_img, num_detections, inference_time


# ============================================================================
# INFERENCE RUNNER
# ============================================================================

def save_inference_results(results, output_dir):
    """Save all inference results as images."""
    output_dir = Path(output_dir)
    
    for model_key, dataset_results in results.items():
        for dataset_name, image_list in dataset_results.items():
            # Create subdirectory for this model-dataset combination
            save_dir = output_dir / model_key / dataset_name
            save_dir.mkdir(parents=True, exist_ok=True)
            
            for img_name, annotated_img in image_list:
                # Save annotated image as PNG
                save_path = save_dir / (Path(img_name).stem + '.png')
                cv2.imwrite(str(save_path), annotated_img[:, :, ::-1])  # RGB to BGR
    
    print(f"\nSaved all inference results to {output_dir}")


def load_inference_results(output_dir):
    """Load previously saved inference results."""
    output_dir = Path(output_dir)
    
    if not output_dir.exists():
        return None
    
    results = defaultdict(lambda: defaultdict(list))
    
    # Iterate through model directories
    for model_dir in output_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_key = model_dir.name
        
        # Iterate through dataset directories
        for dataset_dir in model_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            
            dataset_name = dataset_dir.name
            
            # Load all images in this directory
            for img_path in sorted(dataset_dir.iterdir()):
                if img_path.suffix.lower() == '.png':
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        # Convert BGR to RGB
                        img_rgb = img[:, :, ::-1]
                        results[model_key][dataset_name].append((img_path.name, img_rgb))
    
    if not results:
        return None
    
    return dict(results)


def run_all_inferences(datasets):
    """
    Run all model/dataset combinations and save results.
    
    Returns:
        dict of results with structure:
        {
            "yolo": {
                "yolo_normal": [(annotated_img, stats), ...],
                "yolo_contrast": [...],
            },
            "maskrcnn": {
                "coco_normal": [...],
                "coco_contrast": [...],
            },
            "istr": {
                "coco_normal": [...],
                "coco_contrast": [...],
            },
        }
    """
    results = defaultdict(lambda: defaultdict(list))
    stats = []
    
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # YOLO Inference - Test both model variants on both datasets
    # -------------------------------------------------------------------------
    for model_variant in ["normal", "contrast"]:
        print("\n" + "="*80)
        print(f"Running YOLO Inference (Model trained on {model_variant})")
        print("="*80)
        
        yolo_model = load_yolo_model(variant=model_variant)
        confidence = CONFIG["models"]["yolo"][model_variant]["confidence"]
        
        model_key = f"yolo_{model_variant}"
        
        for dataset_name in ["yolo_normal", "yolo_contrast"]:
            print(f"\nProcessing {dataset_name}...")
            images = datasets[dataset_name]
            
            for img_data in tqdm(images, desc=f"YOLO-{model_variant} on {dataset_name}"):
                # Handle both old (img_path, img_name) and new (img_path, img_name, label_path) formats
                if len(img_data) == 3:
                    img_path, img_name, label_path = img_data
                else:
                    img_path, img_name = img_data
                
                annotated_img, num_detections, inference_time = run_yolo_inference(
                    yolo_model, img_path, confidence
                )
                
                results[model_key][dataset_name].append((img_name, annotated_img))
                
                stats.append({
                    "model": f"YOLO-{model_variant}",
                    "dataset": dataset_name,
                    "image": img_name,
                    "num_detections": num_detections,
                    "inference_time_ms": f"{inference_time * 1000:.2f}",
                })
        
        # Free YOLO model from GPU
        del yolo_model
        torch.cuda.empty_cache()
    
    # -------------------------------------------------------------------------
    # Mask R-CNN Inference - Test both model variants on both datasets
    # -------------------------------------------------------------------------
    for model_variant in ["normal", "contrast"]:
        print("\n" + "="*80)
        print(f"Running Mask R-CNN Inference (Model trained on {model_variant})")
        print("="*80)
        
        maskrcnn_model, maskrcnn_cfg, maskrcnn_metadata = load_detectron2_model("maskrcnn", variant=model_variant)
        confidence = CONFIG["models"]["maskrcnn"][model_variant]["confidence"]
        
        model_key = f"maskrcnn_{model_variant}"
        
        for dataset_name in ["coco_normal", "coco_contrast"]:
            print(f"\nProcessing {dataset_name}...")
            images = datasets[dataset_name]
            
            for img_path, img_name in tqdm(images, desc=f"MaskRCNN-{model_variant} on {dataset_name}"):
                annotated_img, num_detections, inference_time = run_detectron2_inference(
                    maskrcnn_model, maskrcnn_metadata, img_path, confidence
                )
                
                results[model_key][dataset_name].append((img_name, annotated_img))
                
                stats.append({
                    "model": f"MaskRCNN-{model_variant}",
                    "dataset": dataset_name,
                    "image": img_name,
                    "num_detections": num_detections,
                    "inference_time_ms": f"{inference_time * 1000:.2f}",
                })
        
        # Free Mask R-CNN model from GPU
        del maskrcnn_model
        torch.cuda.empty_cache()
    
    # -------------------------------------------------------------------------
    # ISTR Inference - Test both model variants on both datasets
    # -------------------------------------------------------------------------
    for model_variant in ["normal", "contrast"]:
        print("\n" + "="*80)
        print(f"Running ISTR Inference (Model trained on {model_variant})")
        print("="*80)
        
        istr_model, istr_cfg, istr_metadata = load_detectron2_model("istr", variant=model_variant)
        confidence = CONFIG["models"]["istr"][model_variant]["confidence"]
        
        model_key = f"istr_{model_variant}"
        
        for dataset_name in ["coco_normal", "coco_contrast"]:
            print(f"\nProcessing {dataset_name}...")
            images = datasets[dataset_name]
            
            for img_path, img_name in tqdm(images, desc=f"ISTR-{model_variant} on {dataset_name}"):
                annotated_img, num_detections, inference_time = run_detectron2_inference(
                    istr_model, istr_metadata, img_path, confidence
                )
                
                results[model_key][dataset_name].append((img_name, annotated_img))
                
                stats.append({
                    "model": f"ISTR-{model_variant}",
                    "dataset": dataset_name,
                    "image": img_name,
                    "num_detections": num_detections,
                    "inference_time_ms": f"{inference_time * 1000:.2f}",
                })
        
        # Free ISTR model from GPU
        del istr_model
        torch.cuda.empty_cache()
    
    # Save all results as images
    print("\nSaving inference results...")
    save_inference_results(results, output_dir)
    
    return dict(results), stats


# ============================================================================
# VISUALIZATION - Gallery View
# ============================================================================

def create_gallery_view(original_img, annotated_img, results, image_name):
    """
    Create gallery view for comparison.
    
    Layout:
        Left column: Original image (top) + Annotated version (bottom)
        Right grid (3 rows × 4 columns):
            Row 1: YOLO on all 4 dataset types
            Row 2: Mask R-CNN on all 4 dataset types
            Row 3: ISTR on all 4 dataset types
    
    Actually, simpler layout based on your description:
        Left: Original + Annotated below it
        Right top row: YOLO results (normal, contrast)
        Right middle row: MaskRCNN results (normal, contrast)
        Right bottom row: ISTR results (normal, contrast)
    """
    # Prepare images
    original_rgb = original_img[:, :, ::-1]  # BGR to RGB
    annotated_rgb = annotated_img[:, :, ::-1] if len(annotated_img.shape) == 3 else annotated_img
    
    # Helper function to find image in results
    def find_image(model_key, dataset_key):
        if model_key not in results or dataset_key not in results[model_key]:
            return None
        for img_name, img in results[model_key][dataset_key]:
            if img_name == image_name:
                return img
        return None
    
    # Get all model results for this image (12 combinations total)
    # YOLO combinations
    yolo_normal_normal = find_image("yolo_normal", "yolo_normal")
    yolo_normal_contrast = find_image("yolo_normal", "yolo_contrast")
    yolo_contrast_normal = find_image("yolo_contrast", "yolo_normal")
    yolo_contrast_contrast = find_image("yolo_contrast", "yolo_contrast")
    
    # MaskRCNN combinations
    maskrcnn_normal_normal = find_image("maskrcnn_normal", "coco_normal")
    maskrcnn_normal_contrast = find_image("maskrcnn_normal", "coco_contrast")
    maskrcnn_contrast_normal = find_image("maskrcnn_contrast", "coco_normal")
    maskrcnn_contrast_contrast = find_image("maskrcnn_contrast", "coco_contrast")
    
    # ISTR combinations
    istr_normal_normal = find_image("istr_normal", "coco_normal")
    istr_normal_contrast = find_image("istr_normal", "coco_contrast")
    istr_contrast_normal = find_image("istr_contrast", "coco_normal")
    istr_contrast_contrast = find_image("istr_contrast", "coco_contrast")
    
    # Resize all images to same height (larger for better visibility)
    target_height = min(600, original_rgb.shape[0])  # Larger images for better detail
    
    def resize_to_height(img, h):
        if img is None:
            return np.ones((h, int(h * 1.5), 3), dtype=np.uint8) * 200
        old_h, old_w = img.shape[:2]
        new_w = int(old_w * (h / old_h))
        return cv2.resize(img, (new_w, h))
    
    # Add labels with smaller font for compact display
    def add_label(img, text):
        labeled = img.copy()
        label_height = 25
        label_bar = np.ones((label_height, labeled.shape[1], 3), dtype=np.uint8) * 50
        # Use smaller font and adjust text size to fit
        font_scale = 0.4
        cv2.putText(label_bar, text, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
        return np.vstack([label_bar, labeled])
    
    # Resize and label all images (12 total)
    spacing = 8
    
    # YOLO row
    yolo_nn = add_label(resize_to_height(yolo_normal_normal, target_height), "YOLO-N on Normal")
    yolo_nc = add_label(resize_to_height(yolo_normal_contrast, target_height), "YOLO-N on Contrast")
    yolo_cn = add_label(resize_to_height(yolo_contrast_normal, target_height), "YOLO-C on Normal")
    yolo_cc = add_label(resize_to_height(yolo_contrast_contrast, target_height), "YOLO-C on Contrast")
    
    # MaskRCNN row
    mask_nn = add_label(resize_to_height(maskrcnn_normal_normal, target_height), "Mask-N on Normal")
    mask_nc = add_label(resize_to_height(maskrcnn_normal_contrast, target_height), "Mask-N on Contrast")
    mask_cn = add_label(resize_to_height(maskrcnn_contrast_normal, target_height), "Mask-C on Normal")
    mask_cc = add_label(resize_to_height(maskrcnn_contrast_contrast, target_height), "Mask-C on Contrast")
    
    # ISTR row
    istr_nn = add_label(resize_to_height(istr_normal_normal, target_height), "ISTR-N on Normal")
    istr_nc = add_label(resize_to_height(istr_normal_contrast, target_height), "ISTR-N on Contrast")
    istr_cn = add_label(resize_to_height(istr_contrast_normal, target_height), "ISTR-C on Normal")
    istr_cc = add_label(resize_to_height(istr_contrast_contrast, target_height), "ISTR-C on Contrast")
    
    # Create horizontal spacing
    white_space = np.ones((yolo_nn.shape[0], spacing, 3), dtype=np.uint8) * 255
    
    # Create rows (4 images per row)
    row1 = np.hstack([yolo_nn, white_space, yolo_nc, white_space, yolo_cn, white_space, yolo_cc])
    row2 = np.hstack([mask_nn, white_space, mask_nc, white_space, mask_cn, white_space, mask_cc])
    row3 = np.hstack([istr_nn, white_space, istr_nc, white_space, istr_cn, white_space, istr_cc])
    
    # Stack rows vertically
    row_spacing = np.ones((spacing, row1.shape[1], 3), dtype=np.uint8) * 255
    model_results = np.vstack([row1, row_spacing, row2, row_spacing, row3])
    
    # Create left column: original + annotated
    original_resized = resize_to_height(original_rgb, target_height)
    annotated_resized = resize_to_height(annotated_rgb, target_height)
    
    original_labeled = add_label(original_resized, "Original")
    annotated_labeled = add_label(annotated_resized, "Ground Truth")
    
    # Create spacing for left column with correct width
    left_spacing = np.ones((spacing, original_labeled.shape[1], 3), dtype=np.uint8) * 255
    left_column = np.vstack([original_labeled, left_spacing, annotated_labeled])
    
    # Ensure heights match
    if left_column.shape[0] < model_results.shape[0]:
        padding = np.ones((model_results.shape[0] - left_column.shape[0], left_column.shape[1], 3), dtype=np.uint8) * 255
        left_column = np.vstack([left_column, padding])
    elif left_column.shape[0] > model_results.shape[0]:
        padding = np.ones((left_column.shape[0] - model_results.shape[0], model_results.shape[1], 3), dtype=np.uint8) * 255
        model_results = np.vstack([model_results, padding])
    
    # Combine left and right
    column_spacing = np.ones((left_column.shape[0], spacing * 3, 3), dtype=np.uint8) * 255
    gallery = np.hstack([left_column, column_spacing, model_results])
    
    # Add title bar
    title_height = 50
    title_bar = np.ones((title_height, gallery.shape[1], 3), dtype=np.uint8) * 240
    cv2.putText(title_bar, f"Model Comparison - {image_name}", (20, 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    gallery = np.vstack([title_bar, gallery])
    
    return gallery


def display_gallery(datasets, results, metadata=None):
    """Display gallery view for all images."""
    # Get metadata for labels if not provided
    if metadata is None:
        metadata = MetadataCatalog.get("comparison_coco_normal")
    
    # Get first dataset to iterate through (they should all have same images)
    first_dataset = datasets["coco_normal"]
    
    # Define class names for YOLO annotations
    yolo_class_names = {0: 'positive', 1: 'negative', 2: 'lines'}
    
    window_name = "Model Comparison Gallery - Press 'q' to quit, arrow keys to navigate"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    current_idx = 0
    
    while True:
        img_path, img_name = first_dataset[current_idx]
        
        # Read original image
        original_img = cv2.imread(img_path)
        
        # Try to load ground truth from YOLO format labels
        # Find corresponding image in YOLO normal dataset to get label path
        annotated_img = original_img.copy()
        has_gt = False
        
        for yolo_data in datasets.get("yolo_normal", []):
            if len(yolo_data) == 3:
                yolo_img_path, yolo_img_name, label_path = yolo_data
                # Match by filename (without extension)
                if Path(img_name).stem == Path(yolo_img_name).stem:
                    # Load and draw ground truth annotations (only if label file exists)
                    if Path(label_path).exists():
                        gt_annotations = read_yolo_annotations(label_path)
                        if gt_annotations:  # Only draw if there are annotations
                            annotated_img = draw_yolo_annotations(original_img.copy(), gt_annotations, yolo_class_names)
                            has_gt = True
                    break
        
        # If no ground truth, add text overlay
        if not has_gt:
            annotated_img = original_img.copy()
            cv2.putText(annotated_img, "No Ground Truth", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
        
        # Create gallery
        gallery = create_gallery_view(original_img, annotated_img, results, img_name)
        
        # Display
        cv2.imshow(window_name, gallery[:, :, ::-1])  # RGB to BGR
        
        # Wait for key
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q') or key == 27:  # q or ESC
            break
        elif key == 83 or key == ord('d'):  # Right arrow or d
            current_idx = (current_idx + 1) % len(first_dataset)
        elif key == 81 or key == ord('a'):  # Left arrow or a
            current_idx = (current_idx - 1) % len(first_dataset)
    
    cv2.destroyAllWindows()


# ============================================================================
# STATS TABLE EXPORT
# ============================================================================

def save_stats_table(stats):
    """Save statistics as Markdown and CSV."""
    output_dir = Path(CONFIG["output_dir"])
    
    # Save as CSV
    csv_path = output_dir / "inference_stats.csv"
    with open(csv_path, 'w', newline='') as f:
        if stats:
            writer = csv.DictWriter(f, fieldnames=stats[0].keys())
            writer.writeheader()
            writer.writerows(stats)
    print(f"\nStats saved to CSV: {csv_path}")
    
    # Save as Markdown
    md_path = output_dir / "inference_stats.md"
    with open(md_path, 'w') as f:
        f.write("# Model Inference Statistics\n\n")
        
        if stats:
            # Write table header
            headers = list(stats[0].keys())
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
            
            # Write rows
            for row in stats:
                f.write("| " + " | ".join(str(row[h]) for h in headers) + " |\n")
        
        # Calculate summary statistics
        f.write("\n## Summary Statistics\n\n")
        
        # Group by model and dataset
        summary = defaultdict(lambda: defaultdict(list))
        for row in stats:
            model = row["model"]
            dataset = row["dataset"]
            summary[model][dataset].append(float(row["inference_time_ms"]))
        
        f.write("| Model | Dataset | Avg Time (ms) | Min Time (ms) | Max Time (ms) | Num Images |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        
        # Sort models for consistent ordering
        for model in sorted(summary.keys()):
            for dataset in sorted(summary[model].keys()):
                times = summary[model][dataset]
                f.write(f"| {model} | {dataset} | {np.mean(times):.2f} | "
                       f"{np.min(times):.2f} | {np.max(times):.2f} | {len(times)} |\n")
    
    print(f"Stats saved to Markdown: {md_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare YOLO, Mask R-CNN, and ISTR models")
    parser.add_argument("--show", action="store_true", help="Display gallery view")
    parser.add_argument("--force-inference", action="store_true", help="Force rerun inference even if cached results exist")
    parser.add_argument("--load-only", action="store_true", help="Only load and display cached results, don't run inference")
    args = parser.parse_args()
    
    print("="*80)
    print("MODEL COMPARISON SCRIPT")
    print("="*80)
    
    output_dir = Path(CONFIG["output_dir"])
    results = None
    stats = None
    
    # Try to load cached results if not forcing inference
    if not args.force_inference:
        print("\nChecking for cached inference results...")
        results = load_inference_results(output_dir)
        
        if results:
            print(f"Found cached results in {output_dir}")
            # Load stats if they exist
            stats_csv = output_dir / "inference_stats.csv"
            if stats_csv.exists():
                print("Found cached statistics")
        else:
            print("No cached results found")
    
    # Run inference if needed
    if results is None or args.force_inference:
        if args.load_only:
            print("\nError: No cached results found and --load-only was specified")
            print("Run without --load-only to perform inference")
            return
        
        print("\nRunning inference...")
        # Load datasets
        datasets = load_datasets()
        
        # Run all inferences
        results, stats = run_all_inferences(datasets)
        
        # Save statistics
        save_stats_table(stats)
    else:
        print("\nUsing cached inference results")
        # Load datasets for display (we need original images)
        datasets = load_datasets()
    
    # Display gallery if requested
    if args.show:
        if results is None:
            print("\nError: No results available to display")
            return
        
        print("\nDisplaying gallery view...")
        print("Controls: Arrow keys or a/d to navigate, q/ESC to quit")
        # Get metadata from registered dataset for labels
        metadata = MetadataCatalog.get("comparison_coco_normal")
        display_gallery(datasets, results, metadata)
    
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)


if __name__ == "__main__":
    main()
