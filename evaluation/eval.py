"""
Comprehensive Multi-Model Evaluation Script with Gallery Visualization

This script provides a unified pipeline for evaluating multiple object detection and 
instance segmentation models (YOLO, Mask R-CNN, ISTR) with advanced visualization and 
comparison capabilities.

Note: Supports only COCO-format datasets for evaluation.

Features:
- Automatic evaluation of multiple models on multiple datasets
- COCO metrics (AP, AP50, AP75, per-class, scale-based)
- YOLO-style metrics (Precision, Recall, F1 @ IoU=0.5)
- Automatic visualization saving for all predictions
- Interactive gallery viewer for comparing models side-by-side
- Confusion matrices and PR curves
- Comprehensive Markdown and CSV reports
- Smart caching: skip evaluation if results exist
- Support for unlabeled images (automatic prediction)
- Prediction mode for inference on custom images

Configuration:
- All models and datasets are configured in the CONFIG dictionary at the top of the script
- Add/remove models or datasets by modifying the CONFIG dict
- Supports both YOLO and detectron2-based models (Mask R-CNN, ISTR, etc.)

Usage Examples:

    # Full evaluation with report generation
    python eval.py --output-dir eval_results
    
    # Evaluation + interactive gallery viewer
    python eval.py --gallery --output-dir eval_results
    
    # Force re-evaluation (ignore cached results)
    python eval.py --gallery --force-eval --output-dir eval_results
    
    # Gallery only (load existing results)
    python eval.py --gallery --output-dir eval_results
    
    # Predict on unlabeled images (no evaluation)
    python eval.py --predict-only --input /path/to/images/ --output-dir predictions
    python eval.py --predict-only --input img1.jpg img2.jpg img3.jpg --output-dir predictions
    
    # Custom confidence threshold for visualization
    python eval.py --confidence-threshold 0.7 --output-dir eval_results

Arguments:
    --gallery           Display interactive gallery viewer with keyboard navigation
    --force-eval        Force re-evaluation even if cached results exist
    --predict-only      Run inference on custom images without evaluation
    --input             Path(s) to images or directory (used with --predict-only)
    --output-dir        Directory to save all results (default: eval_results)
    --confidence-threshold  Minimum score for visualization (default: 0.5)
    --num_gpus          Number of GPUs to use (default: 1)
    --opts              Additional config options (detectron2 format)

Automatic Features:
- Unlabeled images in ../shared/unlabeled_images/ are automatically predicted during evaluation
- Visualizations are saved automatically for gallery viewing
- Results are cached - subsequent runs load from disk unless --force-eval is used

Gallery Controls:
    Arrow Keys / A,D  - Navigate between images
    Q / ESC          - Quit gallery viewer
    
Important Note:
- The gallery viewer ONLY displays results from full evaluation runs (with ground truth)
- --predict-only mode saves annotated images to disk but does NOT launch the gallery
- To view predict-only results, browse the output directory manually or use an image viewer

Output Structure:
    output_dir/
        comprehensive_evaluation_report.md  # Detailed Markdown report
        evaluation_summary.csv              # Metrics in CSV format
        confusion_matrices_grid.png         # Grid of confusion matrices
        pr_curves_merged_*.png              # Precision-Recall curves
        visualizations/
            {model_key}/
                {dataset_name}/
                    *.png                    # Annotated predictions
                unlabeled/
                    *.png                    # Predictions on unlabeled images
        {model_run_name}/
            inference/
                {dataset_name}/
                    coco_instances_results.json  # COCO format predictions

Configuration Example:
    CONFIG = {
        "models": {
            "maskrcnn_normal": {
                "name": "Mask R-CNN R50-FPN",
                "type": "detectron2",
                "weights": "../shared/models/model_final.pth",
                "evaluate_on": ["big-images-rev", "big-images-rev-contrast"]
            }
        },
        "datasets": {
            "big-images-rev": {
                "path": "../shared/coco-big-images-rev",
                "format": "coco"
            },
            "unlabeled": {
                "path": "../shared/unlabeled_images",
                "format": "unlabeled"
            }
        }
    }
"""

import os
import csv
import cv2
import glob
import time
import json
import yaml
import logging
import argparse
import numpy as np
from tqdm import tqdm
import csv as csv_module
from pathlib import Path
from datetime import datetime
from collections import OrderedDict, defaultdict

import torch
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_test_loader, DatasetCatalog
from detectron2.data.detection_utils import read_image
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, print_csv_format, DatasetEvaluator
from detectron2.utils.logger import setup_logger, log_first_n
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.structures import Boxes, Instances

from istr import ISTRDatasetMapper, add_ISTR_config
from detectron2.data.datasets import register_coco_instances

from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# ISTR imports
try:
    from istr import add_ISTR_config
except ImportError:
    add_ISTR_config = None

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


# ============================================================================
# CONFIGURATION
# ============================================================================

def load_config(config_path=None):
    """Load configuration from YAML file, fallback to hardcoded config."""
    if config_path is None:
        config_path = Path("config.yaml")
    else:
        config_path = Path(config_path)
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            print(f"Loaded configuration from {config_path}")
            return config
    except Exception as e:
        raise RuntimeError(f"Error loading configuration from {config_path}: {e}")


class Trainer(DefaultTrainer):
    """Trainer class with custom evaluator."""

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)


class YOLOWrapper:
    """
    Wrapper to make YOLO model compatible with detectron2 inference API.
    
    This wrapper converts YOLO predictions to detectron2's Instances format,
    allowing YOLO models to be evaluated using detectron2's COCOEvaluator.
    
    The conversion maps:
    - YOLO boxes (xyxy) -> detectron2 Boxes
    - YOLO masks -> detectron2 pred_masks
    - YOLO confidence scores -> detectron2 scores
    - YOLO class predictions -> detectron2 pred_classes
    """
    
    def __init__(self, weights_path):
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics package is required for YOLO models. Install with: pip install ultralytics")
        self.model = YOLO(weights_path)
        self.model.eval()
    
    def __call__(self, batched_inputs):
        """
        Run YOLO inference and convert to detectron2 format.
        
        Args:
            batched_inputs: list of dicts with keys "image", "height", "width"
        
        Returns:
            list of dicts with "instances" key containing detectron2 Instances
        """
        outputs = []
        
        for input_dict in batched_inputs:
            # Get image tensor and convert to numpy (HWC, RGB)
            image_tensor = input_dict["image"]
            if isinstance(image_tensor, torch.Tensor):
                # Convert from CHW to HWC and to numpy
                image_np = image_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            else:
                image_np = image_tensor
            
            # Run YOLO inference
            results = self.model(image_np, verbose=False)[0]
            
            # Convert YOLO results to detectron2 Instances
            instances = Instances((input_dict["height"], input_dict["width"]))
            
            if results.masks is not None and len(results.boxes) > 0:
                # Extract boxes, scores, classes, and masks
                boxes = results.boxes.xyxy.cpu()  # x1, y1, x2, y2
                scores = results.boxes.conf.cpu()
                classes = results.boxes.cls.cpu().long()
                
                # Extract masks (already in image coordinates)
                masks = results.masks.data.cpu()  # [N, H, W]
                
                # Create detectron2 Instances
                instances.pred_boxes = Boxes(boxes)
                instances.scores = scores
                instances.pred_classes = classes
                instances.pred_masks = masks
            else:
                # No detections
                instances.pred_boxes = Boxes(torch.zeros((0, 4)))
                instances.scores = torch.zeros(0)
                instances.pred_classes = torch.zeros(0, dtype=torch.long)
                instances.pred_masks = torch.zeros((0, input_dict["height"], input_dict["width"]))
            
            outputs.append({"instances": instances})
        
        return outputs
    
    def eval(self):
        """Set model to eval mode."""
        self.model.eval()
        return self


def register_all_datasets():
    """Register all datasets from CONFIG configuration."""
    for dataset_key, dataset_info in CONFIG["datasets"].items():
        # Only register COCO format datasets
        if dataset_info.get("format") != "coco":
            continue
            
        dataset_path = dataset_info["path"]
        test_path = os.path.join(dataset_path, "test")
        
        # Register the test dataset
        dataset_name = f"{dataset_key}-test"
        try:
            # Try to register, skip if already registered
            register_coco_instances(
                dataset_name,
                {"thing_classes": ["positive", "negative", "lines"]},
                os.path.join(test_path, "_annotations.coco.json"),
                test_path
            )
        except AssertionError:
            # Dataset already registered, skip
            pass


def setup_for_model(model_key, args):
    """
    Create configs for a specific model.
    
    Args:
        model_key: Key of the model in CONFIG['models'] dict
        args: Command line arguments
    
    Returns:
        cfg: Configuration object (or dict for YOLO models)
    """
    model_config = CONFIG["models"][model_key]
    
    if model_config["type"] == "yolo":
        # For YOLO models, create a minimal config dict
        cfg_dict = {
            "MODEL": {"WEIGHTS": model_config["weights"]},
            "DATASETS": {"TEST": tuple([f"{dataset_key}-test" for dataset_key in model_config["evaluate_on"]])},
            "OUTPUT_DIR": os.path.join(args.output_dir, model_config["run_name"]),
            "type": "yolo"
        }
        os.makedirs(cfg_dict["OUTPUT_DIR"], exist_ok=True)
        return cfg_dict
    else:
        # For detectron2 and ISTR models
        cfg = get_cfg()
        if model_config["type"] == "istr":
            add_ISTR_config(cfg)
        
        cfg.merge_from_file(model_config["config"])
        cfg.merge_from_list(args.opts)
        
        # Set model weights
        cfg.MODEL.WEIGHTS = model_config["weights"]
        
        # Set test datasets based on model's evaluate_on list
        test_datasets = [f"{dataset_key}-test" for dataset_key in model_config["evaluate_on"]]
        cfg.DATASETS.TEST = tuple(test_datasets)
        
        # Set output directory for this model
        cfg.OUTPUT_DIR = os.path.join(args.output_dir, model_config["run_name"])
        
        cfg.freeze()
        default_setup(cfg, args)
        return cfg


def compute_yolo_metrics(gt_json_path, pred_json_path, iou_threshold=0.50):
    """
    Compute YOLO-style precision, recall, F1 for mask predictions.
    
    Args:
        gt_json_path: Path to ground truth COCO JSON
        pred_json_path: Path to predictions COCO JSON
        iou_threshold: IoU threshold for matching (default 0.5)
    
    Returns:
        dict: Metrics including precision, recall, F1, TP, FP, FN
    """
    def compute_mask_iou(rle1, rle2):
        """Compute IoU between two RLE masks."""
        mask1 = maskUtils.decode(rle1)
        mask2 = maskUtils.decode(rle2)
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union > 0 else 0.0
    
    def seg_to_rle(seg, h, w):
        """Convert segmentation to RLE format."""
        if isinstance(seg, dict):
            # Already in RLE format
            if isinstance(seg.get('counts'), list):
                # Uncompressed RLE, need to encode
                return maskUtils.frPyObjects(seg, h, w)
            return seg
        elif isinstance(seg, list):
            # Could be polygon or uncompressed RLE
            if len(seg) > 0:
                if isinstance(seg[0], list):
                    # List of polygons - each polygon is a list of coordinates
                    rles = maskUtils.frPyObjects(seg, h, w)
                    return maskUtils.merge(rles)
                else:
                    # Single polygon as flat list
                    rles = maskUtils.frPyObjects([seg], h, w)
                    return rles if isinstance(rles, dict) else rles[0]
        return None
    
    # Load ground truth and predictions
    with open(gt_json_path, 'r') as f:
        gt_data = json.load(f)
    
    with open(pred_json_path, 'r') as f:
        pred_data = json.load(f)
    
    # Build image id to dimensions mapping
    image_info = {img['id']: (img['height'], img['width']) for img in gt_data['images']}
    
    # Group by image_id and category_id
    preds_by_img_cat = defaultdict(list)
    gts_by_img_cat = defaultdict(list)
    
    for pred in pred_data:
        key = (pred['image_id'], pred['category_id'])
        preds_by_img_cat[key].append(pred)
    
    for gt in gt_data['annotations']:
        key = (gt['image_id'], gt['category_id'])
        gts_by_img_cat[key].append(gt)
    
    TP = 0
    FP = 0
    total_gt = len(gt_data['annotations'])
    
    # Process each image-category combination
    for key in preds_by_img_cat.keys():
        preds = preds_by_img_cat[key]
        gts = gts_by_img_cat.get(key, [])
        image_id = key[0]
        
        if not gts:
            # All predictions are false positives
            FP += len(preds)
            continue
        
        # Get image dimensions
        h, w = image_info.get(image_id, (1024, 1024))
        
        # Sort predictions by score (descending)
        preds = sorted(preds, key=lambda x: x.get('score', 1.0), reverse=True)
        
        matched_gts = set()
        
        for pred in preds:
            best_iou = 0
            best_gt_idx = -1
            
            pred_seg = pred.get('segmentation')
            if pred_seg is None:
                FP += 1
                continue
            
            # Convert to RLE
            try:
                pred_rle = seg_to_rle(pred_seg, h, w)
                if pred_rle is None:
                    FP += 1
                    continue
            except Exception:
                FP += 1
                continue
            
            # Find best matching GT
            for gt_idx, gt in enumerate(gts):
                if gt_idx in matched_gts:
                    continue
                
                gt_seg = gt.get('segmentation')
                if gt_seg is None:
                    continue
                
                # Convert to RLE
                try:
                    gt_rle = seg_to_rle(gt_seg, h, w)
                    if gt_rle is None:
                        continue
                except Exception:
                    continue
                
                iou = compute_mask_iou(pred_rle, gt_rle)
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Check if match is good enough
            if best_iou >= iou_threshold:
                TP += 1
                matched_gts.add(best_gt_idx)
            else:
                FP += 1
    
    FN = total_gt - TP
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision_50": round(precision * 100, 2),  # Convert to percentage
        "recall_50": round(recall * 100, 2),  # Convert to percentage
        "f1_50": round(f1 * 100, 2),  # Convert to percentage
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "total_gt": total_gt  # Total ground truth instances in dataset
    }


def predict_on_images(model, cfg, image_paths, output_dir, args):
    """
    Run inference on unlabeled images and save visualizations.
    
    Args:
        model: trained model (YOLOWrapper or detectron2 model)
        cfg: config object or dict (for YOLO)
        image_paths: list of image file paths
        output_dir: directory to save predictions
        args: command line arguments
    """
    import logging
    logger = logging.getLogger(__name__)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Running predictions on {len(image_paths)} images...")
    logger.info(f"Saving results to {output_path}")
    
    # Handle YOLO models
    if isinstance(model, YOLOWrapper):
        for img_path in tqdm(image_paths, desc="Predicting"):
            img_name = Path(img_path).name
            
            # Run YOLO inference
            results = model.model(img_path, conf=args.confidence_threshold, verbose=False)
            annotated_img = results[0].plot()  # BGR format
            annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            
            # Save
            save_path = output_path / img_name
            cv2.imwrite(str(save_path), cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))
    else:
        # Handle detectron2/ISTR models
        model.eval()  # Ensure model is in eval mode
        
        # Get metadata from first registered dataset or create minimal metadata
        try:
            metadata = MetadataCatalog.get(list(cfg.DATASETS.TEST)[0]) if cfg.DATASETS.TEST else None
        except:
            metadata = None
        
        for img_path in tqdm(image_paths, desc="Predicting"):
            img_name = Path(img_path).name
            
            # Load image
            img = read_image(str(img_path), format="RGB")
            
            # Run inference
            with torch.no_grad():
                predictions = model([{"image": torch.as_tensor(img.transpose(2, 0, 1))}])
            
            # Visualize
            visualizer = Visualizer(img, metadata=metadata, instance_mode=ColorMode.IMAGE)
            if "instances" in predictions[0]:
                instances = predictions[0]["instances"].to("cpu")
                # Filter by confidence threshold
                if hasattr(instances, 'scores'):
                    mask = instances.scores >= args.confidence_threshold
                    instances = instances[mask]
                vis_output = visualizer.draw_instance_predictions(instances)
            else:
                vis_output = visualizer.output
            
            annotated_rgb = vis_output.get_image()
            
            # Save
            save_path = output_path / img_name
            cv2.imwrite(str(save_path), cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))
    
    logger.info(f"Predictions saved to {output_path}")

def create_coco_annotations(image_path, coco_json_path, class_names=None, show_masks=True, show_boxes=True):
    """
    Overlay ground truth annotations on images.

    Args:
        image_path: Path to the image file
        coco_json_path: Path to the COCO JSON annotation file
        class_names: List of class names (default: ["positive", "negative", "lines"])
        show_masks: Whether to show segmentation masks
        show_boxes: Whether to show bounding boxes

    Returns:
        numpy.ndarray: Annotated image in RGB format
    """
    if class_names is None:
        class_names = ["positive", "negative", "lines"]
    
    # Define colors for each class (RGB format) - scale to 0-255
    colors_normalized = sns.color_palette("husl", len(class_names))
    colors = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors_normalized]

    # Load image
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Load COCO annotations
    coco = COCO(str(coco_json_path))

    # Find image in COCO dataset
    img_name = image_path.name
    img_ids = []
    for img_id, img_info in coco.imgs.items():
        if Path(img_info['file_name']).name == img_name:
            img_ids.append(img_id)
            break
    
    if not img_ids:
        print(f"Warning: Image {img_name} not found in COCO annotations")
        return img_rgb
    
    img_id = img_ids[0]
    
    # Get annotations for this image
    ann_ids = coco.getAnnIds(imgIds=[img_id])
    anns = coco.loadAnns(ann_ids)
    
    # Create overlay for masks
    overlay = img_rgb.copy()
    
    # Draw each annotation
    for ann in anns:
        category_id = ann['category_id']
        
        # Ensure category_id is within range
        if category_id >= len(colors):
            color = (128, 128, 128)  # Gray for unknown classes
        else:
            color = colors[category_id]
        
        # Draw segmentation mask
        if show_masks and 'segmentation' in ann:
            seg = ann['segmentation']
            
            # Handle different segmentation formats
            if isinstance(seg, list):
                # Polygon format
                for polygon in seg:
                    if len(polygon) >= 6:  # At least 3 points (x,y pairs)
                        pts = np.array(polygon).reshape(-1, 2).astype(np.int32)
                        cv2.fillPoly(overlay, [pts], color)
            elif isinstance(seg, dict):
                # RLE format
                if isinstance(seg.get('counts'), list):
                    # Uncompressed RLE
                    h, w = ann.get('height', img_rgb.shape[0]), ann.get('width', img_rgb.shape[1])
                    rle = maskUtils.frPyObjects(seg, h, w)
                else:
                    rle = seg
                
                mask = maskUtils.decode(rle)
                overlay[mask > 0] = color
        
        # Draw bounding box
        if show_boxes and 'bbox' in ann:
            x, y, w, h = ann['bbox']
            x, y, w, h = int(x), int(y), int(w), int(h)
            cv2.rectangle(img_rgb, (x, y), (x + w, y + h), color, 2)
    
    # Blend overlay with original image
    if show_masks:
        alpha = 0.4
        img_rgb = cv2.addWeighted(overlay, alpha, img_rgb, 1 - alpha, 0)
    
    return img_rgb

def save_visualizations(model, cfg, dataset_name, output_dir, args):
    """
    Save visualized predictions for gallery view.
    
    Args:
        model: trained model (YOLOWrapper or detectron2 model)
        cfg: config object or dict (for YOLO)
        dataset_name: name of dataset
        output_dir: directory to save visualizations
        args: command line arguments
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # output_dir already contains the model_key path: {output_dir}/visualizations/{model_key}
    output_path = Path(output_dir) / dataset_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get dataset images
    dataset_dict = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)
    
    # Create ground truth visualization directory (save once per dataset)
    gt_output_path = Path(output_dir).parent / "ground_truth" / dataset_name
    gt_output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving visualizations to {output_path}...")
    logger.info(f"Saving ground truth to {gt_output_path}...")
    
    # Handle YOLO models (check if it's a YOLOWrapper)
    if isinstance(model, YOLOWrapper):
        for item in tqdm(dataset_dict, desc=f"Visualizing {dataset_name}"):
            img_path = item["file_name"]
            img_name = Path(img_path).stem + ".png"
            
            # Extract base dataset name (remove '-test' suffix)
            base_dataset_name = dataset_name.replace('-test', '')
            dataset_path = CONFIG["datasets"][base_dataset_name]["path"]
            coco_annotation_path = f"{dataset_path}/test/_annotations.coco.json"

            # Save ground truth visualization (only if not already saved)
            gt_save_path = gt_output_path / img_name
            if not gt_save_path.exists():
                gt_rgb = create_coco_annotations(image_path=Path(img_path),
                                        coco_json_path=coco_annotation_path,
                                        class_names=metadata.get("thing_classes", ["positive", "negative", "lines"]))
                cv2.imwrite(str(gt_save_path), cv2.cvtColor(gt_rgb, cv2.COLOR_RGB2BGR))
            
            # Run YOLO inference - use the wrapped model
            results = model.model(img_path, conf=args.confidence_threshold, verbose=False)
            annotated_img = results[0].plot()  # BGR format
            annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            
            # Save prediction as PNG
            save_path = output_path / img_name
            cv2.imwrite(str(save_path), cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))
    else:
        # Handle detectron2/ISTR models
        model.eval()  # Ensure model is in eval mode
        for item in tqdm(dataset_dict, desc=f"Visualizing {dataset_name}"):
            img_path = item["file_name"]
            img_name = Path(img_path).stem + ".png"
            
            # Extract base dataset name (remove '-test' suffix)
            base_dataset_name = dataset_name.replace('-test', '')
            dataset_path = CONFIG["datasets"][base_dataset_name]["path"]
            coco_annotation_path = f"{dataset_path}/test/_annotations.coco.json"
            
            # Load image
            img = read_image(img_path, format="RGB")
            
            # Save ground truth visualization (only if not already saved)
            gt_save_path = gt_output_path / img_name
            if not gt_save_path.exists():
                gt_rgb = create_coco_annotations(image_path=Path(img_path),
                                        coco_json_path=coco_annotation_path,
                                        class_names=metadata.get("thing_classes", ["positive", "negative", "lines"]))
                cv2.imwrite(str(gt_save_path), cv2.cvtColor(gt_rgb, cv2.COLOR_RGB2BGR))
            
            # Run inference
            with torch.no_grad():
                predictions = model([{"image": torch.as_tensor(img.transpose(2, 0, 1))}])
            
            # Visualize predictions
            visualizer = Visualizer(img, metadata=metadata, instance_mode=ColorMode.IMAGE)
            if "instances" in predictions[0]:
                instances = predictions[0]["instances"].to("cpu")
                # Filter by confidence threshold for visualization
                if hasattr(instances, 'scores'):
                    mask = instances.scores >= args.confidence_threshold
                    instances = instances[mask]
                vis_output = visualizer.draw_instance_predictions(instances)
            else:
                vis_output = visualizer.output
            
            annotated_rgb = vis_output.get_image()
            
            # Save prediction as PNG
            save_path = output_path / img_name
            cv2.imwrite(str(save_path), cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))


def do_evaluation(cfg, model, args):
    """
    Run evaluation on the test dataset.
    
    Args:
        cfg: config object or dict (for YOLO)
        model: trained model
        args: command line arguments
    
    Returns:
        dict: evaluation results (COCO metrics + YOLO-style metrics + inference times)
    """
    results = {}
    logger = logging.getLogger(__name__)
    
    # Handle YOLO models (cfg is a dict)
    if isinstance(cfg, dict):
        datasets_test = cfg["DATASETS"]["TEST"]
        output_dir = cfg["OUTPUT_DIR"]
        
        for dataset_name in datasets_test:
            # Build data loader using a minimal cfg
            temp_cfg = get_cfg()
            temp_cfg.DATASETS.TEST = (dataset_name,)
            temp_cfg.DATALOADER.NUM_WORKERS = 2
            temp_cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            
            data_loader = build_detection_test_loader(temp_cfg, dataset_name)
            evaluator = COCOEvaluator(
                dataset_name, 
                output_dir=os.path.join(output_dir, "inference", dataset_name)
            )
            
            # Track total inference time for the dataset
            start_time = time.time()
            results_i = inference_on_dataset(model, data_loader, evaluator)
            total_inference_time = time.time() - start_time
            
            # Get dataset size for average calculation
            dataset_dict = DatasetCatalog.get(dataset_name)
            num_images = len(dataset_dict) if dataset_dict else 0
            
            # Store timing statistics
            if num_images > 0:
                avg_time_seconds = total_inference_time / num_images
                timing_stats = {
                    'total_time': total_inference_time,
                    'avg_time_per_image_ms': avg_time_seconds * 1000,  # Convert to milliseconds
                    'images_per_second': num_images / total_inference_time,  # Hz
                    'num_images': num_images
                }
                results_i['inference_times'] = timing_stats
                
                if comm.is_main_process():
                    logger.info(f"\nInference Timing:")
                    logger.info(f"  Total time: {timing_stats['total_time']:.2f}s")
                    logger.info(f"  Avg per image: {timing_stats['avg_time_per_image_ms']:.2f}ms")
                    logger.info(f"  Throughput: {timing_stats['images_per_second']:.2f} images/s")
                    logger.info(f"  Images: {timing_stats['num_images']}")
            
            # Compute YOLO-style metrics
            inference_dir = os.path.join(output_dir, "inference", dataset_name)
            pred_json = os.path.join(inference_dir, "coco_instances_results.json")
            
            # Get ground truth JSON path
            dataset_dict = DatasetCatalog.get(dataset_name)
            if dataset_dict and len(dataset_dict) > 0:
                # Extract GT path from dataset metadata
                for dataset_key, dataset_info in CONFIG["datasets"].items():
                    if f"{dataset_key}-test" == dataset_name:
                        gt_json = os.path.join(dataset_info["path"], "test", "_annotations.coco.json")
                        if os.path.exists(pred_json) and os.path.exists(gt_json):
                            yolo_metrics = compute_yolo_metrics(gt_json, pred_json)
                            results_i['yolo_metrics'] = yolo_metrics
                            
                            if comm.is_main_process():
                                logger.info(f"\nYOLO-style Metrics @ IoU=0.5:")
                                logger.info(f"  Precision: {yolo_metrics['precision_50']:.3f}")
                                logger.info(f"  Recall:    {yolo_metrics['recall_50']:.3f}")
                                logger.info(f"  F1-Score:  {yolo_metrics['f1_50']:.3f}")
                                logger.info(f"  TP: {yolo_metrics['TP']}, FP: {yolo_metrics['FP']}, FN: {yolo_metrics['FN']}\n")
                        break
            
            results[dataset_name] = results_i
            if comm.is_main_process():
                print_csv_format(results_i)
    else:
        # Handle detectron2/ISTR models (cfg is CfgNode)
        for dataset_name in cfg.DATASETS.TEST:
            data_loader = build_detection_test_loader(cfg, dataset_name)
            evaluator = Trainer.build_evaluator(
                cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            )
            
            # Track total inference time for the dataset
            start_time = time.time()
            results_i = inference_on_dataset(model, data_loader, evaluator)
            total_inference_time = time.time() - start_time
            
            # Get dataset size for average calculation
            dataset_dict = DatasetCatalog.get(dataset_name)
            num_images = len(dataset_dict) if dataset_dict else 0
            
            # Store timing statistics
            if num_images > 0:
                avg_time_seconds = total_inference_time / num_images
                timing_stats = {
                    'total_time': total_inference_time,
                    'avg_time_per_image_ms': avg_time_seconds * 1000,  # Convert to milliseconds
                    'images_per_second': num_images / total_inference_time,  # Hz
                    'num_images': num_images
                }
                results_i['inference_times'] = timing_stats
                
                if comm.is_main_process():
                    logger.info(f"\nInference Timing:")
                    logger.info(f"  Total time: {timing_stats['total_time']:.2f}s")
                    logger.info(f"  Avg per image: {timing_stats['avg_time_per_image_ms']:.2f}ms")
                    logger.info(f"  Throughput: {timing_stats['images_per_second']:.2f} images/s")
                    logger.info(f"  Images: {timing_stats['num_images']}")
            
            # Compute YOLO-style metrics
            inference_dir = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            pred_json = os.path.join(inference_dir, "coco_instances_results.json")
            
            # Get ground truth JSON path
            for dataset_key, dataset_info in CONFIG["datasets"].items():
                if f"{dataset_key}-test" == dataset_name:
                    gt_json = os.path.join(dataset_info["path"], "test", "_annotations.coco.json")
                    if os.path.exists(pred_json) and os.path.exists(gt_json):
                        yolo_metrics = compute_yolo_metrics(gt_json, pred_json)
                        results_i['yolo_metrics'] = yolo_metrics
                        
                        if comm.is_main_process():
                            logger.info(f"\nYOLO-style Metrics @ IoU=0.5:")
                            logger.info(f"  Precision: {yolo_metrics['precision_50']:.3f}")
                            logger.info(f"  Recall:    {yolo_metrics['recall_50']:.3f}")
                            logger.info(f"  F1-Score:  {yolo_metrics['f1_50']:.3f}")
                            logger.info(f"  TP: {yolo_metrics['TP']}, FP: {yolo_metrics['FP']}, FN: {yolo_metrics['FN']}\n")
                    break
            
            results[dataset_name] = results_i
            if comm.is_main_process():
                print_csv_format(results_i)
    
    return results


# ============================================================================
# REPORTING
# ============================================================================


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


def generate_confusion_matrices(yolo_metrics, coco_predictions, coco_map, output_dir):
    """Generate confusion matrices in a 3-column grid layout.
    
    Creates a proper (num_classes+1) x (num_classes+1) confusion matrix that includes:
    - Rows (True Labels): Positive, Negative, Lines, Background
    - Columns (Predicted): Positive, Negative, Lines, Background
    
    Where Background row captures False Positives (predictions with no matching GT)
    and Background column captures False Negatives (GT with no matching prediction).
    
    IMPORTANT: This uses class-aware matching (same as YOLO metrics):
    - A prediction only matches a GT if they have the SAME class AND IoU >= 0.5
    - Class confusion is NOT counted (prediction of class A cannot match GT of class B)
    - This ensures consistency between confusion matrix diagonal and YOLO TP count
    """
    output_dir = Path(output_dir)
    confusion_paths = []
    all_matrices = {}  # Store all matrices for grid layout
    all_debug_data = {}  # Store debug data for JSON export
    
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
        
        # Build confusion matrix for 3 classes + background
        # Rows: True labels (Positive, Negative, Lines, Background)
        # Columns: Predicted labels (Positive, Negative, Lines, Background)
        # Background row = False Positives (predicted object where there's no GT of same class)
        # Background column = False Negatives (GT object that was not detected)
        num_classes = 3
        background_idx = num_classes  # index for background class
        matrix_size = num_classes + 1  # 4x4 matrix
        cm = np.zeros((matrix_size, matrix_size), dtype=int)
        
        # Debug counters
        debug_info = {
            'model': model_key,
            'dataset': dataset_name,
            'total_predictions': len(pred_boxes),
            'total_gt': len(coco_gt.anns),
            'predictions_by_class': {0: 0, 1: 0, 2: 0},
            'gt_by_class': {0: 0, 1: 0, 2: 0},
            'tp_by_class': {0: 0, 1: 0, 2: 0},
            'fp_by_class': {0: 0, 1: 0, 2: 0},
            'fn_by_class': {0: 0, 1: 0, 2: 0},
            'matched_details': [],
            'unmatched_preds': [],
            'unmatched_gts': [],
        }
        
        # Group predictions by image AND category (class-aware matching like YOLO metrics)
        pred_by_img_cat = defaultdict(list)
        for pred in pred_boxes:
            cat_id = pred.get('category_id', -1)
            if 0 <= cat_id < num_classes:
                pred_by_img_cat[(pred['image_id'], cat_id)].append(pred)
                debug_info['predictions_by_class'][cat_id] += 1
        
        # Group GT by image AND category
        gt_by_img_cat = defaultdict(list)
        for ann_id in coco_gt.anns:
            ann = coco_gt.anns[ann_id]
            cat_id = ann['category_id']
            if 0 <= cat_id < num_classes:
                gt_by_img_cat[(ann['image_id'], cat_id)].append(ann)
                debug_info['gt_by_class'][cat_id] += 1
        
        # Also group GT just by image for FN tracking
        gt_by_img = defaultdict(list)
        for ann_id in coco_gt.anns:
            ann = coco_gt.anns[ann_id]
            gt_by_img[ann['image_id']].append(ann)
        
        # Get all (image_id, category_id) keys from both predictions and GT
        all_keys = set(pred_by_img_cat.keys()) | set(gt_by_img_cat.keys())
        
        # Compute IoU matching for confusion matrix using masks
        def compute_iou(pred, gt, img_id):
            """Compute IoU using masks if available, fallback to bbox."""
            pred_seg = pred.get('segmentation')
            
            # Get GT segmentation
            gt_seg = None
            if 'segmentation' in gt:
                if isinstance(gt['segmentation'], list):
                    # Polygon format - convert to RLE
                    img_info = coco_gt.imgs[img_id]
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
            y2 = min(b1[1] + b1[3], b2[1] + b2[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            union = b1[2] * b1[3] + b2[2] * b2[3] - inter
            return inter / union if union > 0 else 0
        
        matched_gt_global = set()  # Track all matched GT across all classes
        
        # Process each (image_id, category_id) combination - CLASS-AWARE MATCHING
        # This ensures a prediction only matches GT of the SAME class
        for (img_id, cat_id) in all_keys:
            preds = pred_by_img_cat.get((img_id, cat_id), [])
            gts = gt_by_img_cat.get((img_id, cat_id), [])
            
            # Sort predictions by score (descending) for greedy matching
            preds = sorted(preds, key=lambda x: x.get('score', 1.0), reverse=True)
            
            matched_gts_local = set()  # Track matched GT within this (img, cat) group
            
            for pred in preds:
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt in enumerate(gts):
                    if gt_idx in matched_gts_local:
                        continue  # Already matched
                    
                    iou = compute_iou(pred, gt, img_id)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou >= 0.5 and best_gt_idx >= 0:
                    # True Positive: correct class AND IoU >= 0.5
                    matched_gts_local.add(best_gt_idx)
                    matched_gt_global.add((img_id, cat_id, best_gt_idx))
                    cm[cat_id, cat_id] += 1  # On diagonal (same class)
                    debug_info['tp_by_class'][cat_id] += 1
                    debug_info['matched_details'].append({
                        'image_id': int(img_id),
                        'category_id': int(cat_id),
                        'iou': round(best_iou, 3),
                        'pred_score': round(pred.get('score', 1.0), 3)
                    })
                else:
                    # False Positive: no matching GT of same class with IoU >= 0.5
                    cm[background_idx, cat_id] += 1  # Background row, predicted class column
                    debug_info['fp_by_class'][cat_id] += 1
                    debug_info['unmatched_preds'].append({
                        'image_id': int(img_id),
                        'category_id': int(cat_id),
                        'best_iou': round(best_iou, 3),
                        'pred_score': round(pred.get('score', 1.0), 3)
                    })
        
        # Count unmatched GT instances as False Negatives
        for img_id, gts in gt_by_img.items():
            for gt_idx, gt in enumerate(gts):
                gt_class = gt['category_id']
                if not (0 <= gt_class < num_classes):
                    continue
                gt_key = (img_id, gt_class, gt_idx)
                if gt_key not in matched_gt_global:
                    # False Negative: GT with no matching prediction
                    cm[gt_class, background_idx] += 1
                    debug_info['fn_by_class'][gt_class] += 1
                    debug_info['unmatched_gts'].append({
                        'image_id': int(img_id),
                        'category_id': int(gt_class),
                        'gt_idx': int(gt_idx)
                    })
        
        # Compute summary metrics from confusion matrix
        diagonal_sum = sum(cm[i, i] for i in range(num_classes))
        fp_sum = sum(cm[background_idx, i] for i in range(num_classes))
        fn_sum = sum(cm[i, background_idx] for i in range(num_classes))
        
        debug_info['cm_diagonal_sum'] = int(diagonal_sum)
        debug_info['cm_fp_sum'] = int(fp_sum)
        debug_info['cm_fn_sum'] = int(fn_sum)
        debug_info['confusion_matrix'] = cm.tolist()
        
        # Store debug data
        all_debug_data[key] = debug_info
        
        print(f"  {model_key} on {dataset_name}:")
        print(f"    Total GT: {debug_info['total_gt']}, Total Preds: {debug_info['total_predictions']}")
        print(f"    CM Diagonal (TP): {diagonal_sum}, FP: {fp_sum}, FN: {fn_sum}")
        print(f"    TP by class: {debug_info['tp_by_class']}")
        print(f"    FP by class: {debug_info['fp_by_class']}")
        print(f"    FN by class: {debug_info['fn_by_class']}")
        
        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(7, 6))
        
        class_names = ['Positive', 'Negative', 'Lines', 'Background']
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
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
        
        # Flatten axes array for easier indexing
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        class_names = ['Pos', 'Neg', 'Lines', 'BG']
        
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
    
    # Export debug data to JSON for analysis
    if all_debug_data:
        debug_json_path = output_dir / 'confusion_matrix_debug.json'
        with open(debug_json_path, 'w') as f:
            json.dump(all_debug_data, f, indent=2)
        print(f"Exported confusion matrix debug data to: {debug_json_path}")
    
    print(f"Generated confusion matrix grid with {len(all_matrices)} matrices")
    return confusion_paths


def generate_comprehensive_report(all_results, output_dir, args):
    """Generate comprehensive Markdown report with COCO evaluation metrics."""
    import logging
    logger = logging.getLogger(__name__)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect COCO predictions and ground truth paths for PR curves
    coco_predictions = {}
    coco_map = {}
    
    for model_key, dataset_results in all_results.items():
        model_config = CONFIG["models"][model_key]
        run_name = model_config["run_name"]
        
        for dataset_name in dataset_results.keys():
            # Build prediction path
            pred_path = output_dir / run_name / "inference" / dataset_name / "coco_instances_results.json"
            if pred_path.exists():
                dataset_simple = dataset_name.replace("-test", "")
                coco_predictions[f"{model_key}__{dataset_simple}"] = pred_path
            
            # Build GT path and COCO object
            for dataset_key, dataset_info in CONFIG["datasets"].items():
                if f"{dataset_key}-test" == dataset_name and dataset_info.get("format") == "coco":
                    gt_json = Path(dataset_info["path"]) / "test" / "_annotations.coco.json"
                    if gt_json.exists():
                        dataset_simple = dataset_name.replace("-test", "")
                        if dataset_simple not in coco_map:
                            coco_map[dataset_simple] = {
                                'gt_path': gt_json,
                                'coco': COCO(str(gt_json))
                            }
    
    # Collect YOLO metrics for confusion matrices
    yolo_metrics = {}
    for model_key, dataset_results in all_results.items():
        for dataset_name, metrics in dataset_results.items():
            if 'yolo_metrics' in metrics:
                dataset_simple = dataset_name.replace("-test", "")
                yolo_metrics[f"{model_key}__{dataset_simple}"] = metrics['yolo_metrics']
    
    # Generate all plots with error handling
    try:
        confusion_paths = generate_confusion_matrices(yolo_metrics, coco_predictions, coco_map, output_dir)
    except Exception as e:
        logger.warning(f"Error generating confusion matrices: {e}")
        confusion_paths = []
    
    try:
        pr_curve_paths = generate_pr_curves(coco_predictions, coco_map, output_dir)
    except Exception as e:
        logger.warning(f"Error generating PR curves: {e}")
        pr_curve_paths = []
    
    # Format COCO evaluation results from all_results structure
    coco_results = []
    yolo_metrics_list = []
    
    logger.info("\nFormatting COCO evaluation results...")
    for model_key, dataset_results in all_results.items():
        model_name = CONFIG["models"][model_key]["name"]
        
        for dataset_name, metrics in dataset_results.items():
            dataset_display = dataset_name.replace("-test", "")
            
            # Extract bbox metrics
            if 'bbox' in metrics:
                bbox_result = {
                    'model': model_key,
                    'model_key': model_key,
                    'dataset': dataset_display,
                    'iou_type': 'bbox'
                }
                bbox_metrics = metrics['bbox']
                bbox_result['AP'] = bbox_metrics.get('AP', -2)
                bbox_result['AP50'] = bbox_metrics.get('AP50', -2)
                bbox_result['AP75'] = bbox_metrics.get('AP75', -2)
                bbox_result['APs'] = bbox_metrics.get('APs', -2)
                bbox_result['APm'] = bbox_metrics.get('APm', -2)
                bbox_result['APl'] = bbox_metrics.get('APl', -2)
                bbox_result['AP-positive'] = bbox_metrics.get('AP-positive', -2)
                bbox_result['AP-negative'] = bbox_metrics.get('AP-negative', -2)
                bbox_result['AP-lines'] = bbox_metrics.get('AP-lines', -2)
                coco_results.append(bbox_result)
            
            # Extract segmentation metrics
            if 'segm' in metrics:
                segm_result = {
                    'model': model_key,
                    'model_key': model_key,
                    'dataset': dataset_display,
                    'iou_type': 'segm'
                }
                segm_metrics = metrics['segm']
                segm_result['AP'] = segm_metrics.get('AP', -2)
                segm_result['AP50'] = segm_metrics.get('AP50', -2)
                segm_result['AP75'] = segm_metrics.get('AP75', -2)
                segm_result['APs'] = segm_metrics.get('APs', -2)
                segm_result['APm'] = segm_metrics.get('APm', -2)
                segm_result['APl'] = segm_metrics.get('APl', -2)
                segm_result['AP-positive'] = segm_metrics.get('AP-positive', -2)
                segm_result['AP-negative'] = segm_metrics.get('AP-negative', -2)
                segm_result['AP-lines'] = segm_metrics.get('AP-lines', -2)
                coco_results.append(segm_result)
            
            # Extract YOLO metrics
            if 'yolo_metrics' in metrics:
                yolo = metrics['yolo_metrics']
                yolo_result = {
                    'model': model_key,
                    'model_key': model_key,
                    'dataset': dataset_display,
                    'precision_50': yolo.get('precision_50', -2),
                    'recall_50': yolo.get('recall_50', -2),
                    'f1_50': yolo.get('f1_50', -2),
                    'TP': yolo.get('TP', -2),
                    'FP': yolo.get('FP', -2),
                    'FN': yolo.get('FN', -2),
                    'total_gt': yolo.get('total_gt', -2),
                }
                yolo_metrics_list.append(yolo_result)
    
    # Separate bbox and segm results
    bbox_results = [r for r in coco_results if r.get('iou_type') == 'bbox']
    segm_results = [r for r in coco_results if r.get('iou_type') == 'segm']
    
    logger.info(f"Formatted {len(bbox_results)} bbox results and {len(segm_results)} segm results")
    
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
            
            # Use model key (nickname) as the main heading, include run name and human-readable name as details
            f.write(f"### {model_key}\n\n")
            f.write(f"- **Model**: {model_name}\n")
            f.write(f"- **Run name**: {run_name}\n")
            f.write(f"- **Type**: {model_type}\n")
            f.write(f"- **Epoch**: {epoch}\n")
            f.write(f"- **Weights**: {weights}\n")
            f.write(f"- **Evaluated on**: {datasets_eval}\n")
            f.write("\n")
        
        # Table of Contents
        f.write("## Table of Contents\n\n")
        f.write("1. [Models Evaluated](#models-evaluated)\n")
        f.write("2. [COCO Evaluation Metrics](#coco-evaluation-metrics)\n")
        f.write("3. [Summary](#summary)\n\n")
        
        # COCO Evaluation Metrics
        f.write("## 2. COCO Evaluation Metrics\n\n")
        f.write("*Evaluation performed using Detectron2's official COCOEvaluator (matches eval.py and train_net.py)*\n\n")
        
        # BBox Metrics
        f.write("### BBox Detection Metrics\n\n")
        if bbox_results:
            headers = ["Model", "Dataset", "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AP-positive", "AP-negative", "AP-lines"]
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
            
            for res in bbox_results:
                row = [
                    res.get('model', ''),
                    res.get('dataset', ''),
                    f"{res.get('AP', -1):.3f}",
                    f"{res.get('AP50', -1):.3f}",
                    f"{res.get('AP75', -1):.3f}",
                    f"{res.get('APs', -1):.3f}",
                    f"{res.get('APm', -1):.3f}",
                    f"{res.get('APl', -1):.3f}",
                    f"{res.get('AP-positive', -1):.3f}",
                    f"{res.get('AP-negative', -1):.3f}",
                    f"{res.get('AP-lines', -1):.3f}",
                ]
                f.write("| " + " | ".join(row) + " |\n")
            f.write("\n")
        else:
            f.write("*No bbox evaluation results available.*\n\n")
        
        # Segmentation Metrics
        f.write("### Instance Segmentation Metrics\n\n")
        if segm_results:
            headers = ["Model", "Dataset", "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AP-positive", "AP-negative", "AP-lines"]
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
            
            for res in segm_results:
                row = [
                    res.get('model', ''),
                    res.get('dataset', ''),
                    f"{res.get('AP', -1):.3f}",
                    f"{res.get('AP50', -1):.3f}",
                    f"{res.get('AP75', -1):.3f}",
                    f"{res.get('APs', -1):.3f}",
                    f"{res.get('APm', -1):.3f}",
                    f"{res.get('APl', -1):.3f}",
                    f"{res.get('AP-positive', -1):.3f}",
                    f"{res.get('AP-negative', -1):.3f}",
                    f"{res.get('AP-lines', -1):.3f}",
                ]
                f.write("| " + " | ".join(row) + " |\n")
            f.write("\n")
        else:
            f.write("*No segmentation evaluation results available.*\n\n")
        
        # Inference Timing
        f.write("\n## Inference Timing\n\n")
        f.write("*Inference time statistics per model/dataset combination (in seconds)*\n\n")
        
        # Collect timing data
        timing_data = []
        for model_key, dataset_results in all_results.items():
            model_name = CONFIG["models"][model_key]["name"]
            for dataset_name, metrics in dataset_results.items():
                if 'inference_times' in metrics:
                    timing = metrics['inference_times']
                    timing_data.append({
                        'model': model_key,
                        'dataset': dataset_name.replace('-test', ''),
                        'total': timing['total_time'],
                        'avg_ms': timing['avg_time_per_image_ms'],
                        'hz': timing['images_per_second'],
                        'images': timing['num_images']
                    })
        
        if timing_data:
            headers = ["Model", "Dataset", "Total Time (s)", "Avg per Image (ms)", "Throughput (img/s)", "Images"]
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
            
            for t in timing_data:
                row = [
                    t['model'],
                    t['dataset'],
                    f"{t['total']:.2f}",
                    f"{t['avg_ms']:.2f}",
                    f"{t['hz']:.2f}",
                    str(t['images'])
                ]
                f.write("| " + " | ".join(row) + " |\n")
            f.write("\n")
        else:
            f.write("*No timing data available.*\n\n")
        
        # YOLO-Style Metrics
        f.write("\n## YOLO-Style Metrics @ IoU=0.5\n\n")
        f.write("*Precision, Recall, and F1 Score computed at IoU threshold = 0.5 (in %)*\n\n")
        
        if yolo_metrics_list:
            f.write("### Metrics Table\n\n")
            headers = ["Model", "Dataset", "Precision (%)", "Recall (%)", "F1 (%)", "TP", "FP", "FN", "Total GT"]
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
            
            for m in yolo_metrics_list:
                row = [
                    m.get('model', ''),
                    m.get('dataset', ''),
                    f"{m.get('precision_50', -1):.2f}",
                    f"{m.get('recall_50', -1):.2f}",
                    f"{m.get('f1_50', -1):.2f}",
                    str(m.get('TP', -1)),
                    str(m.get('FP', -1)),
                    str(m.get('FN', -1)),
                    str(m.get('total_gt', -1)),
                ]
                f.write("| " + " | ".join(row) + " |\n")
            f.write("\n")
            
            # Best performers
            if yolo_metrics_list:
                f.write("### BBox Metrics Table (IoU@0.5)\n\n")
                headers = ["Model", "Dataset", "Precision", "Recall", "F1", "TP", "FP", "FN"]
                f.write("### Top Performers\n\n")
                best_f1 = max(yolo_metrics_list, key=lambda x: x.get('f1_50', -1))
                best_precision = max(yolo_metrics_list, key=lambda x: x.get('precision_50', -1))
                best_recall = max(yolo_metrics_list, key=lambda x: x.get('recall_50', -1))
                
                f.write(f"- **Best F1 Score**: {best_f1['model']} on {best_f1['dataset']} ({best_f1.get('f1_50', -1):.2f}%)\n")
                f.write(f"- **Best Precision**: {best_precision['model']} on {best_precision['dataset']} ({best_precision.get('precision_50', -1):.2f}%)\n")
                f.write(f"- **Best Recall**: {best_recall['model']} on {best_recall['dataset']} ({best_recall.get('recall_50', -1):.2f}%)\n\n")
        else:
            f.write("*No YOLO-style metrics available.*\n\n")
        
        # Visualizations
        f.write("\n## Visualizations\n\n")
        
        if confusion_paths:
            f.write("### Confusion Matrices\n\n")
            # Only include the grid confusion matrix
            grid_cm = next((cm_path for name, cm_path in confusion_paths if name == 'grid'), None)
            if grid_cm:
                f.write(f"![Confusion Matrix Grid]({grid_cm.name})\n\n")
        
        if pr_curve_paths:
            f.write("### Precision-Recall Curves\n\n")
            for name, pr_path in pr_curve_paths:
                f.write(f"![PR Curves: {name}]({pr_path.name})\n\n")
        
        # Summary Table
        f.write("\n## Summary Table\n\n")
        f.write("*Overview of all model performances across datasets (values in percentage)*\n\n")
        
        if segm_results or bbox_results:
            # Build summary table combining all metrics
            summary_rows = []
            for model_key in CONFIG['models'].keys():
                model_name = CONFIG['models'][model_key]['name']
                for dataset_key in CONFIG['models'][model_key].get('evaluate_on', []):
                    row = {'Model': model_key, 'Dataset': dataset_key}
                    
                    # Find segm results
                    segm = next((r for r in segm_results if r['model_key'] == model_key and r['dataset'] == dataset_key), None)
                    if segm:
                        row['AP'] = f"{segm.get('AP', -1):.1f}"
                        row['AP50'] = f"{segm.get('AP50', -1):.1f}"
                        row['AP75'] = f"{segm.get('AP75', -1):.1f}"
                    
                    # Find yolo metrics
                    yolo = next((r for r in yolo_metrics_list if r['model_key'] == model_key and r['dataset'] == dataset_key), None)
                    if yolo:
                        row['P@0.5'] = f"{yolo.get('precision_50', -1):.1f}"
                        row['R@0.5'] = f"{yolo.get('recall_50', -1):.1f}"
                        row['F1@0.5'] = f"{yolo.get('f1_50', -1):.1f}"
                    
                    summary_rows.append(row)
            
            if summary_rows:
                headers = ["Model", "Dataset", "AP", "AP50", "AP75", "P@0.5", "R@0.5", "F1@0.5"]
                f.write("| " + " | ".join(headers) + " |\n")
                f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
                
                for row in summary_rows:
                    cells = [row.get(h, 'N/A') for h in headers]
                    f.write("| " + " | ".join(cells) + " |\n")
                f.write("\n")
        
        # Summary
        f.write("\n---\n\n")
        f.write("## Summary\n\n")
        if coco_results:
            best_ap = max(coco_results, key=lambda x: x.get('AP', -1))
            f.write(f"**Best Overall AP (COCO)**: {best_ap['model']} on {best_ap['dataset']} "
                   f"({best_ap.get('AP', -1):.3f}%)\n\n")
        
        if yolo_metrics_list:
            best_f1 = max(yolo_metrics_list, key=lambda x: x.get('f1_50', -1))
            f.write(f"**Best Overall F1 (YOLO-style)**: {best_f1['model']} on {best_f1['dataset']} "
                   f"({best_f1.get('f1_50', -1):.2f}%)\n\n")
        
        # Metrics Glossary
        f.write("\n---\n\n")
        f.write("## Metrics Glossary\n\n")
        f.write("### COCO Evaluation Metrics\n\n")
        f.write("COCO (Common Objects in Context) metrics are the standard evaluation metrics for object detection and instance segmentation tasks. "
               "All COCO metrics are computed over multiple IoU (Intersection over Union) thresholds.\n\n")
        
        f.write("**Primary Metrics:**\n\n")
        f.write("- **AP** (Average Precision): Primary challenge metric. Mean Average Precision (mAP) averaged over IoU thresholds from 0.50 to 0.95 with a step size of 0.05 (10 IoU thresholds). This is the most comprehensive metric.\n")
        f.write("- **AP50**: Average Precision at IoU=0.50. More lenient metric that requires 50% overlap between prediction and ground truth.\n")
        f.write("- **AP75**: Average Precision at IoU=0.75. Stricter metric that requires 75% overlap, testing localization accuracy.\n\n")
        
        f.write("**Scale-based Metrics:**\n\n")
        f.write("- **AP_small (APs)**: AP for small objects (area < 32² pixels). Measures performance on small instances.\n")
        f.write("- **AP_medium (APm)**: AP for medium objects (32² < area < 96²). Measures performance on medium-sized instances.\n")
        f.write("- **AP_large (APl)**: AP for large objects (area > 96²). Measures performance on large instances.\n\n")
        
        f.write("**Per-Class Metrics:**\n\n")
        f.write("- **AP-positive**: Average Precision for the 'positive' class across all IoU thresholds.\n")
        f.write("- **AP-negative**: Average Precision for the 'negative' class across all IoU thresholds.\n")
        f.write("- **AP-lines**: Average Precision for the 'lines' class across all IoU thresholds.\n\n")
        
        f.write("**Recall Metrics:**\n\n")
        f.write("- **AR1**: Average Recall given 1 detection per image. Maximum recall given a fixed number of detections.\n")
        f.write("- **AR10**: Average Recall given 10 detections per image.\n")
        f.write("- **AR100**: Average Recall given 100 detections per image. This is the standard recall metric, representing maximum recall.\n")
        f.write("- **ARs**: Average Recall for small objects.\n")
        f.write("- **ARm**: Average Recall for medium objects.\n")
        f.write("- **ARl**: Average Recall for large objects.\n\n")
        
        f.write("### YOLO-Style Metrics @ IoU=0.5\n\n")
        f.write("These metrics provide a simpler, single-threshold evaluation at IoU=0.5, making them easier to interpret "
               "and commonly used for quick model comparison.\n\n")
        
        f.write("**Core Metrics:**\n\n")
        f.write("- **Precision (P@0.5)**: Percentage of correct predictions out of all predictions made. Formula: `TP / (TP + FP)`. "
               "High precision means few false alarms.\n")
        f.write("- **Recall (R@0.5)**: Percentage of ground truth objects that were correctly detected. Formula: `TP / (TP + FN)`. "
               "High recall means few missed detections.\n")
        f.write("- **F1 Score (F1@0.5)**: Harmonic mean of precision and recall. Formula: `2 × (P × R) / (P + R)`. "
               "Balances precision and recall into a single score.\n\n")
        
        f.write("**Detection Counts:**\n\n")
        f.write("- **TP** (True Positives): Number of correct detections (predicted instances matching ground truth with IoU ≥ 0.5).\n")
        f.write("- **FP** (False Positives): Number of incorrect detections (predicted instances with no matching ground truth or IoU < 0.5).\n")
        f.write("- **FN** (False Negatives): Number of missed ground truth instances (ground truth objects that were not detected).\n\n")
        
        f.write("### Key Differences\n\n")
        f.write("- **COCO metrics** (AP) are averaged over multiple IoU thresholds (0.5 to 0.95), providing a more comprehensive evaluation.\n")
        f.write("- **YOLO metrics** use a single IoU threshold (0.5), making them simpler to interpret but less comprehensive.\n")
        f.write("- **AP** is generally more robust and preferred for academic evaluation, while **Precision/Recall/F1** are intuitive for practical applications.\n")
        f.write("- COCO's **AP** emphasizes localization quality more than single-threshold metrics.\n\n")
    
    logger.info(f"\nComprehensive report saved to: {md_path}")
    
    # Save CSV
    csv_path = output_dir / "evaluation_summary.csv"
    csv_headers = ["Model", "Dataset", "Type", "AP", "AP50", "AP75", "AP-positive", "AP-negative", "AP-lines", "P@0.5", "R@0.5", "F1@0.5", "TP", "FP", "FN"]
    
    # Combine COCO and YOLO metrics for CSV
    csv_data = []
    for model_key, dataset_results in all_results.items():
        model_name = CONFIG["models"][model_key]["name"]
        model_config = CONFIG["models"][model_key]
        
        for dataset_name, metrics in dataset_results.items():
            dataset_display = dataset_name.replace("-test", "")
            
            row = {
                "Model": model_key,
                "Dataset": dataset_display,
                "Type": model_config["type"],
            }
            
            # COCO metrics
            if "segm" in metrics:
                row["AP"] = f"{metrics['segm']['AP']:.1f}"
                row["AP50"] = f"{metrics['segm']['AP50']:.1f}"
                row["AP75"] = f"{metrics['segm']['AP75']:.1f}"
                row["AP-positive"] = f"{metrics['segm'].get('AP-positive', -1):.1f}"
                row["AP-negative"] = f"{metrics['segm'].get('AP-negative', -1):.1f}"
                row["AP-lines"] = f"{metrics['segm'].get('AP-lines', -1):.1f}"
            elif "bbox" in metrics:
                row["AP"] = f"{metrics['bbox']['AP']:.1f}"
                row["AP50"] = f"{metrics['bbox']['AP50']:.1f}"
                row["AP75"] = f"{metrics['bbox']['AP75']:.1f}"
                row["AP-positive"] = f"{metrics['bbox'].get('AP-positive', -1):.1f}"
                row["AP-negative"] = f"{metrics['bbox'].get('AP-negative', -1):.1f}"
                row["AP-lines"] = f"{metrics['bbox'].get('AP-lines', -1):.1f}"
            
            # YOLO metrics
            if "yolo_metrics" in metrics:
                yolo = metrics["yolo_metrics"]
                row["P@0.5"] = f"{yolo['precision_50']:.1f}"
                row["R@0.5"] = f"{yolo['recall_50']:.1f}"
                row["F1@0.5"] = f"{yolo['f1_50']:.1f}"
                row["TP"] = yolo['TP']
                row["FP"] = yolo['FP']
                row["FN"] = yolo['FN']
            
            csv_data.append(row)
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(csv_data)
    
    logger.info(f"CSV summary saved to: {csv_path}")
    
    return md_path, csv_path


def load_visualization_results(output_dir):
    """Load previously saved visualization results."""
    output_dir = Path(output_dir) / "visualizations"
    
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
            
            # Load all images
            for img_path in sorted(dataset_dir.iterdir()):
                if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        results[model_key][dataset_name].append((img_path.name, img_rgb))
    
    if not results:
        return None
    
    return dict(results)


def create_gallery_view(results, image_name, original_img=None, gt_img=None):
    """
    Create gallery view for model comparison.
    Shows original image on top, annotated prediction below, with 2 models per row.
    
    Args:
        results: Dict of loaded visualization results {model_key: {dataset_name: [(name, img)]}}
        image_name: Name of image to display
        original_img: Original image (optional)
        gt_img: Ground truth annotated image (optional)
    
    Returns:
        numpy array: Gallery image
    """
    def find_image(model_key, dataset_key):
        """Find image for specific model/dataset combination."""
        if model_key not in results or dataset_key not in results[model_key]:
            return None
        for img_name, img in results[model_key][dataset_key]:
            if img_name == image_name or Path(img_name).stem == Path(image_name).stem:
                return img
        return None
    
    def create_color_legend(width, height=60):
        """Create a color legend for the three classes."""
        legend = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Define class colors using seaborn husl palette (matching ground truth)
        class_names = ["Positive", "Negative", "Lines"]
        colors_normalized = sns.color_palette("husl", len(class_names))
        colors_rgb = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors_normalized]
        
        classes = [
            (class_names[0], colors_rgb[0]),
            (class_names[1], colors_rgb[1]),
            (class_names[2], colors_rgb[2])
        ]
        
        # Calculate spacing
        box_size = 20
        start_x = 10
        spacing_x = width // 3
        y_pos = height // 2 - box_size // 2
        
        for i, (class_name, color) in enumerate(classes):
            x_pos = start_x + i * spacing_x
            
            # Draw colored box
            cv2.rectangle(legend, (x_pos, y_pos), (x_pos + box_size, y_pos + box_size), color, -1)
            cv2.rectangle(legend, (x_pos, y_pos), (x_pos + box_size, y_pos + box_size), (0, 0, 0), 1)
            
            # Draw text label
            cv2.putText(legend, class_name, (x_pos + box_size + 5, y_pos + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        return legend
    
    # Organize results by model and dataset
    # Structure: {model_key: {dataset_key: img}}
    model_dataset_grid = {}
    all_models = []
    all_datasets = []
    
    for model_key, model_info in CONFIG["models"].items():
        if model_key not in results:
            continue
        
        if model_key not in model_dataset_grid:
            model_dataset_grid[model_key] = {}
            all_models.append(model_key)
            
        for dataset_key in model_info.get("evaluate_on", []):
            dataset_test_name = f"{dataset_key}-test"
            
            if dataset_test_name not in results[model_key]:
                continue
                
            img = find_image(model_key, dataset_test_name)
            if img is None:
                continue
            
            model_dataset_grid[model_key][dataset_key] = img
            if dataset_key not in all_datasets:
                all_datasets.append(dataset_key)
        
        # Also check for unlabeled images
        if "unlabeled" in results[model_key]:
            unlabeled_img = find_image(model_key, "unlabeled")
            if unlabeled_img is not None:
                model_dataset_grid[model_key]["unlabeled"] = unlabeled_img
                if "unlabeled" not in all_datasets:
                    all_datasets.append("unlabeled")
    
    if not model_dataset_grid:
        print("No visualization results found")
        return None
    
    # Keep images at original size - no resizing
    spacing = 20
    
    def add_label(img, text):
        """Add label bar to image."""
        if img is None:
            return None
        labeled = img.copy()
        label_height = 30
        label_bar = np.ones((label_height, labeled.shape[1], 3), dtype=np.uint8) * 50
        cv2.putText(label_bar, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return np.vstack([label_bar, labeled])
    
    # Keep original sizes - just organize the grid
    processed_grid = {}
    for model_key in all_models:
        processed_grid[model_key] = {}
        for dataset_key in all_datasets:
            if dataset_key in model_dataset_grid[model_key]:
                img = model_dataset_grid[model_key][dataset_key]
                processed_grid[model_key][dataset_key] = img
            else:
                # Empty cell
                processed_grid[model_key][dataset_key] = None
    
    # Prepare first column: Original and GT at original size
    if original_img is None:
        # Use first available image size as reference
        ref_img = None
        for model_key in all_models:
            for dataset_key in all_datasets:
                if dataset_key in model_dataset_grid[model_key]:
                    ref_img = model_dataset_grid[model_key][dataset_key]
                    break
            if ref_img is not None:
                break
        if ref_img is not None:
            original_img = np.ones_like(ref_img) * 200
        else:
            original_img = np.ones((600, 800, 3), dtype=np.uint8) * 200
    
    if gt_img is None:
        gt_img = np.ones_like(original_img) * 200
    
    # Build the grid table
    # Column 1: Original (top) and GT (bottom)
    # Other columns: Rows of 2 models each, each model showing all dataset variants horizontally
    
    # Create first column with Original and GT stacked vertically
    orig_labeled = add_label(original_img, "Original")
    gt_labeled = add_label(gt_img, "Ground Truth")
    
    # Stack them vertically with spacing
    vert_spacing = np.ones((spacing * 3, orig_labeled.shape[1], 3), dtype=np.uint8) * 255
    first_column = np.vstack([orig_labeled, vert_spacing, gt_labeled])
    
    # Build model rows (2 models per row)
    models_per_row = 2
    model_rows = []
    # Get reference image height for spacing
    ref_height = None
    for model_key in all_models:
        for dataset_key in all_datasets:
            if dataset_key in model_dataset_grid[model_key]:
                ref_height = processed_grid[model_key][dataset_key].shape[0] + 30
                break
        if ref_height is not None:
            break
    if ref_height is None:
        ref_height = 600
    white_space_horiz = np.ones((ref_height, spacing, 3), dtype=np.uint8) * 255
    
    for i in range(0, len(all_models), models_per_row):
        row_models = all_models[i:i+models_per_row]
        
        # Build each model's variants horizontally
        model_strips = []
        for model_key in row_models:
            # Get all dataset variants for this model
            variants = []
            for dataset_key in all_datasets:
                if dataset_key in model_dataset_grid[model_key]:
                    img = processed_grid[model_key][dataset_key]
                    model_short = model_key.replace("_", "-").upper()
                    dataset_short = dataset_key.replace("big-images-rev", "BIR").replace("-contrast", "-C")
                    labeled = add_label(img, f"{model_short}: {dataset_short}")
                    variants.append(labeled)
            
            # Stack variants horizontally
            if variants:
                model_strip = variants[0]
                for v in variants[1:]:
                    model_strip = np.hstack([model_strip, white_space_horiz, v])
                model_strips.append(model_strip)
        
        # If we have less than 2 models in this row, pad with empty space
        if len(model_strips) < models_per_row:
            empty = np.ones_like(model_strips[0]) * 255
            model_strips.append(empty)
        
        # Stack the two model strips horizontally with more spacing
        wider_spacing = np.ones((model_strips[0].shape[0], spacing * 3, 3), dtype=np.uint8) * 255
        row_content = model_strips[0]
        for strip in model_strips[1:]:
            row_content = np.hstack([row_content, wider_spacing, strip])
        
        model_rows.append(row_content)
    
    # Stack model rows vertically
    if model_rows:
        row_spacing_vert = np.ones((spacing * 2, model_rows[0].shape[1], 3), dtype=np.uint8) * 255
        right_side = model_rows[0]
        for row in model_rows[1:]:
            right_side = np.vstack([right_side, row_spacing_vert, row])
        
        # Ensure first column matches the height of right side
        if first_column.shape[0] < right_side.shape[0]:
            padding_height = right_side.shape[0] - first_column.shape[0]
            padding = np.ones((padding_height, first_column.shape[1], 3), dtype=np.uint8) * 255
            first_column = np.vstack([first_column, padding])
        elif first_column.shape[0] > right_side.shape[0]:
            padding_height = first_column.shape[0] - right_side.shape[0]
            padding = np.ones((padding_height, right_side.shape[1], 3), dtype=np.uint8) * 255
            right_side = np.vstack([right_side, padding])
        
        # Combine first column with model rows
        column_spacing = np.ones((first_column.shape[0], spacing * 3, 3), dtype=np.uint8) * 255
        gallery = np.hstack([first_column, column_spacing, right_side])
    else:
        gallery = first_column
    
    # Add title bar
    title_height = 50
    title_bar = np.ones((title_height, gallery.shape[1], 3), dtype=np.uint8) * 240
    cv2.putText(title_bar, f"Model Comparison - {Path(image_name).stem}", (20, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Add color legend at the bottom
    legend = create_color_legend(gallery.shape[1], height=60)
    
    # Combine all parts: title + gallery + legend
    gallery = np.vstack([title_bar, gallery, legend])
    
    return gallery


def display_gallery(args):
    """Display interactive gallery view of saved visualizations."""
    import logging
    logger = logging.getLogger(__name__)
    
    # Load visualization results
    logger.info("Loading visualization results...")
    results = load_visualization_results(args.output_dir)
    
    if results is None:
        logger.error(f"No visualization results found in {args.output_dir}/visualizations/")
        logger.error("Please run evaluation first without --gallery flag")
        return
    
    # Get list of images from first available dataset (excluding unlabeled)
    image_list = []
    for model_key in results:
        for dataset_name in results[model_key]:
            if dataset_name != "unlabeled":
                image_list = [name for name, _ in results[model_key][dataset_name]]
                if image_list:
                    break
        if image_list:
            break
    
    # Add unlabeled images to the list (if any exist)
    unlabeled_images = []
    for model_key in results:
        if "unlabeled" in results[model_key]:
            unlabeled_images = [name for name, _ in results[model_key]["unlabeled"]]
            break
    
    if unlabeled_images:
        logger.info(f"Found {len(unlabeled_images)} unlabeled images")
        image_list.extend(unlabeled_images)
    
    if not image_list:
        logger.error("No images found in visualization results")
        return
    
    logger.info(f"Found {len(image_list)} images")
    logger.info("Controls: Arrow keys or A/D to navigate, Q/ESC to quit")
    
    window_name = "Model Comparison Gallery - Press 'q' to quit, arrow keys to navigate"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    current_idx = 0
    
    while True:
        image_name = image_list[current_idx]
        
        # Check if this is an unlabeled image
        is_unlabeled = False
        for model_key in results:
            if "unlabeled" in results[model_key]:
                unlabeled_names = [name for name, _ in results[model_key]["unlabeled"]]
                if image_name in unlabeled_names:
                    is_unlabeled = True
                    break
        
        # Try to load original and GT images (skip for unlabeled)
        original_img = None
        gt_img = None
        image_name_original = image_name.replace(".png", ".jpg")

        
        # Find original image path and GT visualization
        if not is_unlabeled:
            for dataset_name, dataset_info in CONFIG["datasets"].items():
                if dataset_info.get("format") == "coco":
                    test_path = Path(dataset_info["path"]) / "test"
                    img_path = test_path / image_name_original
                    if img_path.exists():
                        # Load original image
                        original_img = cv2.imread(str(img_path))
                        if original_img is not None:
                            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                        
                        # Load ground truth visualization (with -test suffix to match save location)
                        gt_vis_dir = Path(args.output_dir) / "visualizations" / "ground_truth" / f"{dataset_name}-test"
                        gt_img_name = Path(image_name).stem + ".png"
                        gt_img_path = gt_vis_dir / gt_img_name
                        
                        logger.info(f"Looking for GT at: {gt_img_path}")
                        if gt_img_path.exists():
                            gt_img = cv2.imread(str(gt_img_path))
                            if gt_img is not None:
                                gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
                                logger.info(f"Successfully loaded GT image: {gt_img.shape}")
                            else:
                                logger.warning(f"Failed to read GT image from {gt_img_path}")
                        else:
                            logger.warning(f"GT image not found at {gt_img_path}")
                        break
                    else:
                        logger.warning(f"Original image not found: {img_path}")
        elif is_unlabeled:
            unlabeled_path = Path(CONFIG["datasets"]["unlabeled"]["path"])
            img_path = unlabeled_path / image_name_original
            if img_path.exists():
                original_img = cv2.imread(str(img_path))
                if original_img is not None:
                    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                else:
                    logger.warning(f"Failed to read unlabeled original image from {img_path}")
            else:
                logger.warning(f"Unlabeled original image not found: {img_path}")
        
        # Create gallery
        gallery = create_gallery_view(results, image_name, original_img, gt_img)
        
        if gallery is None:
            logger.error(f"Failed to create gallery for {image_name}")
            break
        
        # Display
        cv2.imshow(window_name, cv2.cvtColor(gallery, cv2.COLOR_RGB2BGR))
        
        # Handle keyboard input
        key = cv2.waitKey(0)
        key_code = key & 0xFF
        
        logger.info(f"Key pressed: {key} (masked: {key_code}) - Image {current_idx + 1}/{len(image_list)}: {image_name}")
        
        if key_code == ord('q') or key_code == 27:  # q or ESC
            break
        elif key_code == ord('d') or key == 65363 or key == 2555904:  # 'd' or Right arrow (different systems)
            old_idx = current_idx
            current_idx = (current_idx + 1) % len(image_list)
            logger.info(f"Moving forward from image {old_idx + 1} to {current_idx + 1}")
        elif key_code == ord('a') or key == 65361 or key == 2424832:  # 'a' or Left arrow (different systems)
            old_idx = current_idx
            current_idx = (current_idx - 1) % len(image_list)
            logger.info(f"Moving backward from image {old_idx + 1} to {current_idx + 1}")
    
    cv2.destroyAllWindows()
    logger.info("Gallery closed")


def print_summary_table(all_results, args):
    """Print a summary table of all evaluation results."""
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "="*100)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*100)
    
    # Collect all results
    summary_data = []
    
    for model_key, dataset_results in all_results.items():
        model_name = CONFIG["models"][model_key]["name"]
        
        for dataset_name, metrics in dataset_results.items():
            # Extract dataset display name
            dataset_display = dataset_name.replace("-test", "")
            
            row = {
                "Model": model_name,
                "Dataset": dataset_display,
            }
            
            # COCO metrics
            if "segm" in metrics:
                row["AP"] = f"{metrics['segm']['AP']:.1f}"
                row["AP50"] = f"{metrics['segm']['AP50']:.1f}"
                row["AP75"] = f"{metrics['segm']['AP75']:.1f}"
            elif "bbox" in metrics:
                row["AP"] = f"{metrics['bbox']['AP']:.1f}"
                row["AP50"] = f"{metrics['bbox']['AP50']:.1f}"
                row["AP75"] = f"{metrics['bbox']['AP75']:.1f}"
            
            # YOLO metrics
            if "yolo_metrics" in metrics:
                yolo = metrics["yolo_metrics"]
                row["P@0.5"] = f"{yolo['precision_50']:.3f}"
                row["R@0.5"] = f"{yolo['recall_50']:.3f}"
                row["F1@0.5"] = f"{yolo['f1_50']:.3f}"
            
            summary_data.append(row)
    
    if not summary_data:
        logger.info("No results to display")
        return
    
    # Print table header
    headers = ["Model", "Dataset", "AP", "AP50", "AP75", "P@0.5", "R@0.5", "F1@0.5"]
    col_widths = {h: max(len(h), max(len(str(row.get(h, ""))) for row in summary_data)) for h in headers}
    
    header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
    separator = "-+-".join("-" * col_widths[h] for h in headers)
    
    logger.info("\n" + header_line)
    logger.info(separator)
    
    # Print rows
    for row in summary_data:
        line = " | ".join(str(row.get(h, "N/A")).ljust(col_widths[h]) for h in headers)
        logger.info(line)
    
    logger.info("="*100 + "\n")
    
    # Generate comprehensive report
    generate_comprehensive_report(all_results, args.output_dir, args)


def check_evaluation_exists(args):
    """Check if evaluation results already exist.
    
    Returns:
        bool: True if all required files exist, False otherwise
    """
    output_dir = Path(args.output_dir)
    
    # Check if report exists
    report_path = output_dir / "comprehensive_evaluation_report.md"
    if not report_path.exists():
        return False
    
    # Check if visualizations exist for all models
    vis_dir = output_dir / "visualizations"
    if not vis_dir.exists():
        return False
    
    # Check each model has visualizations
    for model_key in CONFIG["models"].keys():
        model_vis_dir = vis_dir / model_key
        if not model_vis_dir.exists():
            return False
        
        # Check if at least one dataset has images
        has_images = False
        for dataset_dir in model_vis_dir.iterdir():
            if dataset_dir.is_dir() and any(dataset_dir.iterdir()):
                has_images = True
                break
        
        if not has_images:
            return False
    
    return True


def main(args):
    """Main evaluation function."""
    # Setup logger once for the entire run
    logger = setup_logger(name="eval")
    
    # Load configuration from specified file
    global CONFIG
    CONFIG = load_config(args.config)
    
    # Handle predict-only mode
    if args.predict_only:
        if not args.input:
            logger.error("--input is required when using --predict-only")
            return {}
        
        # Collect image paths
        image_paths = []
        for path_str in args.input:
            path = Path(path_str)
            if path.is_dir():
                # Add all images in directory
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                    image_paths.extend(path.glob(ext))
                    image_paths.extend(path.glob(ext.upper()))
            elif path.is_file():
                image_paths.append(path)
            else:
                logger.warning(f"Path not found: {path_str}")
        
        if not image_paths:
            logger.error("No valid images found")
            return {}
        
        logger.info(f"Found {len(image_paths)} images to process")
        
        # Process each model
        for model_key in CONFIG["models"].keys():
            model_config = CONFIG["models"][model_key]
            logger.info(f"\n{'='*80}")
            logger.info(f"Running predictions with: {model_config['name']} ({model_key})")
            logger.info(f"{'='*80}\n")
            
            # Setup configuration
            cfg = setup_for_model(model_key, args)
            
            # Build model
            if model_config["type"] == "yolo":
                model = YOLOWrapper(model_config["weights"])
            else:
                model = DefaultTrainer.build_model(cfg)
                DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
                model.eval()
            
            # Run predictions
            output_dir = Path(args.output_dir) / "predictions" / model_key
            predict_on_images(model, cfg, image_paths, output_dir, args)
            
            # Cleanup
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        logger.info(f"\nAll predictions saved to {args.output_dir}/predictions/")
        return {}
    
    # If gallery mode, check if evaluation exists
    if args.gallery:
        if not args.force_eval and check_evaluation_exists(args):
            logger.info("Evaluation results found. Loading gallery...")
            display_gallery(args)
            return {}
        else:
            if args.force_eval:
                logger.info("Force evaluation requested. Running evaluation...")
            else:
                logger.info("No evaluation results found. Running evaluation first...")
    
    # Register all datasets once at startup
    register_all_datasets()
    
    # Evaluate all models
    models_to_evaluate = list(CONFIG["models"].keys())
    
    all_results = {}
    
    # Iterate through each model
    for model_key in models_to_evaluate:
        model_config = CONFIG["models"][model_key]
        logger.info(f"\n{'='*80}")
        logger.info(f"Evaluating model: {model_config['name']} ({model_key})")
        logger.info(f"Weights: {model_config['weights']}")
        logger.info(f"Datasets: {model_config['evaluate_on']}")
        logger.info(f"Type: {model_config['type']}")
        logger.info(f"{'='*80}\n")
        
        # Setup configuration for this model
        cfg = setup_for_model(model_key, args)
        
        # Build model based on type
        if model_config["type"] == "yolo":
            # Load YOLO model using wrapper
            model = YOLOWrapper(model_config["weights"])
        else:
            # Build detectron2/ISTR model
            model = Trainer.build_model(cfg)
            
            # Load weights
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=False
            )
        
        # Run evaluation
        results = do_evaluation(cfg, model, args)
        all_results[model_key] = results
        
        # Always save visualizations (like compare_models_unified.py)
        logger.info("Saving visualizations...")
        for dataset_name in (cfg["DATASETS"]["TEST"] if isinstance(cfg, dict) else cfg.DATASETS.TEST):
            vis_output_dir = os.path.join(args.output_dir, "visualizations", model_key)
            save_visualizations(model, cfg, dataset_name, vis_output_dir, args)
        
        # Check for unlabeled dataset and predict on it
        unlabeled_config = CONFIG["datasets"].get("unlabeled")
        if unlabeled_config and unlabeled_config.get("format") == "unlabeled":
            unlabeled_path = Path(unlabeled_config["path"])
            if unlabeled_path.exists():
                # Collect all images
                image_paths = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                    image_paths.extend(unlabeled_path.glob(ext))
                    image_paths.extend(unlabeled_path.glob(ext.upper()))
                
                if image_paths:
                    logger.info(f"\nFound {len(image_paths)} unlabeled images. Running predictions...")
                    output_dir = Path(args.output_dir) / "visualizations" / model_key / "unlabeled"
                    predict_on_images(model, cfg, image_paths, output_dir, args)
        
        # Cleanup
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    
    # Print summary table and generate report
    print_summary_table(all_results, args)
    
    # If gallery mode was requested, show it now
    if args.gallery:
        logger.info("\nEvaluation complete. Opening gallery...")
        display_gallery(args)
    
    return all_results


if __name__ == "__main__":
    parser = default_argument_parser()
    
    # Custom arguments
    parser.add_argument(
        "--output-dir",
        default="eval_results",
        type=str,
        help="Base directory to save output files.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown (for visualization).",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for evaluation.",
    )
    parser.add_argument(
        "--gallery",
        action="store_true",
        help="Display interactive gallery view. If evaluation results exist, loads them. Otherwise, prompts to run evaluation first.",
    )
    parser.add_argument(
        "--force-eval",
        action="store_true",
        help="Force re-evaluation even if results exist. Useful with --gallery to refresh results.",
    )
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Run inference on unlabeled images without evaluation. Use with --input to specify images.",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        type=str,
        help="Path(s) to input images or directory for prediction (used with --predict-only).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file for the evaluation.",
    )
    
    args = parser.parse_args()
    
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
