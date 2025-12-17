#!/usr/bin/env python3
"""Export model predictions for images into COCO-format annotations.json

This script is modeled after `demo.py` but instead of showing
visualizations it saves predicted instance segmentations as COCO-style
annotations (polygons), bboxes and areas.

Note: The exported annotations will have category IDs corresponding to the
      model's predicted class IDs. You may need to map these to your desired
      category IDs depending on the dataset the model has used to train.
      See `categories` at line 247.

Usage examples:
  python export_coco_predictions.py --config-file configs/...yaml --coco input_annotations.json --output annotations.json
  python export_coco_predictions.py --config-file configs/...yaml --input 'images/*.jpg' --output annotations.json
"""
import argparse
import glob
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image

from predictor import VisualizationDemo


def setup_cfg(args):
    cfg = get_cfg()

    # Add ISTR config if in the config path
    config_path = args.config_file.lower()
    if "istr" in config_path:
        from ISTR.projects.ISTR.istr import add_ISTR_config
        add_ISTR_config(cfg)
        
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def masks_to_polygons(mask):
    """Convert a binary mask (H,W) to COCO polygon(s).

    Returns a list of polygons where each polygon is a flat list of floats [x1,y1,x2,y2,...].
    """
    # ensure uint8 0/255
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    if mask_u8.sum() == 0:
        return []

    # find contours
    contours_info = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]

    polys = []
    for contour in contours:
        if contour is None:
            continue
        if contour.size < 6:
            continue
        # flatten and convert to list of floats
        poly = contour.flatten().astype(float).tolist()
        # COCO expects polygons with >= 6 values (3 points)
        if len(poly) >= 6:
            polys.append(poly)
    return polys


def process_image(entry, demo, confidence_threshold, ann_start_id, cat_map):
    """Run model on a single image entry and produce COCO-style annotations list.

    entry: dict with keys `file_name`, `width`, `height`, `id` (image id)
    returns: (list of annotations, next_ann_id)
    """
    image_path = entry["file_name"]
    img = read_image(image_path, format="BGR")
    start = time.time()
    predictions, _ = demo.run_on_image(img, confidence_threshold)
    elapsed = time.time() - start

    anns = []
    if "instances" not in predictions:
        return anns, ann_start_id

    instances = predictions["instances"]
    # ensure on cpu
    instances = instances.to(torch.device("cpu"))

    # boxes
    if hasattr(instances, "pred_boxes"):
        boxes = instances.pred_boxes.tensor.numpy()
    else:
        boxes = None

    # classes
    if hasattr(instances, "pred_classes"):
        classes = instances.pred_classes.numpy()
    else:
        classes = np.zeros((0,), dtype=np.int64)

    # masks
    masks = None
    if hasattr(instances, "pred_masks"):
        masks_field = instances.pred_masks
        # detectron2's BitMasks object has .tensor, else torch.Tensor
        if hasattr(masks_field, "tensor"):
            masks = masks_field.tensor.numpy()
        elif isinstance(masks_field, torch.Tensor):
            masks = masks_field.numpy()
        else:
            try:
                masks = np.asarray(masks_field)
            except Exception:
                masks = None

    num = len(classes)
    for i in range(num):
        ann_id = ann_start_id
        ann_start_id += 1

        category_pred = int(classes[i])
        # allow mapping if provided, otherwise use predicted id
        category_id = cat_map.get(category_pred, category_pred)

        seg = []
        area = 0.0
        if masks is not None:
            mask = masks[i]
            polys = masks_to_polygons(mask)
            seg = polys
            area = float((mask > 0).sum())

        if boxes is not None:
            x1, y1, x2, y2 = boxes[i].tolist()
            w = x2 - x1
            h = y2 - y1
            bbox = [float(x1), float(y1), float(w), float(h)]
        else:
            bbox = [0.0, 0.0, 0.0, 0.0]

        ann = {
            "id": ann_id,
            "image_id": int(entry.get("id", 0)),
            "category_id": int(category_id),
            "iscrowd": 0,
            "segmentation": seg,
            "area": float(area),
            "bbox": bbox,
        }
        anns.append(ann)

    return anns, ann_start_id


def load_images_from_coco(coco_path, images_root=None):
    with open(coco_path, "r") as f:
        data = json.load(f)
    images = data.get("images", [])
    if images_root is not None:
        # update file_name to include root if not absolute
        for im in images:
            fn = im["file_name"]
            if not os.path.isabs(fn):
                im["file_name"] = os.path.join(images_root, fn)
    return images


def main():
    parser = argparse.ArgumentParser(description="Export predictions to COCO annotations.json")
    parser.add_argument("--config-file", required=True, help="path to config file")
    parser.add_argument("--coco", help="Path to input COCO JSON to copy images list from")
    parser.add_argument("--images-root", help="If using --coco, prefix image file_name with this root path")
    parser.add_argument(
        "--input",
        nargs="+",
        help="Image glob(s), directory path(s), or list of image files",
    )
    parser.add_argument("--output", required=True, help="Output annotations.json path")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--opts", default=[], nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)

    # build images list
    images = []
    if args.coco:
        images = load_images_from_coco(args.coco, args.images_root)
    elif args.input:
        # expand glob(s) and directories
        files = []
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        for pattern in args.input:
            p = os.path.expanduser(pattern)
            if os.path.isdir(p):
                # add files in directory with common image extensions
                for fname in sorted(os.listdir(p)):
                    if fname.lower().endswith(exts):
                        files.append(os.path.join(p, fname))
            else:
                files.extend(glob.glob(p))
        files = sorted(files)
        # construct image entries
        for idx, f in enumerate(files):
            h_w = (0, 0)
            try:
                im = cv2.imread(f)
                if im is not None:
                    h, w = im.shape[:2]
                else:
                    w = h = 0
            except Exception:
                w = h = 0
            images.append({"file_name": f, "width": int(w), "height": int(h), "id": idx + 1})
    else:
        raise RuntimeError("Either --coco or --input must be provided")

    # Prepare output structure
    out = {
        "info": {
            "description": "COCO 2017 Dataset",
            "url": "http://cocodataset.org",
            "version": "1.0",
            "year": 2017,
            "contributor": "COCO Consortium",
            "date_created": "2017/09/01",
        },
        "licenses": [{"url": "http://creativecommons.org/licenses/by/2.0/", "id": 4, "name": "Attribution License"}],
        "images": [],
        "annotations": [],
        "categories": [],
    }

    # categories: user asked for exactly three categories: positive, negative, lines
    categories = [
        {"supercategory": "object", "id": 0, "name": "positive"},
        {"supercategory": "object", "id": 1, "name": "negative"},
        {"supercategory": "object", "id": 2, "name": "lines"},
    ]
    out["categories"] = categories

    # cat_map allows mapping model class ids -> desired category ids. For now assume identity.
    cat_map = {0: 0, 1: 1, 2: 2}

    # copy images entries (but ensure file_name absolute or as provided)
    for im in images:
        out["images"].append({
            "id": int(im.get("id", 0)),
            "license": 4,
            "coco_url": im.get("coco_url", ""),
            "flickr_url": im.get("flickr_url", ""),
            "width": int(im.get("width", 0)),
            "height": int(im.get("height", 0)),
            "file_name": im["file_name"],
            "date_captured": im.get("date_captured", ""),
        })

    ann_id = 1
    for im in tqdm.tqdm(out["images"], desc="Predicting images"):
        anns, ann_id = process_image(im, demo, args.confidence_threshold, ann_id, cat_map)
        out["annotations"].extend(anns)

    # write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f)

    print(f"Wrote {len(out['annotations'])} annotations for {len(out['images'])} images to {output_path}")


if __name__ == "__main__":
    main()
