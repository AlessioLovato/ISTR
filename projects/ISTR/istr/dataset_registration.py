"""
Dataset registration utilities for ISTR.
Allows registration of custom COCO-format datasets via command line arguments.
"""

import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json


def register_coco_instances_with_root(name, metadata, json_file, image_root):
    """
    Register a dataset in COCO's json annotation format for instance detection/segmentation.
    
    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2017_train".
        metadata (dict): extra metadata associated with this dataset. You can leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, str), json_file
    assert isinstance(image_root, str), image_root
    
    # Register the dataset
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))
    
    # Set metadata
    MetadataCatalog.get(name).set(
        json_file=json_file,
        image_root=image_root,
        evaluator_type="coco",
        **metadata
    )


def register_dataset_from_args(dataset_name, json_file, image_root, **kwargs):
    """
    Register a dataset with the given parameters.
    
    Args:
        dataset_name (str): Name to register the dataset as
        json_file (str): Path to COCO-format JSON annotation file
        image_root (str): Path to directory containing images
        **kwargs: Additional metadata to associate with the dataset
    """
    # Validate paths
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"Annotation file not found: {json_file}")
    if not os.path.exists(image_root):
        raise FileNotFoundError(f"Image directory not found: {image_root}")
    
    # Check if dataset is already registered
    if dataset_name in DatasetCatalog.list():
        print(f"WARNING: Dataset '{dataset_name}' is already registered. Skipping registration.")
        print(f"  If you want to re-register, use a different name with --train-name or --test-name")
        return
    
    # Extract metadata from kwargs
    metadata = {k: v for k, v in kwargs.items() if k not in ['dataset_name', 'json_file', 'image_root']}
    
    # Register the dataset
    register_coco_instances_with_root(dataset_name, metadata, json_file, image_root)
    
    print(f"Registered dataset '{dataset_name}':")
    print(f"  Annotations: {json_file}")
    print(f"  Images: {image_root}")
    print(f"  Metadata: {metadata}")


def setup_datasets_from_config(cfg):
    """
    Register datasets specified in the config.
    
    Expected config structure:
        cfg.DATASETS.TRAIN_JSON: path to training annotations
        cfg.DATASETS.TRAIN_IMAGES: path to training images
        cfg.DATASETS.TEST_JSON: path to test annotations  
        cfg.DATASETS.TEST_IMAGES: path to test images
    """
    # Register training dataset if paths are provided
    if hasattr(cfg.DATASETS, 'TRAIN_JSON') and hasattr(cfg.DATASETS, 'TRAIN_IMAGES'):
        train_json = cfg.DATASETS.TRAIN_JSON
        train_images = cfg.DATASETS.TRAIN_IMAGES
        
        if train_json and train_images and os.path.exists(train_json):
            train_name = cfg.DATASETS.TRAIN[0] if cfg.DATASETS.TRAIN else "custom_train"
            register_dataset_from_args(train_name, train_json, train_images)
    
    # Register test dataset if paths are provided
    if hasattr(cfg.DATASETS, 'TEST_JSON') and hasattr(cfg.DATASETS, 'TEST_IMAGES'):
        test_json = cfg.DATASETS.TEST_JSON
        test_images = cfg.DATASETS.TEST_IMAGES
        
        if test_json and test_images and os.path.exists(test_json):
            test_name = cfg.DATASETS.TEST[0] if cfg.DATASETS.TEST else "custom_test"
            register_dataset_from_args(test_name, test_json, test_images)
