#!/usr/bin/env python3
"""
Simple script to test custom dataset registration.
Usage: python test_dataset_registration.py --json /path/to/annotations.json --images /path/to/images/
"""

import argparse
import sys
from detectron2.data import DatasetCatalog, MetadataCatalog

# Add the istr module to path
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'istr'))

from istr import register_dataset_from_args


def main():
    parser = argparse.ArgumentParser(description="Test custom dataset registration")
    parser.add_argument("--json", required=True, help="Path to COCO format JSON file")
    parser.add_argument("--images", required=True, help="Path to images directory")
    parser.add_argument("--name", default="test_dataset", help="Name for the dataset")
    args = parser.parse_args()
    
    print("=" * 80)
    print("Testing Custom Dataset Registration")
    print("=" * 80)
    
    # Register the dataset
    try:
        register_dataset_from_args(args.name, args.json, args.images)
        print("\n✓ Dataset registered successfully!")
    except Exception as e:
        print(f"\n✗ Failed to register dataset: {e}")
        return 1
    
    # Verify registration
    print("\n" + "=" * 80)
    print("Verification")
    print("=" * 80)
    
    # Check if dataset is in catalog
    all_datasets = DatasetCatalog.list()
    print(f"\nTotal registered datasets: {len(all_datasets)}")
    if args.name in all_datasets:
        print(f"✓ '{args.name}' found in DatasetCatalog")
    else:
        print(f"✗ '{args.name}' NOT found in DatasetCatalog")
        return 1
    
    # Get metadata
    metadata = MetadataCatalog.get(args.name)
    print(f"\nMetadata for '{args.name}':")
    print(f"  - JSON file: {metadata.json_file}")
    print(f"  - Image root: {metadata.image_root}")
    print(f"  - Evaluator type: {metadata.evaluator_type}")
    
    # Load dataset and print statistics
    try:
        dataset_dicts = DatasetCatalog.get(args.name)
        print(f"\nDataset Statistics:")
        print(f"  - Total images: {len(dataset_dicts)}")
        
        if len(dataset_dicts) > 0:
            first_image = dataset_dicts[0]
            print(f"\nFirst image sample:")
            print(f"  - File: {first_image.get('file_name', 'N/A')}")
            print(f"  - Image ID: {first_image.get('image_id', 'N/A')}")
            print(f"  - Height: {first_image.get('height', 'N/A')}")
            print(f"  - Width: {first_image.get('width', 'N/A')}")
            print(f"  - Annotations: {len(first_image.get('annotations', []))}")
            
            # Count total annotations
            total_annotations = sum(len(img.get('annotations', [])) for img in dataset_dicts)
            print(f"\n  - Total annotations: {total_annotations}")
            print(f"  - Average annotations per image: {total_annotations / len(dataset_dicts):.2f}")
            
            # Category statistics
            categories = set()
            for img in dataset_dicts:
                for ann in img.get('annotations', []):
                    categories.add(ann.get('category_id'))
            print(f"  - Unique categories: {len(categories)}")
            print(f"  - Category IDs: {sorted(categories)}")
        
        print("\n✓ Dataset loaded and verified successfully!")
        print("\n" + "=" * 80)
        print("You can now use this dataset in train_net.py")
        print("=" * 80)
        return 0
        
    except Exception as e:
        print(f"\n✗ Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
