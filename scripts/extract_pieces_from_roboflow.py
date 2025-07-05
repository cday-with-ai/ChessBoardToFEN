#!/usr/bin/env python3
"""
Extract individual chess pieces from Roboflow dataset using bounding box annotations
"""

import os
import sys
import cv2
import yaml
import json
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_yolo_annotations(labels_dir: Path) -> Dict[str, List[Tuple]]:
    """Load YOLO format annotations"""
    annotations = {}
    
    # YOLO class mapping for chess pieces
    class_names = {
        0: 'black_bishop',
        1: 'black_king', 
        2: 'black_knight',
        3: 'black_pawn',
        4: 'black_queen',
        5: 'black_rook',
        6: 'white_bishop',
        7: 'white_king',
        8: 'white_knight', 
        9: 'white_pawn',
        10: 'white_queen',
        11: 'white_rook'
    }
    
    for label_file in labels_dir.glob('*.txt'):
        image_name = label_file.stem
        boxes = []
        
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    # YOLO format: class x_center y_center width height (normalized)
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    if class_id in class_names:
                        boxes.append({
                            'class': class_names[class_id],
                            'x_center': x_center,
                            'y_center': y_center,
                            'width': width,
                            'height': height
                        })
        
        if boxes:
            annotations[image_name] = boxes
    
    return annotations


def extract_pieces_from_image(image_path: Path, annotations: List[Dict], 
                            output_dir: Path, padding: int = 5):
    """Extract individual pieces from image using bounding boxes"""
    
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Failed to load image: {image_path}")
        return 0
    
    h, w = image.shape[:2]
    extracted = 0
    
    for i, box in enumerate(annotations):
        # Convert normalized YOLO coordinates to pixel coordinates
        x_center = int(box['x_center'] * w)
        y_center = int(box['y_center'] * h)
        box_width = int(box['width'] * w)
        box_height = int(box['height'] * h)
        
        # Calculate bounding box corners
        x1 = max(0, x_center - box_width // 2 - padding)
        y1 = max(0, y_center - box_height // 2 - padding)
        x2 = min(w, x_center + box_width // 2 + padding)
        y2 = min(h, y_center + box_height // 2 + padding)
        
        # Extract piece
        piece_img = image[y1:y2, x1:x2]
        
        if piece_img.size == 0:
            continue
        
        # Resize to standard size
        piece_img = cv2.resize(piece_img, (128, 128))
        
        # Save piece
        piece_class = box['class']
        class_dir = output_dir / piece_class
        class_dir.mkdir(parents=True, exist_ok=True)
        
        output_name = f"roboflow_{image_path.stem}_{i}.png"
        output_path = class_dir / output_name
        
        cv2.imwrite(str(output_path), piece_img)
        extracted += 1
    
    return extracted


def process_roboflow_dataset():
    """Process Roboflow chess dataset"""
    
    roboflow_dir = Path('dataset/new_pieces/roboflow')
    
    # Check if dataset exists
    if not roboflow_dir.exists():
        print("❌ Roboflow dataset not found!")
        print(f"   Please download it to: {roboflow_dir}")
        print("   Follow instructions in dataset/new_pieces/DOWNLOAD_GUIDE.md")
        return
    
    # Look for train/valid/test directories
    train_dir = roboflow_dir / 'train'
    valid_dir = roboflow_dir / 'valid'
    test_dir = roboflow_dir / 'test'
    
    if not train_dir.exists():
        print("❌ Expected directory structure not found!")
        print("   Looking for: train/, valid/, test/ directories")
        return
    
    output_dir = Path('training_data_new/pieces')
    total_extracted = 0
    
    # Process each split
    for split_name, split_dir in [('train', train_dir), ('valid', valid_dir), ('test', test_dir)]:
        if not split_dir.exists():
            continue
            
        print(f"\nProcessing {split_name} split...")
        
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'
        
        if not images_dir.exists() or not labels_dir.exists():
            print(f"  Skipping {split_name}: missing images or labels directory")
            continue
        
        # Load annotations
        annotations = load_yolo_annotations(labels_dir)
        print(f"  Found annotations for {len(annotations)} images")
        
        # Process each image
        split_extracted = 0
        for image_file in images_dir.glob('*.jpg'):
            if image_file.stem in annotations:
                extracted = extract_pieces_from_image(
                    image_file, 
                    annotations[image_file.stem],
                    output_dir
                )
                split_extracted += extracted
                
                if split_extracted % 50 == 0 and split_extracted > 0:
                    print(f"  Extracted {split_extracted} pieces...")
        
        print(f"  Total extracted from {split_name}: {split_extracted}")
        total_extracted += split_extracted
    
    print(f"\n✅ Total pieces extracted: {total_extracted}")
    print(f"   Saved to: {output_dir}")
    
    return total_extracted


def check_class_distribution():
    """Check distribution of extracted pieces"""
    
    pieces_dir = Path('training_data_new/pieces')
    
    print("\n=== Piece Distribution ===")
    
    total = 0
    for class_dir in sorted(pieces_dir.iterdir()):
        if class_dir.is_dir():
            count = len(list(class_dir.glob('*.png')))
            total += count
            print(f"{class_dir.name:15} : {count:4} images")
    
    print(f"\nTotal images: {total}")


if __name__ == "__main__":
    print("Roboflow Chess Dataset Extractor\n")
    
    # Process dataset
    extracted = process_roboflow_dataset()
    
    if extracted and extracted > 0:
        # Show distribution
        check_class_distribution()
        
        print("\nNext steps:")
        print("1. Review extracted pieces in training_data_new/pieces/")
        print("2. Download more datasets from DOWNLOAD_GUIDE.md")
        print("3. Train model with: python train_enhanced_model.py")