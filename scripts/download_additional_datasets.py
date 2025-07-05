#!/usr/bin/env python3
"""
Guide for downloading additional chess piece datasets
"""

import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def print_dataset_guide():
    """Print guide for manually downloading datasets"""
    
    guide = """
=== Chess Piece Dataset Download Guide ===

This guide provides instructions for downloading additional chess piece datasets
to enhance our training data.

## 1. Roboflow Chess Dataset
URL: https://public.roboflow.com/object-detection/chess-full

Instructions:
1. Visit the URL above
2. Click "Download This Dataset"
3. Choose format: "YOLOv5 PyTorch" (includes images + annotations)
4. Sign up for free account if needed
5. Download will include:
   - 292 images with chess boards
   - Bounding box annotations for all pieces
   - Various angles and lighting conditions

After downloading:
- Extract to: dataset/new_pieces/roboflow/
- Run: python scripts/extract_pieces_from_roboflow.py


## 2. Kaggle Chess Pieces Dataset
URL: https://www.kaggle.com/datasets/anshulmehtakaggl/chess-pieces-detection-images-dataset

Instructions:
1. Visit the URL above
2. Sign in to Kaggle (free account)
3. Click "Download" (7,000+ images)
4. Dataset includes:
   - Individual piece images
   - Multiple piece styles
   - Various backgrounds

After downloading:
- Extract to: dataset/new_pieces/kaggle/
- Images are already cropped individual pieces


## 3. GitHub: samryan18/chess-dataset
URL: https://github.com/samryan18/chess-dataset

Instructions:
1. Clone the repository:
   git clone https://github.com/samryan18/chess-dataset.git dataset/new_pieces/samryan18

2. Dataset includes:
   - 500 labeled chess board images
   - FEN annotations
   - Can extract pieces using our board detector


## 4. GitHub: ThanosM97/end-to-end-chess-recognition
URL: https://github.com/ThanosM97/end-to-end-chess-recognition

Instructions:
1. Visit the repository
2. Download the dataset folder
3. Contains 10,800 images from 100 games
4. Multiple camera angles


## 5. Chess.com Piece Styles (for reference)
While we can't directly download Chess.com pieces, we can use them as reference
for the types of pieces users might photograph:

Common styles:
- Neo
- Classic
- Wood
- Glass
- Gothic
- Icy Sea
- Marble

Consider creating similar styles with our SVG pieces.


## Next Steps After Downloading:

1. For Roboflow dataset with bounding boxes:
   python scripts/extract_pieces_from_roboflow.py

2. For Kaggle individual pieces:
   python scripts/organize_kaggle_pieces.py

3. To add all new pieces to training data:
   python scripts/merge_training_data.py

4. To train the enhanced model:
   python train_enhanced_model.py
    """
    
    print(guide)
    
    # Create directories
    dirs_to_create = [
        'dataset/new_pieces/roboflow',
        'dataset/new_pieces/kaggle',
        'dataset/new_pieces/samryan18',
        'dataset/new_pieces/manual_crops'
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Save guide to file
    with open('dataset/new_pieces/DOWNLOAD_GUIDE.md', 'w') as f:
        f.write(guide)
    
    print(f"\n✅ Created directories and saved guide to: dataset/new_pieces/DOWNLOAD_GUIDE.md")


if __name__ == "__main__":
    print_dataset_guide()