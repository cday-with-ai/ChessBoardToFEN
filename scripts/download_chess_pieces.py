#!/usr/bin/env python3
"""
Download open-source chess piece images for training data
"""

import os
import sys
import requests
import json
from pathlib import Path
import cv2
import numpy as np
from typing import List, Dict, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def download_lichess_pieces():
    """
    Download popular Lichess piece sets that are free to use
    """
    print("=== Downloading Lichess Piece Sets ===\n")
    
    # These sets are confirmed free to use based on Lichess licensing
    piece_sets = {
        'cburnett': 'https://raw.githubusercontent.com/lichess-org/lila/master/public/piece/cburnett/',
        'merida': 'https://raw.githubusercontent.com/lichess-org/lila/master/public/piece/merida/',
        'alpha': 'https://raw.githubusercontent.com/lichess-org/lila/master/public/piece/alpha/',
        'pirouetti': 'https://raw.githubusercontent.com/lichess-org/lila/master/public/piece/pirouetti/',
        'spatial': 'https://raw.githubusercontent.com/lichess-org/lila/master/public/piece/spatial/',
        'horsey': 'https://raw.githubusercontent.com/lichess-org/lila/master/public/piece/horsey/',
    }
    
    pieces = ['K', 'Q', 'R', 'B', 'N', 'P']  # King, Queen, Rook, Bishop, Knight, Pawn
    colors = ['w', 'b']  # white, black
    
    output_dir = Path('dataset/new_pieces/lichess')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded = 0
    
    for set_name, base_url in piece_sets.items():
        print(f"Downloading {set_name} set...")
        set_dir = output_dir / set_name
        set_dir.mkdir(exist_ok=True)
        
        for color in colors:
            for piece in pieces:
                filename = f"{color}{piece}.svg"
                url = base_url + filename
                
                try:
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        output_file = set_dir / filename
                        with open(output_file, 'wb') as f:
                            f.write(response.content)
                        downloaded += 1
                        print(f"  ✓ Downloaded {filename}")
                    else:
                        print(f"  ✗ Failed to download {filename}: {response.status_code}")
                except Exception as e:
                    print(f"  ✗ Error downloading {filename}: {e}")
    
    print(f"\nDownloaded {downloaded} piece files")
    return output_dir


def download_roboflow_samples():
    """
    Download sample images from Roboflow chess dataset
    Note: Full dataset requires registration, but we can get samples
    """
    print("\n=== Checking Roboflow Dataset ===\n")
    
    # Sample URLs from the public dataset
    sample_info = """
    The Roboflow Chess Pieces dataset contains:
    - 292 images with 2894 annotations
    - 12 classes: white/black × king/queen/bishop/knight/rook/pawn
    - Various board angles and lighting conditions
    
    To download the full dataset:
    1. Visit: https://public.roboflow.com/object-detection/chess-full
    2. Choose format (e.g., YOLOv5 PyTorch)
    3. Download will include images and annotations
    """
    
    print(sample_info)
    
    output_dir = Path('dataset/new_pieces/roboflow_info')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save dataset info
    with open(output_dir / 'dataset_info.txt', 'w') as f:
        f.write(sample_info)
    
    return output_dir


def extract_pieces_from_boards(board_images_dir: Path, output_dir: Path):
    """
    Extract individual pieces from board images
    """
    print(f"\n=== Extracting Pieces from Board Images ===\n")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # This is a placeholder - in practice, you'd need annotations
    # or a detection model to extract pieces accurately
    
    print("To extract pieces from board images:")
    print("1. Use bounding box annotations if available")
    print("2. Or manually crop pieces using an annotation tool")
    print("3. Or use our board detector to extract squares")
    
    return output_dir


def convert_svg_to_png(svg_dir: Path, output_dir: Path, sizes: List[int] = [64, 128]):
    """
    Convert SVG pieces to PNG at different sizes
    """
    print(f"\n=== Converting SVG to PNG ===\n")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Note: Requires cairosvg or similar library
    try:
        import cairosvg
        
        svg_files = list(svg_dir.rglob('*.svg'))
        converted = 0
        
        for svg_file in svg_files:
            for size in sizes:
                output_file = output_dir / f"{svg_file.stem}_{size}px.png"
                
                try:
                    cairosvg.svg2png(
                        url=str(svg_file),
                        write_to=str(output_file),
                        output_width=size,
                        output_height=size
                    )
                    converted += 1
                    print(f"  ✓ Converted {svg_file.name} to {size}px PNG")
                except Exception as e:
                    print(f"  ✗ Error converting {svg_file.name}: {e}")
        
        print(f"\nConverted {converted} images")
        
    except ImportError:
        print("cairosvg not installed. Install with: pip install cairosvg")
        print("Alternatively, use an online converter or image editor")
    
    return output_dir


def organize_training_data():
    """
    Organize downloaded pieces into training data structure
    """
    print("\n=== Organizing Training Data ===\n")
    
    # Expected structure:
    # training_data_new/
    #   ├── pieces/
    #   │   ├── white_king/
    #   │   ├── white_queen/
    #   │   ├── ... (other pieces)
    #   │   ├── black_king/
    #   │   └── ... (other pieces)
    #   └── empty/
    
    base_dir = Path('training_data_new')
    pieces_dir = base_dir / 'pieces'
    
    piece_mapping = {
        'wK': 'white_king',
        'wQ': 'white_queen',
        'wR': 'white_rook',
        'wB': 'white_bishop',
        'wN': 'white_knight',
        'wP': 'white_pawn',
        'bK': 'black_king',
        'bQ': 'black_queen',
        'bR': 'black_rook',
        'bB': 'black_bishop',
        'bN': 'black_knight',
        'bP': 'black_pawn',
    }
    
    # Create directories
    for piece_dir in piece_mapping.values():
        (pieces_dir / piece_dir).mkdir(parents=True, exist_ok=True)
    
    # Also create empty squares directory
    (base_dir / 'empty').mkdir(parents=True, exist_ok=True)
    
    print(f"Created training data structure at: {base_dir}")
    print("\nNext steps:")
    print("1. Convert SVG files to PNG")
    print("2. Place piece images in appropriate directories")
    print("3. Ensure variety in:")
    print("   - Backgrounds (wood, marble, digital)")
    print("   - Lighting conditions")
    print("   - Piece styles")
    print("4. Add empty square samples from various boards")
    
    return base_dir


def create_download_summary():
    """
    Create a summary of available resources
    """
    summary = """
# Chess Piece Training Data Resources

## Downloaded Resources

### 1. Lichess Piece Sets (SVG)
- Location: dataset/new_pieces/lichess/
- Sets: cburnett, merida, alpha, pirouetti, spatial, horsey
- Format: SVG (need conversion to PNG)
- License: Various open source licenses

### 2. Additional Resources to Download Manually

#### Roboflow Chess Dataset
- URL: https://public.roboflow.com/object-detection/chess-full
- 292 images with 2894 annotations
- Includes bounding boxes for all pieces
- Various real-world conditions

#### Kaggle Chess Pieces Dataset  
- URL: https://www.kaggle.com/datasets/anshulmehtakaggl/chess-pieces-detection-images-dataset
- Over 7,000 images of chess pieces
- Six different piece types

#### GitHub Datasets
1. samryan18/chess-dataset
   - 500 labeled chess board images
   - Includes FEN notation

2. ThanosM97/end-to-end-chess-recognition
   - 10,800 images from 100 games
   - Multiple camera angles

## Next Steps

1. Convert SVG files to PNG:
   ```bash
   pip install cairosvg
   python scripts/convert_pieces.py
   ```

2. Download additional datasets manually

3. Organize all pieces into training structure:
   ```
   training_data_new/
   ├── pieces/
   │   ├── white_king/
   │   ├── white_queen/
   │   └── ...
   └── empty/
   ```

4. Augment with various backgrounds and conditions
"""
    
    with open('dataset/new_pieces/RESOURCES_SUMMARY.md', 'w') as f:
        f.write(summary)
    
    print("\n" + summary)


if __name__ == "__main__":
    print("Chess Piece Dataset Downloader\n")
    
    # Download Lichess pieces
    lichess_dir = download_lichess_pieces()
    
    # Get info about Roboflow dataset
    roboflow_dir = download_roboflow_samples()
    
    # Create training data structure
    training_dir = organize_training_data()
    
    # Create summary
    create_download_summary()
    
    print("\n✅ Done! Check dataset/new_pieces/ for downloaded resources")