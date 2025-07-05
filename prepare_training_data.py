#!/usr/bin/env python3
"""
Prepare training data from the dataset manifest
This script extracts individual squares from board images and organizes them for training
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import shutil

from app.models.board_detector import BoardDetector
from app.utils.image_utils import load_image_from_bytes


def create_training_directories():
    """Create directory structure for training data"""
    base_dir = Path("training_data")
    
    # Remove old training data if exists
    if base_dir.exists():
        print("Removing old training data...")
        shutil.rmtree(base_dir)
    
    # Create directories for each piece type
    pieces = ['empty', 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
    
    for piece in pieces:
        piece_dir = base_dir / "squares" / piece
        piece_dir.mkdir(parents=True, exist_ok=True)
    
    return base_dir


def fen_to_board_array(fen: str) -> List[List[str]]:
    """Convert FEN string to 8x8 board array"""
    # Extract just the position part
    position = fen.split()[0]
    
    board = []
    for row in position.split('/'):
        board_row = []
        for char in row:
            if char.isdigit():
                # Add empty squares
                board_row.extend(['empty'] * int(char))
            else:
                # Add piece
                board_row.append(char)
        board.append(board_row)
    
    return board


def extract_and_save_squares(position_data: Dict, base_dir: Path, detector: BoardDetector):
    """Extract squares from a position and save them"""
    print(f"\nProcessing position {position_data['id']}: {position_data.get('description', '')[:50]}...")
    
    # Load image
    image_path = Path("dataset") / position_data['image']
    if not image_path.exists():
        print(f"  Warning: Image not found: {image_path}")
        return 0
    
    # Read image
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    image = load_image_from_bytes(image_bytes)
    
    # Detect board
    try:
        board_image = detector.detect_board(image)
    except Exception as e:
        print(f"  Error detecting board: {e}")
        return 0
    
    # Extract squares
    squares = detector.extract_squares(board_image)
    
    # Parse FEN to get piece positions
    board_array = fen_to_board_array(position_data['fen'])
    
    # Save each square
    saved_count = 0
    for row in range(8):
        for col in range(8):
            square_idx = row * 8 + col
            piece = board_array[row][col]
            square_image = squares[square_idx]
            
            # Generate filename
            square_name = f"pos{position_data['id']}_r{row}c{col}.png"
            save_path = base_dir / "squares" / piece / square_name
            
            # Save square image
            cv2.imwrite(str(save_path), cv2.cvtColor(square_image, cv2.COLOR_RGB2BGR))
            saved_count += 1
    
    print(f"  Saved {saved_count} squares")
    return saved_count


def generate_summary(base_dir: Path):
    """Generate summary of training data"""
    pieces = ['empty', 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
    piece_names = {
        'empty': 'Empty',
        'P': 'White Pawn', 'N': 'White Knight', 'B': 'White Bishop',
        'R': 'White Rook', 'Q': 'White Queen', 'K': 'White King',
        'p': 'Black Pawn', 'n': 'Black Knight', 'b': 'Black Bishop',
        'r': 'Black Rook', 'q': 'Black Queen', 'k': 'Black King'
    }
    
    summary = {"pieces": {}, "total_squares": 0}
    
    print("\n=== Training Data Summary ===")
    for piece in pieces:
        piece_dir = base_dir / "squares" / piece
        count = len(list(piece_dir.glob("*.png")))
        summary["pieces"][piece] = count
        summary["total_squares"] += count
        
        if count > 0:
            print(f"{piece_names[piece]:15} ({piece}): {count:4} examples")
    
    print(f"\nTotal squares: {summary['total_squares']}")
    
    # Save summary
    with open(base_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Check if we have enough data
    min_per_class = min(summary["pieces"].values())
    print(f"\nMinimum examples per class: {min_per_class}")
    
    if min_per_class < 10:
        print("⚠️  Warning: Some pieces have very few examples. Consider adding more positions.")
    elif min_per_class < 50:
        print("📊 You have enough data for initial experiments!")
    else:
        print("✅ Good amount of training data!")


def main():
    print("Preparing training data from dataset...")
    
    # Load manifest
    manifest_path = Path("dataset/manifest.json")
    if not manifest_path.exists():
        print("Error: dataset/manifest.json not found!")
        return
    
    with open(manifest_path, 'r') as f:
        data = json.load(f)
    
    positions = data.get('positions', [])
    if not positions:
        print("No positions found in dataset!")
        return
    
    print(f"Found {len(positions)} positions in dataset")
    
    # Create directories
    base_dir = create_training_directories()
    
    # Initialize board detector
    detector = BoardDetector()
    
    # Process each position
    total_squares = 0
    for position in positions:
        squares_saved = extract_and_save_squares(position, base_dir, detector)
        total_squares += squares_saved
    
    print(f"\nTotal squares extracted: {total_squares}")
    
    # Generate summary
    generate_summary(base_dir)
    
    print("\n✅ Training data prepared!")
    print(f"📁 Data saved in: {base_dir}/")
    print("\nNext steps:")
    print("1. Keep adding more positions to dataset/manifest.json")
    print("2. Run this script again to update training data")
    print("3. When you have enough data, run: python train_model.py")


if __name__ == "__main__":
    main()