#!/usr/bin/env python3
"""
Convert SVG chess pieces to PNG and prepare training data
"""

import os
import sys
from pathlib import Path
import cairosvg
import cv2
import numpy as np
from typing import List, Tuple
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def convert_svg_to_png(svg_path: Path, output_path: Path, size: int = 128, 
                      bg_color: Tuple[int, int, int] = None) -> bool:
    """Convert SVG to PNG with optional background color"""
    try:
        # Convert SVG to PNG
        png_data = cairosvg.svg2png(
            url=str(svg_path),
            output_width=size,
            output_height=size
        )
        
        # Load as numpy array
        nparr = np.frombuffer(png_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            return False
        
        # Handle alpha channel and background
        if img.shape[2] == 4:  # Has alpha channel
            if bg_color:
                # Create colored background
                bgr = img[:, :, :3]
                alpha = img[:, :, 3] / 255.0
                
                # Create background
                background = np.full((size, size, 3), bg_color, dtype=np.uint8)
                
                # Composite
                for c in range(3):
                    background[:, :, c] = (alpha * bgr[:, :, c] + 
                                         (1 - alpha) * background[:, :, c])
                
                img = background
            else:
                # Just remove alpha channel (transparent becomes black)
                img = img[:, :, :3]
        
        # Save PNG
        cv2.imwrite(str(output_path), img)
        return True
        
    except Exception as e:
        print(f"Error converting {svg_path}: {e}")
        return False


def generate_board_backgrounds() -> List[Tuple[int, int, int]]:
    """Generate various chess board square colors"""
    backgrounds = [
        # Classic wooden boards
        (240, 217, 181),  # Light wood
        (181, 136, 99),   # Dark wood
        
        # Digital boards (chess.com style)
        (238, 238, 210),  # Light yellow
        (118, 150, 86),   # Dark green
        
        # Lichess default
        (240, 217, 181),  # Light
        (181, 136, 99),   # Dark
        
        # Blue theme
        (222, 227, 230),  # Light blue
        (140, 162, 173),  # Dark blue
        
        # Gray theme
        (232, 235, 239),  # Light gray
        (125, 135, 150),  # Dark gray
        
        # Brown theme
        (240, 217, 181),  # Light brown
        (181, 136, 99),   # Dark brown
        
        # Green theme
        (238, 238, 210),  # Light green
        (118, 150, 86),   # Dark green
        
        # Purple theme
        (235, 228, 240),  # Light purple
        (180, 160, 200),  # Dark purple
    ]
    
    return backgrounds


def process_lichess_pieces():
    """Convert Lichess SVG pieces to PNG with various backgrounds"""
    print("=== Processing Lichess Pieces ===\n")
    
    svg_dir = Path('dataset/new_pieces/lichess')
    output_base = Path('training_data_new/pieces')
    
    # Piece mapping
    piece_map = {
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
    
    backgrounds = generate_board_backgrounds()
    sizes = [64, 96, 128]  # Different sizes for variety
    
    total_converted = 0
    
    # Process each piece set
    for set_dir in svg_dir.iterdir():
        if not set_dir.is_dir():
            continue
            
        print(f"Processing {set_dir.name} set...")
        
        for svg_file in set_dir.glob('*.svg'):
            piece_code = svg_file.stem  # e.g., 'wK'
            
            if piece_code not in piece_map:
                continue
                
            piece_name = piece_map[piece_code]
            output_dir = output_base / piece_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert with different backgrounds and sizes
            for i, (bg1, bg2) in enumerate(zip(backgrounds[::2], backgrounds[1::2])):
                # Alternate between light and dark backgrounds
                for j, bg_color in enumerate([bg1, bg2]):
                    for size in sizes:
                        output_name = f"{set_dir.name}_{piece_code}_bg{i*2+j}_s{size}.png"
                        output_path = output_dir / output_name
                        
                        if convert_svg_to_png(svg_file, output_path, size, bg_color):
                            total_converted += 1
                            if total_converted % 50 == 0:
                                print(f"  Converted {total_converted} images...")
    
    print(f"\nTotal pieces converted: {total_converted}")
    return total_converted


def generate_empty_squares():
    """Generate empty square samples with various backgrounds"""
    print("\n=== Generating Empty Squares ===\n")
    
    output_dir = Path('training_data_new/empty')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    backgrounds = generate_board_backgrounds()
    sizes = [64, 96, 128]
    
    generated = 0
    
    for i, bg_color in enumerate(backgrounds):
        for size in sizes:
            # Create empty square
            square = np.full((size, size, 3), bg_color, dtype=np.uint8)
            
            # Add slight variations
            variations = [
                ('plain', square),
                ('noise', add_noise(square.copy(), intensity=5)),
                ('gradient', add_gradient(square.copy())),
                ('texture', add_wood_texture(square.copy()) if i < 4 else square),
            ]
            
            for var_name, var_square in variations:
                output_name = f"empty_bg{i}_{var_name}_s{size}.png"
                output_path = output_dir / output_name
                
                cv2.imwrite(str(output_path), var_square)
                generated += 1
    
    print(f"Generated {generated} empty squares")
    return generated


def add_noise(image: np.ndarray, intensity: int = 10) -> np.ndarray:
    """Add subtle noise to image"""
    noise = np.random.randint(-intensity, intensity, image.shape, dtype=np.int16)
    noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy


def add_gradient(image: np.ndarray) -> np.ndarray:
    """Add subtle gradient to image"""
    h, w = image.shape[:2]
    gradient = np.linspace(0, 20, h).reshape(h, 1)
    gradient = np.repeat(gradient, w, axis=1)
    
    # Apply gradient
    for c in range(3):
        image[:, :, c] = np.clip(image[:, :, c].astype(float) + gradient, 0, 255)
    
    return image.astype(np.uint8)


def add_wood_texture(image: np.ndarray) -> np.ndarray:
    """Add wood-like texture to image"""
    h, w = image.shape[:2]
    
    # Create wood grain pattern
    x = np.linspace(0, 4 * np.pi, w)
    y = np.linspace(0, 4 * np.pi, h)
    X, Y = np.meshgrid(x, y)
    
    # Wood grain function
    grain = 5 * np.sin(X * 0.5 + np.sin(Y * 0.1)) + 3 * np.sin(Y * 0.3)
    
    # Apply to image
    for c in range(3):
        image[:, :, c] = np.clip(image[:, :, c].astype(float) + grain, 0, 255)
    
    return image.astype(np.uint8)


def augment_existing_pieces():
    """Apply augmentations to existing piece images"""
    print("\n=== Augmenting Existing Pieces ===\n")
    
    pieces_dir = Path('training_data_new/pieces')
    if not pieces_dir.exists():
        print("No existing pieces to augment")
        return 0
    
    augmented = 0
    
    for piece_dir in pieces_dir.iterdir():
        if not piece_dir.is_dir():
            continue
            
        print(f"Augmenting {piece_dir.name}...")
        
        # Get original images
        original_images = list(piece_dir.glob('*.png'))[:5]  # Limit to avoid too many
        
        for img_path in original_images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # Apply augmentations
            augmentations = [
                ('flip', cv2.flip(img, 1)),  # Horizontal flip
                ('bright', adjust_brightness(img.copy(), 1.2)),
                ('dark', adjust_brightness(img.copy(), 0.8)),
                ('blur', cv2.GaussianBlur(img, (3, 3), 0)),
            ]
            
            for aug_name, aug_img in augmentations:
                output_name = f"{img_path.stem}_{aug_name}.png"
                output_path = piece_dir / output_name
                
                if not output_path.exists():
                    cv2.imwrite(str(output_path), aug_img)
                    augmented += 1
    
    print(f"Created {augmented} augmented images")
    return augmented


def adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    """Adjust image brightness"""
    adjusted = cv2.convertScaleAbs(image, alpha=factor, beta=0)
    return adjusted


def create_training_summary():
    """Create summary of training data"""
    print("\n=== Training Data Summary ===\n")
    
    training_dir = Path('training_data_new')
    
    summary_lines = ["# Training Data Summary\n"]
    total_images = 0
    
    # Count pieces
    pieces_dir = training_dir / 'pieces'
    if pieces_dir.exists():
        summary_lines.append("\n## Piece Images\n")
        
        for piece_dir in sorted(pieces_dir.iterdir()):
            if piece_dir.is_dir():
                count = len(list(piece_dir.glob('*.png')))
                total_images += count
                summary_lines.append(f"- {piece_dir.name}: {count} images")
    
    # Count empty squares
    empty_dir = training_dir / 'empty'
    if empty_dir.exists():
        count = len(list(empty_dir.glob('*.png')))
        total_images += count
        summary_lines.append(f"\n## Empty Squares: {count} images")
    
    summary_lines.append(f"\n## Total Images: {total_images}")
    
    # Add recommendations
    summary_lines.append("\n## Recommendations")
    summary_lines.append("- Minimum 100 images per class recommended")
    summary_lines.append("- Include variety in lighting, angles, and backgrounds")
    summary_lines.append("- Balance between piece types")
    
    summary_text = '\n'.join(summary_lines)
    
    # Save summary
    with open(training_dir / 'TRAINING_SUMMARY.md', 'w') as f:
        f.write(summary_text)
    
    print(summary_text)
    
    return total_images


if __name__ == "__main__":
    print("Preparing Chess Piece Training Data\n")
    
    # Convert Lichess pieces
    pieces_count = process_lichess_pieces()
    
    # Generate empty squares
    empty_count = generate_empty_squares()
    
    # Augment existing pieces
    aug_count = augment_existing_pieces()
    
    # Create summary
    total = create_training_summary()
    
    print(f"\n✅ Done! Prepared {total} training images")
    print(f"   Location: training_data_new/")
    print(f"\nNext steps:")
    print(f"1. Review images in training_data_new/")
    print(f"2. Add more piece sets from other sources")
    print(f"3. Train model with: python train_enhanced_model.py")