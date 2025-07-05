#!/usr/bin/env python3
"""
Test the smart margin detector to remove rank/file labels
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from app.models.smart_margin_detector import SmartMarginDetector
from app.models.validated_board_detector import ValidatedBoardDetector
from app.models.piece_classifier import PieceClassifier
from app.models.fen_builder import build_fen_from_squares


def test_margin_detection(image_paths):
    """Test margin detection on images with labels"""
    
    margin_detector = SmartMarginDetector()
    board_detector = ValidatedBoardDetector()
    classifier = PieceClassifier()
    
    for img_path in image_paths:
        print(f"\nTesting margin detection on: {img_path}")
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Could not load {img_path}")
            continue
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        try:
            # First detect the board
            board_image = board_detector.detect_board(image)
            board_rgb = cv2.cvtColor(board_image, cv2.COLOR_BGR2RGB)
            
            # Detect margins
            margins = margin_detector.detect_board_margins(board_image)
            print(f"Detected margins: {margins}")
            
            # Apply smart crop
            cropped_board = margin_detector.apply_smart_crop(board_image, margins)
            cropped_rgb = cv2.cvtColor(cropped_board, cv2.COLOR_BGR2RGB)
            
            # Extract squares from both original and cropped
            squares_orig = board_detector.extract_squares(board_image)
            squares_cropped = board_detector.extract_squares(cropped_board)
            
            # Classify and build FEN
            class_orig = classifier.classify_board(squares_orig)
            class_cropped = classifier.classify_board(squares_cropped)
            
            fen_orig = build_fen_from_squares(class_orig, 0.3)
            fen_cropped = build_fen_from_squares(class_cropped, 0.3)
            
            # Visualize results
            visualize_margin_detection(
                image_rgb, board_rgb, cropped_rgb, 
                margins, fen_orig, fen_cropped,
                Path(img_path).name
            )
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")


def visualize_margin_detection(original, board, cropped, margins, fen_orig, fen_cropped, filename):
    """Visualize the margin detection results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original image
    axes[0, 0].imshow(original)
    axes[0, 0].set_title(f"Original: {filename}")
    axes[0, 0].axis('off')
    
    # Detected board with margin indicators
    axes[0, 1].imshow(board)
    
    # Draw margin lines
    h, w = board.shape[:2]
    # Top margin
    axes[0, 1].axhline(y=margins['top'], color='red', linewidth=2, label='Top margin')
    # Bottom margin
    axes[0, 1].axhline(y=h-margins['bottom'], color='red', linewidth=2, label='Bottom margin')
    # Left margin
    axes[0, 1].axvline(x=margins['left'], color='blue', linewidth=2, label='Left margin')
    # Right margin
    axes[0, 1].axvline(x=w-margins['right'], color='blue', linewidth=2, label='Right margin')
    
    axes[0, 1].set_title(f"Detected Board with Margins\nT:{margins['top']} B:{margins['bottom']} L:{margins['left']} R:{margins['right']}")
    axes[0, 1].axis('off')
    
    # Cropped board
    axes[1, 0].imshow(cropped)
    axes[1, 0].set_title("Cropped Board (margins removed)")
    axes[1, 0].axis('off')
    
    # FEN comparison
    axes[1, 1].text(0.1, 0.7, "FEN Comparison:", fontsize=12, weight='bold')
    axes[1, 1].text(0.1, 0.5, f"Original:\n{fen_orig.split()[0]}", fontsize=10, family='monospace')
    axes[1, 1].text(0.1, 0.3, f"Cropped:\n{fen_cropped.split()[0]}", fontsize=10, family='monospace')
    
    # Check if FEN improved
    if fen_orig != fen_cropped:
        axes[1, 1].text(0.1, 0.1, "✅ FEN changed after margin removal", fontsize=10, color='green')
    else:
        axes[1, 1].text(0.1, 0.1, "⚠️ FEN unchanged", fontsize=10, color='orange')
    
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # Save
    output_path = f"margin_test_{Path(filename).stem}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")
    plt.close()


def test_margin_improvement():
    """Test if margin detection improves accuracy on known problem images"""
    
    # Images known to have margin/label issues
    problem_images = [
        'dataset/images/1.jpeg',   # Streaming UI with labels - 78.1% accuracy
        'dataset/images/3.png',    # If has labels
        'dataset/images/10.png',   # Browser screenshot
        'dataset/images/12.png',   # If has labels
    ]
    
    # Add any images from dataset that might have visible coordinates
    dataset_path = Path('dataset/images')
    if dataset_path.exists():
        # Check first few images
        for i in range(1, 20):
            for ext in ['.png', '.jpeg', '.jpg']:
                img_path = dataset_path / f"{i}{ext}"
                if img_path.exists() and str(img_path) not in problem_images:
                    problem_images.append(str(img_path))
                    if len(problem_images) >= 8:  # Test up to 8 images
                        break
    
    # Filter to existing images
    test_images = [img for img in problem_images if Path(img).exists()][:5]
    
    if not test_images:
        print("No test images found!")
        return
    
    print(f"Testing margin detection on {len(test_images)} images...")
    test_margin_detection(test_images)


def visualize_text_detection():
    """Visualize the text detection process"""
    
    margin_detector = SmartMarginDetector()
    board_detector = ValidatedBoardDetector()
    
    # Test on an image likely to have labels
    test_image = 'dataset/images/1.jpeg'
    if not Path(test_image).exists():
        # Try to find any image
        for img in Path('dataset/images').glob('*'):
            if img.suffix in ['.png', '.jpeg', '.jpg']:
                test_image = str(img)
                break
    
    print(f"\nVisualizing text detection on: {test_image}")
    
    image = cv2.imread(test_image)
    if image is None:
        print("Could not load test image")
        return
    
    try:
        # Detect board first
        board_image = board_detector.detect_board(image)
        gray = cv2.cvtColor(board_image, cv2.COLOR_BGR2GRAY)
        
        # Get text mask
        text_mask = margin_detector._detect_text_regions(gray)
        
        # Visualize
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(cv2.cvtColor(board_image, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Detected Board")
        axes[0].axis('off')
        
        axes[1].imshow(gray, cmap='gray')
        axes[1].set_title("Grayscale")
        axes[1].axis('off')
        
        axes[2].imshow(text_mask, cmap='hot')
        axes[2].set_title("Text Regions Detected")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig("text_detection_visualization.png", dpi=150, bbox_inches='tight')
        print("Saved text detection visualization")
        plt.close()
        
    except Exception as e:
        print(f"Error in text detection visualization: {e}")


if __name__ == "__main__":
    print("=== Smart Margin Detection Testing ===\n")
    
    # Test 1: Margin detection on problem images
    test_margin_improvement()
    
    print("\n" + "="*60 + "\n")
    
    # Test 2: Visualize text detection
    visualize_text_detection()
    
    print("\n✅ Testing complete! Check the generated images for results.")