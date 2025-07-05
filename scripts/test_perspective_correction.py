#!/usr/bin/env python3
"""
Test perspective correction on angled board images
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from app.models.board_detector import BoardDetector
from app.models.improved_board_detector import ImprovedBoardDetector
from app.models.perspective_board_detector import PerspectiveBoardDetector
from app.models.piece_classifier import PieceClassifier
from app.models.fen_builder import build_fen_from_squares


def calculate_fen_accuracy(predicted_fen: str, expected_fen: str) -> float:
    """Calculate accuracy between two FEN strings (position only)"""
    pred_position = predicted_fen.split()[0]
    exp_position = expected_fen.split()[0]
    
    # Expand FEN notation
    def expand_fen(fen_pos):
        expanded = ""
        for char in fen_pos:
            if char.isdigit():
                expanded += "-" * int(char)
            elif char == "/":
                pass
            else:
                expanded += char
        return expanded
    
    pred_expanded = expand_fen(pred_position)
    exp_expanded = expand_fen(exp_position)
    
    if len(pred_expanded) != 64 or len(exp_expanded) != 64:
        return 0.0
    
    correct = sum(1 for p, e in zip(pred_expanded, exp_expanded) if p == e)
    return correct / 64


def find_angled_images():
    """Find images that are likely angled/tilted based on poor performance"""
    
    # Load manifest
    with open('dataset/manifest.json', 'r') as f:
        manifest = json.load(f)
    
    # Test a subset with original detector to find poor performers
    print("Finding angled images by testing performance...")
    
    detector = BoardDetector()
    classifier = PieceClassifier()
    
    angled_candidates = []
    
    # Test first 100 images
    for i, position in enumerate(manifest['positions'][:100]):
        img_path = Path('dataset') / position['image']
        expected_fen = position['fen']
        
        if not img_path.exists():
            continue
            
        image = cv2.imread(str(img_path))
        if image is None:
            continue
            
        try:
            board = detector.detect_board(image)
            squares = detector.extract_squares(board)
            classifications = classifier.classify_board(squares)
            predicted_fen = build_fen_from_squares(classifications, 0.3)
            accuracy = calculate_fen_accuracy(predicted_fen, expected_fen)
            
            # Images with poor accuracy might be angled
            if accuracy < 0.6:  # Less than 60% accuracy
                angled_candidates.append({
                    'path': str(img_path),
                    'accuracy': accuracy,
                    'expected_fen': expected_fen
                })
                
        except:
            pass
    
    # Sort by accuracy (worst first)
    angled_candidates.sort(key=lambda x: x['accuracy'])
    
    return angled_candidates[:10]  # Return 10 worst performers


def test_perspective_correction(test_images):
    """Test perspective correction on specific images"""
    
    print("\n=== Testing Perspective Correction ===\n")
    
    # Initialize detectors
    original_detector = BoardDetector()
    improved_detector = ImprovedBoardDetector()
    perspective_detector = PerspectiveBoardDetector()
    classifier = PieceClassifier()
    
    print("Image          Original  Improved  Perspective  Notes")
    print("-" * 70)
    
    for test_image in test_images:
        img_path = test_image['path']
        expected_fen = test_image['expected_fen']
        img_name = Path(img_path).name
        
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        # Test original
        orig_accuracy = test_image.get('accuracy', 0.0)
        
        # Test improved
        imp_accuracy = 0.0
        try:
            board = improved_detector.detect_board(image)
            squares = improved_detector.extract_squares(board)
            classifications = classifier.classify_board(squares)
            predicted_fen = build_fen_from_squares(classifications, 0.3)
            imp_accuracy = calculate_fen_accuracy(predicted_fen, expected_fen)
        except:
            pass
        
        # Test perspective
        persp_accuracy = 0.0
        notes = ""
        try:
            board = perspective_detector.detect_board(image, debug=True)
            squares = perspective_detector.extract_squares(board)
            classifications = classifier.classify_board(squares)
            predicted_fen = build_fen_from_squares(classifications, 0.3)
            persp_accuracy = calculate_fen_accuracy(predicted_fen, expected_fen)
            
            # Check if perspective correction was applied
            if "perspective correction" in str(board):
                notes = "corrected"
        except Exception as e:
            notes = "failed"
        
        # Format output
        orig_str = f"{orig_accuracy:6.1%}"
        imp_str = f"{imp_accuracy:6.1%}"
        persp_str = f"{persp_accuracy:6.1%}"
        
        # Highlight improvements
        if persp_accuracy > orig_accuracy + 0.1:
            persp_str += " ↑↑"
        elif persp_accuracy > orig_accuracy + 0.05:
            persp_str += " ↑"
        
        print(f"{img_name:14s} {orig_str}   {imp_str}   {persp_str}     {notes}")


def visualize_perspective_correction(image_path):
    """Visualize the perspective correction process"""
    
    print(f"\n=== Visualizing Perspective Correction: {Path(image_path).name} ===\n")
    
    image = cv2.imread(image_path)
    if image is None:
        print("Could not load image")
        return
        
    # Initialize detectors
    original_detector = BoardDetector()
    perspective_detector = PerspectiveBoardDetector()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original image
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    # Original detection
    try:
        board_orig = original_detector.detect_board(image)
        axes[0, 1].imshow(cv2.cvtColor(board_orig, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title("Original Detection")
    except:
        axes[0, 1].text(0.5, 0.5, "Detection Failed", ha='center', va='center')
        axes[0, 1].set_title("Original Detection")
    axes[0, 1].axis('off')
    
    # Perspective detection
    try:
        board_persp = perspective_detector.detect_board(image, debug=True)
        axes[1, 0].imshow(cv2.cvtColor(board_persp, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title("Perspective Corrected")
    except:
        axes[1, 0].text(0.5, 0.5, "Detection Failed", ha='center', va='center')
        axes[1, 0].set_title("Perspective Corrected")
    axes[1, 0].axis('off')
    
    # Grid overlay to show correction
    try:
        # Draw grid on corrected board
        board_grid = board_persp.copy()
        h, w = board_grid.shape[:2]
        step = h // 8
        
        # Draw grid lines
        for i in range(9):
            # Horizontal lines
            cv2.line(board_grid, (0, i*step), (w, i*step), (0, 255, 0), 2)
            # Vertical lines
            cv2.line(board_grid, (i*step, 0), (i*step, h), (0, 255, 0), 2)
            
        axes[1, 1].imshow(cv2.cvtColor(board_grid, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title("Grid Overlay (8x8)")
    except:
        axes[1, 1].text(0.5, 0.5, "No Grid", ha='center', va='center')
        axes[1, 1].set_title("Grid Overlay")
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    output_file = f"perspective_correction_{Path(image_path).stem}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_file}")
    plt.close()


def create_test_angled_board():
    """Create a synthetic angled board for testing"""
    
    print("\n=== Creating Synthetic Angled Board ===\n")
    
    # Create a simple chess board pattern
    board_size = 400
    square_size = board_size // 8
    board = np.zeros((board_size, board_size, 3), dtype=np.uint8)
    
    # Create checkerboard pattern
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 0:
                color = (240, 217, 181)  # Light square
            else:
                color = (181, 136, 99)   # Dark square
                
            y1 = i * square_size
            y2 = (i + 1) * square_size
            x1 = j * square_size
            x2 = (j + 1) * square_size
            
            board[y1:y2, x1:x2] = color
    
    # Apply perspective transform to create angled view
    src_pts = np.float32([[0, 0], [board_size, 0], 
                         [board_size, board_size], [0, board_size]])
    
    # Create tilted perspective
    dst_pts = np.float32([[50, 30], [board_size - 30, 50],
                         [board_size - 50, board_size - 30], [30, board_size - 50]])
    
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # Create larger canvas for transformed image
    canvas_size = int(board_size * 1.2)
    angled = cv2.warpPerspective(board, M, (canvas_size, canvas_size))
    
    # Save test image
    cv2.imwrite('test_angled_board.png', angled)
    print("Created test_angled_board.png")
    
    return 'test_angled_board.png'


if __name__ == "__main__":
    print("Testing perspective correction for angled chess boards\n")
    
    # Create synthetic test
    test_board = create_test_angled_board()
    visualize_perspective_correction(test_board)
    
    # Find real angled images
    print("\nSearching for angled images in dataset...")
    angled_images = find_angled_images()
    
    if angled_images:
        print(f"\nFound {len(angled_images)} potentially angled images")
        
        # Test perspective correction
        test_perspective_correction(angled_images)
        
        # Visualize best candidate
        if angled_images:
            visualize_perspective_correction(angled_images[0]['path'])
    else:
        print("No angled images found in dataset")