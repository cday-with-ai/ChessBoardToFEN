#!/usr/bin/env python3
"""
Diagnose board extraction issues
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from app.models.board_detector import BoardDetector
from app.models.hybrid_board_detector import HybridBoardDetector
from app.models.piece_classifier import PieceClassifier

def diagnose_extraction(image_path):
    """Show detailed extraction process"""
    # Load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create detector
    detector = HybridBoardDetector()  # Using hybrid version
    
    print(f"Original image shape: {image.shape}")
    
    try:
        # Detect board
        board = detector.detect_board(image)
        board_rgb = cv2.cvtColor(board, cv2.COLOR_BGR2RGB)
        print(f"Detected board shape: {board.shape}")
        
        # Extract squares
        squares = detector.extract_squares(board)
        print(f"Number of squares: {len(squares)}")
        print(f"Square size: {squares[0].shape}")
        
        # Create visualization
        fig = plt.figure(figsize=(20, 16))
        
        # Original image
        ax1 = plt.subplot(3, 1, 1)
        ax1.imshow(image_rgb)
        ax1.set_title("Original Image", fontsize=16)
        ax1.axis('off')
        
        # Detected board with grid overlay
        ax2 = plt.subplot(3, 1, 2)
        board_with_grid = board_rgb.copy()
        h, w = board_with_grid.shape[:2]
        sq_h, sq_w = h // 8, w // 8
        
        # Draw grid lines
        for i in range(9):
            # Vertical lines
            cv2.line(board_with_grid, (i * sq_w, 0), (i * sq_w, h), (255, 0, 0), 2)
            # Horizontal lines
            cv2.line(board_with_grid, (0, i * sq_h), (w, i * sq_h), (255, 0, 0), 2)
        
        ax2.imshow(board_with_grid)
        ax2.set_title("Detected Board with Grid Overlay", fontsize=16)
        ax2.axis('off')
        
        # Show all 64 squares
        ax3 = plt.subplot(3, 1, 3)
        
        # Create grid of squares
        grid_size = int(np.ceil(np.sqrt(64)))
        square_display_size = 80
        display_grid = np.ones((grid_size * square_display_size, 
                                grid_size * square_display_size, 3), dtype=np.uint8) * 255
        
        for i, square in enumerate(squares):
            row = i // 8
            col = i % 8
            
            # Resize square for display
            square_resized = cv2.resize(square, (square_display_size-2, square_display_size-2))
            if len(square_resized.shape) == 3:
                square_rgb = cv2.cvtColor(square_resized, cv2.COLOR_BGR2RGB)
            else:
                square_rgb = cv2.cvtColor(square_resized, cv2.COLOR_GRAY2RGB)
            
            # Place in grid
            y = row * square_display_size + 1
            x = col * square_display_size + 1
            display_grid[y:y+square_display_size-2, x:x+square_display_size-2] = square_rgb
            
            # Add labels
            label = f"{chr(97+col)}{8-row}"
            cv2.putText(display_grid, label, 
                       (x+5, y+15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, (255, 0, 0), 1)
        
        ax3.imshow(display_grid)
        ax3.set_title("All 64 Extracted Squares (a8 to h1)", fontsize=16)
        ax3.axis('off')
        
        plt.tight_layout()
        plt.savefig('board_extraction_diagnosis.png', dpi=150, bbox_inches='tight')
        print("\nDiagnosis saved to board_extraction_diagnosis.png")
        
        # Also save a few individual squares for closer inspection
        for i, idx in enumerate([0, 4, 56, 60]):  # corners
            cv2.imwrite(f'square_{idx}.png', squares[idx])
        print("Saved corner squares: square_0.png (a8), square_4.png (e8), square_56.png (a1), square_60.png (e1)")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    image_path = sys.argv[1] if len(sys.argv) > 1 else "dataset/images/1.jpeg"
    diagnose_extraction(image_path)