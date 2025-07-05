#!/usr/bin/env python3
"""
Test board detection and visualization
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from app.models.board_detector import BoardDetector
from app.models.piece_classifier import PieceClassifier

def visualize_board_detection(image_path):
    """Visualize the board detection process"""
    # Load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create detector
    detector = BoardDetector()
    classifier = PieceClassifier()
    
    try:
        # Detect board
        board = detector.detect_board(image)
        board_rgb = cv2.cvtColor(board, cv2.COLOR_BGR2RGB)
        
        # Extract squares
        squares = detector.extract_squares(board)
        
        # Classify squares
        predictions = []
        for square in squares:
            piece, conf = classifier.classify_square(square)
            predictions.append((piece, conf))
        
        # Visualize
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(image_rgb)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        # Detected board
        axes[0, 1].imshow(board_rgb)
        axes[0, 1].set_title("Detected Board")
        axes[0, 1].axis('off')
        
        # First few squares
        axes[0, 2].set_title("Sample Squares")
        axes[0, 2].axis('off')
        
        # Show first 4 squares in a 2x2 grid
        square_grid = np.zeros((128, 128, 3), dtype=np.uint8)
        for i in range(min(4, len(squares))):
            row = i // 2
            col = i % 2
            sq = cv2.resize(squares[i], (64, 64))
            if len(sq.shape) == 3:
                sq_rgb = cv2.cvtColor(sq, cv2.COLOR_BGR2RGB)
            else:
                sq_rgb = cv2.cvtColor(sq, cv2.COLOR_GRAY2RGB)
            square_grid[row*64:(row+1)*64, col*64:(col+1)*64] = sq_rgb
        axes[0, 2].imshow(square_grid)
        
        # Board with predictions
        board_with_labels = board_rgb.copy()
        h, w = board_with_labels.shape[:2]
        sq_h, sq_w = h // 8, w // 8
        
        for i, (piece, conf) in enumerate(predictions):
            row = i // 8
            col = i % 8
            if piece != 'empty':
                # Draw piece label
                cv2.putText(board_with_labels, 
                           piece, 
                           (col * sq_w + sq_w//3, row * sq_h + sq_h//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           1.0, 
                           (255, 0, 0), 
                           2)
        
        axes[1, 0].imshow(board_with_labels)
        axes[1, 0].set_title("Detected Pieces")
        axes[1, 0].axis('off')
        
        # Confidence heatmap
        conf_map = np.zeros((8, 8))
        for i, (_, conf) in enumerate(predictions):
            row = i // 8
            col = i % 8
            conf_map[row, col] = conf
        
        im = axes[1, 1].imshow(conf_map, cmap='RdYlGn', vmin=0, vmax=1)
        axes[1, 1].set_title("Confidence Heatmap")
        axes[1, 1].set_xlabel("File (a-h)")
        axes[1, 1].set_ylabel("Rank (8-1)")
        axes[1, 1].set_xticks(range(8))
        axes[1, 1].set_xticklabels(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
        axes[1, 1].set_yticks(range(8))
        axes[1, 1].set_yticklabels(['8', '7', '6', '5', '4', '3', '2', '1'])
        plt.colorbar(im, ax=axes[1, 1])
        
        # FEN
        board_array = []
        for i in range(0, 64, 8):
            row = [p for p, _ in predictions[i:i+8]]
            board_array.append(row)
        
        fen = build_fen(board_array)
        axes[1, 2].text(0.5, 0.5, f"FEN:\n{fen}", 
                        ha='center', va='center', 
                        fontsize=12, 
                        transform=axes[1, 2].transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        axes[1, 2].axis('off')
        axes[1, 2].set_title("Resulting FEN")
        
        plt.tight_layout()
        plt.savefig('board_detection_test.png', dpi=150, bbox_inches='tight')
        print("Visualization saved to board_detection_test.png")
        
        # Print detailed info
        print(f"\nBoard size: {board.shape}")
        print(f"Number of squares: {len(squares)}")
        print(f"Square size: {squares[0].shape if squares else 'N/A'}")
        print(f"\nFEN: {fen}")
        
        # Print piece distribution
        piece_counts = {}
        for piece, _ in predictions:
            piece_counts[piece] = piece_counts.get(piece, 0) + 1
        print(f"\nPiece distribution:")
        for piece, count in sorted(piece_counts.items()):
            print(f"  {piece}: {count}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def build_fen(board_array):
    """Convert board array to FEN string"""
    fen_parts = []
    
    for row in board_array:
        fen_row = ""
        empty_count = 0
        
        for piece in row:
            if piece == 'empty':
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += piece
        
        if empty_count > 0:
            fen_row += str(empty_count)
        
        fen_parts.append(fen_row)
    
    return '/'.join(fen_parts) + ' w KQkq - 0 1'

if __name__ == "__main__":
    # Test with a sample image
    import sys
    test_image = sys.argv[1] if len(sys.argv) > 1 else "dataset/images/84.png"
    print(f"Testing with: {test_image}")
    visualize_board_detection(test_image)