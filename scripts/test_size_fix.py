#!/usr/bin/env python3
"""
Test the square extraction sizing fix
"""

import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.board_detector import BoardDetector
from app.models.validated_board_detector import ValidatedBoardDetector


def test_size_consistency():
    """Test that all extracted squares have consistent sizes"""
    
    print("=== Testing Square Size Consistency ===\n")
    
    # Create test boards of different sizes
    test_sizes = [58, 64, 100, 200, 431]  # Including your 58px case and the 431px that gives 53.9333
    
    detectors = {
        'BoardDetector': BoardDetector(),
        'ValidatedBoardDetector': ValidatedBoardDetector()
    }
    
    for size in test_sizes:
        print(f"Testing {size}x{size} board:")
        
        # Create a test board with a checkerboard pattern
        test_board = create_checkerboard_pattern(size, size)
        
        for name, detector in detectors.items():
            try:
                squares = detector.extract_squares(test_board)
                
                # Check sizes
                square_sizes = [s.shape for s in squares]
                unique_sizes = set(square_sizes)
                
                print(f"  {name}:")
                print(f"    Extracted {len(squares)} squares")
                print(f"    Unique sizes: {unique_sizes}")
                
                if len(unique_sizes) == 1:
                    print(f"    ✅ All squares have consistent size: {list(unique_sizes)[0]}")
                else:
                    print(f"    ❌ Inconsistent square sizes detected!")
                    
                # Test the specific values you mentioned
                if size == 58:
                    expected_square_size = 58 / 8.0  # 7.25
                    actual_size = squares[0].shape[0]  # Should be 64 after resize
                    print(f"    Original calculation: {size}/8 = {expected_square_size}")
                    print(f"    Actual square size: {actual_size}x{actual_size}")
                elif size == 431:
                    expected_square_size = 431 / 8.0  # 53.875 (close to your 53.9333)
                    actual_size = squares[0].shape[0]
                    print(f"    Original calculation: {size}/8 = {expected_square_size}")
                    print(f"    Actual square size: {actual_size}x{actual_size}")
                    
            except Exception as e:
                print(f"  {name}: Error - {e}")
        
        print()


def create_checkerboard_pattern(height: int, width: int) -> np.ndarray:
    """Create a checkerboard pattern for testing"""
    board = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Calculate square size
    square_h = height // 8
    square_w = width // 8
    
    for row in range(8):
        for col in range(8):
            # Checkerboard pattern: white squares where (row + col) is even
            if (row + col) % 2 == 0:
                color = 240  # Light
            else:
                color = 120  # Dark
            
            y1 = row * square_h
            y2 = min((row + 1) * square_h, height)
            x1 = col * square_w
            x2 = min((col + 1) * square_w, width)
            
            board[y1:y2, x1:x2] = color
    
    return board


def test_real_board_extraction():
    """Test extraction on a real board image if available"""
    
    print("=== Testing Real Board Extraction ===\n")
    
    # Look for a test image
    test_images = [
        'dataset/images/1.png',
        'dataset/images/2.png',
        'test_angled_board.png'
    ]
    
    detector = ValidatedBoardDetector()
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"Testing: {img_path}")
            
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            try:
                board = detector.detect_board(image)
                squares = detector.extract_squares(board)
                
                print(f"  Board size: {board.shape[:2]}")
                print(f"  Extracted {len(squares)} squares")
                
                square_sizes = [s.shape for s in squares]
                unique_sizes = set(square_sizes)
                print(f"  Square sizes: {unique_sizes}")
                
                if len(unique_sizes) == 1:
                    print(f"  ✅ Consistent square sizes")
                else:
                    print(f"  ❌ Inconsistent square sizes!")
                
            except Exception as e:
                print(f"  Error: {e}")
            
            break
    else:
        print("No test images found")


if __name__ == "__main__":
    test_size_consistency()
    test_real_board_extraction()
    
    print("="*60)
    print("SIZE FIX SUMMARY:")
    print("- Replaced integer division (//) with float division (/)")
    print("- Added uniform resizing to 64x64 for all squares")
    print("- Added boundary checks to prevent extraction errors")
    print("- This should fix the 58px vs 53.9333px size discrepancy")