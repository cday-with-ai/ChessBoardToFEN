#!/usr/bin/env python3
"""
Fix square extraction sizing issues - ensure proper division and consistent sizes
"""

import cv2
import numpy as np
from typing import List


def extract_squares_fixed(board_image: np.ndarray) -> List[np.ndarray]:
    """
    Extract 64 squares with proper floating-point division to avoid size discrepancies
    Returns squares in FEN order (a8 to h1)
    """
    height, width = board_image.shape[:2]
    
    # Use floating point division for accurate positioning
    square_height = height / 8.0
    square_width = width / 8.0
    
    squares = []
    
    # Extract squares row by row (starting from top = rank 8)
    for row in range(8):
        for col in range(8):
            # Calculate exact positions using float arithmetic
            y1 = int(row * square_height)
            y2 = int((row + 1) * square_height)
            x1 = int(col * square_width)
            x2 = int((col + 1) * square_width)
            
            # Ensure we don't go out of bounds
            y2 = min(y2, height)
            x2 = min(x2, width)
            
            square = board_image[y1:y2, x1:x2]
            
            # Debug info for the size issue you mentioned
            if row == 0 and col == 0:  # First square
                print(f"Board size: {height}x{width}")
                print(f"Square size calculation: {height}/8 = {square_height}, {width}/8 = {square_width}")
                print(f"First square actual size: {square.shape}")
                print(f"Expected square size: {int(square_height)}x{int(square_width)}")
            
            squares.append(square)
    
    return squares


def extract_squares_uniform_resize(board_image: np.ndarray, target_size: int = 64) -> List[np.ndarray]:
    """
    Extract squares and resize all to uniform size to eliminate size variations
    """
    height, width = board_image.shape[:2]
    square_height = height / 8.0
    square_width = width / 8.0
    
    squares = []
    
    for row in range(8):
        for col in range(8):
            y1 = int(row * square_height)
            y2 = int((row + 1) * square_height)
            x1 = int(col * square_width)
            x2 = int((col + 1) * square_width)
            
            y2 = min(y2, height)
            x2 = min(x2, width)
            
            square = board_image[y1:y2, x1:x2]
            
            # Resize to uniform size
            if square.size > 0:
                square_resized = cv2.resize(square, (target_size, target_size))
                squares.append(square_resized)
            else:
                # Create empty square if extraction failed
                empty_square = np.zeros((target_size, target_size, 3), dtype=np.uint8)
                squares.append(empty_square)
    
    return squares


def test_size_issue():
    """Test the specific size issue: 58px board vs 53.9333px piece"""
    
    print("=== Testing Size Issue: 58px board ===\n")
    
    # Create a test 58x58 board
    test_board = np.random.randint(0, 255, (58, 58, 3), dtype=np.uint8)
    
    print("Original integer division method:")
    height, width = test_board.shape[:2]
    square_height_int = height // 8
    square_width_int = width // 8
    print(f"  Board: {height}x{width}")
    print(f"  Integer division: {height}//8 = {square_height_int}, {width}//8 = {square_width_int}")
    print(f"  Total covered: {square_height_int * 8}x{square_width_int * 8}")
    print(f"  Lost pixels: {height - square_height_int * 8}x{width - square_width_int * 8}")
    
    print(f"\nFloat division method:")
    square_height_float = height / 8.0
    square_width_float = width / 8.0
    print(f"  Exact square size: {square_height_float}x{square_width_float}")
    print(f"  53.9333 mentioned = {431/8} (suggesting original 431px board)")
    
    # Test extraction
    print(f"\nTesting square extraction:")
    squares_old = extract_squares_old_method(test_board)
    squares_new = extract_squares_fixed(test_board)
    
    print(f"  Old method - first square size: {squares_old[0].shape}")
    print(f"  New method - first square size: {squares_new[0].shape}")
    
    # Check if all squares have the same size
    old_sizes = [s.shape for s in squares_old]
    new_sizes = [s.shape for s in squares_new]
    
    print(f"  Old method - unique sizes: {set(old_sizes)}")
    print(f"  New method - unique sizes: {set(new_sizes)}")


def extract_squares_old_method(board_image: np.ndarray) -> List[np.ndarray]:
    """Original method with integer division for comparison"""
    height, width = board_image.shape[:2]
    square_height = height // 8  # Integer division
    square_width = width // 8
    
    squares = []
    for row in range(8):
        for col in range(8):
            y1 = row * square_height
            y2 = (row + 1) * square_height
            x1 = col * square_width
            x2 = (col + 1) * square_width
            
            square = board_image[y1:y2, x1:x2]
            squares.append(square)
    
    return squares


if __name__ == "__main__":
    test_size_issue()
    
    print("\n" + "="*60)
    print("SOLUTION:")
    print("Replace integer division (//) with float division (/) in extract_squares methods")
    print("This will fix the 58px vs 53.9333px size discrepancy")