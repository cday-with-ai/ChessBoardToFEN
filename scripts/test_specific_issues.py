#!/usr/bin/env python3
"""
Test specific issues with our improvements
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from pathlib import Path
from app.models.board_detector import BoardDetector
from app.models.validated_board_detector import ValidatedBoardDetector


def test_specific_images():
    """Test on specific problem cases"""
    
    original_detector = BoardDetector()
    validated_detector = ValidatedBoardDetector()
    
    # Test cases
    test_cases = [
        ('dataset/images/10.png', 'UI screenshot - should be rejected'),
        ('dataset/images/12.png', 'Good board with UI elements - should detect and crop'),
        ('dataset/images/4.png', 'Clean board - should work well'),
        ('dataset/images/22.png', 'Score 0.350 - check why failing'),
        ('dataset/images/19.png', 'Score 0.150 - very low score'),
    ]
    
    for img_path, description in test_cases:
        if not Path(img_path).exists():
            continue
            
        print(f"\n{'='*60}")
        print(f"Testing: {img_path}")
        print(f"Description: {description}")
        print('='*60)
        
        image = cv2.imread(str(img_path))
        
        # Test original
        print("\nOriginal Detector:")
        try:
            board = original_detector.detect_board(image)
            print(f"  ✅ Detected board: {board.shape}")
        except Exception as e:
            print(f"  ❌ Failed: {e}")
        
        # Test validated with debug info
        print("\nValidated Detector (with debug):")
        try:
            board = validated_detector.detect_board(image, debug=True)
            print(f"  ✅ Detected board: {board.shape}")
        except Exception as e:
            print(f"  ❌ Failed: {e}")
        
        # Test without margins
        print("\nValidated Detector (no margins):")
        validated_detector.apply_smart_margins = False
        try:
            board = validated_detector.detect_board(image, debug=False)
            print(f"  ✅ Detected board: {board.shape}")
        except Exception as e:
            print(f"  ❌ Failed: {e}")
        validated_detector.apply_smart_margins = True


if __name__ == "__main__":
    test_specific_images()