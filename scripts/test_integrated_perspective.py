#!/usr/bin/env python3
"""
Test the integrated perspective correction in ImprovedBoardDetector
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import json
from pathlib import Path
from app.models.improved_board_detector import ImprovedBoardDetector
from app.models.piece_classifier import PieceClassifier
from app.models.fen_builder import build_fen_from_squares


def test_api_with_perspective():
    """Test if API benefits from perspective correction"""
    
    print("=== Testing API with Perspective Correction ===\n")
    
    # Test images that might benefit from perspective correction
    test_images = [
        'test_angled_board.png',  # Our synthetic test
        'dataset/images/25.png',   # Showed improvement in earlier test
        'dataset/images/51.webp',  # Low accuracy image
        'dataset/images/17.png',   # Another low accuracy
    ]
    
    detector = ImprovedBoardDetector()
    classifier = PieceClassifier()
    
    for img_path in test_images:
        if not Path(img_path).exists():
            continue
            
        print(f"\nTesting: {img_path}")
        print("-" * 40)
        
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        # Test with perspective enabled (default)
        print("With perspective correction:")
        try:
            board = detector.detect_board(image, debug=True)
            squares = detector.extract_squares(board)
            classifications = classifier.classify_board(squares)
            fen = build_fen_from_squares(classifications, 0.3)
            print(f"  FEN: {fen.split()[0]}")
        except Exception as e:
            print(f"  Failed: {e}")
        
        # Test without perspective
        print("\nWithout perspective correction:")
        detector.disable_perspective_correction()
        try:
            board = detector.detect_board(image, debug=True)
            squares = detector.extract_squares(board)
            classifications = classifier.classify_board(squares)
            fen = build_fen_from_squares(classifications, 0.3)
            print(f"  FEN: {fen.split()[0]}")
        except Exception as e:
            print(f"  Failed: {e}")
        
        # Re-enable for next test
        detector.enable_perspective_correction()


def test_performance_impact():
    """Test if perspective correction improves overall performance"""
    
    print("\n\n=== Performance Impact Test ===\n")
    
    # Load manifest
    with open('dataset/manifest.json', 'r') as f:
        manifest = json.load(f)
    
    detector = ImprovedBoardDetector()
    classifier = PieceClassifier()
    
    # Test on images that had low accuracy
    low_accuracy_threshold = 0.5
    test_count = 0
    improved_count = 0
    
    print("Testing on low-accuracy images...")
    
    for position in manifest['positions'][:100]:  # Test first 100
        img_path = Path('dataset') / position['image']
        
        if not img_path.exists():
            continue
            
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        # Quick test without perspective to find low accuracy images
        detector.disable_perspective_correction()
        try:
            board = detector.detect_board(image, debug=False)
            squares = detector.extract_squares(board)
            classifications = classifier.classify_board(squares)
            fen_without = build_fen_from_squares(classifications, 0.3)
            
            # Simple accuracy check (just count non-empty squares)
            pieces_without = sum(1 for p, _ in classifications if p != 'empty')
            
            # Now test with perspective
            detector.enable_perspective_correction()
            board = detector.detect_board(image, debug=False)
            squares = detector.extract_squares(board)
            classifications = classifier.classify_board(squares)
            fen_with = build_fen_from_squares(classifications, 0.3)
            
            pieces_with = sum(1 for p, _ in classifications if p != 'empty')
            
            # Check if perspective helped
            if pieces_without < 20:  # Low piece count might indicate problem
                test_count += 1
                if pieces_with > pieces_without:
                    improved_count += 1
                    print(f"  ✅ {Path(img_path).name}: {pieces_without} → {pieces_with} pieces")
                    
        except:
            pass
    
    if test_count > 0:
        print(f"\nSummary:")
        print(f"  Tested {test_count} low-accuracy images")
        print(f"  Improved {improved_count} ({improved_count/test_count:.1%})")
    else:
        print("No low-accuracy images found in test set")


if __name__ == "__main__":
    print("Testing integrated perspective correction\n")
    
    # Load model
    print("Loading model...")
    
    # Test specific images
    test_api_with_perspective()
    
    # Test performance impact
    test_performance_impact()
    
    print("\n✅ Testing complete!")