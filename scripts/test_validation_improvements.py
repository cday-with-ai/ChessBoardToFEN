#!/usr/bin/env python3
"""
Test the improved board detector with validation scoring
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from app.models.board_detector import BoardDetector
from app.models.validated_board_detector import ValidatedBoardDetector
from app.models.piece_classifier import PieceClassifier
from app.models.fen_builder import build_fen_from_squares


def test_detector_comparison(image_paths):
    """Compare original vs validated detector"""
    
    # Initialize detectors
    original_detector = BoardDetector()
    validated_detector = ValidatedBoardDetector()
    classifier = PieceClassifier()
    
    results = []
    
    for img_path in image_paths:
        print(f"\nTesting: {img_path}")
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Could not load {img_path}")
            continue
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        result = {
            'path': img_path,
            'name': Path(img_path).name
        }
        
        # Test original detector
        try:
            board_orig = original_detector.detect_board(image)
            squares_orig = original_detector.extract_squares(board_orig)
            classifications_orig = classifier.classify_board(squares_orig)
            fen_orig = build_fen_from_squares(classifications_orig, 0.3)
            
            result['original'] = {
                'detected': True,
                'fen': fen_orig,
                'board_shape': board_orig.shape
            }
        except Exception as e:
            result['original'] = {
                'detected': False,
                'error': str(e)
            }
        
        # Test validated detector
        try:
            board_val = validated_detector.detect_board(image, debug=True)
            squares_val = validated_detector.extract_squares(board_val)
            classifications_val = classifier.classify_board(squares_val)
            fen_val = build_fen_from_squares(classifications_val, 0.3)
            
            result['validated'] = {
                'detected': True,
                'fen': fen_val,
                'board_shape': board_val.shape
            }
        except Exception as e:
            result['validated'] = {
                'detected': False,
                'error': str(e)
            }
        
        results.append(result)
        
        # Visualize comparison
        visualize_comparison(image_rgb, result)
    
    return results


def visualize_comparison(image, result):
    """Create visualization comparing detectors"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title(f"Original: {result['name']}")
    axes[0].axis('off')
    
    # Original detector result
    if result['original']['detected']:
        axes[1].set_title("Original Detector\n✅ Detected")
        axes[1].text(0.5, 0.5, f"FEN: {result['original']['fen'].split()[0][:20]}...", 
                    ha='center', va='center', transform=axes[1].transAxes,
                    fontsize=10, wrap=True)
    else:
        axes[1].set_title("Original Detector\n❌ Failed")
        axes[1].text(0.5, 0.5, f"Error: {result['original']['error'][:50]}", 
                    ha='center', va='center', transform=axes[1].transAxes,
                    fontsize=10, wrap=True, color='red')
    axes[1].axis('off')
    
    # Validated detector result
    if result['validated']['detected']:
        axes[2].set_title("Validated Detector\n✅ Detected")
        axes[2].text(0.5, 0.5, f"FEN: {result['validated']['fen'].split()[0][:20]}...", 
                    ha='center', va='center', transform=axes[2].transAxes,
                    fontsize=10, wrap=True)
    else:
        axes[2].set_title("Validated Detector\n❌ Failed")
        axes[2].text(0.5, 0.5, f"Error: {result['validated']['error'][:50]}", 
                    ha='center', va='center', transform=axes[2].transAxes,
                    fontsize=10, wrap=True, color='red')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save individual comparison
    output_path = f"validation_test_{Path(result['name']).stem}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison to {output_path}")
    plt.close()


def test_ui_rejection():
    """Test if validated detector properly rejects UI elements"""
    
    # Load manifest to get expected FENs
    with open('dataset/manifest.json', 'r') as f:
        manifest = json.load(f)
    
    # Create mapping of image name to FEN
    fen_map = {}
    for pos in manifest['positions']:
        img_name = Path(pos['image']).name
        fen_map[img_name] = pos['fen']
    
    # Test on problematic images
    problem_images = [
        'dataset/images/9.png',   # Chess.com UI - 28.1% accuracy
        'dataset/images/10.png',  # Browser screenshot - 45.3%
        'dataset/images/1.jpeg',  # Streaming UI - 78.1%
        'dataset/images/16.png',  # If exists
        'dataset/images/15.png',  # If exists
    ]
    
    # Filter to existing images
    test_images = [img for img in problem_images if Path(img).exists()]
    
    print("Testing UI rejection on problematic screenshots...")
    results = test_detector_comparison(test_images)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for result in results:
        print(f"\n{result['name']}:")
        
        # Get expected FEN
        expected_fen = fen_map.get(result['name'], 'Unknown')
        print(f"  Expected FEN: {expected_fen.split()[0]}")
        
        # Original detector
        if result['original']['detected']:
            print(f"  Original: ✅ Detected")
            print(f"    FEN: {result['original']['fen'].split()[0]}")
        else:
            print(f"  Original: ❌ Failed - {result['original']['error'][:50]}")
        
        # Validated detector
        if result['validated']['detected']:
            print(f"  Validated: ✅ Detected")
            print(f"    FEN: {result['validated']['fen'].split()[0]}")
        else:
            print(f"  Validated: ❌ Failed - {result['validated']['error'][:50]}")


def test_on_good_boards():
    """Test that validated detector still works on good boards"""
    
    good_images = [
        'dataset/images/84.png',   # ChessVision clean - 100% accuracy
        'dataset/images/4.png',    # Screenshot but good - 100%
        'dataset/images/100.png',  # ChessVision clean
    ]
    
    # Filter to existing images
    test_images = [img for img in good_images if Path(img).exists()]
    
    print("\nTesting on good quality boards...")
    results = test_detector_comparison(test_images)
    
    # Check that we didn't break anything
    for result in results:
        if result['original']['detected'] and not result['validated']['detected']:
            print(f"\n⚠️  WARNING: {result['name']} worked with original but failed with validated!")


if __name__ == "__main__":
    print("=== Board Detector Validation Testing ===\n")
    
    # Test 1: UI rejection
    test_ui_rejection()
    
    print("\n" + "="*60 + "\n")
    
    # Test 2: Good boards still work
    test_on_good_boards()
    
    print("\n✅ Testing complete! Check the generated images for visual comparison.")