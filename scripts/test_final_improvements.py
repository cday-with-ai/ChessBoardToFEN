#!/usr/bin/env python3
"""
Test the final improved detector with all refinements
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import json
from pathlib import Path
from app.models.board_detector import BoardDetector
from app.models.improved_board_detector import ImprovedBoardDetector
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


def test_final_improvements():
    """Test the final improved detector"""
    
    # Load manifest
    with open('dataset/manifest.json', 'r') as f:
        manifest = json.load(f)
    
    # Initialize detectors
    original_detector = BoardDetector()
    improved_detector = ImprovedBoardDetector()
    classifier = PieceClassifier()
    
    # Test on full dataset
    results = {
        'original': {'detected': 0, 'total_accuracy': 0.0, 'ui_rejected': 0},
        'improved': {'detected': 0, 'total_accuracy': 0.0, 'ui_rejected': 0, 
                    'used_validated': 0, 'used_fallback': 0}
    }
    
    # Known UI screenshots
    ui_images = ['9.png', '10.png']
    
    test_positions = manifest['positions'][:50]  # Test first 50
    
    print(f"Testing on {len(test_positions)} images...")
    print("Legend: V=Validated detector, F=Fallback to original, X=Failed\n")
    
    for i, position in enumerate(test_positions):
        img_path = Path('dataset') / position['image']
        expected_fen = position['fen']
        img_name = Path(position['image']).name
        
        if not img_path.exists():
            continue
        
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        # Test original
        try:
            board = original_detector.detect_board(image)
            squares = original_detector.extract_squares(board)
            classifications = classifier.classify_board(squares)
            predicted_fen = build_fen_from_squares(classifications, 0.3)
            accuracy = calculate_fen_accuracy(predicted_fen, expected_fen)
            
            results['original']['detected'] += 1
            results['original']['total_accuracy'] += accuracy
            
            # Check if UI was incorrectly detected
            if img_name in ui_images:
                print(f"[{i+1:2d}] {img_name:12s} Original: {accuracy:5.1%} ⚠️  (UI not rejected)")
            else:
                print(f"[{i+1:2d}] {img_name:12s} Original: {accuracy:5.1%}", end='')
        except:
            if img_name in ui_images:
                results['original']['ui_rejected'] += 1
                print(f"[{i+1:2d}] {img_name:12s} Original: FAILED ✅ (UI rejected)", end='')
            else:
                print(f"[{i+1:2d}] {img_name:12s} Original: FAILED", end='')
        
        # Test improved with debug
        print(" | Improved: ", end='')
        
        # Capture debug output
        import io
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        try:
            with redirect_stdout(f):
                board = improved_detector.detect_board(image, debug=True)
            debug_output = f.getvalue()
            
            squares = improved_detector.extract_squares(board)
            classifications = classifier.classify_board(squares)
            predicted_fen = build_fen_from_squares(classifications, 0.3)
            accuracy = calculate_fen_accuracy(predicted_fen, expected_fen)
            
            results['improved']['detected'] += 1
            results['improved']['total_accuracy'] += accuracy
            
            # Check which detector was used
            if "Used validated detector" in debug_output:
                results['improved']['used_validated'] += 1
                detector_used = "V"
            else:
                results['improved']['used_fallback'] += 1
                detector_used = "F"
            
            if img_name in ui_images:
                print(f"{accuracy:5.1%} [{detector_used}] ⚠️  (UI not rejected)")
            else:
                print(f"{accuracy:5.1%} [{detector_used}]")
        except:
            if img_name in ui_images:
                results['improved']['ui_rejected'] += 1
                print("FAILED [X] ✅ (UI rejected)")
            else:
                print("FAILED [X]")
    
    # Print summary
    total = len(test_positions)
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    # Original
    orig_detected = results['original']['detected']
    orig_avg_acc = results['original']['total_accuracy'] / total if total > 0 else 0
    print(f"\nOriginal Detector:")
    print(f"  Detection rate: {orig_detected}/{total} ({orig_detected/total:.1%})")
    print(f"  Average accuracy: {orig_avg_acc:.1%}")
    print(f"  UI screenshots rejected: {results['original']['ui_rejected']}/2")
    
    # Improved
    imp_detected = results['improved']['detected']
    imp_avg_acc = results['improved']['total_accuracy'] / total if total > 0 else 0
    print(f"\nImproved Detector:")
    print(f"  Detection rate: {imp_detected}/{total} ({imp_detected/total:.1%})")
    print(f"  Average accuracy: {imp_avg_acc:.1%}")
    print(f"  UI screenshots rejected: {results['improved']['ui_rejected']}/2")
    print(f"  Used validated detector: {results['improved']['used_validated']}")
    print(f"  Used fallback (original): {results['improved']['used_fallback']}")
    
    # Improvements
    print(f"\nOverall Impact:")
    print(f"  Detection rate change: {(imp_detected - orig_detected)/total:+.1%}")
    print(f"  Accuracy change: {(imp_avg_acc - orig_avg_acc):+.1%}")
    
    # Specific improvements
    print(f"\nKey Improvements:")
    if results['improved']['ui_rejected'] > results['original']['ui_rejected']:
        print(f"  ✅ Better UI rejection ({results['improved']['ui_rejected']} vs {results['original']['ui_rejected']})")
    if imp_detected >= orig_detected:
        print(f"  ✅ Maintained detection rate with fallback")
    if imp_avg_acc > orig_avg_acc:
        print(f"  ✅ Improved average accuracy")


def test_specific_improvements():
    """Test specific known problem cases"""
    print("\n" + "="*60)
    print("SPECIFIC CASE TESTING")
    print("="*60)
    
    improved_detector = ImprovedBoardDetector()
    
    test_cases = [
        ('dataset/images/10.png', 'UI screenshot - should be rejected or handled carefully'),
        ('dataset/images/12.png', 'Board with UI elements - should detect and crop'),
        ('dataset/images/1.jpeg', 'Wooden board - improved pattern detection should help'),
    ]
    
    for img_path, description in test_cases:
        if not Path(img_path).exists():
            continue
        
        print(f"\n{Path(img_path).name}: {description}")
        
        image = cv2.imread(img_path)
        
        try:
            board = improved_detector.detect_board(image, debug=True)
            print("  Result: Successfully detected")
        except Exception as e:
            print(f"  Result: Failed - {e}")


if __name__ == "__main__":
    print("=== Final Improvements Test ===\n")
    test_final_improvements()
    test_specific_improvements()