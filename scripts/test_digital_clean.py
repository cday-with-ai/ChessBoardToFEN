#!/usr/bin/env python3
"""
Test the improved detector specifically on digital_clean images
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


def test_digital_clean_subset():
    """Test on digital_clean images specifically"""
    
    print("=== Testing Digital Clean Images ===\n")
    
    # Load manifest
    with open('dataset/manifest.json', 'r') as f:
        manifest = json.load(f)
    
    # Filter for digital_clean images
    digital_clean_positions = []
    for position in manifest['positions']:
        # Check if this is a clean digital board (ChessVision.ai style)
        img_name = Path(position['image']).name
        # ChessVision images are typically numbered 50+
        if img_name.endswith('.png'):
            try:
                img_num = int(Path(img_name).stem)
                if img_num >= 50:  # ChessVision dataset starts around 50
                    digital_clean_positions.append(position)
            except:
                pass
    
    print(f"Found {len(digital_clean_positions)} digital clean images\n")
    
    # Initialize detectors
    original_detector = BoardDetector()
    improved_detector = ImprovedBoardDetector()
    classifier = PieceClassifier()
    
    # Test on these images
    results = {
        'original': {'detected': 0, 'total_accuracy': 0.0, 'perfect': 0},
        'improved': {'detected': 0, 'total_accuracy': 0.0, 'perfect': 0, 
                    'used_validated': 0, 'used_fallback': 0}
    }
    
    # Limit to first 20 for quick test
    test_positions = digital_clean_positions[:20]
    
    print(f"Testing on {len(test_positions)} images...\n")
    print("Image        Original  Improved  Detector  Notes")
    print("-" * 60)
    
    for position in test_positions:
        img_path = Path('dataset') / position['image']
        expected_fen = position['fen']
        img_name = Path(position['image']).name
        
        if not img_path.exists():
            continue
        
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        # Test original
        orig_accuracy = 0.0
        try:
            board = original_detector.detect_board(image)
            squares = original_detector.extract_squares(board)
            classifications = classifier.classify_board(squares)
            predicted_fen = build_fen_from_squares(classifications, 0.3)
            orig_accuracy = calculate_fen_accuracy(predicted_fen, expected_fen)
            
            results['original']['detected'] += 1
            results['original']['total_accuracy'] += orig_accuracy
            if orig_accuracy == 1.0:
                results['original']['perfect'] += 1
        except:
            pass
        
        # Test improved
        imp_accuracy = 0.0
        detector_used = "X"
        notes = ""
        
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
            imp_accuracy = calculate_fen_accuracy(predicted_fen, expected_fen)
            
            results['improved']['detected'] += 1
            results['improved']['total_accuracy'] += imp_accuracy
            if imp_accuracy == 1.0:
                results['improved']['perfect'] += 1
            
            # Check which detector was used
            if "Used validated detector" in debug_output:
                results['improved']['used_validated'] += 1
                detector_used = "V"
                # Check if margins were applied
                if "Detected margins:" in debug_output:
                    notes = "margins"
            else:
                results['improved']['used_fallback'] += 1
                detector_used = "F"
        except:
            pass
        
        # Format output
        orig_str = f"{orig_accuracy:6.1%}" if orig_accuracy > 0 else "FAILED"
        imp_str = f"{imp_accuracy:6.1%}" if imp_accuracy > 0 else "FAILED"
        
        # Highlight significant changes
        if imp_accuracy > orig_accuracy + 0.05:
            imp_str += " ↑"
        elif imp_accuracy < orig_accuracy - 0.05:
            imp_str += " ↓"
        
        print(f"{img_name:12s} {orig_str}  {imp_str}     [{detector_used}]    {notes}")
    
    # Summary
    total = len(test_positions)
    print("\n" + "="*60)
    print("SUMMARY - Digital Clean Images")
    print("="*60)
    
    # Original
    if total > 0:
        orig_detected = results['original']['detected']
        orig_avg = results['original']['total_accuracy'] / total
        orig_perfect_rate = results['original']['perfect'] / total
        
        print(f"\nOriginal Detector:")
        print(f"  Detection rate: {orig_detected}/{total} ({orig_detected/total:.1%})")
        print(f"  Average accuracy: {orig_avg:.1%}")
        print(f"  Perfect (100%) rate: {results['original']['perfect']}/{total} ({orig_perfect_rate:.1%})")
    
    # Improved
    if total > 0:
        imp_detected = results['improved']['detected']
        imp_avg = results['improved']['total_accuracy'] / total
        imp_perfect_rate = results['improved']['perfect'] / total
        
        print(f"\nImproved Detector:")
        print(f"  Detection rate: {imp_detected}/{total} ({imp_detected/total:.1%})")
        print(f"  Average accuracy: {imp_avg:.1%}")
        print(f"  Perfect (100%) rate: {results['improved']['perfect']}/{total} ({imp_perfect_rate:.1%})")
        print(f"  Used validated: {results['improved']['used_validated']}")
        print(f"  Used fallback: {results['improved']['used_fallback']}")
        
        # Compare
        print(f"\nChanges:")
        print(f"  Detection rate: {(imp_detected - orig_detected)/total:+.1%}")
        print(f"  Accuracy: {(imp_avg - orig_avg):+.1%}")
        print(f"  Perfect rate: {(imp_perfect_rate - orig_perfect_rate):+.1%}")
    
    print("\nNotes:")
    print("  [V] = Validated detector used")
    print("  [F] = Fallback to original detector")
    print("  [X] = Failed to detect")
    print("  'margins' = Smart margin detection applied")


if __name__ == "__main__":
    test_digital_clean_subset()