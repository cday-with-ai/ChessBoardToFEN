#!/usr/bin/env python3
"""
Test all improvements from QUICK_FIXES.md:
1. Board validation scoring
2. Smart margin detection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import json
from pathlib import Path
from app.models.board_detector import BoardDetector
from app.models.validated_board_detector import ValidatedBoardDetector
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


def test_improvements_on_dataset():
    """Test improvements on the full dataset"""
    
    # Load manifest
    with open('dataset/manifest.json', 'r') as f:
        manifest = json.load(f)
    
    # Initialize detectors
    original_detector = BoardDetector()
    improved_detector = ValidatedBoardDetector()
    classifier = PieceClassifier()
    
    results = {
        'original': {'detected': 0, 'total_accuracy': 0.0, 'results': []},
        'improved': {'detected': 0, 'total_accuracy': 0.0, 'results': []},
        'improved_no_margins': {'detected': 0, 'total_accuracy': 0.0, 'results': []}
    }
    
    # Test subset of images
    test_positions = manifest['positions'][:30]  # Test first 30 images
    
    print(f"Testing on {len(test_positions)} images...")
    
    for i, position in enumerate(test_positions):
        img_path = Path('dataset') / position['image']
        expected_fen = position['fen']
        
        if not img_path.exists():
            continue
        
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        print(f"\n[{i+1}/{len(test_positions)}] Testing {Path(img_path).name}...", end='')
        
        # Test original detector
        try:
            board = original_detector.detect_board(image)
            squares = original_detector.extract_squares(board)
            classifications = classifier.classify_board(squares)
            predicted_fen = build_fen_from_squares(classifications, 0.3)
            accuracy = calculate_fen_accuracy(predicted_fen, expected_fen)
            
            results['original']['detected'] += 1
            results['original']['total_accuracy'] += accuracy
            results['original']['results'].append({
                'image': Path(img_path).name,
                'accuracy': accuracy,
                'detected': True
            })
        except Exception as e:
            results['original']['results'].append({
                'image': Path(img_path).name,
                'accuracy': 0.0,
                'detected': False,
                'error': str(e)
            })
        
        # Test improved detector with margins
        try:
            board = improved_detector.detect_board(image)
            squares = improved_detector.extract_squares(board)
            classifications = classifier.classify_board(squares)
            predicted_fen = build_fen_from_squares(classifications, 0.3)
            accuracy = calculate_fen_accuracy(predicted_fen, expected_fen)
            
            results['improved']['detected'] += 1
            results['improved']['total_accuracy'] += accuracy
            results['improved']['results'].append({
                'image': Path(img_path).name,
                'accuracy': accuracy,
                'detected': True
            })
        except Exception as e:
            results['improved']['results'].append({
                'image': Path(img_path).name,
                'accuracy': 0.0,
                'detected': False,
                'error': str(e)
            })
        
        # Test improved detector without margins (to isolate validation impact)
        try:
            improved_detector.apply_smart_margins = False
            board = improved_detector.detect_board(image)
            squares = improved_detector.extract_squares(board)
            classifications = classifier.classify_board(squares)
            predicted_fen = build_fen_from_squares(classifications, 0.3)
            accuracy = calculate_fen_accuracy(predicted_fen, expected_fen)
            improved_detector.apply_smart_margins = True  # Re-enable
            
            results['improved_no_margins']['detected'] += 1
            results['improved_no_margins']['total_accuracy'] += accuracy
            results['improved_no_margins']['results'].append({
                'image': Path(img_path).name,
                'accuracy': accuracy,
                'detected': True
            })
        except Exception as e:
            improved_detector.apply_smart_margins = True  # Re-enable
            results['improved_no_margins']['results'].append({
                'image': Path(img_path).name,
                'accuracy': 0.0,
                'detected': False,
                'error': str(e)
            })
        
        # Print progress
        orig_acc = results['original']['results'][-1]['accuracy']
        imp_acc = results['improved']['results'][-1]['accuracy']
        print(f" Original: {orig_acc:.1%}, Improved: {imp_acc:.1%}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    total = len(test_positions)
    
    # Original detector
    orig_detected = results['original']['detected']
    orig_avg_acc = results['original']['total_accuracy'] / total if total > 0 else 0
    print(f"\nOriginal Detector:")
    print(f"  Detected: {orig_detected}/{total} ({orig_detected/total:.1%})")
    print(f"  Average accuracy: {orig_avg_acc:.1%}")
    
    # Improved (validation only)
    imp_no_margin_detected = results['improved_no_margins']['detected']
    imp_no_margin_avg_acc = results['improved_no_margins']['total_accuracy'] / total if total > 0 else 0
    print(f"\nImproved Detector (validation only):")
    print(f"  Detected: {imp_no_margin_detected}/{total} ({imp_no_margin_detected/total:.1%})")
    print(f"  Average accuracy: {imp_no_margin_avg_acc:.1%}")
    
    # Improved (validation + margins)
    imp_detected = results['improved']['detected']
    imp_avg_acc = results['improved']['total_accuracy'] / total if total > 0 else 0
    print(f"\nImproved Detector (validation + margins):")
    print(f"  Detected: {imp_detected}/{total} ({imp_detected/total:.1%})")
    print(f"  Average accuracy: {imp_avg_acc:.1%}")
    
    # Show improvements
    print(f"\nImprovements:")
    print(f"  Detection rate: {(imp_detected - orig_detected)/total:.1%}")
    print(f"  Accuracy gain: {(imp_avg_acc - orig_avg_acc):.1%}")
    
    # Find specific improvements
    print(f"\nBiggest improvements:")
    improvements = []
    for i in range(len(results['original']['results'])):
        orig = results['original']['results'][i]
        imp = results['improved']['results'][i]
        
        if imp['accuracy'] > orig['accuracy']:
            gain = imp['accuracy'] - orig['accuracy']
            improvements.append((orig['image'], orig['accuracy'], imp['accuracy'], gain))
    
    improvements.sort(key=lambda x: x[3], reverse=True)
    for img, orig_acc, imp_acc, gain in improvements[:5]:
        print(f"  {img}: {orig_acc:.1%} → {imp_acc:.1%} (+{gain:.1%})")
    
    # Find regressions
    print(f"\nAny regressions:")
    regressions = []
    for i in range(len(results['original']['results'])):
        orig = results['original']['results'][i]
        imp = results['improved']['results'][i]
        
        if orig['detected'] and not imp['detected']:
            regressions.append((orig['image'], "Failed to detect"))
        elif imp['accuracy'] < orig['accuracy'] - 0.05:  # More than 5% worse
            loss = orig['accuracy'] - imp['accuracy']
            regressions.append((orig['image'], f"{orig['accuracy']:.1%} → {imp['accuracy']:.1%} (-{loss:.1%})"))
    
    if regressions:
        for img, issue in regressions[:5]:
            print(f"  {img}: {issue}")
    else:
        print("  None found!")
    
    # Save detailed results
    with open('improvement_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to improvement_test_results.json")


if __name__ == "__main__":
    print("=== Testing All Improvements ===\n")
    test_improvements_on_dataset()