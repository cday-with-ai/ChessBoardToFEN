#!/usr/bin/env python3
"""
Analyze failure modes in detail
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import json
from pathlib import Path
from app.models.validated_board_detector import ValidatedBoardDetector
import matplotlib.pyplot as plt


def analyze_failed_detections():
    """Analyze why certain images are failing validation"""
    
    # Load test results
    with open('improvement_test_results.json', 'r') as f:
        results = json.load(f)
    
    detector = ValidatedBoardDetector()
    
    # Find images that failed with improved detector but worked with original
    failures = []
    for i, orig_result in enumerate(results['original']['results']):
        imp_result = results['improved']['results'][i]
        
        if orig_result['detected'] and not imp_result['detected']:
            failures.append({
                'image': orig_result['image'],
                'error': imp_result.get('error', 'Unknown')
            })
    
    print(f"Found {len(failures)} images that failed validation but worked originally\n")
    
    # Analyze each failure in detail
    for failure in failures[:5]:  # Analyze first 5
        img_path = Path('dataset/images') / failure['image']
        if not img_path.exists():
            continue
            
        print(f"\n{'='*60}")
        print(f"Analyzing: {failure['image']}")
        print(f"Error: {failure['error']}")
        print('='*60)
        
        image = cv2.imread(str(img_path))
        
        # Run detector with debug to get detailed scores
        try:
            # Temporarily increase threshold to see all candidates
            original_threshold = detector.min_score_threshold
            detector.min_score_threshold = 0.0
            
            board = detector.detect_board(image, debug=True)
            
            detector.min_score_threshold = original_threshold
        except Exception as e:
            print(f"Analysis failed: {e}")
            detector.min_score_threshold = original_threshold
            
        # Visualize the image
        visualize_problem_image(image, failure['image'])


def visualize_problem_image(image, filename):
    """Create visualization to understand the problem"""
    
    # Convert to RGB for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create analysis plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    # Original image
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title(f"Original: {filename}")
    axes[0, 0].axis('off')
    
    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    axes[0, 1].imshow(gray, cmap='gray')
    axes[0, 1].set_title("Grayscale")
    axes[0, 1].axis('off')
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    axes[1, 0].imshow(edges, cmap='gray')
    axes[1, 0].set_title("Edges (Canny)")
    axes[1, 0].axis('off')
    
    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)
    axes[1, 1].imshow(thresh, cmap='gray')
    axes[1, 1].set_title("Adaptive Threshold")
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    output_file = f"failure_analysis_{Path(filename).stem}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved analysis to {output_file}")
    plt.close()


def analyze_ui_detection():
    """Analyze why UI screenshots are passing validation"""
    
    print("\n" + "="*60)
    print("Analyzing UI Detection Issues")
    print("="*60)
    
    # Test on known UI screenshots
    ui_images = [
        ('dataset/images/10.png', 'Browser UI screenshot'),
        ('dataset/images/9.png', 'Chess.com UI'),
    ]
    
    detector = ValidatedBoardDetector()
    
    for img_path, description in ui_images:
        if not Path(img_path).exists():
            continue
            
        print(f"\n{description}: {img_path}")
        
        image = cv2.imread(img_path)
        
        # Get detailed scores
        try:
            detector.min_score_threshold = 0.0  # See all scores
            board = detector.detect_board(image, debug=True)
            detector.min_score_threshold = 0.25  # Reset
            print("  ⚠️  This UI screenshot was accepted!")
        except Exception as e:
            detector.min_score_threshold = 0.25  # Reset
            print("  ✅ Correctly rejected")


def suggest_improvements():
    """Based on analysis, suggest specific improvements"""
    
    print("\n" + "="*60)
    print("SUGGESTED IMPROVEMENTS")
    print("="*60)
    
    print("""
1. JPEG Artifact Handling:
   - Add image quality detection
   - Reduce checkerboard pattern weight for low-quality images
   - Use multiple pattern detection methods
   
2. UI Detection Improvements:
   - Check for UI-specific elements (buttons, text boxes, menus)
   - Analyze color distribution (UIs often have solid color regions)
   - Check for non-square aspect ratios in parent regions
   
3. Scoring Adjustments:
   - Make checkerboard detection more robust to compression
   - Add "board likelihood" based on overall appearance
   - Consider multiple candidates more carefully
   
4. Margin Detection:
   - Be more conservative with cropping
   - Validate that cropped region still looks like a board
   - Add option to disable margins for problematic images
""")


if __name__ == "__main__":
    print("=== Failure Mode Analysis ===\n")
    
    analyze_failed_detections()
    analyze_ui_detection()
    suggest_improvements()