#!/usr/bin/env python3
"""
Generate a comprehensive report showing board detection and FEN recognition results
for all images in the dataset
"""

import cv2
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from app.models.board_detector import BoardDetector
from app.models.piece_classifier import PieceClassifier
from app.models.image_type_classifier import ImageTypeClassifier
from app.models.fen_builder import build_fen_from_squares
import os


def calculate_fen_accuracy(fen1, fen2):
    """Calculate accuracy between two FEN strings"""
    # Split FEN into parts
    pos1 = fen1.split()[0]
    pos2 = fen2.split()[0]
    
    # Expand FEN notation
    def expand_fen(fen):
        expanded = ""
        for char in fen:
            if char.isdigit():
                expanded += '.' * int(char)
            elif char == '/':
                pass
            else:
                expanded += char
        return expanded
    
    expanded1 = expand_fen(pos1)
    expanded2 = expand_fen(pos2)
    
    # Calculate accuracy
    if len(expanded1) != len(expanded2):
        return 0.0
    
    correct = sum(1 for a, b in zip(expanded1, expanded2) if a == b)
    total = len(expanded1)
    
    return (correct / total) * 100 if total > 0 else 0.0


def process_image(image_path, expected_fen, detector, classifier, image_classifier, output_dir):
    """Process a single image and generate visualization"""
    print(f"Processing {image_path}...")
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Classify image type
    image_type, type_conf, features = image_classifier.classify_image(image)
    
    result = {
        'path': str(image_path),
        'expected_fen': expected_fen,
        'image_type': image_type.value,
        'type_confidence': type_conf
    }
    
    try:
        # Detect board
        board = detector.detect_board(image)
        board_rgb = cv2.cvtColor(board, cv2.COLOR_BGR2RGB)
        
        # Extract squares
        squares = detector.extract_squares(board)
        
        # Classify pieces
        classifications = classifier.classify_board(squares)
        
        # Build FEN
        actual_fen = build_fen_from_squares(classifications, 0.3)
        
        # Calculate accuracy
        accuracy = calculate_fen_accuracy(expected_fen, actual_fen)
        
        result['actual_fen'] = actual_fen
        result['accuracy'] = accuracy
        result['board_detected'] = True
        result['avg_confidence'] = float(np.mean([conf for _, conf in classifications]))
        
        # Create visualization
        fig = plt.figure(figsize=(20, 8))
        
        # Original image
        ax1 = plt.subplot(1, 4, 1)
        # Resize for display if too large
        display_img = image_rgb
        max_size = 600
        h, w = display_img.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            display_img = cv2.resize(display_img, (new_w, new_h))
        ax1.imshow(display_img)
        ax1.set_title(f"Original\n({image_type.value})")
        ax1.axis('off')
        
        # Detected board
        ax2 = plt.subplot(1, 4, 2)
        ax2.imshow(board_rgb)
        ax2.set_title("Detected Board")
        ax2.axis('off')
        
        # Board with grid overlay
        ax3 = plt.subplot(1, 4, 3)
        board_with_grid = board_rgb.copy()
        h, w = board_with_grid.shape[:2]
        sq_h, sq_w = h // 8, w // 8
        
        # Draw grid
        for i in range(9):
            cv2.line(board_with_grid, (i * sq_w, 0), (i * sq_w, h), (255, 0, 0), 2)
            cv2.line(board_with_grid, (0, i * sq_h), (w, i * sq_h), (255, 0, 0), 2)
        
        ax3.imshow(board_with_grid)
        ax3.set_title("Grid Overlay")
        ax3.axis('off')
        
        # FEN comparison
        ax4 = plt.subplot(1, 4, 4)
        ax4.text(0.1, 0.9, "Expected FEN:", fontsize=12, fontweight='bold', transform=ax4.transAxes)
        ax4.text(0.1, 0.8, expected_fen.split()[0], fontsize=10, family='monospace', 
                transform=ax4.transAxes, wrap=True)
        
        ax4.text(0.1, 0.6, "Actual FEN:", fontsize=12, fontweight='bold', transform=ax4.transAxes)
        ax4.text(0.1, 0.5, actual_fen.split()[0], fontsize=10, family='monospace', 
                transform=ax4.transAxes, wrap=True)
        
        ax4.text(0.1, 0.3, f"Accuracy: {accuracy:.1f}%", fontsize=14, 
                color='green' if accuracy > 90 else 'orange' if accuracy > 70 else 'red',
                fontweight='bold', transform=ax4.transAxes)
        
        ax4.text(0.1, 0.2, f"Avg Confidence: {result['avg_confidence']:.3f}", 
                fontsize=12, transform=ax4.transAxes)
        
        ax4.axis('off')
        
        plt.tight_layout()
        
        # Save figure
        img_name = Path(image_path).stem
        fig_path = output_dir / f"{img_name}_analysis.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        result['visualization'] = str(fig_path)
        
    except Exception as e:
        print(f"  Error: {e}")
        result['board_detected'] = False
        result['error'] = str(e)
        result['actual_fen'] = "8/8/8/8/8/8/8/8 w KQkq - 0 1"
        result['accuracy'] = 0.0
        result['avg_confidence'] = 0.0
    
    return result


def generate_markdown_report(results, output_path):
    """Generate markdown report from results"""
    with open(output_path, 'w') as f:
        f.write("# Chess Board Recognition Dataset Report\n\n")
        f.write("This report shows the board detection and FEN recognition results for all images in the dataset.\n\n")
        
        # Summary statistics
        total = len(results)
        detected = sum(1 for r in results if r.get('board_detected', False))
        avg_accuracy = np.mean([r.get('accuracy', 0) for r in results])
        perfect = sum(1 for r in results if r.get('accuracy', 0) == 100)
        good = sum(1 for r in results if r.get('accuracy', 0) >= 90)
        
        f.write("## Summary Statistics\n\n")
        f.write(f"- **Total Images**: {total}\n")
        f.write(f"- **Boards Detected**: {detected} ({detected/total*100:.1f}%)\n")
        f.write(f"- **Average Accuracy**: {avg_accuracy:.1f}%\n")
        f.write(f"- **Perfect Matches**: {perfect} ({perfect/total*100:.1f}%)\n")
        f.write(f"- **Good Matches (≥90%)**: {good} ({good/total*100:.1f}%)\n\n")
        
        # Image type breakdown
        f.write("## Results by Image Type\n\n")
        type_stats = {}
        for r in results:
            img_type = r.get('image_type', 'unknown')
            if img_type not in type_stats:
                type_stats[img_type] = {'count': 0, 'accuracy': [], 'detected': 0}
            type_stats[img_type]['count'] += 1
            type_stats[img_type]['accuracy'].append(r.get('accuracy', 0))
            if r.get('board_detected', False):
                type_stats[img_type]['detected'] += 1
        
        for img_type, stats in sorted(type_stats.items()):
            avg_acc = np.mean(stats['accuracy']) if stats['accuracy'] else 0
            f.write(f"### {img_type}\n")
            f.write(f"- Count: {stats['count']}\n")
            f.write(f"- Detection Rate: {stats['detected']}/{stats['count']} ({stats['detected']/stats['count']*100:.1f}%)\n")
            f.write(f"- Average Accuracy: {avg_acc:.1f}%\n\n")
        
        # Individual results
        f.write("## Individual Image Results\n\n")
        
        # Sort by accuracy (worst first)
        results.sort(key=lambda x: x.get('accuracy', 0))
        
        for i, result in enumerate(results, 1):
            img_name = Path(result['path']).name
            f.write(f"### {i}. {img_name}\n\n")
            
            if 'visualization' in result and Path(result['visualization']).exists():
                # Use relative path for the image
                vis_path = Path(result['visualization']).name
                f.write(f"![{img_name} analysis]({vis_path})\n\n")
            
            f.write(f"- **Image Type**: {result.get('image_type', 'unknown')}\n")
            f.write(f"- **Board Detected**: {'✅' if result.get('board_detected', False) else '❌'}\n")
            
            if result.get('board_detected', False):
                f.write(f"- **Accuracy**: {result.get('accuracy', 0):.1f}%\n")
                f.write(f"- **Average Confidence**: {result.get('avg_confidence', 0):.3f}\n")
                f.write(f"- **Expected FEN**: `{result['expected_fen'].split()[0]}`\n")
                f.write(f"- **Actual FEN**: `{result.get('actual_fen', 'N/A').split()[0]}`\n")
            else:
                f.write(f"- **Error**: {result.get('error', 'Unknown error')}\n")
            
            f.write("\n---\n\n")


def main():
    # Load manifest
    with open('dataset/manifest.json', 'r') as f:
        manifest = json.load(f)
    
    # Create output directory
    output_dir = Path('dataset_report')
    output_dir.mkdir(exist_ok=True)
    
    # Initialize components
    detector = BoardDetector()
    classifier = PieceClassifier()
    image_classifier = ImageTypeClassifier()
    
    # Process all images
    results = []
    positions = manifest['positions']
    
    # Limit to first N images for testing (remove this for full dataset)
    positions = positions[:50]  # Process first 50 images
    
    print(f"Processing {len(positions)} images...")
    
    for i, position in enumerate(positions):
        print(f"\n[{i+1}/{len(positions)}] ", end='')
        
        image_path = Path('dataset') / position['image']
        
        if not image_path.exists():
            print(f"Skipping {image_path} - not found")
            continue
        
        result = process_image(
            image_path, 
            position['fen'],
            detector,
            classifier,
            image_classifier,
            output_dir
        )
        
        if result:
            results.append(result)
    
    # Generate report
    print("\n\nGenerating markdown report...")
    report_path = output_dir / 'dataset_analysis.md'
    generate_markdown_report(results, report_path)
    
    print(f"\nReport generated: {report_path}")
    print(f"Visualizations saved in: {output_dir}")
    
    # Also save raw results as JSON
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()