#!/usr/bin/env python3
"""
Generate a quick summary report without individual visualizations
"""

import cv2
import numpy as np
import json
from pathlib import Path
from app.models.board_detector import BoardDetector
from app.models.piece_classifier import PieceClassifier
from app.models.image_type_classifier import ImageTypeClassifier
from app.models.fen_builder import build_fen_from_squares


def calculate_fen_accuracy(fen1, fen2):
    """Calculate accuracy between two FEN strings"""
    pos1 = fen1.split()[0]
    pos2 = fen2.split()[0]
    
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
    
    if len(expanded1) != len(expanded2):
        return 0.0
    
    correct = sum(1 for a, b in zip(expanded1, expanded2) if a == b)
    total = len(expanded1)
    
    return (correct / total) * 100 if total > 0 else 0.0


def process_batch(positions, detector, classifier, image_classifier):
    """Process a batch of positions"""
    results = []
    
    for i, position in enumerate(positions):
        image_path = Path('dataset') / position['image']
        
        if not image_path.exists():
            continue
        
        # Load and classify image
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        
        image_type, type_conf, _ = image_classifier.classify_image(image)
        
        result = {
            'id': position['id'],
            'path': str(image_path),
            'expected_fen': position['fen'],
            'image_type': image_type.value
        }
        
        try:
            # Quick processing
            board = detector.detect_board(image)
            squares = detector.extract_squares(board)
            classifications = classifier.classify_board(squares)
            actual_fen = build_fen_from_squares(classifications, 0.3)
            accuracy = calculate_fen_accuracy(position['fen'], actual_fen)
            
            result['board_detected'] = True
            result['actual_fen'] = actual_fen
            result['accuracy'] = accuracy
            result['avg_confidence'] = float(np.mean([conf for _, conf in classifications]))
            
        except Exception as e:
            result['board_detected'] = False
            result['error'] = str(e)
            result['accuracy'] = 0.0
        
        results.append(result)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(positions)}...")
    
    return results


def generate_summary_report(results, output_path):
    """Generate a summary markdown report"""
    with open(output_path, 'w') as f:
        f.write("# Chess Board Recognition Summary Report\n\n")
        
        # Overall statistics
        total = len(results)
        detected = sum(1 for r in results if r.get('board_detected', False))
        avg_accuracy = np.mean([r.get('accuracy', 0) for r in results])
        
        f.write("## Overall Statistics\n\n")
        f.write(f"- **Total Images**: {total}\n")
        f.write(f"- **Boards Detected**: {detected} ({detected/total*100:.1f}%)\n")
        f.write(f"- **Average Accuracy**: {avg_accuracy:.1f}%\n\n")
        
        # Accuracy distribution
        accuracy_ranges = {
            '100%': 0,
            '90-99%': 0,
            '80-89%': 0,
            '70-79%': 0,
            '60-69%': 0,
            '50-59%': 0,
            '40-49%': 0,
            '30-39%': 0,
            '20-29%': 0,
            '10-19%': 0,
            '0-9%': 0
        }
        
        for r in results:
            acc = r.get('accuracy', 0)
            if acc == 100:
                accuracy_ranges['100%'] += 1
            elif acc >= 90:
                accuracy_ranges['90-99%'] += 1
            elif acc >= 80:
                accuracy_ranges['80-89%'] += 1
            elif acc >= 70:
                accuracy_ranges['70-79%'] += 1
            elif acc >= 60:
                accuracy_ranges['60-69%'] += 1
            elif acc >= 50:
                accuracy_ranges['50-59%'] += 1
            elif acc >= 40:
                accuracy_ranges['40-49%'] += 1
            elif acc >= 30:
                accuracy_ranges['30-39%'] += 1
            elif acc >= 20:
                accuracy_ranges['20-29%'] += 1
            elif acc >= 10:
                accuracy_ranges['10-19%'] += 1
            else:
                accuracy_ranges['0-9%'] += 1
        
        f.write("## Accuracy Distribution\n\n")
        for range_name, count in accuracy_ranges.items():
            percentage = count / total * 100 if total > 0 else 0
            bar = '█' * int(percentage / 2)
            f.write(f"- {range_name}: {count:3d} ({percentage:5.1f}%) {bar}\n")
        
        # By image type
        f.write("\n## Results by Image Type\n\n")
        type_stats = {}
        for r in results:
            img_type = r.get('image_type', 'unknown')
            if img_type not in type_stats:
                type_stats[img_type] = {
                    'count': 0, 
                    'detected': 0,
                    'accuracies': [],
                    'perfect': 0
                }
            
            type_stats[img_type]['count'] += 1
            if r.get('board_detected', False):
                type_stats[img_type]['detected'] += 1
                type_stats[img_type]['accuracies'].append(r.get('accuracy', 0))
                if r.get('accuracy', 0) == 100:
                    type_stats[img_type]['perfect'] += 1
        
        for img_type, stats in sorted(type_stats.items()):
            avg_acc = np.mean(stats['accuracies']) if stats['accuracies'] else 0
            f.write(f"### {img_type}\n")
            f.write(f"- Count: {stats['count']}\n")
            f.write(f"- Detection Rate: {stats['detected']/stats['count']*100:.1f}%\n")
            f.write(f"- Average Accuracy: {avg_acc:.1f}%\n")
            f.write(f"- Perfect Matches: {stats['perfect']}\n\n")
        
        # Worst performers
        f.write("## Worst Performing Images (< 50% accuracy)\n\n")
        worst = [r for r in results if r.get('accuracy', 0) < 50]
        worst.sort(key=lambda x: x.get('accuracy', 0))
        
        for r in worst[:20]:  # Show top 20 worst
            f.write(f"- **{Path(r['path']).name}** ({r['image_type']}): ")
            f.write(f"{r.get('accuracy', 0):.1f}% accuracy")
            if not r.get('board_detected', False):
                f.write(f" - Board not detected: {r.get('error', 'Unknown error')}")
            f.write("\n")
        
        # Best performers
        f.write("\n## Best Performing Images (100% accuracy)\n\n")
        perfect = [r for r in results if r.get('accuracy', 0) == 100]
        
        for r in perfect[:20]:  # Show top 20
            f.write(f"- **{Path(r['path']).name}** ({r['image_type']})\n")


def main():
    # Load manifest
    with open('dataset/manifest.json', 'r') as f:
        manifest = json.load(f)
    
    # Initialize components
    print("Initializing components...")
    detector = BoardDetector()
    classifier = PieceClassifier()
    image_classifier = ImageTypeClassifier()
    
    # Process all positions
    positions = manifest['positions']
    print(f"\nProcessing {len(positions)} images...")
    
    results = process_batch(positions, detector, classifier, image_classifier)
    
    # Save results
    output_dir = Path('dataset_report')
    output_dir.mkdir(exist_ok=True)
    
    # Save raw results
    with open(output_dir / 'quick_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate report
    print("\nGenerating summary report...")
    report_path = output_dir / 'summary_report.md'
    generate_summary_report(results, report_path)
    
    print(f"\nReport generated: {report_path}")


if __name__ == "__main__":
    main()