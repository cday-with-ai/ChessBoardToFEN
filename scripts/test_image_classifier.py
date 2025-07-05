#!/usr/bin/env python3
"""
Test image type classification
"""

import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from app.models.image_type_classifier import ImageTypeClassifier, ImageType
from app.models.adaptive_board_processor import AdaptiveBoardProcessor

def test_classifier():
    classifier = ImageTypeClassifier()
    
    # Test images
    test_images = [
        ("dataset/images/1.jpeg", "Streaming screenshot"),
        ("dataset/images/10.png", "Browser screenshot"),
        ("dataset/images/84.png", "ChessVision clean"),
        ("dataset/images/100.png", "ChessVision clean 2"),
    ]
    
    results = []
    
    for img_path, description in test_images:
        if not Path(img_path).exists():
            print(f"Skipping {img_path} - not found")
            continue
            
        image = cv2.imread(img_path)
        image_type, confidence, features = classifier.classify_image(image)
        
        results.append({
            'path': img_path,
            'description': description,
            'type': image_type,
            'confidence': confidence,
            'features': features
        })
        
        print(f"\n{description} ({img_path}):")
        print(f"  Type: {image_type.value}")
        print(f"  Confidence: {confidence:.2f}")
        print(f"  Features:")
        feature_order = ['has_perfect_grid', 'uniform_squares', 'has_ui_elements', 
                        'has_wood_texture', 'has_3d_pieces', 'perspective_distortion',
                        'color_variance', 'edge_sharpness']
        for key in feature_order:
            if key in features:
                value = features[key]
                if isinstance(value, bool):
                    print(f"    {key}: {value}")
                elif isinstance(value, float):
                    print(f"    {key}: {value:.3f}")
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, result in enumerate(results[:4]):
        image = cv2.imread(result['path'])
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize for display
        max_dim = 400
        h, w = image_rgb.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            image_rgb = cv2.resize(image_rgb, (new_w, new_h))
        
        axes[i].imshow(image_rgb)
        axes[i].set_title(f"{result['description']}\n{result['type'].value} ({result['confidence']:.2f})")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('image_classification_results.png')
    print("\nVisualization saved to image_classification_results.png")

def test_adaptive_processor():
    """Test the full adaptive processing pipeline"""
    processor = AdaptiveBoardProcessor()
    
    print("\n" + "="*60)
    print("Testing Adaptive Board Processor")
    print("="*60)
    
    # Test on a clean digital image
    print("\nTesting on clean digital board...")
    image = cv2.imread("dataset/images/84.png")
    
    try:
        classifications, metadata = processor.process_image(image)
        
        print(f"Successfully processed!")
        print(f"Image type: {metadata['image_type']}")
        print(f"Type confidence: {metadata['type_confidence']:.2f}")
        
        # Count pieces
        piece_counts = {}
        for piece, conf in classifications:
            piece_counts[piece] = piece_counts.get(piece, 0) + 1
        
        print("Piece distribution:")
        for piece, count in sorted(piece_counts.items()):
            if piece != 'empty':
                print(f"  {piece}: {count}")
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_classifier()
    test_adaptive_processor()