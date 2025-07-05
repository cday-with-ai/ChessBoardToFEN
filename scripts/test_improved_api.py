#!/usr/bin/env python3
"""
Test the API with the improved board detector
"""

import requests
import base64
from pathlib import Path
import json


def test_api_endpoint(image_path: str, endpoint: str = "http://localhost:8000/api/recognize-position"):
    """Test the API with a specific image"""
    
    # Check if file exists
    if not Path(image_path).exists():
        print(f"❌ File not found: {image_path}")
        return None
    
    # Read and encode image
    with open(image_path, 'rb') as f:
        files = {'file': (Path(image_path).name, f, 'image/png')}
        
        try:
            # Make request
            response = requests.post(endpoint, files=files)
            
            if response.status_code == 200:
                result = response.json()
                return result
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.ConnectionError:
            print("❌ Could not connect to API. Make sure the server is running.")
            return None
        except Exception as e:
            print(f"❌ Error: {e}")
            return None


def test_specific_improvements():
    """Test specific cases that should be improved"""
    
    print("=== Testing Improved API ===\n")
    
    test_cases = [
        {
            'path': 'dataset/images/12.png',
            'description': 'Board with UI elements - should improve with margin detection',
            'expected_improvement': True
        },
        {
            'path': 'dataset/images/10.png',
            'description': 'UI screenshot - should be handled better',
            'expected_improvement': True
        },
        {
            'path': 'dataset/images/1.jpeg',
            'description': 'Wooden board - improved pattern detection',
            'expected_improvement': True
        },
        {
            'path': 'dataset/images/4.png',
            'description': 'Clean board - should maintain high accuracy',
            'expected_improvement': False
        }
    ]
    
    # Test both standard and debug endpoints
    endpoints = {
        'standard': 'http://localhost:8000/api/recognize-position',
        'debug': 'http://localhost:8000/api/recognize-position/debug'
    }
    
    for test_case in test_cases:
        print(f"\n{'='*60}")
        print(f"Testing: {test_case['path']}")
        print(f"Description: {test_case['description']}")
        print('='*60)
        
        # Test standard endpoint
        print("\nStandard endpoint:")
        result = test_api_endpoint(test_case['path'], endpoints['standard'])
        if result:
            print(f"✅ FEN: {result['fen']}")
            print(f"   Confidence: {result['confidence']:.2%}")
            print(f"   Processing time: {result['processing_time']:.3f}s")
            if 'image_type' in result:
                print(f"   Image type: {result['image_type']}")
        
        # Test debug endpoint
        print("\nDebug endpoint:")
        result = test_api_endpoint(test_case['path'], endpoints['debug'])
        if result:
            print(f"✅ Board detected: {result['debug_info']['board_detected']}")
            print(f"   Average confidence: {result['confidence']:.2%}")
            
            # Show confidence heatmap
            if 'square_confidences' in result['debug_info']:
                print("\n   Confidence heatmap (low squares):")
                confidences = result['debug_info']['square_confidences']
                for i in range(8):
                    for j in range(8):
                        conf = confidences[i][j]
                        if conf < 0.5:
                            square = chr(ord('a') + j) + str(8 - i)
                            print(f"     {square}: {conf:.2%}")


def test_api_performance():
    """Test overall API performance with improved detector"""
    
    print("\n\n=== Performance Test ===\n")
    
    # Test on a batch of images
    test_images = [
        f'dataset/images/{i}.png' for i in range(1, 21)
        if Path(f'dataset/images/{i}.png').exists()
    ]
    
    test_images.extend([
        f'dataset/images/{i}.jpeg' for i in range(1, 21)
        if Path(f'dataset/images/{i}.jpeg').exists()
    ])
    
    successful = 0
    total_confidence = 0
    total_time = 0
    
    endpoint = 'http://localhost:8000/api/recognize-position'
    
    for image_path in test_images[:10]:  # Test first 10
        result = test_api_endpoint(image_path, endpoint)
        if result:
            successful += 1
            total_confidence += result['confidence']
            total_time += result['processing_time']
            print(f"✅ {Path(image_path).name}: {result['confidence']:.1%} confidence, {result['processing_time']:.2f}s")
        else:
            print(f"❌ {Path(image_path).name}: Failed")
    
    if successful > 0:
        print(f"\nSummary:")
        print(f"  Success rate: {successful}/{len(test_images)} ({successful/len(test_images):.1%})")
        print(f"  Average confidence: {total_confidence/successful:.1%}")
        print(f"  Average processing time: {total_time/successful:.3f}s")


if __name__ == "__main__":
    print("Testing improved API...\n")
    print("Make sure the API server is running:")
    print("  uvicorn app.main:app --reload\n")
    
    test_specific_improvements()
    test_api_performance()