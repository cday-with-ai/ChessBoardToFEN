#!/usr/bin/env python3
"""
Validates the chess position dataset.
Checks FEN strings and ensures all images exist.
"""

import json
import hashlib
from pathlib import Path


def validate_fen(fen):
    """Validate a FEN string"""
    parts = fen.split()
    
    # Basic structure check
    if len(parts) < 1:
        return False, "FEN must have at least the board position"
    
    # Validate board position
    rows = parts[0].split('/')
    if len(rows) != 8:
        return False, "Board must have 8 rows"
    
    # Check each row
    for i, row in enumerate(rows):
        count = 0
        for char in row:
            if char.isdigit():
                if char == '0':
                    return False, f"Row {i+1}: Zero is not valid"
                count += int(char)
            elif char in 'kqrbnpKQRBNP':
                count += 1
            else:
                return False, f"Row {i+1}: Invalid character '{char}'"
        
        if count != 8:
            return False, f"Row {i+1}: Has {count} squares, should have 8"
    
    # Check for exactly one king per side
    board_str = parts[0].replace('/', '')
    if board_str.count('K') != 1:
        return False, "White must have exactly one king"
    if board_str.count('k') != 1:
        return False, "Black must have exactly one king"
    
    # If full FEN, validate other parts
    if len(parts) >= 2:
        if parts[1] not in ['w', 'b']:
            return False, "Active color must be 'w' or 'b'"
    
    if len(parts) >= 3:
        castling = parts[2]
        valid_castling = set('KQkq-')
        if not all(c in valid_castling for c in castling):
            return False, f"Invalid castling rights: {castling}"
    
    return True, "Valid"


def get_file_hash(file_path):
    """Calculate SHA256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def validate_dataset():
    """Validate the entire dataset"""
    manifest_path = Path(__file__).parent / 'manifest.json'
    dataset_dir = Path(__file__).parent
    
    print("Validating chess position dataset...\n")
    
    # Load manifest
    try:
        with open(manifest_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading manifest.json: {e}")
        return
    
    positions = data.get('positions', [])
    if not positions:
        print("No positions found in dataset")
        return
    
    # Track statistics
    total = len(positions)
    valid = 0
    errors = []
    seen_fens = {}
    seen_images = {}
    image_hashes = {}  # Track image content hashes
    
    # Validate each position
    for i, pos in enumerate(positions):
        position_id = pos.get('id', i + 1)
        issues = []
        
        # Check required fields
        if 'fen' not in pos:
            issues.append("Missing FEN")
        else:
            # Validate FEN
            is_valid, msg = validate_fen(pos['fen'])
            if not is_valid:
                issues.append(f"Invalid FEN: {msg}")
            else:
                # Check for duplicate FENs
                if pos['fen'] in seen_fens:
                    issues.append(f"Duplicate FEN (also in {seen_fens[pos['fen']]})")
                seen_fens[pos['fen']] = position_id
        
        if 'image' not in pos:
            issues.append("Missing image path")
        else:
            # Check image exists
            image_path = dataset_dir / pos['image']
            if not image_path.exists():
                issues.append(f"Image not found: {pos['image']}")
            else:
                # Check for duplicate image filenames
                if pos['image'] in seen_images:
                    issues.append(f"Duplicate image filename (also in {seen_images[pos['image']]})")
                seen_images[pos['image']] = position_id
                
                # Check for duplicate image content
                try:
                    img_hash = get_file_hash(image_path)
                    if img_hash in image_hashes:
                        issues.append(f"Duplicate image content (same as {image_hashes[img_hash]})")
                    else:
                        image_hashes[img_hash] = position_id
                except Exception as e:
                    issues.append(f"Could not check image hash: {e}")
        
        # Report issues
        if issues:
            errors.append(f"{position_id}: " + "; ".join(issues))
        else:
            valid += 1
    
    # Print results
    print(f"Total positions: {total}")
    print(f"Valid positions: {valid}")
    print(f"Errors found: {len(errors)}\n")
    
    if errors:
        print("Errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("✓ All positions are valid!")
    
    # Additional statistics
    print(f"\nDataset statistics:")
    print(f"  Unique FEN positions: {len(seen_fens)}")
    print(f"  Unique image files: {len(seen_images)}")
    print(f"  Unique image content: {len(image_hashes)}")
    
    # Check for orphaned images
    images_dir = dataset_dir / 'images'
    if images_dir.exists():
        actual_images = set(f"images/{img.name}" for img in images_dir.iterdir() 
                          if img.is_file() and img.suffix.lower() in ['.png', '.jpg', '.jpeg'])
        referenced_images = set(pos.get('image', '') for pos in positions)
        orphaned = actual_images - referenced_images
        
        if orphaned:
            print(f"\nOrphaned images (not referenced in manifest):")
            for img in sorted(orphaned):
                print(f"  - {img}")


if __name__ == '__main__':
    validate_dataset()