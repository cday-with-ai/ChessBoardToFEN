#!/usr/bin/env python3
"""
Helper script to add new chess positions to the dataset.
Usage: python add_position.py --image path/to/image.png --fen "FEN string" --description "Description"

Supports absolute paths, relative paths, and ~ (home directory) for images.
Examples:
  python add_position.py --image /Users/name/chess.png --fen "..." 
  python add_position.py --image ./screenshots/position.jpg --fen "..."
  python add_position.py --image ~/Downloads/game.png --fen "..."
  python add_position.py --image ../chess_images/board.jpg --fen "..."
"""

import json
import shutil
import argparse
import hashlib
from pathlib import Path
from datetime import datetime


def validate_fen(fen):
    """Basic FEN validation"""
    parts = fen.split()
    if len(parts) < 1:
        return False
    
    # Check board part
    rows = parts[0].split('/')
    if len(rows) != 8:
        return False
    
    # Check each row has 8 squares
    for row in rows:
        count = 0
        for char in row:
            if char.isdigit():
                count += int(char)
            elif char in 'kqrbnpKQRBNP':
                count += 1
            else:
                return False
        if count != 8:
            return False
    
    return True


def get_file_hash(file_path):
    """Calculate SHA256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def find_duplicate_image(new_image_path, images_dir, manifest_data):
    """Check if an image with the same content already exists"""
    new_image_hash = get_file_hash(new_image_path)
    
    for pos in manifest_data['positions']:
        existing_image_path = images_dir.parent / pos['image']
        if existing_image_path.exists():
            existing_hash = get_file_hash(existing_image_path)
            if existing_hash == new_image_hash:
                return pos
    
    return None


def get_next_id(manifest_path):
    """Get the next position ID"""
    with open(manifest_path, 'r') as f:
        data = json.load(f)
    
    if not data['positions']:
        return 1
    
    # Extract numbers from IDs and find max
    max_num = 0
    for pos in data['positions']:
        try:
            # Handle both old format (position_XXX) and new format (just number)
            if isinstance(pos['id'], int):
                num = pos['id']
            else:
                num = int(pos['id'].split('_')[1]) if '_' in str(pos['id']) else int(pos['id'])
            max_num = max(max_num, num)
        except:
            pass
    
    return max_num + 1


def add_position(image_path, fen, description=None, tags=None):
    """Add a new position to the dataset"""
    manifest_path = Path(__file__).parent / 'manifest.json'
    images_dir = Path(__file__).parent / 'images'
    
    # Validate FEN
    if not validate_fen(fen):
        print(f"Error: Invalid FEN string: {fen}")
        return False
    
    # Check image exists - handle absolute, relative, and ~ paths
    image_path = Path(image_path).expanduser()  # Expands ~ to home directory
    if not image_path.is_absolute():
        # If relative path, make it relative to current working directory
        image_path = Path.cwd() / image_path
    
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return False
    
    # Load manifest
    with open(manifest_path, 'r') as f:
        data = json.load(f)
    
    # Check for duplicate image
    duplicate = find_duplicate_image(image_path, images_dir, data)
    if duplicate:
        print(f"Error: This image already exists in the dataset!")
        print(f"  Existing position: {duplicate['id']}")
        print(f"  Existing image: {duplicate['image']}")
        print(f"  Existing FEN: {duplicate['fen']}")
        if 'description' in duplicate:
            print(f"  Description: {duplicate['description']}")
        return False
    
    # Get next ID
    position_id = get_next_id(manifest_path)
    
    # Copy image to dataset
    image_ext = image_path.suffix
    new_image_name = f"{position_id}{image_ext}"
    new_image_path = images_dir / new_image_name
    shutil.copy2(image_path, new_image_path)
    
    # Create new entry
    new_entry = {
        "id": position_id,
        "image": f"images/{new_image_name}",
        "fen": fen,
        "date_added": datetime.now().strftime("%Y-%m-%d")
    }
    
    if description:
        new_entry["description"] = description
    
    if tags:
        new_entry["tags"] = tags
    
    # Add to manifest
    data['positions'].append(new_entry)
    
    # Save manifest
    with open(manifest_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Successfully added {position_id}:")
    print(f"  Image: {new_image_path}")
    print(f"  FEN: {fen}")
    if description:
        print(f"  Description: {description}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Add a chess position to the dataset')
    parser.add_argument('--image', required=True, help='Path to the chess board image (absolute or relative)')
    parser.add_argument('--fen', required=True, help='FEN string for the position')
    parser.add_argument('--description', help='Optional description of the position')
    parser.add_argument('--tags', nargs='+', help='Optional tags (space-separated)')
    
    args = parser.parse_args()
    
    add_position(args.image, args.fen, args.description, args.tags)


if __name__ == '__main__':
    main()