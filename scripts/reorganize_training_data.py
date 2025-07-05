#!/usr/bin/env python3
"""
Reorganize training data into a unified structure with consistent naming
"""

import os
import sys
import shutil
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_unified_structure():
    """Create new unified training data structure"""
    
    print("=== Creating Unified Training Data Structure ===\n")
    
    # Define the new structure
    unified_dir = Path('training_data')
    
    # Standard naming convention: color-piece
    piece_dirs = [
        'white-king',
        'white-queen',
        'white-rook',
        'white-bishop',
        'white-knight',
        'white-pawn',
        'black-king',
        'black-queen',
        'black-rook',
        'black-bishop',
        'black-knight',
        'black-pawn',
        'empty'
    ]
    
    # Mapping from old names to new names
    name_mapping = {
        # From training_data_new structure
        'white_king': 'white-king',
        'white_queen': 'white-queen',
        'white_rook': 'white-rook',
        'white_bishop': 'white-bishop',
        'white_knight': 'white-knight',
        'white_pawn': 'white-pawn',
        'black_king': 'black-king',
        'black_queen': 'black-queen',
        'black_rook': 'black-rook',
        'black_bishop': 'black-bishop',
        'black_knight': 'black-knight',
        'black_pawn': 'black-pawn',
        'empty': 'empty',
        
        # From old single-letter structure
        'K': 'white-king',
        'Q': 'white-queen',
        'R': 'white-rook',
        'B': 'white-bishop',
        'N': 'white-knight',
        'P': 'white-pawn',
        'k_black': 'black-king',
        'q_black': 'black-queen',
        'r_black': 'black-rook',
        'b_black': 'black-bishop',
        'n_black': 'black-knight',
        'p_black': 'black-pawn'
    }
    
    # Backup existing training_data if it exists
    if unified_dir.exists():
        backup_dir = Path('training_data_old')
        if backup_dir.exists():
            print(f"Removing old backup: {backup_dir}")
            shutil.rmtree(backup_dir)
        print(f"Backing up existing training_data to {backup_dir}")
        shutil.move(str(unified_dir), str(backup_dir))
    
    # Create new structure
    print(f"\nCreating new directory structure...")
    unified_dir.mkdir(exist_ok=True)
    
    for piece_dir in piece_dirs:
        (unified_dir / piece_dir).mkdir(exist_ok=True)
    
    return unified_dir, name_mapping


def copy_from_training_data_new(unified_dir: Path, name_mapping: dict):
    """Copy data from training_data_new (our latest synthetic data)"""
    
    print("\n=== Copying from training_data_new ===")
    
    source_dir = Path('training_data_new')
    if not source_dir.exists():
        print(f"Source directory {source_dir} not found!")
        return 0
    
    total_copied = 0
    
    # Copy pieces
    pieces_dir = source_dir / 'pieces'
    if pieces_dir.exists():
        for old_dir in pieces_dir.iterdir():
            if old_dir.is_dir() and old_dir.name in name_mapping:
                new_name = name_mapping[old_dir.name]
                dest_dir = unified_dir / new_name
                
                # Copy all images
                count = 0
                for img_file in old_dir.glob('*.png'):
                    dest_file = dest_dir / img_file.name
                    shutil.copy2(img_file, dest_file)
                    count += 1
                
                print(f"  {old_dir.name} → {new_name}: {count} images")
                total_copied += count
    
    # Copy empty squares
    empty_dir = source_dir / 'empty'
    if empty_dir.exists():
        dest_dir = unified_dir / 'empty'
        count = 0
        for img_file in empty_dir.glob('*.png'):
            dest_file = dest_dir / img_file.name
            shutil.copy2(img_file, dest_file)
            count += 1
        
        print(f"  empty → empty: {count} images")
        total_copied += count
    
    return total_copied


def create_summary_report(unified_dir: Path):
    """Create a summary of the unified training data"""
    
    print("\n=== Training Data Summary ===\n")
    
    summary_lines = ["# Unified Training Data Structure\n"]
    summary_lines.append("## Directory Structure\n")
    summary_lines.append("```")
    summary_lines.append("training_data/")
    
    total_images = 0
    class_counts = {}
    
    # Count images in each class
    for piece_dir in sorted(unified_dir.iterdir()):
        if piece_dir.is_dir():
            count = len(list(piece_dir.glob('*.png')))
            class_counts[piece_dir.name] = count
            total_images += count
            summary_lines.append(f"├── {piece_dir.name}/ ({count} images)")
    
    summary_lines.append("```\n")
    
    # Statistics
    summary_lines.append("## Statistics\n")
    summary_lines.append(f"- **Total Images**: {total_images:,}")
    summary_lines.append(f"- **Classes**: {len(class_counts)}")
    summary_lines.append(f"- **Average per class**: {total_images // len(class_counts) if class_counts else 0}")
    
    # Class distribution
    summary_lines.append("\n## Class Distribution\n")
    summary_lines.append("| Class | Count |")
    summary_lines.append("|-------|-------|")
    
    for class_name in sorted(class_counts.keys()):
        summary_lines.append(f"| {class_name} | {class_counts[class_name]} |")
    
    # Naming convention
    summary_lines.append("\n## Naming Convention\n")
    summary_lines.append("- Format: `color-piece` (e.g., `white-bishop`, `black-knight`)")
    summary_lines.append("- Special case: `empty` for empty squares")
    summary_lines.append("- All lowercase with hyphen separator")
    
    # Model compatibility
    summary_lines.append("\n## Model Compatibility\n")
    summary_lines.append("The class indices for model training should be:")
    summary_lines.append("```python")
    summary_lines.append("CLASS_INDICES = {")
    
    # Create consistent ordering
    piece_order = [
        'empty',
        'white-king', 'white-queen', 'white-rook', 
        'white-bishop', 'white-knight', 'white-pawn',
        'black-king', 'black-queen', 'black-rook',
        'black-bishop', 'black-knight', 'black-pawn'
    ]
    
    for i, class_name in enumerate(piece_order):
        summary_lines.append(f"    '{class_name}': {i},")
    
    summary_lines.append("}")
    summary_lines.append("```")
    
    # Save summary
    summary_text = '\n'.join(summary_lines)
    
    with open(unified_dir / 'TRAINING_DATA_SUMMARY.md', 'w') as f:
        f.write(summary_text)
    
    print(summary_text)
    
    return total_images


def update_model_config():
    """Update model configuration for new class names"""
    
    config_file = Path('app/models/piece_classifier.py')
    
    if not config_file.exists():
        print("\n⚠️  Could not find piece_classifier.py to update")
        return
    
    print("\n=== Updating Model Configuration ===")
    
    # New class configuration
    new_config = """
# Updated class names for unified training data structure
PIECE_CLASSES = [
    'empty',
    'white-king', 'white-queen', 'white-rook',
    'white-bishop', 'white-knight', 'white-pawn',
    'black-king', 'black-queen', 'black-rook', 
    'black-bishop', 'black-knight', 'black-pawn'
]

# Mapping for backward compatibility
PIECE_MAPPING = {
    'white-king': 'K', 'white-queen': 'Q', 'white-rook': 'R',
    'white-bishop': 'B', 'white-knight': 'N', 'white-pawn': 'P',
    'black-king': 'k', 'black-queen': 'q', 'black-rook': 'r',
    'black-bishop': 'b', 'black-knight': 'n', 'black-pawn': 'p',
    'empty': 'empty'
}
"""
    
    print("Note: You'll need to update piece_classifier.py with the new class names")
    print("when training a new model.")
    
    # Save config reference
    with open('training_data/MODEL_CONFIG.py', 'w') as f:
        f.write(new_config)
    
    print(f"Saved configuration reference to: training_data/MODEL_CONFIG.py")


def cleanup_old_directories():
    """Remove old training data directories"""
    
    print("\n=== Cleanup Options ===")
    
    old_dirs = [
        Path('training_data_new'),
        Path('training_data_clean'),
        Path('training_data_old')  # Created during backup
    ]
    
    existing_old_dirs = [d for d in old_dirs if d.exists()]
    
    if existing_old_dirs:
        print("\nOld directories found:")
        for d in existing_old_dirs:
            size = sum(f.stat().st_size for f in d.rglob('*') if f.is_file()) / (1024 * 1024)
            print(f"  - {d}: {size:.1f} MB")
        
        print("\nTo remove these directories, run:")
        print("  rm -rf training_data_new training_data_clean training_data_old")
    else:
        print("No old directories to clean up")


if __name__ == "__main__":
    print("Training Data Reorganization Script\n")
    
    # Create unified structure
    unified_dir, name_mapping = create_unified_structure()
    
    # Copy from latest training data
    copied = copy_from_training_data_new(unified_dir, name_mapping)
    
    if copied > 0:
        print(f"\n✅ Successfully copied {copied} images")
        
        # Create summary
        total = create_summary_report(unified_dir)
        
        # Update model config
        update_model_config()
        
        # Cleanup suggestions
        cleanup_old_directories()
        
        print("\n✅ Reorganization complete!")
        print(f"   New structure: training_data/")
        print(f"   Total images: {total:,}")
        print("\nNext steps:")
        print("1. Review the new structure in training_data/")
        print("2. Remove old directories if satisfied")
        print("3. Update model training script to use new class names")
    else:
        print("\n❌ No images were copied. Check source directories.")