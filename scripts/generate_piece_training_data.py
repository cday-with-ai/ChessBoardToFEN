import os
import cv2
import numpy as np
from pathlib import Path
import cairosvg
from PIL import Image
import io
import shutil
from datetime import datetime

# Configuration
PIECE_SETS_PATH = "/Users/carsonday/IdeaProjects/Simple-FICS-Interface/pieces"
OUTPUT_DIR = "training_data_clean"
SQUARE_SIZE = 128  # Size of each square image
PIECE_SCALE = 0.85  # How much of the square the piece should occupy

# Board color themes (light, dark)
BOARD_THEMES = [
    # Standard themes
    {"name": "standard", "light": (240, 217, 181), "dark": (181, 136, 99)},
    {"name": "blue", "light": (222, 227, 230), "dark": (140, 162, 173)},
    {"name": "green", "light": (235, 236, 208), "dark": (119, 149, 86)},
    {"name": "purple", "light": (230, 230, 230), "dark": (163, 130, 180)},
    # Dark themes
    {"name": "dark_blue", "light": (90, 90, 100), "dark": (40, 40, 50)},
    {"name": "dark_gray", "light": (80, 80, 80), "dark": (50, 50, 50)},
    # Light themes
    {"name": "light_cream", "light": (255, 248, 220), "dark": (205, 183, 142)},
    {"name": "light_gray", "light": (245, 245, 245), "dark": (200, 200, 200)},
]

# Piece notation mapping
PIECE_MAPPING = {
    'wP': 'P', 'wN': 'N', 'wB': 'B', 'wR': 'R', 'wQ': 'Q', 'wK': 'K',
    'bP': 'p', 'bN': 'n', 'bB': 'b', 'bR': 'r', 'bQ': 'q', 'bK': 'k'
}

def svg_to_png(svg_path, size):
    """Convert SVG to PNG with transparency"""
    png_data = cairosvg.svg2png(
        url=str(svg_path),
        output_width=int(size * PIECE_SCALE),
        output_height=int(size * PIECE_SCALE)
    )
    img = Image.open(io.BytesIO(png_data))
    return np.array(img)

def create_square_with_piece(piece_img, bg_color, square_size):
    """Create a square with a centered piece"""
    # Create background
    square = np.full((square_size, square_size, 3), bg_color, dtype=np.uint8)
    
    # If no piece (empty square), just return the background
    if piece_img is None:
        return square
    
    # Ensure piece has alpha channel
    if piece_img.shape[2] == 3:
        piece_img = cv2.cvtColor(piece_img, cv2.COLOR_RGB2RGBA)
    
    # Get piece dimensions
    piece_h, piece_w = piece_img.shape[:2]
    
    # Calculate position to center the piece
    y_offset = (square_size - piece_h) // 2
    x_offset = (square_size - piece_w) // 2
    
    # Create an overlay for the piece
    overlay = square.copy()
    
    # Extract alpha channel
    alpha = piece_img[:, :, 3] / 255.0
    
    # Blend the piece with the background
    for c in range(3):
        square[y_offset:y_offset+piece_h, x_offset:x_offset+piece_w, c] = \
            (1 - alpha) * square[y_offset:y_offset+piece_h, x_offset:x_offset+piece_w, c] + \
            alpha * piece_img[:, :, c]
    
    return square

def create_training_directories():
    """Create directory structure for training data"""
    base_dir = Path(OUTPUT_DIR)
    
    # Remove old training data if exists
    if base_dir.exists():
        print("Removing old training data...")
        shutil.rmtree(base_dir)
    
    # Create directories for each piece type
    # Use different naming for black pieces to avoid case-insensitive filesystem issues
    piece_dirs = {
        'empty': 'empty',
        'P': 'P', 'N': 'N', 'B': 'B', 'R': 'R', 'Q': 'Q', 'K': 'K',
        'p': 'p_black', 'n': 'n_black', 'b': 'b_black', 'r': 'r_black', 'q': 'q_black', 'k': 'k_black'
    }
    
    for piece_label, dir_name in piece_dirs.items():
        piece_dir = base_dir / "squares" / dir_name
        piece_dir.mkdir(parents=True, exist_ok=True)
    
    return base_dir

def generate_training_data():
    """Generate training data from SVG piece sets"""
    base_dir = create_training_directories()
    
    # Define piece directory mapping (same as in create_training_directories)
    piece_dirs = {
        'empty': 'empty',
        'P': 'P', 'N': 'N', 'B': 'B', 'R': 'R', 'Q': 'Q', 'K': 'K',
        'p': 'p_black', 'n': 'n_black', 'b': 'b_black', 'r': 'r_black', 'q': 'q_black', 'k': 'k_black'
    }
    
    # Get all piece sets
    piece_sets = [d for d in os.listdir(PIECE_SETS_PATH) 
                  if os.path.isdir(os.path.join(PIECE_SETS_PATH, d))]
    
    print(f"Found {len(piece_sets)} piece sets")
    print(f"Using {len(BOARD_THEMES)} board themes")
    print(f"Square size: {SQUARE_SIZE}x{SQUARE_SIZE}")
    print()
    
    total_generated = 0
    
    # Generate empty squares first
    print("Generating empty squares...")
    empty_count = 0
    for theme in BOARD_THEMES:
        for is_light in [True, False]:
            color = theme['light'] if is_light else theme['dark']
            square = create_square_with_piece(None, color, SQUARE_SIZE)
            
            filename = f"empty_{theme['name']}_{'light' if is_light else 'dark'}_{empty_count}.png"
            output_path = base_dir / "squares" / "empty" / filename
            cv2.imwrite(str(output_path), cv2.cvtColor(square, cv2.COLOR_RGB2BGR))
            empty_count += 1
    
    print(f"  Generated {empty_count} empty square variations")
    total_generated += empty_count
    
    # Generate pieces for each set
    for set_idx, piece_set in enumerate(piece_sets):
        print(f"\nProcessing piece set {set_idx+1}/{len(piece_sets)}: {piece_set}")
        set_path = Path(PIECE_SETS_PATH) / piece_set
        
        piece_count = 0
        
        # Process each piece type
        for svg_file in set_path.glob("*.svg"):
            piece_notation = svg_file.stem  # e.g., 'wK', 'bP'
            
            if piece_notation not in PIECE_MAPPING:
                continue
            
            piece_label = PIECE_MAPPING[piece_notation]
            
            try:
                # Load SVG as PNG
                piece_img = svg_to_png(svg_file, SQUARE_SIZE)
                
                # Generate on different backgrounds
                for theme_idx, theme in enumerate(BOARD_THEMES):
                    # Determine if this should be on light or dark square
                    # Mix it up - some pieces on light, some on dark
                    for is_light in [True, False]:
                        color = theme['light'] if is_light else theme['dark']
                        
                        # Create square with piece
                        square = create_square_with_piece(piece_img, color, SQUARE_SIZE)
                        
                        # Save image
                        filename = f"{piece_set}_{piece_notation}_{theme['name']}_{'light' if is_light else 'dark'}.png"
                        # Use the mapped directory name
                        dir_name = piece_dirs.get(piece_label, piece_label)
                        output_path = base_dir / "squares" / dir_name / filename
                        cv2.imwrite(str(output_path), cv2.cvtColor(square, cv2.COLOR_RGB2BGR))
                        piece_count += 1
                
            except Exception as e:
                print(f"  Error processing {svg_file}: {e}")
        
        print(f"  Generated {piece_count} piece images")
        total_generated += piece_count
    
    # Summary statistics
    print(f"\n{'='*60}")
    print(f"TRAINING DATA GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total images generated: {total_generated}")
    print(f"\nImages per category:")
    
    for piece_label, dir_name in piece_dirs.items():
        piece_dir = base_dir / "squares" / dir_name
        count = len(list(piece_dir.glob("*.png")))
        print(f"  {piece_label}: {count} images")
    
    # Create a sample grid to visualize
    create_sample_visualization(base_dir)

def create_sample_visualization(base_dir):
    """Create a visualization showing sample pieces"""
    print("\nCreating sample visualization...")
    
    # Define piece directory mapping
    piece_dirs = {
        'empty': 'empty',
        'P': 'P', 'N': 'N', 'B': 'B', 'R': 'R', 'Q': 'Q', 'K': 'K',
        'p': 'p_black', 'n': 'n_black', 'b': 'b_black', 'r': 'r_black', 'q': 'q_black', 'k': 'k_black'
    }
    
    # Get one sample from each category
    samples = []
    labels = []
    
    for piece_label, dir_name in piece_dirs.items():
        piece_dir = base_dir / "squares" / dir_name
        sample_files = list(piece_dir.glob("*.png"))[:5]  # Get up to 5 samples
        
        for f in sample_files:
            img = cv2.imread(str(f))
            if img is not None:
                samples.append(img)
                labels.append(piece_label)
    
    if samples:
        # Create grid
        rows = len(samples) // 13 + (1 if len(samples) % 13 else 0)
        grid_width = min(13, len(samples)) * SQUARE_SIZE
        grid_height = rows * SQUARE_SIZE
        
        grid = np.full((grid_height, grid_width, 3), 255, dtype=np.uint8)
        
        for idx, (img, label) in enumerate(zip(samples, labels)):
            row = idx // 13
            col = idx % 13
            y = row * SQUARE_SIZE
            x = col * SQUARE_SIZE
            grid[y:y+SQUARE_SIZE, x:x+SQUARE_SIZE] = img
        
        cv2.imwrite("training_data_samples.png", grid)
        print("Sample visualization saved to training_data_samples.png")

if __name__ == "__main__":
    # Check dependencies
    try:
        import cairosvg
    except ImportError:
        print("Error: cairosvg is required. Install with: pip install cairosvg")
        exit(1)
    
    generate_training_data()