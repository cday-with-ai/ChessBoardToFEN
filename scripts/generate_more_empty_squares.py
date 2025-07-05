#!/usr/bin/env python3
"""
Generate more empty square variations to balance the dataset
"""

import cv2
import numpy as np
from pathlib import Path
import random

# Configuration
OUTPUT_DIR = Path("training_data_clean/squares/empty")
SQUARE_SIZE = 128
TARGET_COUNT = 500  # Generate enough to match other pieces

# Extended board color themes with more variations
BOARD_THEMES = [
    # Standard themes
    {"name": "standard", "light": (240, 217, 181), "dark": (181, 136, 99)},
    {"name": "blue", "light": (222, 227, 230), "dark": (140, 162, 173)},
    {"name": "green", "light": (235, 236, 208), "dark": (119, 149, 86)},
    {"name": "purple", "light": (230, 230, 230), "dark": (163, 130, 180)},
    # Dark themes
    {"name": "dark_blue", "light": (90, 90, 100), "dark": (40, 40, 50)},
    {"name": "dark_gray", "light": (80, 80, 80), "dark": (50, 50, 50)},
    {"name": "dark_brown", "light": (100, 80, 60), "dark": (60, 40, 30)},
    # Light themes
    {"name": "light_cream", "light": (255, 248, 220), "dark": (205, 183, 142)},
    {"name": "light_gray", "light": (245, 245, 245), "dark": (200, 200, 200)},
    {"name": "light_blue", "light": (240, 248, 255), "dark": (176, 196, 222)},
    # High contrast
    {"name": "high_contrast", "light": (255, 255, 255), "dark": (0, 0, 0)},
    {"name": "inverse", "light": (50, 50, 50), "dark": (250, 250, 250)},
    # Natural wood colors
    {"name": "wood_maple", "light": (255, 228, 196), "dark": (139, 90, 43)},
    {"name": "wood_walnut", "light": (222, 184, 135), "dark": (101, 67, 33)},
    {"name": "wood_cherry", "light": (255, 218, 185), "dark": (139, 69, 19)},
]

def add_texture_variation(square, variation_type):
    """Add subtle texture variations to squares"""
    h, w = square.shape[:2]
    
    if variation_type == 'gradient':
        # Add subtle gradient
        gradient = np.linspace(0, 20, h).reshape(-1, 1)
        gradient = np.repeat(gradient, w, axis=1)
        for c in range(3):
            square[:, :, c] = np.clip(square[:, :, c].astype(float) + gradient, 0, 255).astype(np.uint8)
    
    elif variation_type == 'noise':
        # Add subtle noise
        noise = np.random.normal(0, 5, square.shape)
        square = np.clip(square.astype(float) + noise, 0, 255).astype(np.uint8)
    
    elif variation_type == 'vignette':
        # Add subtle vignette effect
        center_x, center_y = w // 2, h // 2
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        vignette = 1 - (dist_from_center / max_dist) * 0.2
        for c in range(3):
            square[:, :, c] = np.clip(square[:, :, c] * vignette, 0, 255).astype(np.uint8)
    
    return square

def generate_empty_squares():
    """Generate many variations of empty squares"""
    print(f"Generating empty squares to reach {TARGET_COUNT} total...")
    
    # Count existing
    existing_count = len(list(OUTPUT_DIR.glob("*.png")))
    print(f"Existing empty squares: {existing_count}")
    
    if existing_count >= TARGET_COUNT:
        print("Already have enough empty squares!")
        return
    
    to_generate = TARGET_COUNT - existing_count
    print(f"Need to generate: {to_generate} more")
    
    count = existing_count
    variations = ['none', 'gradient', 'noise', 'vignette']
    
    while count < TARGET_COUNT:
        # Pick random theme
        theme = random.choice(BOARD_THEMES)
        
        # Pick light or dark
        is_light = random.choice([True, False])
        color = theme['light'] if is_light else theme['dark']
        
        # Add slight color variation
        color_var = tuple(
            np.clip(c + random.randint(-10, 10), 0, 255) 
            for c in color
        )
        
        # Create square
        square = np.full((SQUARE_SIZE, SQUARE_SIZE, 3), color_var, dtype=np.uint8)
        
        # Add texture variation
        variation = random.choice(variations)
        if variation != 'none':
            square = add_texture_variation(square, variation)
        
        # Save
        filename = f"empty_{theme['name']}_{'light' if is_light else 'dark'}_{variation}_{count}.png"
        cv2.imwrite(str(OUTPUT_DIR / filename), cv2.cvtColor(square, cv2.COLOR_RGB2BGR))
        
        count += 1
        
        if count % 100 == 0:
            print(f"  Generated {count}/{TARGET_COUNT}...")
    
    print(f"\nTotal empty squares now: {count}")

if __name__ == "__main__":
    generate_empty_squares()