# Training Data Structure

## Current Structure (Unified)

The project now uses a single, unified training data directory with consistent naming:

```
training_data/
├── black-bishop/    (308 images)
├── black-king/      (308 images)
├── black-knight/    (308 images)
├── black-pawn/      (308 images)
├── black-queen/     (308 images)
├── black-rook/      (308 images)
├── empty/           (192 images)
├── white-bishop/    (308 images)
├── white-king/      (308 images)
├── white-knight/    (308 images)
├── white-pawn/      (308 images)
├── white-queen/     (308 images)
├── white-rook/      (308 images)
├── MODEL_CONFIG.py
└── TRAINING_DATA_SUMMARY.md
```

**Total**: 3,888 images

## Naming Convention

- **Format**: `color-piece` (e.g., `white-bishop`, `black-knight`)
- **Special case**: `empty` for empty squares
- **All lowercase** with hyphen separator

## Class Indices for Model Training

```python
CLASS_INDICES = {
    'empty': 0,
    'white-king': 1,
    'white-queen': 2,
    'white-rook': 3,
    'white-bishop': 4,
    'white-knight': 5,
    'white-pawn': 6,
    'black-king': 7,
    'black-queen': 8,
    'black-rook': 9,
    'black-bishop': 10,
    'black-knight': 11,
    'black-pawn': 12,
}
```

## Data Sources

The current training data includes:

1. **Lichess Piece Sets** (Primary Source)
   - 6 different piece styles: cburnett, merida, alpha, pirouetti, spatial, horsey
   - Rendered on various chess board backgrounds
   - Multiple sizes: 64×64, 96×96, 128×128
   - Total: 3,456 piece images

2. **Empty Squares**
   - Generated with various board colors
   - Includes plain, noise, gradient, and wood texture variations
   - Total: 192 empty square images

3. **Augmentations**
   - Horizontal flips
   - Brightness adjustments
   - Blur effects
   - Total: 240 augmented images

## Model Compatibility

When training a new model, update `app/models/piece_classifier.py` to use the new class names:

```python
PIECE_CLASSES = [
    'empty',
    'white-king', 'white-queen', 'white-rook',
    'white-bishop', 'white-knight', 'white-pawn',
    'black-king', 'black-queen', 'black-rook', 
    'black-bishop', 'black-knight', 'black-pawn'
]

# Mapping for FEN notation
PIECE_MAPPING = {
    'white-king': 'K', 'white-queen': 'Q', 'white-rook': 'R',
    'white-bishop': 'B', 'white-knight': 'N', 'white-pawn': 'P',
    'black-king': 'k', 'black-queen': 'q', 'black-rook': 'r',
    'black-bishop': 'b', 'black-knight': 'n', 'black-pawn': 'p',
    'empty': 'empty'
}
```

## Previous Structure (Deprecated)

The project previously had multiple training data directories:
- `training_data/` - Original with single-letter notation
- `training_data_clean/` - Cleaned version with mixed notation
- `training_data_new/` - Latest synthetic data (now integrated)

These have been consolidated into the single `training_data/` directory with the new naming convention.