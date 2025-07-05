# Chess Position Dataset

This directory contains chess board images paired with their FEN strings for training and testing the chess recognition system.

## Structure

- `manifest.json` - Main file linking images to FEN strings
- `images/` - Directory containing all chess board images
- `add_position.py` - Helper script to add new positions
- `validate_dataset.py` - Validates all FEN strings and checks images exist

## Adding New Positions

### Method 1: Manual Entry
1. Add your image to the `images/` directory
2. Edit `manifest.json` and add an entry:
```json
{
  "id": "position_XXX",
  "image": "images/your_image.png",
  "fen": "your FEN string here",
  "description": "Optional description"
}
```

### Method 2: Using the Helper Script
```bash
python add_position.py --image path/to/image.png --fen "FEN string" --description "Description"
```

## FEN Format Reminder

FEN (Forsyth-Edwards Notation) format:
- Pieces: K (King), Q (Queen), R (Rook), B (Bishop), N (Knight), P (Pawn)
- Uppercase = White, lowercase = black
- Numbers = empty squares
- Rows separated by `/` starting from rank 8 (black's back rank)
- Example: Starting position = `rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1`

## Validation

Run the validation script to check all positions:
```bash
python validate_dataset.py
```

This will:
- Verify all FEN strings are valid
- Check that all referenced images exist
- Report any duplicates or issues