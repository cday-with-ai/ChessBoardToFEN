#!/usr/bin/env python3
"""Quick check of training data distribution"""

from pathlib import Path
import json

# Check what we have
base_dir = Path("training_data/squares")
pieces = ['empty', 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
piece_names = {
    'empty': 'Empty',
    'P': 'White Pawn', 'N': 'White Knight', 'B': 'White Bishop',
    'R': 'White Rook', 'Q': 'White Queen', 'K': 'White King',
    'p': 'Black Pawn', 'n': 'Black Knight', 'b': 'Black Bishop',
    'r': 'Black Rook', 'q': 'Black Queen', 'k': 'Black King'
}

print("=== Training Data Summary ===\n")

total = 0
min_count = float('inf')
piece_counts = {}

for piece in pieces:
    piece_dir = base_dir / piece
    if piece_dir.exists():
        count = len(list(piece_dir.glob("*.png")))
        piece_counts[piece] = count
        total += count
        if count > 0:
            min_count = min(min_count, count)
            print(f"{piece_names[piece]:15} ({piece}): {count:4} examples")

print(f"\nTotal squares: {total}")
print(f"Positions processed: {total // 64} (out of 20)")
print(f"Minimum examples per class: {min_count if min_count != float('inf') else 0}")

# Check balance
white_pieces = sum(piece_counts.get(p, 0) for p in ['P', 'N', 'B', 'R', 'Q', 'K'])
black_pieces = sum(piece_counts.get(p, 0) for p in ['p', 'n', 'b', 'r', 'q', 'k'])
empty_squares = piece_counts.get('empty', 0)

print(f"\nPiece distribution:")
print(f"  White pieces: {white_pieces}")
print(f"  Black pieces: {black_pieces}")
print(f"  Empty squares: {empty_squares}")

# Recommendation
print("\n📊 Analysis:")
if total >= 1000:
    print("✅ Good dataset size for initial experiments!")
    if min_count < 20:
        print("⚠️  Some pieces have few examples - consider adding more diverse positions")
    else:
        print("✅ Decent distribution across piece types")
else:
    print("🔄 Keep adding positions - you're making good progress!")

if black_pieces == 0:
    print("❌ No black pieces found - check if lowercase pieces are being saved correctly")