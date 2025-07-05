# Training Guide for Chess Piece Recognition

## 🎯 How to Build Your Training Dataset

### Step 1: Keep Adding Images
Continue using the helper script as you've been doing:
```bash
python dataset/add_position.py --image ~/Downloads/chess_position.png --fen "FEN_STRING"
```

### Step 2: Prepare Training Data
When you have enough positions (aim for 50-100 to start), run:
```bash
python prepare_training_data.py
```

This will:
- Extract all 64 squares from each board image
- Organize them by piece type (K, Q, R, B, N, P, etc.)
- Create a summary showing how many examples you have

### Step 3: Check Your Progress
After running the preparation script, you'll see something like:
```
=== Training Data Summary ===
White Pawn     (P): 45 examples
White Knight   (N): 12 examples
White Bishop   (B): 15 examples
...
Empty          (empty): 180 examples

Minimum examples per class: 12
📊 You have enough data for initial experiments!
```

## 📊 Data Collection Tips

### What Makes Good Training Data?

1. **Variety of Positions**
   - Opening positions (lots of pieces)
   - Middlegame (normal distribution)
   - Endgame (few pieces, more empty squares)

2. **Different Sources**
   - Screenshots from different chess websites
   - Different board themes/colors
   - Various piece styles

3. **Clear Images**
   - Board fills most of the image
   - Good contrast between pieces and squares
   - Not blurry or distorted

### Target Numbers

| Phase | Positions | When to Train |
|-------|-----------|---------------|
| Testing | 20-50 | Just to see if it works |
| Basic | 100-200 | Decent for digital boards |
| Good | 500-1000 | Handles most cases |
| Excellent | 2000+ | Professional quality |

## 🚀 Quick Start Checklist

1. ✅ Add images with `add_position.py`
2. ✅ Run `prepare_training_data.py` periodically
3. ✅ Check the summary to see your progress
4. ✅ When ready, we'll train your model!

## 💡 Pro Tips

- **Start Simple**: Focus on one type of board/pieces first
- **Balanced Data**: Try to get similar counts for each piece type
- **Save Everything**: Keep all your images, even if detection fails initially

## Current Status

You have 5 positions so far. Keep going! 
- Next milestone: 20 positions (good for testing)
- First training: 50-100 positions

The more diverse positions you add, the better your model will perform! 🎯