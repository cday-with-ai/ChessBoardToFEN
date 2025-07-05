# Side to Move Detection - Technical Challenge

## The Problem

A complete FEN string includes whose turn it is (w/b), but this information isn't inherent in the piece positions alone. This is a fundamental challenge in chess position recognition.

## Possible Detection Strategies

### 1. Visual Indicators in the Image

Many chess interfaces provide visual clues:

#### **Clock/Timer Position**
- Active player's clock is often highlighted or running
- Look for visual differences in timer displays
- Examples:
  - Chess.com: Green highlight on active timer
  - Lichess: Running timer shows seconds ticking
  - ChessBase: Bold text on active side

#### **Turn Indicators**
- Dots, arrows, or text showing whose move
- "White to move" / "Black to move" text
- Colored indicators next to the board
- Examples:
  - Some apps show a small dot next to active side
  - Analysis boards often have explicit text

#### **Board Orientation**
- If we can detect which side is at bottom, it might hint at whose turn
- Not reliable alone, but useful as additional signal

### 2. OCR (Optical Character Recognition)

Extract text from the image that might indicate:
- "White to move" / "Black to move"
- Player names with indicators
- Move numbers (odd = white, even = black)
- Analysis notation showing whose turn

### 3. Context Clues

#### **Piece Positions**
- Check if one side is in check (must be their move)
- Look for pieces in motion blur (if captured mid-move)
- En passant possibilities suggest last move

#### **Material Balance**
- In puzzle positions, the side to move often has a winning combination
- Significant material imbalance might hint at puzzle orientation

### 4. Machine Learning Approach

Train a separate model to recognize turn indicators:
- Input: Full board image (not just squares)
- Output: 'w' or 'b' for whose turn
- Training data: Images with known side-to-move

## Implementation Plan

### Phase 1: Basic Heuristics
```python
def detect_side_to_move(image, board_region):
    # 1. Look for check - if a king is in check, it's that side's move
    if is_white_in_check(board_position):
        return 'w'
    if is_black_in_check(board_position):
        return 'b'
    
    # 2. Default assumption
    return 'w'  # Most puzzles/positions show white to move
```

### Phase 2: Visual Indicator Detection
```python
def detect_visual_indicators(full_image, board_bbox):
    # Look for common UI elements around the board
    # - Timer highlights
    # - Turn indicator dots
    # - Text regions with "to move"
    
    indicators = []
    
    # Check regions around board for indicators
    left_region = extract_left_of_board(full_image, board_bbox)
    right_region = extract_right_of_board(full_image, board_bbox)
    bottom_region = extract_below_board(full_image, board_bbox)
    
    # Apply various detection methods
    indicators.extend(detect_timer_highlights(left_region, right_region))
    indicators.extend(detect_turn_dots(left_region, right_region))
    indicators.extend(detect_text_indicators(bottom_region))
    
    return vote_on_indicators(indicators)
```

### Phase 3: OCR Integration
```python
def detect_with_ocr(image_regions):
    # Use tesseract or cloud OCR to find text
    for region in image_regions:
        text = ocr.extract_text(region)
        
        # Look for patterns
        if "white to move" in text.lower():
            return 'w', confidence=0.95
        if "black to move" in text.lower():
            return 'b', confidence=0.95
        
        # Check for move numbers
        move_match = re.search(r'(\d+)\.\s*([.]{3}|\w+)', text)
        if move_match:
            # Analyze move notation
            pass
```

### Phase 4: ML Model for UI Recognition
```python
class TurnIndicatorDetector:
    """
    CNN trained on chess UI screenshots to recognize whose turn it is
    based on visual interface elements
    """
    def __init__(self):
        self.model = self.load_ui_model()
    
    def predict_turn(self, full_image, board_bbox):
        # Extract regions of interest
        ui_features = self.extract_ui_regions(full_image, board_bbox)
        
        # Predict
        prediction = self.model.predict(ui_features)
        return 'w' if prediction[0] > 0.5 else 'b'
```

## Training Data Collection

When collecting positions, also capture:

1. **Full Screenshot** - Not just the board
2. **UI Elements** - Include timers, indicators
3. **Source Label** - Which app/website (each has patterns)
4. **Turn Information** - Ground truth for whose move

## Dataset Enhancement

Update the manifest to include:
```json
{
  "id": 12,
  "image": "images/12.png",
  "fen": "position w KQkq - 0 1",
  "side_to_move": "w",
  "ui_indicators": {
    "has_timer": true,
    "has_text": true,
    "source": "chess.com"
  }
}
```

## Fallback Strategy

When detection fails:
1. Return position-only FEN (without side to move)
2. Provide both possibilities
3. Ask user to specify
4. Default to white (statistically more common in puzzles)

## Accuracy Expectations

- **Check positions**: 100% (deterministic)
- **With clear UI indicators**: 90-95%
- **With OCR**: 85-90%
- **Heuristics only**: 60-70%
- **Combined approach**: 85-95%

## Next Steps

1. Start annotating which chess platform each image is from
2. Include full screenshots (not just cropped boards)
3. Build platform-specific indicator detectors
4. Train ensemble model combining all methods