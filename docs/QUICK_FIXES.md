# Quick Fixes for Immediate Improvement

## 1. Board Detection: Add Validation Scoring (1-2 days)

### Problem
Board detector accepts any quadrilateral, including UI panels and menus.

### Quick Fix
```python
def validate_board_candidate(image, corners):
    score = 0.0
    
    # Check 1: Aspect ratio (chess boards are square)
    aspect_ratio = calculate_aspect_ratio(corners)
    if 0.9 < aspect_ratio < 1.1:
        score += 0.3
    
    # Check 2: Contains alternating pattern
    warped = four_point_transform(image, corners)
    if has_checkerboard_pattern(warped):
        score += 0.4
    
    # Check 3: No text detected
    if not has_text_regions(warped):
        score += 0.3
    
    return score
```

## 2. Smart Margin Detection (1 day)

### Problem
Fixed margins include rank/file labels in some boards.

### Quick Fix
```python
def detect_board_margins(board_image):
    # Detect text regions (likely labels)
    text_mask = detect_text_regions(board_image)
    
    # Find actual playing surface
    margins = {
        'top': find_first_non_text_row(text_mask),
        'bottom': find_last_non_text_row(text_mask),
        'left': find_first_non_text_col(text_mask),
        'right': find_last_non_text_col(text_mask)
    }
    
    return margins
```

## 3. Collect Quick Training Data (2-3 days)

### Immediate Sources
1. **Lichess piece sets** - Download their open-source SVGs:
   - https://github.com/lichess-org/lila/tree/master/public/piece
   
2. **Chess.com pieces** - Screenshot and extract from:
   - https://www.chess.com/settings/board
   
3. **Popular apps** - Extract pieces from:
   - Chess.com app
   - Lichess app
   - Chess24

### Quick Collection Script
```python
# 1. Screenshot each piece on light/dark squares
# 2. Use current model to detect board
# 3. Extract and label pieces
# 4. Manual verification
# 5. Add to training set
```

## 4. Add Basic Perspective Correction (1 day)

### Current Issue
```
Expected square:  Actual detection:
+---+            +----+
|   |            |    /
|   |    -->     |   /
+---+            +--+
```

### Quick Fix
Use OpenCV's getPerspectiveTransform even for slight tilts:
```python
def extract_square_corrected(board, row, col):
    # Get four corners of this square
    corners = get_square_corners(board, row, col)
    
    # Define target square (perfect rectangle)
    target = np.array([[0,0], [64,0], [64,64], [0,64]])
    
    # Get transform matrix
    M = cv2.getPerspectiveTransform(corners, target)
    
    # Extract with perspective correction
    square = cv2.warpPerspective(board, M, (64, 64))
    
    return square
```

## 5. Screenshot-Specific Preprocessing (2 days)

### For Chess.com/Lichess screenshots:
```python
def preprocess_screenshot(image):
    # 1. Detect the game board area (usually largest square region)
    # 2. Look for common UI patterns to exclude:
    #    - Player info boxes (usually above/below board)
    #    - Move lists (usually right side)
    #    - Chat boxes (usually below)
    
    # Common chess.com board location
    if is_chesscom_screenshot(image):
        # Board is typically centered, ~60% of width
        h, w = image.shape[:2]
        x1 = int(w * 0.2)
        x2 = int(w * 0.8)
        y1 = int(h * 0.1)
        y2 = int(h * 0.9)
        return image[y1:y2, x1:x2]
    
    return image
```

## Estimated Impact

These quick fixes should improve:
- Screenshot accuracy: 55% → 70%+ 
- Angled photos: 47% → 60%+
- Overall accuracy: 89% → 92%+

## Implementation Order

1. **Day 1**: Add board validation scoring
2. **Day 2**: Implement smart margin detection  
3. **Day 3-4**: Collect initial real piece images
4. **Day 5**: Add perspective correction
5. **Day 6**: Test and refine

Total time: ~1 week for noticeable improvement