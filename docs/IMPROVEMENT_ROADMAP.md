# Chess Board Recognition Improvement Roadmap

## Executive Summary

After analyzing 752 chess board images, we've identified key areas for improvement in our chess position recognition system. While the system performs excellently on clean digital boards (90%+ accuracy), it struggles with real-world scenarios including screenshots with UI elements, photographed boards, and perspective-distorted images.

## Current Performance Analysis

### Strengths
- **99.6% board detection rate** - Almost always finds *something* it thinks is a board
- **90.3% accuracy on clean digital boards** - Excellent performance on ideal inputs
- **69% perfect FEN matches** overall - Shows the model can work very well

### Weaknesses
- **54.9% accuracy on screenshots** - UI elements confuse board detection
- **47.3% accuracy on angled photos** - Perspective distortion not handled well
- **Poor piece recognition on real boards** - Model only trained on synthetic SVG pieces

## Critical Issues Identified

### 1. Board Detection Problems

#### Issue: UI Element Confusion
- Board detector sometimes identifies UI panels, menus, or buttons as the chess board
- Example: Chess.com interface where "Play/Puzzles" menu was detected as board
- Rank/file labels (a-h, 1-8) around boards get included in detected region

#### Issue: Board Border Inclusion
- Physical board frames and decorative borders included in detection
- Causes misalignment when dividing into 8x8 grid
- Square extraction gets offset, leading to pieces being cut off or misaligned

#### Issue: Perspective Handling
- Current detector assumes boards are perfectly square and aligned
- Angled photos need perspective correction before square extraction
- Even small tilts can cause significant accuracy drops

### 2. Training Data Limitations

#### Current State
- Model trained only on 31 chess piece SVG sets rendered on solid backgrounds
- No real board textures, shadows, or lighting variations
- No perspective distortion in training data
- Missing many popular chess set styles (Staunton variations, themed sets, etc.)

#### Gap Analysis
- **Digital boards**: Many online platforms use unique piece designs
- **Physical boards**: Wood grain, marble, plastic textures not represented
- **Lighting conditions**: No shadows, reflections, or uneven lighting in training
- **Piece styles**: Tournament sets, decorative sets, regional variations missing

### 3. Square Extraction Issues

#### Problem: Fixed Grid Division
- Current approach: Divide detected board by 8 equally
- Assumes perfect square board with no borders
- Doesn't account for:
  - Non-square board detections
  - Boards with uneven spacing
  - Perspective distortion

#### Problem: No Validation
- No verification that extracted squares actually contain chess squares
- Can't detect when grid is misaligned
- No confidence scoring for individual square extractions

## Proposed Improvements

### Phase 1: Enhanced Board Detection

#### 1.1 Multi-Strategy Board Detection
```python
class SmartBoardDetector:
    def detect_board(self, image):
        strategies = [
            self.detect_by_chess_pattern,      # Look for 8x8 alternating pattern
            self.detect_by_piece_arrangement,  # Find pieces, infer board
            self.detect_by_corner_detection,   # Find board corners specifically
            self.detect_by_line_intersection,  # Find grid intersections
        ]
        
        candidates = []
        for strategy in strategies:
            result = strategy(image)
            if result:
                score = self.validate_board_candidate(result)
                candidates.append((result, score))
        
        return self.select_best_candidate(candidates)
```

#### 1.2 Board Validation Scoring
- Check for alternating square colors
- Verify 8x8 grid structure
- Look for piece-like objects in reasonable positions
- Penalize candidates with text/UI elements

#### 1.3 Intelligent Cropping
- Detect and remove rank/file labels
- Identify and exclude UI elements
- Find the actual playing surface within board frame

### Phase 2: Robust Training Data

#### 2.1 Real Chess Set Collection
Sources for diverse piece images:
- **Photography sessions**: Photograph real chess sets in various conditions
- **Online chess platforms**: Capture pieces from Chess.com, Lichess, chess24
- **Chess set databases**: 
  - House of Staunton catalog
  - Chess set collectors' forums
  - Museum collections (public domain)
- **3D rendered sets**: Use Blender/Unity to create realistic pieces

#### 2.2 Data Augmentation Pipeline
```python
augmentations = [
    # Lighting variations
    RandomBrightness(0.5, 1.5),
    RandomShadow(num_shadows_limit=3),
    RandomSunFlare(p=0.1),
    
    # Perspective transforms
    RandomPerspective(distortion_scale=0.3),
    RandomRotation(limit=15),
    
    # Board textures
    ApplyBoardTexture(wood_grains, marble_patterns),
    
    # Realistic conditions
    RandomBlur(blur_limit=3),
    RandomNoise(var_limit=(10, 50)),
    RandomReflection(p=0.2),
]
```

#### 2.3 Synthetic Board Generation
- Generate full board positions with realistic rendering
- Include common board defects:
  - Worn squares
  - Piece shadows
  - Reflections on glossy boards
  - Uneven lighting

### Phase 3: Adaptive Square Extraction

#### 3.1 Smart Grid Detection
```python
def extract_squares_adaptive(board_image):
    # Step 1: Find actual grid lines
    grid_lines = detect_grid_lines(board_image)
    
    # Step 2: Find intersection points
    intersections = find_line_intersections(grid_lines)
    
    # Step 3: Validate 9x9 intersection grid
    if not validate_chess_grid(intersections):
        # Fallback to corner-based extraction
        intersections = infer_grid_from_corners(board_image)
    
    # Step 4: Extract warped squares
    squares = []
    for row in range(8):
        for col in range(8):
            square = extract_warped_square(
                board_image, 
                intersections, 
                row, col
            )
            squares.append(square)
    
    return squares
```

#### 3.2 Perspective-Aware Extraction
- Use homography to map each square individually
- Handle non-uniform grids (common in photos)
- Maintain square aspect ratio after extraction

### Phase 4: Ensemble Approach

#### 4.1 Multiple Models
- **Model A**: Trained on synthetic clean data (current model)
- **Model B**: Trained on real chess set photos
- **Model C**: Trained on screenshot data with UI
- **Model D**: Specialized for wooden/tournament boards

#### 4.2 Intelligent Model Selection
```python
def select_model(image_type, confidence):
    if image_type == ImageType.DIGITAL_CLEAN and confidence > 0.9:
        return model_synthetic
    elif image_type == ImageType.SCREENSHOT:
        return model_screenshot
    elif image_type == ImageType.PHOTO_OVERHEAD:
        return model_photo
    else:
        # Use ensemble voting
        return ensemble_models
```

## Implementation Priority

### High Priority (Immediate Impact)
1. **Improve board detection validation** - Reduce false positives on UI elements
2. **Add perspective correction** - Handle tilted boards
3. **Collect real chess piece images** - Start with 10 common sets

### Medium Priority (Significant Improvement)
1. **Implement smart grid detection** - Better square extraction
2. **Create screenshot-specific detector** - Handle Chess.com/Lichess UI
3. **Build data augmentation pipeline** - Generate realistic training data

### Low Priority (Nice to Have)
1. **Train ensemble models** - Specialized models for different scenarios
2. **Add board orientation detection** - Handle boards from black's perspective
3. **Implement confidence calibration** - Better uncertainty estimates

## Validation Metrics

### Primary Metrics
- **FEN Accuracy**: Percentage of correctly identified pieces
- **Board Detection Rate**: Successfully found chess board
- **Square Alignment Score**: How well extracted squares align with actual squares

### Secondary Metrics
- **Processing Time**: Keep under 500ms for API responses
- **Model Size**: Keep under 100MB for deployment
- **Memory Usage**: Optimize for edge devices

## Testing Strategy

### Test Dataset Categories
1. **Clean Digital** (baseline): ChessVision.ai, pure chess diagrams
2. **Screenshots Easy**: Full-screen chess boards, minimal UI
3. **Screenshots Hard**: Boards with heavy UI, multiple boards visible
4. **Photos Overhead**: Tournament/home games from above
5. **Photos Angle**: Casual photos at 15-45° angles
6. **Edge Cases**: Partial boards, unusual sets, poor lighting

### Benchmark Goals
- Clean Digital: >95% accuracy (currently 90%)
- Screenshots: >80% accuracy (currently 55%)
- Photos Overhead: >85% accuracy (currently 92%)
- Photos Angle: >70% accuracy (currently 47%)

## Resource Requirements

### Data Collection
- 100+ real chess set photographs (various styles)
- 500+ screenshot samples from major platforms
- 50+ angled board photos
- Licensing for commercial chess set images

### Compute Resources
- GPU for model training (40+ hours estimated)
- Storage for augmented dataset (~50GB)
- CI/CD pipeline for continuous validation

### Human Resources
- Chess players to validate FEN accuracy
- Photographers for chess set images
- ML engineer for model optimization

## Success Criteria

1. **Overall accuracy >85%** on diverse test set
2. **Board detection >95%** even with UI elements
3. **Processing time <300ms** for average image
4. **Model size <50MB** for mobile deployment
5. **Works with 90% of popular online chess platforms**

## Next Steps

1. **Create GitHub issues** for each high-priority item
2. **Set up data collection pipeline** with proper licensing
3. **Build validation framework** for continuous testing
4. **Design API v2** with confidence scores and diagnostic info
5. **Plan incremental rollout** with A/B testing

## Appendix: Technical Details

### Useful Libraries/Tools
- **OpenCV**: Advanced contour detection, perspective transforms
- **Albumentations**: Image augmentation pipeline
- **imgaug**: Alternative augmentation library
- **Weights & Biases**: Experiment tracking
- **Label Studio**: Data annotation tool

### Research Papers to Review
- "Robust Chess Board Recognition from Photos" (2019)
- "Deep Learning for Board Game Position Recognition" (2021)
- "Perspective-Invariant Object Detection" (2020)

### Open Source Projects to Study
- `python-chess`: Board representation and validation
- `ChessVision.ai`: Their approach to board detection
- `chess-board-recognition`: Academic implementation

---

*This document is a living roadmap and should be updated as we learn more about the problem space and discover new edge cases.*