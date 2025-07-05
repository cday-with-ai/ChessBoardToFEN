# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ChessBoardToFEN is a web service that accepts chess board images and returns FEN (Forsyth-Edwards Notation) strings representing the position. The project is currently in the planning phase with comprehensive documentation in `docs/CHESS_RECOGNITION_API.md`.

## Technical Stack

- **FastAPI** - REST API framework
- **OpenCV** - Board detection and square extraction
- **TensorFlow/PyTorch** - Piece classification
- **Docker** - Deployment
- **Python 3.13** - Development environment

## Project Structure

```
chess-recognition-api/
├── app/
│   ├── main.py                 # FastAPI application entry point
│   ├── api/
│   │   └── endpoints.py        # API routes
│   ├── core/
│   │   ├── config.py           # Configuration management
│   │   └── exceptions.py       # Custom exceptions
│   ├── models/
│   │   ├── board_detector.py   # Board detection logic
│   │   ├── piece_classifier.py # Piece recognition
│   │   └── fen_builder.py      # Board to FEN conversion
│   └── utils/
│       └── image_utils.py      # Image preprocessing
├── tests/
│   └── test_images/            # Test board images
├── models/                     # Trained model files
└── docs/
    └── CHESS_RECOGNITION_API.md  # Detailed implementation guide
```

## Development Commands

### Setting up the environment
```bash
# Create virtual environment (already exists in .venv)
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux

# Install dependencies (once requirements.txt is created)
pip install -r requirements.txt
```

### Running the application
```bash
# Development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Docker commands
```bash
# Build Docker image
docker build -t chess-recognition-api .

# Run with docker-compose
docker-compose up -d

# View logs
docker-compose logs -f
```

## API Endpoints

- **POST /api/recognize-position** - Main endpoint for position recognition
  - Accepts: image/jpeg, image/png
  - Returns: `{"fen": "...", "confidence": 0.95}`
  
- **POST /api/recognize-position/debug** - Debug endpoint with visualization
  - Returns visualization of detected board and pieces

## Implementation Phases

1. **Phase 1: Board Detection** - Detect and normalize chess board in images
2. **Phase 2: Piece Classification** - Identify pieces on each square
3. **Phase 3: API Development** - Create FastAPI endpoints

## Key Implementation Notes

### Board Detection Process
1. Convert to grayscale
2. Find edges using Canny edge detection
3. Detect lines using Hough transform
4. Find board corners
5. Apply perspective transform
6. Extract 64 individual squares

### Piece Classification
- 13 classes: empty + 12 piece types (P,N,B,R,Q,K for both colors)
- Square images extracted in FEN order (a8 to h1)
- Confidence threshold configurable (default 0.8)

### FEN Building
- Convert 8x8 board array to standard FEN notation
- Handle empty squares with digit compression
- Default to white to move with full castling rights

## Testing Strategy

When implementing tests:
- Unit tests for each component (board detector, classifier, FEN builder)
- Integration tests for full pipeline
- Test with various board angles, lighting, piece sets
- Include edge cases: partial boards, blurry images

## Performance Considerations

- Resize large images before processing
- Consider caching for identical images
- GPU support for faster inference if available
- Maximum image size limit: 10MB (configurable)

## Current Status

The project is in the planning phase. The next steps are:
1. Create requirements.txt with dependencies from docs
2. Implement basic project structure
3. Start with Phase 1 (board detection)