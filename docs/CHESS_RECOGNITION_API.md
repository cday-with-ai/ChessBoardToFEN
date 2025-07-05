# Chess Position Recognition API

## Overview
A web service that accepts chess board images and returns FEN (Forsyth-Edwards Notation) strings representing the position.

## Technical Approach

### Architecture
- **FastAPI** backend for REST API
- **OpenCV** for board detection and square extraction
- **TensorFlow/PyTorch** for piece classification
- **Docker** for deployment

### Two-Stage Processing
1. **Board Detection**: Find and normalize the chess board in the image
2. **Piece Classification**: Identify the piece (if any) on each square

## Implementation Plan

### Phase 1: Basic Board Detection (2-3 days)
```python
# Core functionality needed:
- Detect board corners using Hough lines or corner detection
- Perspective transformation to square view
- Extract 64 individual square images
- Handle different board orientations
```

### Phase 2: Piece Classification (3-4 days)
```python
# Approaches to evaluate:
1. Template matching (simple but limited)
2. Pre-trained CNN (recommended)
3. Custom trained model on chess piece dataset
```

### Phase 3: API Development (1-2 days)
```python
# Endpoints:
POST /api/recognize-position
  - Accept: image/jpeg, image/png
  - Return: { "fen": "...", "confidence": 0.95 }

POST /api/recognize-position/debug
  - Return: visualization of detected board and pieces
```

## Technical Stack

### Backend Requirements
```txt
fastapi>=0.104.0
uvicorn>=0.24.0
python-multipart>=0.0.6
opencv-python>=4.8.0
numpy>=1.24.0
pillow>=10.0.0
tensorflow>=2.13.0  # or pytorch>=2.0.0
```

### Project Structure
```
chess-recognition-api/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── api/
│   │   ├── __init__.py
│   │   └── endpoints.py        # API routes
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py           # Configuration
│   │   └── exceptions.py       # Custom exceptions
│   ├── models/
│   │   ├── __init__.py
│   │   ├── board_detector.py   # Board detection logic
│   │   ├── piece_classifier.py # Piece recognition
│   │   └── fen_builder.py      # Board to FEN conversion
│   └── utils/
│       ├── __init__.py
│       └── image_utils.py      # Image preprocessing
├── tests/
│   └── test_images/            # Test board images
├── models/                     # Trained model files
│   └── piece_classifier.h5
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## Implementation Details

### Board Detection (`board_detector.py`)
```python
class BoardDetector:
    def detect_board(self, image: np.ndarray) -> np.ndarray:
        """
        Returns perspective-corrected board image
        Steps:
        1. Convert to grayscale
        2. Find edges (Canny)
        3. Detect lines (Hough)
        4. Find board corners
        5. Apply perspective transform
        """
        pass
    
    def extract_squares(self, board_image: np.ndarray) -> List[np.ndarray]:
        """
        Returns 64 square images in FEN order (a8 to h1)
        """
        pass
```

### Piece Classification (`piece_classifier.py`)
```python
class PieceClassifier:
    PIECES = ['empty', 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
    
    def __init__(self, model_path: str):
        self.model = self.load_model(model_path)
    
    def classify_square(self, square_image: np.ndarray) -> Tuple[str, float]:
        """
        Returns (piece_symbol, confidence)
        """
        pass
```

### FEN Builder (`fen_builder.py`)
```python
def board_array_to_fen(board: List[List[str]], 
                      turn: str = 'w',
                      castling: str = 'KQkq',
                      en_passant: str = '-',
                      halfmove: int = 0,
                      fullmove: int = 1) -> str:
    """
    Convert 8x8 board array to FEN string
    """
    pass
```

### API Endpoint (`endpoints.py`)
```python
@router.post("/recognize-position")
async def recognize_position(
    file: UploadFile = File(...),
    confidence_threshold: float = Query(0.8, ge=0.0, le=1.0)
) -> PositionResponse:
    # Validate file type
    # Read and decode image
    # Detect board
    # Classify pieces
    # Build FEN
    # Return response
    pass
```

## Training Data Generation

### Approach 1: Synthetic Data
```python
# Generate training data from:
1. Various chess GUI screenshots
2. Different piece sets (Staunton, etc.)
3. Multiple board themes
4. Various angles and lighting
```

### Approach 2: Real Board Photos
```python
# Collect and annotate:
1. Photos of physical chess boards
2. Tournament broadcast screenshots  
3. Chess book/magazine scans
```

## Client Integration

### JavaScript/TypeScript Client
```typescript
interface ChessRecognitionResponse {
  fen: string;
  confidence: number;
  processingTime: number;
  debugInfo?: {
    boardDetected: boolean;
    squareConfidences: number[][];
  };
}

async function recognizeChessPosition(
  imageFile: File | Blob
): Promise<ChessRecognitionResponse> {
  const formData = new FormData();
  formData.append('file', imageFile);
  
  const response = await fetch('http://localhost:8000/api/recognize-position', {
    method: 'POST',
    body: formData,
  });
  
  if (!response.ok) {
    throw new Error(`Recognition failed: ${response.statusText}`);
  }
  
  return response.json();
}

// Usage in your chess app
const handleImagePaste = async (e: ClipboardEvent) => {
  const items = e.clipboardData?.items;
  const imageItem = Array.from(items || []).find(
    item => item.type.startsWith('image/')
  );
  
  if (imageItem) {
    const blob = imageItem.getAsFile();
    if (blob) {
      try {
        const result = await recognizeChessPosition(blob);
        if (result.confidence > 0.8) {
          gameStore.loadPosition(result.fen);
        } else {
          console.warn('Low confidence recognition:', result.confidence);
        }
      } catch (error) {
        console.error('Failed to recognize position:', error);
      }
    }
  }
};
```

## Deployment

### Docker Configuration
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY ./app ./app
COPY ./models ./models

# Run
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml
```yaml
version: '3.8'

services:
  chess-recognition:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models/piece_classifier.h5
      - MAX_IMAGE_SIZE=10485760  # 10MB
    volumes:
      - ./models:/app/models:ro
    restart: unless-stopped
```

## Performance Considerations

1. **Image Size Limits**: Resize large images before processing
2. **Caching**: Cache results for identical images
3. **Batch Processing**: Support multiple positions in one request
4. **GPU Support**: Use CUDA for faster inference if available

## Accuracy Improvements

1. **Ensemble Models**: Use multiple models and vote
2. **Board Theme Detection**: Identify board style first
3. **Piece Set Recognition**: Detect piece style
4. **Confidence Thresholds**: Return low-confidence squares for manual correction

## Testing Strategy

1. **Unit Tests**: Test each component separately
2. **Integration Tests**: Full pipeline tests
3. **Test Dataset**: 
   - Various board angles
   - Different lighting conditions
   - Multiple piece sets
   - Partial boards
   - Blurry images

## Future Enhancements

1. **Real-time Recognition**: WebSocket support for live video
2. **Move Detection**: Identify the last move made
3. **Clock Detection**: Read chess clock times
4. **Multi-board Support**: Handle multiple boards in one image
5. **Mobile App**: Native iOS/Android SDKs

## Resources and References

### Datasets
- [Chess Position Images on Kaggle](https://www.kaggle.com/datasets)
- [Lichess Open Database](https://database.lichess.org/)

### Similar Projects
- [chess-recognition](https://github.com/topics/chess-recognition)
- [ChessVision API](https://chessvision.ai/)
- [TensorFlow Chess](https://github.com/tensorflow/models)

### Research Papers
- "Automatic Chess Board Detection" (2018)
- "Deep Learning for Chess Position Recognition" (2020)

## Notes for Implementation

- Start with a simple template matching approach to get the API working
- Focus on standard Staunton pieces first
- Handle only top-down views initially
- Add support for angled views and real boards later
- Consider using YOLO for single-shot board and piece detection