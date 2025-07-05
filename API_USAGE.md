# Chess Board Recognition API Usage

## Quick Start

### 1. Start the API Server
```bash
# Activate virtual environment
source .venv/bin/activate

# Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Test with cURL

#### Basic Recognition
```bash
curl -X POST "http://localhost:8000/api/recognize-position" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/chess/board/image.png" \
  | jq
```

#### With Adaptive Processing (recommended)
```bash
curl -X POST "http://localhost:8000/api/recognize-position?adaptive=true" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/chess/board/image.png" \
  | jq
```

#### Debug Endpoint (includes visualization data)
```bash
curl -X POST "http://localhost:8000/api/recognize-position/debug" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/chess/board/image.png" \
  | jq
```

## API Endpoints

### POST /api/recognize-position
Main endpoint for chess position recognition.

**Parameters:**
- `file`: Image file (JPEG, PNG)
- `confidence_threshold`: Minimum confidence for piece detection (0.0-1.0, default: 0.5)
- `adaptive`: Use adaptive processing based on image type (boolean, default: true)

**Response:**
```json
{
  "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  "confidence": 0.95,
  "processing_time": 0.234,
  "image_type": "digital_clean"
}
```

### POST /api/recognize-position/debug
Same as above but includes detailed debug information.

**Additional Response Fields:**
```json
{
  "debug_info": {
    "board_detected": true,
    "square_confidences": [[0.9, 0.8, ...], ...],
    "piece_predictions": [["r", "n", "b", ...], ...]
  }
}
```

### GET /api/health
Health check endpoint.

## Supported Image Types

The API automatically detects and optimizes processing for:
- **digital_clean**: Pure computer-generated chess boards
- **screenshot**: Chess websites/apps with UI elements  
- **photo_overhead**: Real boards photographed from above
- **photo_angle**: Real boards photographed at an angle

## Performance Tips

1. **Clean digital boards** work best (90%+ accuracy)
2. **Avoid heavy UI** in screenshots when possible
3. **Center the board** in the image
4. **Good lighting** helps for photographed boards
5. **Image size**: Resize large images to ~1000px max dimension for faster processing

## Example Usage

### Python
```python
import requests

url = "http://localhost:8000/api/recognize-position"
files = {"file": open("chess_board.png", "rb")}
params = {"adaptive": True}

response = requests.post(url, files=files, params=params)
result = response.json()

print(f"FEN: {result['fen']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Image type: {result['image_type']}")
```

### JavaScript
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8000/api/recognize-position?adaptive=true', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => {
  console.log('FEN:', data.fen);
  console.log('Confidence:', data.confidence);
});
```

## Testing with Dataset Images

Test with various image types from the dataset:

```bash
# Clean digital board (works great)
curl -X POST "http://localhost:8000/api/recognize-position" \
  -F "file=@dataset/images/84.png" | jq

# Screenshot with UI (challenging)
curl -X POST "http://localhost:8000/api/recognize-position" \
  -F "file=@dataset/images/1.jpeg" | jq

# Real photo
curl -X POST "http://localhost:8000/api/recognize-position" \
  -F "file=@dataset/images/7.png" | jq
```

## API Documentation

When the server is running, visit:
- Interactive API docs: http://localhost:8000/docs
- Alternative API docs: http://localhost:8000/redoc

## Troubleshooting

### Low Confidence Scores
- Image type might be misclassified - check the `image_type` field
- Board detection might be including UI elements
- Pieces might not match training data styles

### Wrong FEN
- Check if board is detected correctly using debug endpoint
- Board might be from black's perspective
- UI elements might be confusing the detector

### Slow Processing
- Large images take longer - resize to ~1000px max
- First request loads the model - subsequent requests are faster
- Adaptive processing adds overhead but improves accuracy