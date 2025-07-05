# Chess Board to FEN - Recognition API

A web service that converts chess board images to FEN (Forsyth-Edwards Notation) strings.

## Quick Start

### Using Python directly

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the API:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

3. Visit http://localhost:8000/docs for interactive API documentation

### Using Docker

1. Build and run:
```bash
docker-compose up -d
```

2. Check status:
```bash
docker-compose ps
docker-compose logs
```

3. Stop:
```bash
docker-compose down
```

## API Endpoints

- `POST /api/recognize-position` - Upload an image and get FEN notation
- `POST /api/recognize-position/debug` - Get FEN with debug information
- `GET /api/health` - Health check
- `GET /docs` - Interactive API documentation

## Testing the API

```bash
python test_api.py
```

Or use curl:
```bash
curl -X POST "http://localhost:8000/api/recognize-position" \
  -H "accept: application/json" \
  -F "file=@dataset/images/1.jpeg"
```

## Current Status

The API currently uses a simple color-based classifier for piece detection. This works for:
- Digital chess boards with clear colors
- Well-lit images with good contrast

Future improvements:
- Train a proper CNN model for piece classification
- Improve board detection for various angles
- Support for real physical boards

## Dataset Management

Use the dataset tools to manage training images:

```bash
# Add a new position
python dataset/add_position.py --image path/to/image.png --fen "FEN string"

# Validate dataset
python dataset/validate_dataset.py
```