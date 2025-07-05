# Chess Board to FEN - Recognition API

A web service that converts chess board images to FEN (Forsyth-Edwards Notation) strings.

## Features

- 🖼️ **Multiple Image Formats**: Supports JPEG, PNG, and WebP
- 🎯 **High Accuracy**: 99.92% piece classification accuracy  
- ⚡ **Fast Processing**: Sub-second inference time
- 🔧 **REST API**: Simple HTTP endpoints for easy integration
- 📊 **Confidence Scores**: Per-square confidence ratings
- 🐳 **Docker Support**: Easy deployment with containerization
- 🔍 **Smart Board Detection**: Validates boards and handles UI screenshots
- ✂️ **Automatic Margin Removal**: Removes rank/file labels and UI elements
- 🔄 **Adaptive Processing**: Different strategies for different image types

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

The API uses a trained PyTorch model with 99.92% accuracy for piece classification and an improved board detector with:
- **Validation scoring**: Multi-criteria evaluation to reject false positives
- **Smart margin detection**: Automatically removes UI elements and labels  
- **Fallback logic**: Gracefully handles edge cases
- **Adaptive processing**: Different strategies for screenshots vs photos

The improved detector works well for:
- Digital chess boards from popular chess websites
- Screenshots with UI elements (automatically cropped)
- Wooden/textured boards (improved pattern detection)
- Various image qualities and compression levels

## Dataset Management

Use the dataset tools to manage training images:

```bash
# Add a new position
python dataset/add_position.py --image path/to/image.png --fen "FEN string"

# Validate dataset
python dataset/validate_dataset.py
```