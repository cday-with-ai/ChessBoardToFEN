# Project Structure

## Core Application
```
app/
├── api/
│   ├── endpoints.py          # FastAPI endpoints
│   └── models.py            # Pydantic models for API
├── core/
│   ├── config.py            # Configuration settings
│   └── exceptions.py        # Custom exceptions
├── models/
│   ├── board_detector.py    # Chess board detection
│   ├── piece_classifier.py  # Piece recognition using PyTorch
│   ├── fen_builder.py       # Convert board to FEN notation
│   ├── image_type_classifier.py  # Classify image types
│   ├── adaptive_board_processor.py  # Adaptive processing pipeline
│   └── hybrid_board_detector.py     # Enhanced board detection
├── utils/
│   └── image_utils.py       # Image processing utilities
└── main.py                  # FastAPI application entry point
```

## Training & Data
```
├── train_model.py           # Train the PyTorch model
├── prepare_training_data.py # Prepare training data from dataset
├── check_training_data.py   # Verify training data
├── training_data_clean/     # High-quality synthetic training data
│   └── squares/
│       ├── P/              # White pieces
│       ├── p_black/        # Black pieces (renamed for case-insensitive systems)
│       └── empty/          # Empty squares
├── models/                  # Trained model files
│   ├── chess_piece_model.pth
│   └── model_info.json
└── dataset/                 # Test dataset with real board images
    ├── manifest.json        # Dataset metadata
    └── images/              # Board images for testing
```

## Scripts & Utilities
```
scripts/
├── README.md                        # Scripts documentation
├── generate_dataset_report.py       # Comprehensive visual analysis
├── generate_quick_report.py         # Quick summary statistics
├── diagnose_board_extraction.py     # Debug board detection issues
├── test_board_detection.py          # Test full pipeline
├── test_image_classifier.py         # Test image type classification
├── test_model_simple.py             # Test piece classifier
├── generate_piece_training_data.py  # Generate synthetic training data
└── generate_more_empty_squares.py   # Balance training dataset
```

## Documentation
```
docs/
├── CHESS_RECOGNITION_API.md    # Original API specification
├── IMPROVEMENT_ROADMAP.md      # Comprehensive improvement plan
└── QUICK_FIXES.md             # Immediate improvements
├── API_USAGE.md               # API usage examples
├── PROJECT_STRUCTURE.md       # This file
└── CLAUDE.md                  # Claude Code instructions
```

## Reports & Analysis
```
dataset_report/
├── dataset_analysis.md        # Detailed analysis with visualizations
├── summary_report.md          # Quick statistics summary
├── results.json              # Raw analysis results
└── *_analysis.png            # Individual image analysis visualizations
```

## Configuration Files
```
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker container definition
├── docker-compose.yml       # Docker compose configuration
├── .dockerignore           # Docker ignore patterns
└── .gitignore              # Git ignore patterns
```

## Key Commands

### Development
```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python train_model.py

# Start API server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Testing
```bash
# Test API
curl -X POST "http://localhost:8000/api/recognize-position" \
  -F "file=@dataset/images/84.png" | jq

# Run analysis
python scripts/generate_quick_report.py

# Test specific image
python scripts/test_board_detection.py dataset/images/1.jpeg
```

### Docker
```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f
```