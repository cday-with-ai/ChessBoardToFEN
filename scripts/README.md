# Utility Scripts

This directory contains various utility scripts for testing, analyzing, and improving the chess board recognition system.

## Dataset Analysis

### `generate_dataset_report.py`
Generates a comprehensive visual report showing board detection and FEN recognition results for dataset images.
```bash
python scripts/generate_dataset_report.py
```
- Creates visualizations showing original image, detected board, grid overlay, and FEN comparison
- Outputs to `dataset_report/dataset_analysis.md`

### `generate_quick_report.py`
Generates a quick summary report without individual visualizations (much faster for large datasets).
```bash
python scripts/generate_quick_report.py
```
- Provides statistics by image type
- Lists worst and best performing images
- Outputs to `dataset_report/summary_report.md`

### `diagnose_board_extraction.py`
Diagnoses board extraction issues for specific images.
```bash
python scripts/diagnose_board_extraction.py dataset/images/1.jpeg
```
- Shows detailed extraction process
- Visualizes all 64 extracted squares
- Helps identify alignment issues

## Testing Tools

### `test_board_detection.py`
Tests board detection and FEN recognition on individual images.
```bash
python scripts/test_board_detection.py dataset/images/84.png
```
- Visualizes the entire pipeline
- Shows confidence heatmap
- Compares expected vs actual FEN

### `test_image_classifier.py`
Tests the image type classifier that determines if an image is a clean digital board, screenshot, photo, etc.
```bash
python scripts/test_image_classifier.py
```
- Shows classification results for sample images
- Tests adaptive processing pipeline

### `test_model_simple.py`
Simple test of the trained piece classification model.
```bash
python scripts/test_model_simple.py
```
- Tests model on sample squares
- Shows per-class accuracy
- Generates prediction visualization

## Training Data Generation

### `generate_piece_training_data.py`
Generates synthetic training data from SVG chess piece sets.
```bash
python scripts/generate_piece_training_data.py
```
- Uses piece sets from `/Users/carsonday/IdeaProjects/Simple-FICS-Interface/pieces`
- Generates pieces on various board color themes
- Creates ~6000 training images

### `generate_more_empty_squares.py`
Generates additional empty square variations to balance the training dataset.
```bash
python scripts/generate_more_empty_squares.py
```
- Creates empty squares with various colors and textures
- Balances the dataset for better training

## Usage Notes

1. Most scripts expect to be run from the project root directory
2. Ensure the virtual environment is activated: `source .venv/bin/activate`
3. Some scripts require the API server to be running
4. Output files are typically saved to `dataset_report/` directory

## Dependencies

All scripts use the main project dependencies. Key requirements:
- OpenCV (cv2)
- PyTorch
- NumPy
- Matplotlib
- PIL/Pillow