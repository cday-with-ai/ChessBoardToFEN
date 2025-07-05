
# Chess Piece Training Data Resources

## Downloaded Resources

### 1. Lichess Piece Sets (SVG)
- Location: dataset/new_pieces/lichess/
- Sets: cburnett, merida, alpha, pirouetti, spatial, horsey
- Format: SVG (need conversion to PNG)
- License: Various open source licenses

### 2. Additional Resources to Download Manually

#### Roboflow Chess Dataset
- URL: https://public.roboflow.com/object-detection/chess-full
- 292 images with 2894 annotations
- Includes bounding boxes for all pieces
- Various real-world conditions

#### Kaggle Chess Pieces Dataset  
- URL: https://www.kaggle.com/datasets/anshulmehtakaggl/chess-pieces-detection-images-dataset
- Over 7,000 images of chess pieces
- Six different piece types

#### GitHub Datasets
1. samryan18/chess-dataset
   - 500 labeled chess board images
   - Includes FEN notation

2. ThanosM97/end-to-end-chess-recognition
   - 10,800 images from 100 games
   - Multiple camera angles

## Next Steps

1. Convert SVG files to PNG:
   ```bash
   pip install cairosvg
   python scripts/convert_pieces.py
   ```

2. Download additional datasets manually

3. Organize all pieces into training structure:
   ```
   training_data_new/
   ├── pieces/
   │   ├── white_king/
   │   ├── white_queen/
   │   └── ...
   └── empty/
   ```

4. Augment with various backgrounds and conditions
