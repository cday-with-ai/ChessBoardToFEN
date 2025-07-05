from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse
import time
import numpy as np
from typing import Optional

from app.api.models import (
    PositionResponse, 
    PositionDebugResponse, 
    ErrorResponse,
    DebugInfo
)
from app.core.config import settings
from app.core.exceptions import BoardDetectionError, InvalidImageError
from app.utils.image_utils import load_image_from_bytes
from app.models.board_detector import BoardDetector
from app.models.piece_classifier import PieceClassifier
from app.models.adaptive_board_processor import AdaptiveBoardProcessor
from app.models.fen_builder import build_fen_from_squares


router = APIRouter(prefix="/api", tags=["chess-recognition"])

# Initialize components
board_detector = BoardDetector()
piece_classifier = PieceClassifier()
adaptive_processor = AdaptiveBoardProcessor()


@router.post("/recognize-position", 
             response_model=PositionResponse,
             responses={400: {"model": ErrorResponse}})
async def recognize_position(
    file: UploadFile = File(...),
    confidence_threshold: float = Query(
        default=settings.confidence_threshold, 
        ge=0.0, 
        le=1.0,
        description="Minimum confidence threshold for piece detection"
    ),
    adaptive: bool = Query(
        default=True,
        description="Use adaptive processing based on image type"
    )
) -> PositionResponse:
    """
    Recognize chess position from an uploaded image.
    
    Returns the position in FEN notation along with confidence score.
    """
    start_time = time.time()
    
    # Validate file type
    if file.content_type not in settings.allowed_image_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {settings.allowed_image_types}"
        )
    
    # Check file size
    contents = await file.read()
    if len(contents) > settings.max_image_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {settings.max_image_size / 1024 / 1024}MB"
        )
    
    try:
        # Load image
        image = load_image_from_bytes(contents)
        
        if adaptive:
            # Use adaptive processor
            classifications, metadata = adaptive_processor.process_image(image)
            
            # Build FEN
            fen = build_fen_from_squares(classifications, confidence_threshold)
            
            # Calculate average confidence
            confidences = [conf for _, conf in classifications]
            avg_confidence = np.mean(confidences)
            
            processing_time = time.time() - start_time
            
            return PositionResponse(
                fen=fen,
                confidence=float(avg_confidence),
                processing_time=processing_time,
                image_type=metadata.get('image_type')
            )
        else:
            # Use standard processing
            # Detect board
            board_image = board_detector.detect_board(image)
            
            # Extract squares
            squares = board_detector.extract_squares(board_image)
            
            # Classify pieces
            classifications = piece_classifier.classify_board(squares)
            
            # Build FEN
            fen = build_fen_from_squares(classifications, confidence_threshold)
            
            # Calculate average confidence
            confidences = [conf for _, conf in classifications]
            avg_confidence = np.mean(confidences)
            
            processing_time = time.time() - start_time
            
            return PositionResponse(
                fen=fen,
                confidence=float(avg_confidence),
                processing_time=processing_time
            )
        
    except BoardDetectionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@router.post("/recognize-position/debug", 
             response_model=PositionDebugResponse,
             responses={400: {"model": ErrorResponse}})
async def recognize_position_debug(
    file: UploadFile = File(...),
    confidence_threshold: float = Query(
        default=settings.confidence_threshold, 
        ge=0.0, 
        le=1.0,
        description="Minimum confidence threshold for piece detection"
    )
) -> PositionDebugResponse:
    """
    Recognize chess position with detailed debug information.
    
    Returns the position in FEN notation along with debug visualization data.
    """
    start_time = time.time()
    
    # Validate file type
    if file.content_type not in settings.allowed_image_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {settings.allowed_image_types}"
        )
    
    # Check file size
    contents = await file.read()
    if len(contents) > settings.max_image_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {settings.max_image_size / 1024 / 1024}MB"
        )
    
    try:
        # Load image
        image = load_image_from_bytes(contents)
        
        # Detect board
        board_detected = True
        try:
            board_image = board_detector.detect_board(image)
        except BoardDetectionError:
            board_detected = False
            raise
        
        # Extract squares
        squares = board_detector.extract_squares(board_image)
        
        # Classify pieces
        classifications = piece_classifier.classify_board(squares)
        
        # Build FEN
        fen = build_fen_from_squares(classifications, confidence_threshold)
        
        # Prepare debug info
        square_confidences = []
        piece_predictions = []
        
        for i in range(8):
            conf_row = []
            pred_row = []
            for j in range(8):
                idx = i * 8 + j
                piece, conf = classifications[idx]
                conf_row.append(float(conf))
                pred_row.append(piece if piece != 'empty' else '')
            square_confidences.append(conf_row)
            piece_predictions.append(pred_row)
        
        # Calculate average confidence
        confidences = [conf for _, conf in classifications]
        avg_confidence = np.mean(confidences)
        
        processing_time = time.time() - start_time
        
        return PositionDebugResponse(
            fen=fen,
            confidence=float(avg_confidence),
            processing_time=processing_time,
            debug_info=DebugInfo(
                board_detected=board_detected,
                square_confidences=square_confidences,
                piece_predictions=piece_predictions
            )
        )
        
    except BoardDetectionError as e:
        raise HTTPException(
            status_code=400, 
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Processing error: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": settings.api_version}