from pydantic import BaseModel, Field
from typing import Optional, List, Dict


class PositionResponse(BaseModel):
    fen: str = Field(..., description="FEN notation of the chess position")
    confidence: float = Field(..., description="Overall confidence score (0-1)")
    processing_time: float = Field(..., description="Processing time in seconds")
    image_type: Optional[str] = Field(None, description="Detected image type (when using adaptive processing)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "confidence": 0.95,
                "processing_time": 0.234,
                "image_type": "digital_clean"
            }
        }


class DebugInfo(BaseModel):
    board_detected: bool
    square_confidences: List[List[float]]
    piece_predictions: List[List[str]]
    
    class Config:
        json_schema_extra = {
            "example": {
                "board_detected": True,
                "square_confidences": [[0.9, 0.8, ...], ...],
                "piece_predictions": [["r", "n", "b", ...], ...]
            }
        }


class PositionDebugResponse(PositionResponse):
    debug_info: Optional[DebugInfo] = None


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "Board detection failed",
                "detail": "Could not find chess board in the uploaded image"
            }
        }