import cv2
import numpy as np
from typing import List, Optional, Tuple
from app.models.board_detector import BoardDetector
from app.models.validated_board_detector import ValidatedBoardDetector
from app.core.exceptions import BoardDetectionError


class ImprovedBoardDetector:
    """
    Improved board detector that combines validation with fallback logic
    """
    
    def __init__(self):
        self.validated_detector = ValidatedBoardDetector()
        self.original_detector = BoardDetector()
        self.prefer_validated = True
        
    def detect_board(self, image: np.ndarray, debug: bool = False) -> np.ndarray:
        """
        Detect chess board with validation and fallback
        """
        # First try validated detector
        if self.prefer_validated:
            try:
                board = self.validated_detector.detect_board(image, debug=debug)
                if debug:
                    print("✅ Used validated detector")
                return board
            except BoardDetectionError as e:
                if debug:
                    print(f"⚠️  Validated detector failed: {e}")
                    print("   Falling back to original detector...")
        
        # Fallback to original detector
        try:
            board = self.original_detector.detect_board(image)
            if debug:
                print("✅ Used original detector (fallback)")
            return board
        except BoardDetectionError as e:
            # If both fail, provide more informative error
            raise BoardDetectionError(
                f"Both detectors failed. Validated: {str(e)}, "
                f"Original: Board detection failed"
            )
    
    def extract_squares(self, board_image: np.ndarray) -> List[np.ndarray]:
        """Extract 64 individual square images from the board"""
        # Both detectors use the same extraction method
        return self.validated_detector.extract_squares(board_image)
    
    def set_validation_threshold(self, threshold: float):
        """Adjust the validation threshold"""
        self.validated_detector.min_score_threshold = threshold
    
    def disable_smart_margins(self):
        """Disable smart margin detection"""
        self.validated_detector.apply_smart_margins = False
    
    def enable_smart_margins(self):
        """Enable smart margin detection"""
        self.validated_detector.apply_smart_margins = True