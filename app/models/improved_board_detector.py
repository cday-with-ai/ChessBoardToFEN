import cv2
import numpy as np
from typing import List, Optional, Tuple
from app.models.board_detector import BoardDetector
from app.models.validated_board_detector import ValidatedBoardDetector
from app.models.perspective_board_detector import PerspectiveBoardDetector
from app.core.exceptions import BoardDetectionError


class ImprovedBoardDetector:
    """
    Improved board detector that combines validation with fallback logic
    """
    
    def __init__(self):
        self.perspective_detector = PerspectiveBoardDetector()
        self.validated_detector = ValidatedBoardDetector()
        self.original_detector = BoardDetector()
        self.prefer_validated = True
        self.enable_perspective = True
        
    def detect_board(self, image: np.ndarray, debug: bool = False) -> np.ndarray:
        """
        Detect chess board with validation, perspective correction, and fallback
        """
        # First try perspective-aware detector if enabled
        if self.enable_perspective and self.prefer_validated:
            try:
                board = self.perspective_detector.detect_board(image, debug=debug)
                if debug:
                    print("✅ Used perspective detector")
                return board
            except BoardDetectionError as e:
                if debug:
                    print(f"⚠️  Perspective detector failed: {e}")
                    print("   Trying validated detector...")
        
        # Try validated detector
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
            # If all fail, provide informative error
            raise BoardDetectionError(
                f"All detectors failed. Last error: {str(e)}"
            )
    
    def extract_squares(self, board_image: np.ndarray) -> List[np.ndarray]:
        """Extract 64 individual square images from the board"""
        # All detectors use the same extraction method
        return self.perspective_detector.extract_squares(board_image)
    
    def set_validation_threshold(self, threshold: float):
        """Adjust the validation threshold"""
        self.validated_detector.min_score_threshold = threshold
        self.perspective_detector.min_score_threshold = threshold
    
    def disable_smart_margins(self):
        """Disable smart margin detection"""
        self.validated_detector.apply_smart_margins = False
        self.perspective_detector.apply_smart_margins = False
    
    def enable_smart_margins(self):
        """Enable smart margin detection"""
        self.validated_detector.apply_smart_margins = True
        self.perspective_detector.apply_smart_margins = True
    
    def disable_perspective_correction(self):
        """Disable perspective correction"""
        self.enable_perspective = False
        self.perspective_detector.enable_perspective_correction = False
    
    def enable_perspective_correction(self):
        """Enable perspective correction"""
        self.enable_perspective = True
        self.perspective_detector.enable_perspective_correction = True