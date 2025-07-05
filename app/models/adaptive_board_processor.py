import cv2
import numpy as np
from typing import Tuple, List, Optional
from app.models.board_detector import BoardDetector
from app.models.improved_board_detector import ImprovedBoardDetector
from app.models.piece_classifier import PieceClassifier
from app.models.image_type_classifier import ImageTypeClassifier, ImageType
from app.utils.image_utils import four_point_transform, resize_image
from app.core.exceptions import BoardDetectionError


class AdaptiveBoardProcessor:
    """
    Adaptive processor that uses different strategies based on image type
    """
    
    def __init__(self):
        self.image_classifier = ImageTypeClassifier()
        self.board_detector = ImprovedBoardDetector()  # Using improved detector
        self.piece_classifier = PieceClassifier()
    
    def process_image(self, image: np.ndarray) -> Tuple[List[Tuple[str, float]], dict]:
        """
        Process chess board image adaptively based on its type
        Returns: (piece_classifications, metadata)
        """
        # Step 1: Classify image type
        image_type, type_confidence, features = self.image_classifier.classify_image(image)
        
        # Step 2: Get preprocessing parameters
        params = self.image_classifier.get_preprocessing_params(image_type)
        
        # Step 3: Preprocess image based on type
        processed_image = self._preprocess_image(image, image_type, params)
        
        # Step 4: Detect board with type-specific strategy
        board_image = self._detect_board_adaptive(processed_image, image_type, params)
        
        # Step 5: Extract squares with appropriate margins
        squares = self._extract_squares_adaptive(board_image, params['edge_margin'])
        
        # Step 6: Enhance squares if needed
        if image_type in [ImageType.PHOTO_OVERHEAD, ImageType.PHOTO_ANGLE]:
            squares = self._enhance_squares(squares)
        
        # Step 7: Classify pieces
        classifications = self.piece_classifier.classify_board(squares)
        
        # Prepare metadata
        metadata = {
            'image_type': image_type.value,
            'type_confidence': type_confidence,
            'preprocessing_applied': params,
            'features': features
        }
        
        return classifications, metadata
    
    def _preprocess_image(self, image: np.ndarray, image_type: ImageType, 
                         params: dict) -> np.ndarray:
        """Apply type-specific preprocessing"""
        processed = image.copy()
        
        # Denoise if needed
        if params.get('denoise', False):
            processed = cv2.fastNlMeansDenoisingColored(processed, None, 10, 10, 7, 21)
        
        # Enhance contrast if needed
        if params.get('enhance_contrast', False):
            # Convert to LAB color space
            lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge and convert back
            processed = cv2.merge([l, a, b])
            processed = cv2.cvtColor(processed, cv2.COLOR_LAB2BGR)
        
        # Sharpen if needed
        if params.get('sharpen', False):
            kernel = np.array([[-1,-1,-1],
                              [-1, 9,-1],
                              [-1,-1,-1]])
            processed = cv2.filter2D(processed, -1, kernel)
        
        return processed
    
    def _detect_board_adaptive(self, image: np.ndarray, image_type: ImageType, 
                              params: dict) -> np.ndarray:
        """Detect board using type-specific strategy"""
        if image_type == ImageType.DIGITAL_CLEAN:
            # For clean digital boards, use simple detection
            return self._detect_digital_board(image)
        
        elif image_type == ImageType.DIGITAL_SCREENSHOT:
            # For screenshots, try to crop out UI first
            cropped = self._crop_ui_elements(image)
            return self._detect_digital_board(cropped)
        
        elif image_type in [ImageType.PHOTO_OVERHEAD, ImageType.PHOTO_ANGLE]:
            # For photos, use more robust detection
            return self._detect_photo_board(image, params['perspective_correction'])
        
        else:
            # Fallback to standard detection
            return self.board_detector.detect_board(image)
    
    def _detect_digital_board(self, image: np.ndarray) -> np.ndarray:
        """Simplified detection for digital boards"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # For digital boards, look for the largest square region
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise BoardDetectionError("No board found")
        
        # Find largest quadrilateral
        largest_area = 0
        best_contour = None
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > largest_area:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                if len(approx) == 4:
                    largest_area = area
                    best_contour = approx.reshape(4, 2)
        
        if best_contour is None:
            # Fallback: use bounding rectangle
            x, y, w, h = cv2.boundingRect(contours[0])
            best_contour = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.float32)
        
        # Transform to square
        board = four_point_transform(image, best_contour)
        size = min(board.shape[:2])
        board = cv2.resize(board, (size, size))
        
        return board
    
    def _crop_ui_elements(self, image: np.ndarray) -> np.ndarray:
        """Try to crop out UI elements from screenshots"""
        # Simple strategy: find the largest rectangular region with chess-like pattern
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Look for regions with alternating patterns
        h, w = gray.shape
        best_region = None
        best_score = 0
        
        # Try different crop regions
        for top in range(0, h//4, 20):
            for bottom in range(3*h//4, h, 20):
                for left in range(0, w//4, 20):
                    for right in range(3*w//4, w, 20):
                        region = gray[top:bottom, left:right]
                        if region.size > 0:
                            score = self._score_chess_pattern(region)
                            if score > best_score:
                                best_score = score
                                best_region = (top, bottom, left, right)
        
        if best_region:
            top, bottom, left, right = best_region
            return image[top:bottom, left:right]
        
        return image
    
    def _detect_photo_board(self, image: np.ndarray, correction_level: str) -> np.ndarray:
        """More sophisticated detection for photographed boards"""
        # This would ideally use the board detector with enhanced parameters
        # For now, use the standard detector
        try:
            return self.board_detector.detect_board(image)
        except:
            # If standard detection fails, try alternative approach
            return self._detect_board_by_pieces(image)
    
    def _detect_board_by_pieces(self, image: np.ndarray) -> np.ndarray:
        """Try to detect board by finding piece patterns"""
        # This is a fallback method that looks for piece-like objects
        # and infers board location from their arrangement
        
        # For now, just return a centered square region
        h, w = image.shape[:2]
        size = min(h, w) * 3 // 4
        
        center_y, center_x = h // 2, w // 2
        y1 = max(0, center_y - size // 2)
        x1 = max(0, center_x - size // 2)
        y2 = min(h, y1 + size)
        x2 = min(w, x1 + size)
        
        board = image[y1:y2, x1:x2]
        board = cv2.resize(board, (size, size))
        
        return board
    
    def _extract_squares_adaptive(self, board_image: np.ndarray, margin: float) -> List[np.ndarray]:
        """Extract squares with adaptive margins"""
        height, width = board_image.shape[:2]
        square_height = height // 8
        square_width = width // 8
        
        squares = []
        
        # Calculate pixel margins
        margin_y = int(square_height * margin)
        margin_x = int(square_width * margin)
        
        for row in range(8):
            for col in range(8):
                y1 = row * square_height + margin_y
                y2 = (row + 1) * square_height - margin_y
                x1 = col * square_width + margin_x
                x2 = (col + 1) * square_width - margin_x
                
                # Ensure valid bounds
                y1, y2 = max(0, y1), min(height, y2)
                x1, x2 = max(0, x1), min(width, x2)
                
                square = board_image[y1:y2, x1:x2]
                squares.append(square)
        
        return squares
    
    def _enhance_squares(self, squares: List[np.ndarray]) -> List[np.ndarray]:
        """Enhance individual squares for better piece recognition"""
        enhanced = []
        
        for square in squares:
            # Resize to standard size if too small
            if square.shape[0] < 64 or square.shape[1] < 64:
                square = cv2.resize(square, (64, 64))
            
            # Enhance contrast
            lab = cv2.cvtColor(square, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            square = cv2.merge([l, a, b])
            square = cv2.cvtColor(square, cv2.COLOR_LAB2BGR)
            
            enhanced.append(square)
        
        return enhanced
    
    def _score_chess_pattern(self, region: np.ndarray) -> float:
        """Score how chess-like a region is"""
        # Simple scoring based on alternating pattern
        h, w = region.shape
        if h < 100 or w < 100:
            return 0
        
        # Sample grid points
        score = 0
        samples = 10
        step_y, step_x = h // samples, w // samples
        
        for i in range(samples - 1):
            for j in range(samples - 1):
                y, x = i * step_y, j * step_x
                
                # Check 2x2 block
                tl = region[y, x]
                tr = region[y, x + step_x]
                bl = region[y + step_y, x]
                br = region[y + step_y, x + step_x]
                
                # Good if diagonal similar, adjacent different
                if abs(int(tl) - int(br)) < 30 and abs(int(tr) - int(bl)) < 30:
                    if abs(int(tl) - int(tr)) > 50:
                        score += 1
        
        return score / ((samples - 1) ** 2)