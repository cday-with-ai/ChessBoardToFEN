import cv2
import numpy as np
from typing import List, Tuple, Optional
from app.utils.image_utils import (
    preprocess_for_detection, 
    four_point_transform,
    resize_image
)
from app.core.exceptions import BoardDetectionError


class BoardDetector:
    def __init__(self):
        self.min_board_area_ratio = 0.1  # Board should be at least 10% of image
        self.canny_threshold1 = 50
        self.canny_threshold2 = 150
        
    def detect_board(self, image: np.ndarray) -> np.ndarray:
        """
        Detect chess board in the image and return perspective-corrected board
        """
        # Resize if too large
        original_image = image.copy()
        image = resize_image(image, max_dimension=1000)
        
        # Get grayscale and blurred versions
        gray, blurred = preprocess_for_detection(image)
        
        # Try multiple detection methods
        board_corners = self._detect_board_corners(image, gray, blurred)
        
        if board_corners is None:
            raise BoardDetectionError("Could not detect chess board in the image")
        
        # Apply perspective transform to get square board
        board_image = four_point_transform(original_image, board_corners)
        
        # Ensure the board is square
        height, width = board_image.shape[:2]
        size = min(height, width)
        board_image = cv2.resize(board_image, (size, size))
        
        return board_image
    
    def _detect_board_corners(self, image: np.ndarray, gray: np.ndarray, 
                            blurred: np.ndarray) -> Optional[np.ndarray]:
        """Try multiple methods to detect board corners"""
        # Method 1: Edge detection + contour finding
        corners = self._detect_by_edges(image, gray, blurred)
        if corners is not None:
            return corners
        
        # Method 2: Hough line detection
        corners = self._detect_by_lines(image, gray)
        if corners is not None:
            return corners
        
        # Method 3: Simple largest quadrilateral
        corners = self._detect_largest_quad(image, gray)
        return corners
    
    def _detect_by_edges(self, image: np.ndarray, gray: np.ndarray, 
                        blurred: np.ndarray) -> Optional[np.ndarray]:
        """Detect board using edge detection and contours"""
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, self.canny_threshold1, self.canny_threshold2)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        image_area = image.shape[0] * image.shape[1]
        min_area = image_area * self.min_board_area_ratio
        
        for contour in contours[:5]:  # Check top 5 contours
            # Approximate contour to polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # Look for quadrilaterals
            if len(approx) == 4 and cv2.contourArea(approx) > min_area:
                # Check if it's roughly square-ish
                corners = approx.reshape(4, 2)
                if self._is_square_like(corners):
                    return corners
        
        return None
    
    def _detect_by_lines(self, image: np.ndarray, gray: np.ndarray) -> Optional[np.ndarray]:
        """Detect board using Hough line detection"""
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 
                               minLineLength=100, maxLineGap=10)
        
        if lines is None:
            return None
        
        # Find horizontal and vertical lines
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            if angle < 10 or angle > 170:  # Horizontal
                horizontal_lines.append(line[0])
            elif 80 < angle < 100:  # Vertical
                vertical_lines.append(line[0])
        
        # Try to find board boundaries from lines
        if len(horizontal_lines) >= 2 and len(vertical_lines) >= 2:
            # Sort lines by position
            horizontal_lines.sort(key=lambda l: (l[1] + l[3]) / 2)
            vertical_lines.sort(key=lambda l: (l[0] + l[2]) / 2)
            
            # Get outer lines
            top_line = horizontal_lines[0]
            bottom_line = horizontal_lines[-1]
            left_line = vertical_lines[0]
            right_line = vertical_lines[-1]
            
            # Find intersections
            corners = []
            for h_line in [top_line, bottom_line]:
                for v_line in [left_line, right_line]:
                    intersection = self._line_intersection(h_line, v_line)
                    if intersection is not None:
                        corners.append(intersection)
            
            if len(corners) == 4:
                return np.array(corners, dtype=np.float32)
        
        return None
    
    def _detect_largest_quad(self, image: np.ndarray, gray: np.ndarray) -> Optional[np.ndarray]:
        """Fallback: detect the largest quadrilateral in the image"""
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Sort by area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for contour in contours[:10]:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            if len(approx) == 4:
                return approx.reshape(4, 2)
        
        return None
    
    def _is_square_like(self, corners: np.ndarray, tolerance: float = 0.3) -> bool:
        """Check if the quadrilateral is roughly square-shaped"""
        # Calculate side lengths
        sides = []
        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]
            length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            sides.append(length)
        
        # Check if all sides are similar in length
        avg_side = np.mean(sides)
        for side in sides:
            if abs(side - avg_side) / avg_side > tolerance:
                return False
        
        return True
    
    def _line_intersection(self, line1: np.ndarray, line2: np.ndarray) -> Optional[np.ndarray]:
        """Find intersection point of two lines"""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        
        return np.array([x, y])
    
    def extract_squares(self, board_image: np.ndarray) -> List[np.ndarray]:
        """
        Extract 64 individual square images from the board
        Returns squares in FEN order (a8 to h1)
        """
        height, width = board_image.shape[:2]
        
        # Use float division for precise positioning
        square_height = height / 8.0
        square_width = width / 8.0
        
        squares = []
        
        # Extract squares row by row (starting from top = rank 8)
        for row in range(8):
            for col in range(8):
                # Calculate precise boundaries
                y1 = int(row * square_height)
                y2 = int((row + 1) * square_height)
                x1 = int(col * square_width)
                x2 = int((col + 1) * square_width)
                
                # Ensure we don't exceed board boundaries
                y2 = min(y2, height)
                x2 = min(x2, width)
                
                square = board_image[y1:y2, x1:x2]
                
                # Resize to uniform size to eliminate size variations
                if square.size > 0:
                    square = cv2.resize(square, (64, 64))
                else:
                    # Create empty square if extraction failed
                    square = np.zeros((64, 64, 3), dtype=np.uint8)
                
                squares.append(square)
        
        return squares