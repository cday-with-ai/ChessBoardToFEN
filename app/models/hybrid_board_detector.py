import cv2
import numpy as np
from typing import List, Tuple, Optional
from app.utils.image_utils import (
    preprocess_for_detection, 
    four_point_transform,
    resize_image
)
from app.core.exceptions import BoardDetectionError


class HybridBoardDetector:
    """
    Hybrid approach: Use original detector's methods but add validation
    to ensure we're getting actual chess boards
    """
    
    def __init__(self):
        self.min_board_area_ratio = 0.1
        self.max_board_area_ratio = 0.9
        self.canny_threshold1 = 50
        self.canny_threshold2 = 150
        
    def detect_board(self, image: np.ndarray) -> np.ndarray:
        """Detect chess board with validation"""
        original_image = image.copy()
        image = resize_image(image, max_dimension=1000)
        
        gray, blurred = preprocess_for_detection(image)
        
        # Try multiple detection methods
        candidates = []
        
        # Method 1: Edge detection
        corners = self._detect_by_edges(image, gray, blurred)
        if corners is not None:
            score = self._score_board_candidate(image, corners)
            if score > 0:
                candidates.append((corners, score, "edges"))
        
        # Method 2: Largest quad with validation
        corners = self._detect_largest_quad(image, gray)
        if corners is not None:
            score = self._score_board_candidate(image, corners)
            if score > 0:
                candidates.append((corners, score, "quad"))
        
        # Method 3: Color-based detection
        corners = self._detect_by_color_clustering(image)
        if corners is not None:
            score = self._score_board_candidate(image, corners)
            if score > 0:
                candidates.append((corners, score, "color"))
        
        if not candidates:
            raise BoardDetectionError("Could not detect chess board in the image")
        
        # Use best candidate
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_corners, best_score, method = candidates[0]
        print(f"Board detected using {method} method with score {best_score:.2f}")
        
        # Scale back to original size
        scale = original_image.shape[0] / image.shape[0]
        best_corners = best_corners * scale
        
        # Transform and return
        board_image = four_point_transform(original_image, best_corners)
        height, width = board_image.shape[:2]
        size = min(height, width)
        board_image = cv2.resize(board_image, (size, size))
        
        return board_image
    
    def _detect_by_edges(self, image, gray, blurred):
        """Original edge detection method"""
        edges = cv2.Canny(blurred, self.canny_threshold1, self.canny_threshold2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        image_area = image.shape[0] * image.shape[1]
        min_area = image_area * self.min_board_area_ratio
        max_area = image_area * self.max_board_area_ratio
        
        for contour in contours[:5]:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            if len(approx) == 4:
                area = cv2.contourArea(approx)
                if min_area < area < max_area:
                    corners = approx.reshape(4, 2)
                    if self._is_square_like(corners):
                        return corners
        return None
    
    def _detect_largest_quad(self, image, gray):
        """Detect largest quadrilateral"""
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        image_area = image.shape[0] * image.shape[1]
        
        for contour in contours[:10]:
            area = cv2.contourArea(contour)
            if area < image_area * 0.1 or area > image_area * 0.9:
                continue
                
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            if len(approx) == 4:
                return approx.reshape(4, 2)
        
        return None
    
    def _detect_by_color_clustering(self, image):
        """Detect board by finding large uniform regions"""
        # Convert to LAB color space for better color separation
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Simple k-means clustering to find dominant colors
        Z = lab.reshape((-1, 3))
        Z = np.float32(Z)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 5  # Look for 5 dominant colors
        _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        
        # Reshape back
        labels = labels.reshape((image.shape[0], image.shape[1]))
        
        # Find largest connected regions
        for label_id in range(K):
            mask = (labels == label_id).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue
                
            largest = max(contours, key=cv2.contourArea)
            peri = cv2.arcLength(largest, True)
            approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
            
            if len(approx) == 4:
                corners = approx.reshape(4, 2)
                if self._is_square_like(corners):
                    return corners
        
        return None
    
    def _score_board_candidate(self, image, corners):
        """Score how likely this region is a chess board"""
        try:
            # Transform to square for analysis
            warped = four_point_transform(image, corners)
            warped = cv2.resize(warped, (400, 400))
        except:
            return 0
        
        score = 0
        
        # Check aspect ratio
        if self._is_square_like(corners, tolerance=0.2):
            score += 2
        elif self._is_square_like(corners, tolerance=0.4):
            score += 1
        
        # Check for alternating pattern
        pattern_score = self._check_checkerboard_pattern(warped)
        score += pattern_score * 3
        
        # Check for regular grid lines
        grid_score = self._check_grid_lines(warped)
        score += grid_score * 2
        
        # Penalty for too much text/UI elements
        text_score = self._check_for_text(warped)
        score -= text_score * 2
        
        return max(0, score)
    
    def _check_checkerboard_pattern(self, image):
        """Check for alternating squares pattern"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Sample grid points
        h, w = gray.shape
        step = h // 8
        
        alternating_count = 0
        total_checks = 0
        
        for i in range(7):
            for j in range(7):
                # Get 2x2 block
                y, x = i * step + step//2, j * step + step//2
                
                # Sample 4 squares
                try:
                    tl = gray[y, x]
                    tr = gray[y, x + step]
                    bl = gray[y + step, x]
                    br = gray[y + step, x + step]
                    
                    # Check alternating pattern
                    if abs(int(tl) - int(br)) < 50 and abs(int(tr) - int(bl)) < 50:
                        if abs(int(tl) - int(tr)) > 30 and abs(int(tl) - int(bl)) > 30:
                            alternating_count += 1
                    
                    total_checks += 1
                except:
                    pass
        
        return alternating_count / max(1, total_checks)
    
    def _check_grid_lines(self, image):
        """Check for regular grid pattern"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, 
                               minLineLength=100, maxLineGap=10)
        
        if lines is None:
            return 0
        
        # Count horizontal and vertical lines
        h_lines = 0
        v_lines = 0
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            if angle < 10 or angle > 170:
                h_lines += 1
            elif 80 < angle < 100:
                v_lines += 1
        
        # Good if we have multiple lines in both directions
        return min(1.0, (h_lines + v_lines) / 20)
    
    def _check_for_text(self, image):
        """Check for text/UI elements"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Look for high frequency content (text)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        # High variance might indicate text/UI
        return min(1.0, variance / 5000)
    
    def _is_square_like(self, corners, tolerance=0.3):
        """Check if roughly square"""
        sides = []
        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]
            length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            sides.append(length)
        
        avg_side = np.mean(sides)
        for side in sides:
            if abs(side - avg_side) / avg_side > tolerance:
                return False
        return True
    
    def extract_squares(self, board_image):
        """Extract 64 squares with slight margin"""
        height, width = board_image.shape[:2]
        square_height = height // 8
        square_width = width // 8
        
        squares = []
        margin_percent = 0.05  # 5% margin
        
        for row in range(8):
            for col in range(8):
                # Calculate margins
                margin_y = int(square_height * margin_percent)
                margin_x = int(square_width * margin_percent)
                
                y1 = row * square_height + margin_y
                y2 = (row + 1) * square_height - margin_y
                x1 = col * square_width + margin_x
                x2 = (col + 1) * square_width - margin_x
                
                square = board_image[y1:y2, x1:x2]
                squares.append(square)
        
        return squares