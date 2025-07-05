import cv2
import numpy as np
from typing import List, Tuple, Optional
from app.utils.image_utils import four_point_transform, resize_image
from app.core.exceptions import BoardDetectionError


class ImprovedBoardDetector:
    def __init__(self):
        self.min_board_area_ratio = 0.05  # Board should be at least 5% of image
        self.max_board_area_ratio = 0.9   # Board shouldn't be more than 90% of image
        
    def detect_board(self, image: np.ndarray) -> np.ndarray:
        """
        Detect chess board using pattern matching for checkerboard
        """
        # Resize if too large
        original_image = image.copy()
        scale = 1.0
        if max(image.shape[:2]) > 1000:
            scale = 1000 / max(image.shape[:2])
            image = resize_image(image, max_dimension=1000)
        
        # Try multiple detection strategies
        corners = None
        
        # Method 1: Detect by checkerboard pattern
        corners = self._detect_checkerboard_pattern(image)
        
        # Method 2: Detect by color clustering
        if corners is None:
            corners = self._detect_by_color_pattern(image)
        
        # Method 3: Enhanced edge detection
        if corners is None:
            corners = self._detect_by_enhanced_edges(image)
        
        if corners is None:
            raise BoardDetectionError("Could not detect chess board in the image")
        
        # Scale corners back to original size
        if scale != 1.0:
            corners = corners / scale
        
        # Apply perspective transform
        board_image = four_point_transform(original_image, corners)
        
        # Ensure square
        height, width = board_image.shape[:2]
        size = min(height, width)
        board_image = cv2.resize(board_image, (size, size))
        
        # Validate it's actually a chess board
        if not self._validate_chess_board(board_image):
            raise BoardDetectionError("Detected area doesn't appear to be a chess board")
        
        return board_image
    
    def _detect_checkerboard_pattern(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect chess board by finding checkerboard pattern"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Try to find actual checkerboard corners
        for size in [(7,7), (6,6), (5,5)]:  # Try different internal corner counts
            ret, corners = cv2.findChessboardCorners(
                gray, size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            if ret:
                # Extrapolate to full board edges
                return self._extrapolate_board_corners(corners, size)
        
        return None
    
    def _detect_by_color_pattern(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect board by finding alternating color pattern"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive threshold to get binary image
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 21, 5
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter contours
        image_area = image.shape[0] * image.shape[1]
        min_area = image_area * self.min_board_area_ratio
        max_area = image_area * self.max_board_area_ratio
        
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                # Approximate to polygon
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                if len(approx) == 4:
                    # Check if it's roughly square
                    corners = approx.reshape(4, 2)
                    if self._is_square_like(corners):
                        # Check if interior has checkerboard pattern
                        if self._has_checkerboard_pattern(image, corners):
                            candidates.append((corners, area))
        
        # Return largest valid candidate
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
        
        return None
    
    def _detect_by_enhanced_edges(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Enhanced edge detection focusing on grid patterns"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Detect edges
        edges = cv2.Canny(enhanced, 30, 100)
        
        # Detect lines
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, 50,
            minLineLength=50, maxLineGap=10
        )
        
        if lines is None:
            return None
        
        # Find grid-like patterns
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            if angle < 10 or angle > 170:  # Horizontal
                horizontal_lines.append(line[0])
            elif 80 < angle < 100:  # Vertical
                vertical_lines.append(line[0])
        
        # Look for regular grid pattern
        if len(horizontal_lines) >= 8 and len(vertical_lines) >= 8:
            # Find board boundaries
            h_positions = [(l[1] + l[3]) / 2 for l in horizontal_lines]
            v_positions = [(l[0] + l[2]) / 2 for l in vertical_lines]
            
            h_positions.sort()
            v_positions.sort()
            
            # Look for evenly spaced lines (indicating grid)
            board_regions = self._find_grid_region(h_positions, v_positions, image.shape)
            
            if board_regions:
                return board_regions
        
        return None
    
    def _find_grid_region(self, h_positions: List[float], v_positions: List[float], 
                         image_shape: Tuple) -> Optional[np.ndarray]:
        """Find region with evenly spaced lines (chess board grid)"""
        # Look for sequences of evenly spaced lines
        min_lines = 7  # At least 7 lines to separate 8 squares
        
        for h_start in range(len(h_positions) - min_lines):
            for h_end in range(h_start + min_lines, len(h_positions)):
                h_subset = h_positions[h_start:h_end+1]
                if self._are_evenly_spaced(h_subset):
                    for v_start in range(len(v_positions) - min_lines):
                        for v_end in range(v_start + min_lines, len(v_positions)):
                            v_subset = v_positions[v_start:v_end+1]
                            if self._are_evenly_spaced(v_subset):
                                # Found potential board
                                corners = np.array([
                                    [v_subset[0], h_subset[0]],
                                    [v_subset[-1], h_subset[0]],
                                    [v_subset[-1], h_subset[-1]],
                                    [v_subset[0], h_subset[-1]]
                                ], dtype=np.float32)
                                return corners
        
        return None
    
    def _are_evenly_spaced(self, positions: List[float], tolerance: float = 0.2) -> bool:
        """Check if positions are evenly spaced"""
        if len(positions) < 2:
            return False
        
        spacings = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
        avg_spacing = np.mean(spacings)
        
        for spacing in spacings:
            if abs(spacing - avg_spacing) / avg_spacing > tolerance:
                return False
        
        return True
    
    def _has_checkerboard_pattern(self, image: np.ndarray, corners: np.ndarray) -> bool:
        """Check if the region has a checkerboard pattern"""
        # Transform region to square for analysis
        warped = four_point_transform(image, corners)
        warped = cv2.resize(warped, (400, 400))
        
        # Convert to grayscale
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        
        # Check multiple squares for alternating pattern
        square_size = 50  # 400/8
        alternating_score = 0
        
        for row in range(7):  # Check 7x7 grid intersections
            for col in range(7):
                # Get 2x2 block of squares
                x = col * square_size
                y = row * square_size
                
                # Sample center of each square in 2x2 block
                samples = []
                for dy in [25, 75]:
                    for dx in [25, 75]:
                        samples.append(gray[y+dy, x+dx])
                
                # Check if diagonal squares are similar and adjacent are different
                if abs(samples[0] - samples[3]) < 50 and abs(samples[1] - samples[2]) < 50:
                    if abs(samples[0] - samples[1]) > 30 and abs(samples[0] - samples[2]) > 30:
                        alternating_score += 1
        
        # If most intersections show alternating pattern, it's likely a chess board
        return alternating_score > 20
    
    def _validate_chess_board(self, board_image: np.ndarray) -> bool:
        """Validate that the extracted region is actually a chess board"""
        # Quick validation: check for alternating pattern in a few places
        h, w = board_image.shape[:2]
        square_h, square_w = h // 8, w // 8
        
        gray = cv2.cvtColor(board_image, cv2.COLOR_BGR2GRAY)
        
        # Sample centers of a few squares
        samples = []
        positions = [(1, 1), (1, 2), (2, 1), (2, 2), (4, 4), (4, 5), (5, 4), (5, 5)]
        
        for row, col in positions:
            x = col * square_w + square_w // 2
            y = row * square_h + square_h // 2
            samples.append(gray[y, x])
        
        # Check for alternating pattern
        # Adjacent squares should be different, diagonal should be similar
        diff1 = abs(samples[0] - samples[1])  # Adjacent
        diff2 = abs(samples[0] - samples[2])  # Adjacent
        diff3 = abs(samples[0] - samples[3])  # Diagonal
        
        return diff1 > 20 and diff2 > 20 and diff3 < 30
    
    def _is_square_like(self, corners: np.ndarray, tolerance: float = 0.3) -> bool:
        """Check if quadrilateral is roughly square"""
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
    
    def _extrapolate_board_corners(self, inner_corners: np.ndarray, 
                                  pattern_size: Tuple[int, int]) -> np.ndarray:
        """Extrapolate full board corners from internal checkerboard corners"""
        # This is a simplified version - in practice would need more sophisticated extrapolation
        corners = inner_corners.reshape(-1, 2)
        
        # Find bounding box
        min_x, min_y = corners.min(axis=0)
        max_x, max_y = corners.max(axis=0)
        
        # Estimate square size
        width = max_x - min_x
        height = max_y - min_y
        
        square_w = width / pattern_size[0]
        square_h = height / pattern_size[1]
        
        # Extrapolate to full board
        margin_x = square_w / 2
        margin_y = square_h / 2
        
        board_corners = np.array([
            [min_x - margin_x, min_y - margin_y],
            [max_x + margin_x, min_y - margin_y],
            [max_x + margin_x, max_y + margin_y],
            [min_x - margin_x, max_y + margin_y]
        ], dtype=np.float32)
        
        return board_corners
    
    def extract_squares(self, board_image: np.ndarray) -> List[np.ndarray]:
        """Extract 64 individual square images from the board"""
        height, width = board_image.shape[:2]
        square_height = height // 8
        square_width = width // 8
        
        squares = []
        
        # Extract squares row by row (starting from top = rank 8)
        for row in range(8):
            for col in range(8):
                # Add small margin to avoid board lines
                margin = 2
                y1 = row * square_height + margin
                y2 = (row + 1) * square_height - margin
                x1 = col * square_width + margin
                x2 = (col + 1) * square_width - margin
                
                square = board_image[y1:y2, x1:x2]
                squares.append(square)
        
        return squares