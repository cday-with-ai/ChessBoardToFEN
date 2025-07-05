import cv2
import numpy as np
from typing import Tuple, Optional, List
from app.models.validated_board_detector import ValidatedBoardDetector
from app.utils.image_utils import four_point_transform
from app.core.exceptions import BoardDetectionError


class PerspectiveBoardDetector(ValidatedBoardDetector):
    """
    Enhanced board detector with perspective correction for angled photos
    """
    
    def __init__(self):
        super().__init__()
        self.enable_perspective_correction = True
        self.perspective_threshold = 0.15  # How much perspective distortion to tolerate
        
    def detect_board(self, image: np.ndarray, debug: bool = False) -> np.ndarray:
        """
        Detect chess board with perspective correction
        """
        # First try standard detection
        try:
            board = super().detect_board(image, debug=False)
            
            # Check if board needs perspective correction
            if self.enable_perspective_correction:
                corrected = self._check_and_correct_perspective(board, image, debug)
                if corrected is not None:
                    if debug:
                        print("Applied perspective correction")
                    return corrected
                    
            return board
            
        except BoardDetectionError:
            # If standard detection fails, try perspective-aware detection
            if debug:
                print("Standard detection failed, trying perspective-aware detection...")
                
            return self._detect_tilted_board(image, debug)
    
    def _detect_tilted_board(self, image: np.ndarray, debug: bool = False) -> np.ndarray:
        """
        Specialized detection for tilted/angled boards
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Find chess pattern using Hough lines
        board_corners = self._find_board_by_grid_lines(image, gray, debug)
        
        if board_corners is None:
            # Method 2: Find by strong corners (board corners)
            board_corners = self._find_board_by_corners(image, gray, debug)
            
        if board_corners is None:
            # Method 3: Find largest quadrilateral with chess-like properties
            board_corners = self._find_board_by_quadrilateral(image, gray, debug)
            
        if board_corners is None:
            raise BoardDetectionError("Could not detect tilted board")
            
        # Apply perspective transform
        board = four_point_transform(image, board_corners)
        
        # Make it square
        height, width = board.shape[:2]
        size = max(height, width)  # Use max to avoid cutting content
        board = cv2.resize(board, (size, size))
        
        return board
    
    def _find_board_by_grid_lines(self, image: np.ndarray, gray: np.ndarray, 
                                  debug: bool = False) -> Optional[np.ndarray]:
        """
        Find board by detecting the chess grid pattern
        """
        # Edge detection with lower threshold for subtle lines
        edges = cv2.Canny(gray, 30, 100)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, 
                               minLineLength=50, maxLineGap=10)
        
        if lines is None:
            return None
            
        # Classify lines as horizontal or vertical
        h_lines = []
        v_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            if angle < 30 or angle > 150:  # Roughly horizontal
                h_lines.append(line[0])
            elif 60 < angle < 120:  # Roughly vertical
                v_lines.append(line[0])
        
        if debug:
            print(f"Found {len(h_lines)} horizontal and {len(v_lines)} vertical lines")
            
        # Need at least 8 lines in each direction for a chess board
        if len(h_lines) < 8 or len(v_lines) < 8:
            return None
            
        # Find the bounding quadrilateral of the grid
        return self._find_grid_boundary(h_lines, v_lines)
    
    def _find_grid_boundary(self, h_lines: List, v_lines: List) -> Optional[np.ndarray]:
        """
        Find the outer boundary of the chess grid
        """
        # Get all intersection points
        intersections = []
        
        for h_line in h_lines:
            x1, y1, x2, y2 = h_line
            for v_line in v_lines:
                x3, y3, x4, y4 = v_line
                
                # Find intersection point
                pt = self._line_intersection(
                    (x1, y1), (x2, y2), (x3, y3), (x4, y4)
                )
                if pt is not None:
                    intersections.append(pt)
        
        if len(intersections) < 4:
            return None
            
        # Convert to numpy array
        points = np.array(intersections, dtype=np.float32)
        
        # Find convex hull
        hull = cv2.convexHull(points)
        
        # If hull has more than 4 points, find the best quadrilateral
        if len(hull) > 4:
            # Find minimum area rectangle
            rect = cv2.minAreaRect(hull)
            corners = cv2.boxPoints(rect)
            corners = np.array(corners, dtype=np.float32)
        else:
            corners = hull.reshape(-1, 2)
            
        return self._order_corners(corners)
    
    def _find_board_by_corners(self, image: np.ndarray, gray: np.ndarray,
                               debug: bool = False) -> Optional[np.ndarray]:
        """
        Find board by detecting strong corner features
        """
        # Harris corner detection
        corners = cv2.cornerHarris(gray, 2, 3, 0.04)
        corners = cv2.dilate(corners, None)
        
        # Threshold for strong corners
        threshold = 0.01 * corners.max()
        corner_points = np.where(corners > threshold)
        corner_points = np.column_stack((corner_points[1], corner_points[0]))
        
        if len(corner_points) < 4:
            return None
            
        # Cluster corners to find board corners
        # Use k-means to find 4 main corner regions
        if len(corner_points) > 4:
            try:
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=4, random_state=42)
                kmeans.fit(corner_points)
                board_corners = kmeans.cluster_centers_
            except ImportError:
                # Fallback: find 4 corners that form largest quadrilateral
                board_corners = self._find_extreme_corners(corner_points)
        else:
            board_corners = corner_points
            
        return self._order_corners(board_corners.astype(np.float32))
    
    def _find_board_by_quadrilateral(self, image: np.ndarray, gray: np.ndarray,
                                      debug: bool = False) -> Optional[np.ndarray]:
        """
        Find largest quadrilateral that could be a chess board
        """
        # Use adaptive threshold for better edge detection
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for contour in contours[:10]:  # Check top 10 largest
            # Approximate polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            if len(approx) == 4:
                # Check if it could be a board
                area = cv2.contourArea(approx)
                if area > (image.shape[0] * image.shape[1]) * 0.1:  # At least 10% of image
                    corners = approx.reshape(4, 2).astype(np.float32)
                    
                    # Validate it's roughly square-ish (allowing for perspective)
                    if self._is_board_like(corners):
                        return self._order_corners(corners)
                        
        return None
    
    def _check_and_correct_perspective(self, board: np.ndarray, original: np.ndarray,
                                        debug: bool = False) -> Optional[np.ndarray]:
        """
        Check if board has perspective distortion and correct it
        """
        # Detect if board is tilted by analyzing grid lines
        gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
        
        # Detect internal grid lines
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
        
        if lines is None:
            return None
            
        # Analyze line angles
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = theta * 180 / np.pi
            angles.append(angle)
            
        # Check if lines are not perfectly horizontal/vertical
        h_angles = [a for a in angles if a < 10 or a > 170]
        v_angles = [a for a in angles if 80 < a < 100]
        
        if not h_angles or not v_angles:
            return None
            
        # Calculate average deviation from perfect grid
        h_deviation = min(np.mean(h_angles), 180 - np.mean(h_angles))
        v_deviation = abs(90 - np.mean(v_angles))
        
        total_deviation = (h_deviation + v_deviation) / 2
        
        if debug:
            print(f"Grid deviation: {total_deviation:.1f} degrees")
            
        # If deviation is significant, apply correction
        if total_deviation > 5:  # More than 5 degrees average tilt
            return self._correct_perspective_by_grid(board, debug)
            
        return None
    
    def _correct_perspective_by_grid(self, board: np.ndarray, 
                                     debug: bool = False) -> np.ndarray:
        """
        Correct perspective by detecting and aligning the internal grid
        """
        # This is a simplified version - full implementation would
        # detect all 81 grid intersections and map to perfect grid
        
        h, w = board.shape[:2]
        
        # For now, assume the board just needs slight rotation correction
        # Detect the dominant angle
        gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
        
        if lines is not None and len(lines) > 0:
            # Find most common angle
            angles = [line[0][1] * 180 / np.pi for line in lines]
            h_angles = [a for a in angles if a < 45 or a > 135]
            
            if h_angles:
                avg_angle = np.mean(h_angles)
                if avg_angle > 90:
                    avg_angle = avg_angle - 180
                    
                if abs(avg_angle) > 2:  # More than 2 degrees rotation
                    if debug:
                        print(f"Rotating by {-avg_angle:.1f} degrees")
                    
                    # Rotate image
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, -avg_angle, 1.0)
                    rotated = cv2.warpAffine(board, M, (w, h))
                    
                    return rotated
                    
        return board
    
    def _line_intersection(self, p1: Tuple, p2: Tuple, p3: Tuple, p4: Tuple) -> Optional[Tuple]:
        """
        Find intersection point of two lines
        """
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            return None
            
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        
        return (int(x), int(y))
    
    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """
        Order corners as top-left, top-right, bottom-right, bottom-left
        """
        # Calculate center
        center = np.mean(corners, axis=0)
        
        # Sort by angle from center
        angles = []
        for corner in corners:
            angle = np.arctan2(corner[1] - center[1], corner[0] - center[0])
            angles.append(angle)
            
        # Sort corners by angle
        sorted_indices = np.argsort(angles)
        sorted_corners = corners[sorted_indices]
        
        # Find top-left (minimum sum of coordinates)
        sums = [c[0] + c[1] for c in sorted_corners]
        tl_idx = np.argmin(sums)
        
        # Reorder starting from top-left
        ordered = np.roll(sorted_corners, -tl_idx, axis=0)
        
        return ordered
    
    def _is_board_like(self, corners: np.ndarray) -> bool:
        """
        Check if quadrilateral could be a chess board
        """
        # Calculate side lengths
        sides = []
        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]
            length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            sides.append(length)
            
        # Check if roughly square (allowing for perspective)
        min_side = min(sides)
        max_side = max(sides)
        
        if min_side > 0:
            ratio = max_side / min_side
            return ratio < 2.0  # Allow up to 2:1 ratio for perspective
            
        return False
    
    def _find_extreme_corners(self, points: np.ndarray) -> np.ndarray:
        """
        Find 4 extreme corners from a set of points (fallback for no sklearn)
        """
        # Find corners by extremes
        top_left = points[np.argmin(points[:, 0] + points[:, 1])]
        bottom_right = points[np.argmax(points[:, 0] + points[:, 1])]
        top_right = points[np.argmax(points[:, 0] - points[:, 1])]
        bottom_left = points[np.argmax(points[:, 1] - points[:, 0])]
        
        return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)