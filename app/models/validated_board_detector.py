import cv2
import numpy as np
from typing import List, Tuple, Optional
from app.utils.image_utils import (
    preprocess_for_detection, 
    four_point_transform,
    resize_image
)
from app.core.exceptions import BoardDetectionError
from app.models.smart_margin_detector import SmartMarginDetector


class ValidatedBoardDetector:
    """
    Board detector with validation scoring to avoid false positives
    """
    
    def __init__(self):
        self.min_board_area_ratio = 0.1
        self.max_board_area_ratio = 0.9
        self.canny_threshold1 = 50
        self.canny_threshold2 = 150
        self.min_score_threshold = 0.25  # Minimum score to accept a board
        self.margin_detector = SmartMarginDetector()
        self.apply_smart_margins = True  # Can be disabled if needed
        
    def detect_board(self, image: np.ndarray, debug: bool = False) -> np.ndarray:
        """
        Detect chess board with validation
        """
        original_image = image.copy()
        image = resize_image(image, max_dimension=1000)
        
        gray, blurred = preprocess_for_detection(image)
        
        # Find all potential board candidates
        candidates = []
        
        # Method 1: Edge detection
        edge_candidates = self._find_candidates_by_edges(image, gray, blurred)
        candidates.extend(edge_candidates)
        
        # Method 2: Find largest squares
        square_candidates = self._find_candidates_by_squares(image, gray)
        candidates.extend(square_candidates)
        
        # Method 3: Color-based detection (works better for wooden/textured boards)
        color_candidates = self._find_candidates_by_color(image)
        candidates.extend(color_candidates)
        
        if not candidates:
            raise BoardDetectionError("No potential chess boards found")
        
        # Score and rank candidates
        scored_candidates = []
        for corners in candidates:
            score, details = self._score_board_candidate(image, corners)
            scored_candidates.append((corners, score, details))
        
        # Sort by score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        if debug:
            print(f"Found {len(scored_candidates)} candidates")
            for i, (_, score, details) in enumerate(scored_candidates[:3]):
                print(f"Candidate {i+1}: score={score:.3f}, details={details}")
        
        # Select best candidate above threshold
        best_corners = None
        best_score = 0
        best_details = {}
        
        for corners, score, details in scored_candidates:
            if score >= self.min_score_threshold:
                best_corners = corners
                best_score = score
                best_details = details
                break
        
        if best_corners is None:
            raise BoardDetectionError(
                f"No valid chess board found. Best score: {scored_candidates[0][1]:.3f} "
                f"(threshold: {self.min_score_threshold})"
            )
        
        # Scale corners back to original size
        scale = original_image.shape[0] / image.shape[0]
        best_corners = best_corners * scale
        
        # Transform and return
        board_image = four_point_transform(original_image, best_corners)
        height, width = board_image.shape[:2]
        size = min(height, width)
        board_image = cv2.resize(board_image, (size, size))
        
        # Apply smart margin detection if enabled
        if self.apply_smart_margins:
            try:
                # Check if this looks like a clean digital board
                # High confidence + good checkerboard pattern = likely clean
                is_clean_digital = (
                    best_score > 0.7 and 
                    best_details.get('checkerboard_pattern', 0) > 0.6 and
                    best_details.get('ui_elements', 0) < 0.1
                )
                
                if not is_clean_digital:
                    margins = self.margin_detector.detect_board_margins(board_image)
                    if debug:
                        print(f"Detected margins: {margins}")
                    
                    # Only apply if significant margins detected
                    margin_threshold = 20  # Require at least 20 pixels to crop
                    if any(m > margin_threshold for m in margins.values()):
                        board_image = self.margin_detector.apply_smart_crop(board_image, margins)
                        # Resize again to ensure square
                        height, width = board_image.shape[:2]
                        size = min(height, width)
                        board_image = cv2.resize(board_image, (size, size))
                elif debug:
                    print("Skipping margin detection for clean digital board")
            except Exception as e:
                if debug:
                    print(f"Smart margin detection failed: {e}")
        
        return board_image
    
    def _find_candidates_by_edges(self, image, gray, blurred):
        """Find board candidates using edge detection"""
        candidates = []
        
        edges = cv2.Canny(blurred, self.canny_threshold1, self.canny_threshold2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        image_area = image.shape[0] * image.shape[1]
        min_area = image_area * self.min_board_area_ratio
        max_area = image_area * self.max_board_area_ratio
        
        for contour in contours[:10]:  # Check top 10
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                if len(approx) == 4:
                    corners = approx.reshape(4, 2)
                    candidates.append(corners)
        
        return candidates
    
    def _find_candidates_by_squares(self, image, gray):
        """Find candidates by looking for square shapes"""
        candidates = []
        
        # Use adaptive threshold
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        image_area = image.shape[0] * image.shape[1]
        
        for contour in contours[:15]:
            area = cv2.contourArea(contour)
            if area > image_area * 0.05:  # At least 5% of image
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                if len(approx) == 4:
                    corners = approx.reshape(4, 2)
                    candidates.append(corners)
        
        return candidates
    
    def _find_candidates_by_color(self, image):
        """Find candidates using color segmentation - works for wooden boards"""
        candidates = []
        
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Try different color ranges that might represent chess boards
        # Brown/wooden boards
        lower_brown = np.array([10, 30, 30])
        upper_brown = np.array([25, 255, 200])
        mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
        
        # Light/beige boards
        lower_beige = np.array([15, 20, 100])
        upper_beige = np.array([30, 100, 255])
        mask_beige = cv2.inRange(hsv, lower_beige, upper_beige)
        
        # Combine masks
        mask = cv2.bitwise_or(mask_brown, mask_beige)
        
        # Apply morphology to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        image_area = image.shape[0] * image.shape[1]
        
        for contour in contours[:5]:
            area = cv2.contourArea(contour)
            if area > image_area * 0.1:  # At least 10% of image
                # Get bounding rect
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check if roughly square
                aspect = w / h if h > 0 else 0
                if 0.7 < aspect < 1.3:
                    # Create corners from bounding rect
                    corners = np.array([
                        [x, y],
                        [x + w, y],
                        [x + w, y + h],
                        [x, y + h]
                    ], dtype=np.float32)
                    candidates.append(corners)
        
        return candidates
    
    def _score_board_candidate(self, image, corners):
        """
        Score how likely this region is a chess board
        Returns: (score, details_dict)
        """
        details = {}
        score = 0.0
        
        # 1. Aspect ratio check (chess boards are square)
        aspect_score = self._check_aspect_ratio(corners)
        details['aspect_ratio'] = aspect_score
        score += aspect_score * 0.2
        
        # 2. Check for checkerboard pattern
        pattern_score = self._check_checkerboard_pattern(image, corners)
        details['checkerboard_pattern'] = pattern_score
        score += pattern_score * 0.4  # Most important feature
        
        # 3. Check for text/UI elements (negative score)
        text_score = self._check_for_text(image, corners)
        details['text_detected'] = text_score
        score -= text_score * 0.3
        
        # 4. Check edge regularity
        edge_score = self._check_edge_regularity(image, corners)
        details['edge_regularity'] = edge_score
        score += edge_score * 0.2
        
        # 5. Check size appropriateness
        size_score = self._check_size_appropriateness(image, corners)
        details['size_appropriate'] = size_score
        score += size_score * 0.1
        
        # 6. Check for grid lines
        grid_score = self._check_for_grid_lines(image, corners)
        details['grid_lines'] = grid_score
        score += grid_score * 0.1
        
        # 7. Check for UI elements (negative score)
        ui_score = self._check_for_ui_elements(image, corners)
        details['ui_elements'] = ui_score
        score -= ui_score * 0.4  # Strong penalty for UI elements
        
        # Ensure score is between 0 and 1
        score = max(0.0, min(1.0, score))
        
        return score, details
    
    def _check_aspect_ratio(self, corners):
        """Check if the shape is roughly square"""
        # Calculate side lengths
        sides = []
        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]
            length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            sides.append(length)
        
        # Check aspect ratio
        width = (sides[0] + sides[2]) / 2
        height = (sides[1] + sides[3]) / 2
        
        if height > 0:
            aspect_ratio = width / height
        else:
            return 0.0
        
        # Score based on how close to 1.0 (perfect square)
        # Allow some tolerance for perspective
        if 0.8 < aspect_ratio < 1.2:
            return 1.0
        elif 0.7 < aspect_ratio < 1.3:
            return 0.5
        else:
            return 0.0
    
    def _check_checkerboard_pattern(self, image, corners):
        """Check for alternating square pattern - improved version"""
        try:
            # Transform to square for analysis
            warped = four_point_transform(image, corners)
            warped = cv2.resize(warped, (400, 400))
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        except:
            return 0.0
        
        # Method 1: Original intersection-based check
        pattern_score1 = self._check_pattern_intersections(gray)
        
        # Method 2: FFT-based pattern detection (works better for textured boards)
        pattern_score2 = self._check_pattern_fft(gray)
        
        # Method 3: Block variance check (works for any alternating pattern)
        pattern_score3 = self._check_block_variance(gray)
        
        # Return best score with weights
        return max(pattern_score1, pattern_score2 * 0.8, pattern_score3 * 0.9)
    
    def _check_pattern_intersections(self, gray):
        """Original intersection-based pattern check"""
        h, w = gray.shape
        step = h // 8
        
        alternating_count = 0
        total_checks = 0
        
        for i in range(1, 7):  # Avoid edges
            for j in range(1, 7):
                y, x = i * step, j * step
                
                try:
                    # Sample 4 squares around intersection
                    tl = gray[y - step//3, x - step//3]
                    tr = gray[y - step//3, x + step//3]
                    bl = gray[y + step//3, x - step//3]
                    br = gray[y + step//3, x + step//3]
                    
                    # Check if diagonal squares are similar and adjacent are different
                    diag_diff = abs(int(tl) - int(br))
                    diag_diff2 = abs(int(tr) - int(bl))
                    adj_diff1 = abs(int(tl) - int(tr))
                    adj_diff2 = abs(int(tl) - int(bl))
                    
                    # Good checkerboard pattern - more lenient for compressed images
                    if diag_diff < 50 and diag_diff2 < 50 and adj_diff1 > 20 and adj_diff2 > 20:
                        alternating_count += 1
                    
                    total_checks += 1
                except:
                    pass
        
        if total_checks > 0:
            return alternating_count / total_checks
        return 0.0
    
    def _check_pattern_fft(self, gray):
        """Use FFT to detect regular grid pattern"""
        try:
            # Apply FFT
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.abs(f_shift)
            
            # Look for peaks at chess board frequencies (8x8 grid)
            h, w = gray.shape
            center_h, center_w = h // 2, w // 2
            
            # Expected frequencies for 8x8 grid
            freq_h = h // 8
            freq_w = w // 8
            
            # Check for peaks at expected frequencies
            peak_region = magnitude[center_h-freq_h-5:center_h-freq_h+5, 
                                  center_w-freq_w-5:center_w-freq_w+5]
            
            if peak_region.size > 0:
                peak_strength = np.max(peak_region) / (np.mean(magnitude) + 1e-6)
                return min(1.0, peak_strength / 100)  # Normalize
            
            return 0.0
        except:
            return 0.0
    
    def _check_block_variance(self, gray):
        """Check variance between blocks"""
        h, w = gray.shape
        block_size = h // 8
        
        variances = []
        
        for i in range(8):
            for j in range(8):
                y1, y2 = i * block_size, (i + 1) * block_size
                x1, x2 = j * block_size, (j + 1) * block_size
                
                block = gray[y1:y2, x1:x2]
                if block.size > 0:
                    variances.append(np.mean(block))
        
        if len(variances) >= 32:  # At least half the squares
            # Check if we have alternating pattern
            try:
                variances = np.array(variances[:64]).reshape(8, 8)
            except:
                return 0.0
            
            # Calculate differences between adjacent squares
            h_diff = np.abs(np.diff(variances, axis=1))
            v_diff = np.abs(np.diff(variances, axis=0))
            
            # Good boards have consistent differences
            if h_diff.size > 0 and v_diff.size > 0:
                avg_diff = (np.mean(h_diff) + np.mean(v_diff)) / 2
                if avg_diff > 10:  # Significant contrast
                    return min(1.0, avg_diff / 50)
        
        return 0.0
    
    def _check_for_text(self, image, corners):
        """Check for text/UI elements in the region"""
        try:
            warped = four_point_transform(image, corners)
            warped = cv2.resize(warped, (400, 400))
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        except:
            return 0.0
        
        # Look for high frequency content (text)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        # Also check for many small contours (text characters)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Many small contours suggest text
        small_contours = sum(1 for c in contours if cv2.contourArea(c) < 500)
        
        # Combine metrics
        text_score = 0.0
        
        # High Laplacian variance suggests text
        if variance > 5000:
            text_score += 0.5
        
        # Many small contours suggest text
        if small_contours > 50:
            text_score += 0.5
        
        return min(1.0, text_score)
    
    def _check_edge_regularity(self, image, corners):
        """Check if edges are clean and regular"""
        try:
            warped = four_point_transform(image, corners)
            warped = cv2.resize(warped, (400, 400))
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        except:
            return 0.0
        
        # Check edges of the board
        edges = cv2.Canny(gray, 50, 150)
        
        # Look for straight lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, 
                               minLineLength=100, maxLineGap=10)
        
        if lines is None:
            return 0.0
        
        # Count horizontal and vertical lines
        h_lines = 0
        v_lines = 0
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            if angle < 10 or angle > 170:  # Horizontal
                h_lines += 1
            elif 80 < angle < 100:  # Vertical
                v_lines += 1
        
        # Good boards have many regular lines
        total_lines = h_lines + v_lines
        if total_lines > 20:
            return 1.0
        elif total_lines > 10:
            return 0.5
        else:
            return 0.0
    
    def _check_size_appropriateness(self, image, corners):
        """Check if the size is appropriate for a chess board"""
        # Calculate area
        area = cv2.contourArea(corners)
        image_area = image.shape[0] * image.shape[1]
        area_ratio = area / image_area
        
        # Chess boards typically take up 20-80% of the image
        if 0.2 < area_ratio < 0.8:
            return 1.0
        elif 0.1 < area_ratio < 0.9:
            return 0.5
        else:
            return 0.0
    
    def _check_for_grid_lines(self, image, corners):
        """Check for presence of grid lines"""
        try:
            warped = four_point_transform(image, corners)
            warped = cv2.resize(warped, (400, 400))
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        except:
            return 0.0
        
        # Look for regular grid pattern
        edges = cv2.Canny(gray, 30, 100)
        
        # Check for evenly spaced lines
        h, w = edges.shape
        
        # Sample horizontal and vertical lines
        h_spacing = []
        v_spacing = []
        
        # Find horizontal lines
        for y in range(0, h, h//20):
            row = edges[y, :]
            line_positions = np.where(row > 0)[0]
            if len(line_positions) > 1:
                spacings = np.diff(line_positions)
                h_spacing.extend(spacings)
        
        # Find vertical lines
        for x in range(0, w, w//20):
            col = edges[:, x]
            line_positions = np.where(col > 0)[0]
            if len(line_positions) > 1:
                spacings = np.diff(line_positions)
                v_spacing.extend(spacings)
        
        # Check for regular spacing (chess board characteristic)
        if h_spacing and v_spacing:
            h_std = np.std(h_spacing)
            v_std = np.std(v_spacing)
            
            # Low standard deviation means regular spacing
            if h_std < 20 and v_std < 20:
                return 1.0
            elif h_std < 40 and v_std < 40:
                return 0.5
        
        return 0.0
    
    def _check_for_ui_elements(self, image, corners):
        """Check for UI-specific elements that indicate this is not a real board"""
        try:
            warped = four_point_transform(image, corners)
            warped = cv2.resize(warped, (400, 400))
        except:
            return 0.0
        
        ui_score = 0.0
        
        # 1. Check for solid color regions (common in UIs)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        
        # Check borders for solid colors
        border_size = 20
        top_border = gray[:border_size, :]
        bottom_border = gray[-border_size:, :]
        left_border = gray[:, :border_size]
        right_border = gray[:, -border_size:]
        
        # Low variance in borders suggests UI chrome
        if np.var(top_border) < 100 or np.var(bottom_border) < 100:
            ui_score += 0.3
        if np.var(left_border) < 100 or np.var(right_border) < 100:
            ui_score += 0.3
        
        # 2. Check color distribution - UIs often have limited colors
        unique_colors = len(np.unique(warped.reshape(-1, 3), axis=0))
        if unique_colors < 100:  # Very few unique colors
            ui_score += 0.2
        
        # 3. Check for rectangular regions (buttons, panels)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
        
        if lines is not None:
            # Count perfectly horizontal/vertical lines
            perfect_lines = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x1 == x2 or y1 == y2:  # Perfect vertical or horizontal
                    perfect_lines += 1
            
            # Many perfect lines suggest UI elements
            if perfect_lines > 20:
                ui_score += 0.2
        
        # 4. Check if the "board" region contains the actual board
        # Real boards fill most of their bounding box
        h, w = gray.shape
        center_region = gray[h//4:3*h//4, w//4:3*w//4]
        if np.var(center_region) < 500:  # Center is too uniform
            ui_score += 0.3
        
        return min(1.0, ui_score)
    
    def extract_squares(self, board_image: np.ndarray) -> List[np.ndarray]:
        """Extract 64 individual square images from the board"""
        height, width = board_image.shape[:2]
        
        # Use float division for precise positioning
        square_height = height / 8.0
        square_width = width / 8.0
        
        squares = []
        
        # Small margin to avoid board lines
        margin = 2
        
        for row in range(8):
            for col in range(8):
                # Calculate precise boundaries
                y1 = int(row * square_height) + margin
                y2 = int((row + 1) * square_height) - margin
                x1 = int(col * square_width) + margin
                x2 = int((col + 1) * square_width) - margin
                
                # Ensure boundaries are valid
                y1 = max(0, y1)
                x1 = max(0, x1)
                y2 = min(height, y2)
                x2 = min(width, x2)
                
                if y2 > y1 and x2 > x1:
                    square = board_image[y1:y2, x1:x2]
                    # Resize to uniform size to eliminate size variations
                    square = cv2.resize(square, (64, 64))
                else:
                    # Create empty square if extraction failed
                    square = np.zeros((64, 64, 3), dtype=np.uint8)
                
                squares.append(square)
        
        return squares