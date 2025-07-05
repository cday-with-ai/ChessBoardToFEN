import cv2
import numpy as np
from typing import Dict, Tuple


class SmartMarginDetector:
    """
    Detect and remove margins containing rank/file labels or other non-board elements
    """
    
    def detect_board_margins(self, board_image: np.ndarray) -> Dict[str, int]:
        """
        Detect margins to crop from the board image
        Returns dict with 'top', 'bottom', 'left', 'right' pixel counts to crop
        """
        gray = cv2.cvtColor(board_image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Method 1: Detect text regions (likely labels)
        text_mask = self._detect_text_regions(gray)
        
        # Method 2: Detect solid color bands (UI elements)
        ui_margins = self._detect_ui_bands(gray)
        
        # Find board edges by looking for the actual playing surface
        margins = {
            'top': max(self._find_top_margin(gray, text_mask), ui_margins['top']),
            'bottom': max(self._find_bottom_margin(gray, text_mask), ui_margins['bottom']),
            'left': max(self._find_left_margin(gray, text_mask), ui_margins['left']),
            'right': max(self._find_right_margin(gray, text_mask), ui_margins['right'])
        }
        
        # Validate margins aren't too aggressive
        max_margin = min(h, w) * 0.10  # Don't crop more than 10% from any side
        for side in margins:
            margins[side] = min(margins[side], int(max_margin))
            
        # For clean digital boards, ensure margins are symmetric
        # If margins are very small and uneven, likely false positives
        if max(margins.values()) < 15:
            # Small margins detected, check if they're noise
            margin_variance = np.var([margins['top'], margins['bottom'], margins['left'], margins['right']])
            if margin_variance > 25:  # High variance in small margins = likely noise
                # Reset to zero
                margins = {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}
        
        return margins
    
    def _detect_text_regions(self, gray: np.ndarray) -> np.ndarray:
        """Detect regions likely to contain text (labels, coordinates)"""
        h, w = gray.shape
        text_mask = np.zeros_like(gray)
        
        # Method 1: Look for non-chess patterns in margins
        # Check top and bottom margins (15% of image)
        margin_size = int(h * 0.15)
        
        # Top margin analysis
        top_region = gray[:margin_size, :]
        top_variance = np.var(top_region)
        
        # Bottom margin analysis  
        bottom_region = gray[-margin_size:, :]
        bottom_variance = np.var(bottom_region)
        
        # High variance in margins often indicates text/UI
        if top_variance > 1000:
            text_mask[:margin_size, :] = 255
            
        if bottom_variance > 1000:
            text_mask[-margin_size:, :] = 255
        
        # Method 2: Edge density - text has many edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Check edge density in margins
        top_edges = edges[:margin_size, :]
        if np.sum(top_edges > 0) > (margin_size * w * 0.05):  # More than 5% edges
            text_mask[:margin_size, :] = 255
            
        bottom_edges = edges[-margin_size:, :]
        if np.sum(bottom_edges > 0) > (margin_size * w * 0.05):
            text_mask[-margin_size:, :] = 255
            
        # Check left/right margins too
        left_edges = edges[:, :margin_size]
        if np.sum(left_edges > 0) > (h * margin_size * 0.05):
            text_mask[:, :margin_size] = 255
            
        right_edges = edges[:, -margin_size:]
        if np.sum(right_edges > 0) > (h * margin_size * 0.05):
            text_mask[:, -margin_size:] = 255
        
        return text_mask
    
    def _detect_ui_bands(self, gray: np.ndarray) -> Dict[str, int]:
        """Detect solid color bands that indicate UI elements"""
        h, w = gray.shape
        margins = {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}
        
        # Check for solid horizontal bands (common in chess UIs)
        for y in range(h // 4):
            row = gray[y, :]
            # Low variance indicates solid color
            if np.var(row) < 100:  # Solid band
                margins['top'] = y + 1
            else:
                break  # Stop at first non-solid row
        
        # Check bottom
        for y in range(h - 1, 3 * h // 4, -1):
            row = gray[y, :]
            if np.var(row) < 100:
                margins['bottom'] = h - y
            else:
                break
                
        # Check left side
        for x in range(w // 8):
            col = gray[:, x]
            if np.var(col) < 100:
                margins['left'] = x + 1
            else:
                break
                
        # Check right side
        for x in range(w - 1, 7 * w // 8, -1):
            col = gray[:, x]
            if np.var(col) < 100:
                margins['right'] = w - x
            else:
                break
        
        return margins
    
    def _find_top_margin(self, gray: np.ndarray, text_mask: np.ndarray) -> int:
        """Find top margin by detecting where the board pattern starts"""
        h, w = gray.shape
        
        # Look for the first row that has chess board characteristics
        for y in range(h // 3):  # Check top third
            row = gray[y, :]
            
            # Skip if this row has text
            if np.sum(text_mask[y, :] > 0) > w * 0.5:
                continue
            
            # Check for board pattern
            if self._has_board_pattern(row):
                # Found board pattern, but check a bit more to be sure
                confirmed = 0
                for offset in range(1, min(10, h - y)):
                    if self._has_board_pattern(gray[y + offset, :]):
                        confirmed += 1
                
                if confirmed >= 3:  # At least 3 more rows with pattern
                    return max(0, y - 2)  # Small buffer
        
        # If text was detected in top region, be more aggressive
        if np.sum(text_mask[:h//4, :] > 0) > 0:
            # Find first row without text
            for y in range(h // 3):
                if np.sum(text_mask[y, :] > 0) == 0:
                    # Check next few rows also have no text
                    if y + 10 < h and np.sum(text_mask[y:y+10, :] > 0) == 0:
                        return y
        
        return 0
    
    def _find_bottom_margin(self, gray: np.ndarray, text_mask: np.ndarray) -> int:
        """Find bottom margin"""
        h, w = gray.shape
        
        for y in range(h - 1, 3 * h // 4, -1):  # Check bottom quarter
            row = gray[y, :]
            row_text = text_mask[y, :]
            
            # Skip rows with significant text
            if np.sum(row_text > 0) > w * 0.1:
                continue
            
            # Check for board pattern
            if self._has_board_pattern(row):
                return max(0, h - y - 5)  # Small buffer
        
        return 0
    
    def _find_left_margin(self, gray: np.ndarray, text_mask: np.ndarray) -> int:
        """Find left margin"""
        h, w = gray.shape
        
        for x in range(w // 4):  # Check left quarter
            col = gray[:, x]
            col_text = text_mask[:, x]
            
            # Skip columns with significant text
            if np.sum(col_text > 0) > h * 0.1:
                continue
            
            # Check for board pattern
            if self._has_board_pattern(col):
                return max(0, x - 5)  # Small buffer
        
        return 0
    
    def _find_right_margin(self, gray: np.ndarray, text_mask: np.ndarray) -> int:
        """Find right margin"""
        h, w = gray.shape
        
        for x in range(w - 1, 3 * w // 4, -1):  # Check right quarter
            col = gray[:, x]
            col_text = text_mask[:, x]
            
            # Skip columns with significant text
            if np.sum(col_text > 0) > h * 0.1:
                continue
            
            # Check for board pattern
            if self._has_board_pattern(col):
                return max(0, w - x - 5)  # Small buffer
        
        return 0
    
    def _has_board_pattern(self, line: np.ndarray) -> bool:
        """Check if a line (row or column) shows chess board pattern"""
        if len(line) < 8:
            return False
        
        # Look for alternating intensity pattern
        # Smooth the line first
        smoothed = cv2.GaussianBlur(line.reshape(1, -1), (1, 5), 0).flatten()
        
        # Find peaks and valleys
        gradient = np.diff(smoothed)
        sign_changes = np.diff(np.sign(gradient))
        
        # Count transitions (should be ~8 for a chess board)
        transitions = np.sum(np.abs(sign_changes) > 0)
        
        return 6 <= transitions <= 20  # Allow some variance
    
    def apply_smart_crop(self, board_image: np.ndarray, margins: Dict[str, int]) -> np.ndarray:
        """Apply the detected margins to crop the board"""
        h, w = board_image.shape[:2]
        
        y1 = margins['top']
        y2 = h - margins['bottom']
        x1 = margins['left']
        x2 = w - margins['right']
        
        # Ensure valid bounds
        y1, y2 = max(0, y1), min(h, y2)
        x1, x2 = max(0, x1), min(w, x2)
        
        # Ensure we still have a valid image
        if y2 <= y1 or x2 <= x1:
            return board_image  # Return original if margins are invalid
        
        return board_image[y1:y2, x1:x2]