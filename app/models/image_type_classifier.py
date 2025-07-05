import cv2
import numpy as np
from typing import Tuple, Dict
from enum import Enum


class ImageType(Enum):
    DIGITAL_CLEAN = "digital_clean"      # Clean computer-generated boards
    DIGITAL_SCREENSHOT = "screenshot"     # Screenshots with UI elements
    PHOTO_OVERHEAD = "photo_overhead"     # Real board from above
    PHOTO_ANGLE = "photo_angle"          # Real board at an angle
    UNKNOWN = "unknown"


class ImageTypeClassifier:
    """
    Classify chess board images into different types to route to appropriate processors
    """
    
    def classify_image(self, image: np.ndarray) -> Tuple[ImageType, float, Dict]:
        """
        Classify the type of chess board image
        Returns: (image_type, confidence, features)
        """
        features = self._extract_features(image)
        
        # Decision tree based on features
        # Digital images have very high edge sharpness
        if features['edge_sharpness'] > 2000:
            if features['has_ui_elements']:
                return ImageType.DIGITAL_SCREENSHOT, 0.85, features
            else:
                return ImageType.DIGITAL_CLEAN, 0.95, features
        
        # Screenshots have medium sharpness and UI elements
        elif features['edge_sharpness'] > 500 and features['has_ui_elements']:
            return ImageType.DIGITAL_SCREENSHOT, 0.9, features
        
        # Check for uniform digital squares
        elif features['uniform_squares'] and features['has_perfect_grid']:
            return ImageType.DIGITAL_CLEAN, 0.85, features
        
        # Real photos
        elif features['has_wood_texture'] or features['has_3d_pieces']:
            if features['perspective_distortion'] > 0.3:
                return ImageType.PHOTO_ANGLE, 0.85, features
            else:
                return ImageType.PHOTO_OVERHEAD, 0.85, features
        
        else:
            # Default to photo if unclear
            return ImageType.PHOTO_OVERHEAD, 0.5, features
    
    def _extract_features(self, image: np.ndarray) -> Dict:
        """Extract features to determine image type"""
        features = {}
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. Check for perfect grid lines
        features['has_perfect_grid'] = self._detect_perfect_grid(gray)
        
        # 2. Check for uniform square colors
        features['uniform_squares'] = self._check_uniform_squares(image)
        
        # 3. Check for UI elements (text, buttons)
        features['has_ui_elements'] = self._detect_ui_elements(image)
        
        # 4. Check for wood texture
        features['has_wood_texture'] = self._detect_wood_texture(image)
        
        # 5. Check for 3D piece shadows
        features['has_3d_pieces'] = self._detect_3d_pieces(gray)
        
        # 6. Measure perspective distortion
        features['perspective_distortion'] = self._measure_perspective(gray)
        
        # 7. Color statistics
        features['color_variance'] = np.std(image)
        features['edge_sharpness'] = self._measure_edge_sharpness(gray)
        
        return features
    
    def _detect_perfect_grid(self, gray: np.ndarray) -> bool:
        """Detect if image has perfect digital grid lines"""
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 
                               minLineLength=100, maxLineGap=10)
        
        if lines is None:
            return False
        
        # Check for perfectly horizontal/vertical lines
        perfect_lines = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Check if perfectly horizontal or vertical
            if x1 == x2 or y1 == y2:
                perfect_lines += 1
        
        # Digital boards tend to have many perfect lines
        return perfect_lines > 10
    
    def _check_uniform_squares(self, image: np.ndarray) -> bool:
        """Check if squares have uniform digital colors"""
        h, w = image.shape[:2]
        
        # Sample a few regions
        samples = []
        step = min(h, w) // 10
        
        for i in range(2, 8, 2):
            for j in range(2, 8, 2):
                y, x = i * step, j * step
                region = image[y:y+step//2, x:x+step//2]
                
                # Calculate color variance within region
                variance = np.std(region)
                samples.append(variance)
        
        # Digital boards have very uniform colors
        avg_variance = np.mean(samples)
        return avg_variance < 20
    
    def _detect_ui_elements(self, image: np.ndarray) -> bool:
        """Detect UI elements like text, buttons, menus"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Look for text-like regions using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        morph = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        
        # Threshold to find potential text regions
        _, thresh = cv2.threshold(morph, 30, 255, cv2.THRESH_BINARY)
        
        # Count connected components
        num_labels, _ = cv2.connectedComponents(thresh)
        
        # Many small components suggest text/UI
        return num_labels > 100
    
    def _detect_wood_texture(self, image: np.ndarray) -> bool:
        """Detect wood grain texture patterns"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gabor filters to detect texture
        ksize = 21
        theta = np.pi / 4
        lamda = 10.0
        gamma = 0.5
        phi = 0
        
        kernel = cv2.getGaborKernel((ksize, ksize), 4.0, theta, lamda, gamma, phi)
        filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
        
        # Wood texture creates specific patterns
        texture_variance = np.var(filtered)
        
        # Also check color - wood tends to be brown/tan
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        brown_mask = cv2.inRange(hsv, (10, 50, 50), (25, 255, 200))
        brown_ratio = np.sum(brown_mask > 0) / brown_mask.size
        
        return texture_variance > 500 and brown_ratio > 0.3
    
    def _detect_3d_pieces(self, gray: np.ndarray) -> bool:
        """Detect 3D pieces by looking for shadows and gradients"""
        # Calculate gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(grad_x**2 + grad_y**2)
        
        # 3D pieces create complex gradient patterns
        gradient_complexity = np.std(gradient)
        
        # Also look for shadows (darker regions adjacent to pieces)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        shadows = dilated - binary
        shadow_ratio = np.sum(shadows > 0) / shadows.size
        
        return gradient_complexity > 30 and shadow_ratio > 0.05
    
    def _measure_perspective(self, gray: np.ndarray) -> float:
        """Measure perspective distortion"""
        # Find corners/edges
        corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
        
        if corners is None or len(corners) < 4:
            return 0.0
        
        # Simplified: check if corners form a non-rectangular shape
        corners = corners.reshape(-1, 2)
        
        # Get bounding box
        x_min, y_min = corners.min(axis=0)
        x_max, y_max = corners.max(axis=0)
        
        # Calculate aspect ratio
        aspect = (x_max - x_min) / (y_max - y_min + 1e-6)
        
        # Perfect square has aspect ratio 1.0
        distortion = abs(1.0 - aspect)
        
        return min(distortion, 1.0)
    
    def _measure_edge_sharpness(self, gray: np.ndarray) -> float:
        """Measure edge sharpness - digital images have sharper edges"""
        # Laplacian variance is a good measure of sharpness
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        return sharpness
    
    def get_preprocessing_params(self, image_type: ImageType) -> Dict:
        """Get recommended preprocessing parameters for each image type"""
        params = {
            ImageType.DIGITAL_CLEAN: {
                'enhance_contrast': False,
                'denoise': False,
                'sharpen': False,
                'perspective_correction': 'minimal',
                'edge_margin': 0.02  # 2% margin
            },
            ImageType.DIGITAL_SCREENSHOT: {
                'enhance_contrast': False,
                'denoise': False,
                'sharpen': False,
                'perspective_correction': 'minimal',
                'edge_margin': 0.05,  # 5% margin to avoid UI
                'crop_ui': True
            },
            ImageType.PHOTO_OVERHEAD: {
                'enhance_contrast': True,
                'denoise': True,
                'sharpen': True,
                'perspective_correction': 'moderate',
                'edge_margin': 0.08  # 8% margin for board edges
            },
            ImageType.PHOTO_ANGLE: {
                'enhance_contrast': True,
                'denoise': True,
                'sharpen': True,
                'perspective_correction': 'aggressive',
                'edge_margin': 0.10  # 10% margin
            },
            ImageType.UNKNOWN: {
                'enhance_contrast': True,
                'denoise': True,
                'sharpen': False,
                'perspective_correction': 'moderate',
                'edge_margin': 0.05
            }
        }
        
        return params.get(image_type, params[ImageType.UNKNOWN])