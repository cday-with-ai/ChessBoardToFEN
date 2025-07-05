import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import json
from app.core.config import settings
from app.core.exceptions import ModelNotFoundError


class ChessPieceCNN(nn.Module):
    """Same CNN architecture as used in training"""
    
    def __init__(self, num_classes=13):
        super(ChessPieceCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Activation
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Conv block 1
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        
        # Conv block 2
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        
        # Conv block 3
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        
        # Flatten
        x = x.view(-1, 128 * 8 * 8)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class PieceClassifier:
    # Piece labels in order
    PIECES = ['empty', 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or Path('models/chess_piece_model.pth')
        self.model = None
        self.label_encoder = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.use_template_matching = True  # Start with template matching
        
        # Define transform for preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Try to load ML model if available
        if self.model_path.exists():
            try:
                checkpoint = torch.load(str(self.model_path), map_location=self.device, weights_only=False)
                self.model = ChessPieceCNN(num_classes=checkpoint['num_classes']).to(self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                self.label_encoder = checkpoint['label_encoder']
                self.use_template_matching = False
                print(f"Loaded PyTorch model with {checkpoint['val_accuracy']:.2f}% accuracy")
            except Exception as e:
                print(f"Could not load model: {e}. Falling back to template matching.")
    
    def classify_square(self, square_image: np.ndarray) -> Tuple[str, float]:
        """
        Classify a single square image
        Returns: (piece_symbol, confidence)
        """
        if self.model and not self.use_template_matching:
            return self._classify_with_model(square_image)
        else:
            return self._classify_with_templates(square_image)
    
    def _classify_with_model(self, square_image: np.ndarray) -> Tuple[str, float]:
        """Use trained neural network for classification"""
        # Convert numpy array to PIL Image
        if square_image.dtype != np.uint8:
            square_image = (square_image * 255).astype(np.uint8)
        
        # Handle both RGB and BGR
        if len(square_image.shape) == 3 and square_image.shape[2] == 3:
            # Assume BGR from OpenCV, convert to RGB
            square_image = cv2.cvtColor(square_image, cv2.COLOR_BGR2RGB)
        
        img = Image.fromarray(square_image)
        
        # Preprocess
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            output = self.model(img_tensor)
            probs = torch.softmax(output, dim=1)
            pred_idx = output.argmax(dim=1).item()
            confidence = probs[0, pred_idx].item()
        
        # Get piece label
        piece = self.label_encoder.inverse_transform([pred_idx])[0]
        return piece, confidence
    
    def _classify_with_templates(self, square_image: np.ndarray) -> Tuple[str, float]:
        """
        Simple classification based on color analysis
        This is a placeholder for template matching or basic heuristics
        """
        # Convert to grayscale
        gray = cv2.cvtColor(square_image, cv2.COLOR_BGR2GRAY) if len(square_image.shape) == 3 else square_image
        
        # Calculate average intensity
        mean_intensity = np.mean(gray)
        
        # Simple heuristic: empty squares tend to have uniform color
        std_intensity = np.std(gray)
        
        # Check if square is likely empty
        if std_intensity < 20:  # Low variation suggests empty square
            return 'empty', 0.8
        
        # Detect if piece is white or black based on intensity
        # This is very basic - in practice we'd use better features
        if mean_intensity > 200:  # Likely white piece
            # Guess based on size/shape (very rough)
            return 'P', 0.3  # Low confidence
        elif mean_intensity < 100:  # Likely black piece
            return 'p', 0.3  # Low confidence
        else:
            return 'empty', 0.5
    
    def classify_board(self, squares: List[np.ndarray]) -> List[Tuple[str, float]]:
        """
        Classify all squares on the board
        Returns list of (piece, confidence) tuples
        """
        results = []
        for square in squares:
            piece, confidence = self.classify_square(square)
            results.append((piece, confidence))
        return results


class SimpleColorClassifier(PieceClassifier):
    """
    A simple classifier based on color detection
    This can work reasonably well for digital boards with clear colors
    """
    
    def __init__(self):
        super().__init__()
        # Force template matching for simple classifier
        self.use_template_matching = True
        self.model = None
    
    def _classify_with_templates(self, square_image: np.ndarray) -> Tuple[str, float]:
        """
        Improved classification using color analysis
        """
        # Handle both RGB and BGR
        if len(square_image.shape) == 3:
            # Convert to RGB if needed
            if square_image.shape[2] == 3:
                rgb = cv2.cvtColor(square_image, cv2.COLOR_BGR2RGB)
            else:
                rgb = square_image
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        else:
            gray = square_image
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
        # Get center region (pieces are usually in center)
        h, w = gray.shape[:2]
        center_gray = gray[h//4:3*h//4, w//4:3*w//4]
        
        # Calculate features
        mean_intensity = np.mean(center_gray)
        std_intensity = np.std(center_gray)
        
        # Edge detection to find piece outlines
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Empty square detection
        if std_intensity < 15 and edge_density < 0.05:
            return 'empty', 0.9
        
        # Detect piece color
        if mean_intensity > 180:  # White piece
            # Try to distinguish piece types based on shape
            # This is still basic but better than random
            if edge_density > 0.15:
                return 'Q', 0.4  # Complex pieces have more edges
            elif edge_density > 0.10:
                return 'N', 0.4
            else:
                return 'P', 0.5
        elif mean_intensity < 80:  # Black piece
            if edge_density > 0.15:
                return 'q', 0.4
            elif edge_density > 0.10:
                return 'n', 0.4
            else:
                return 'p', 0.5
        else:
            # Uncertain - could be shadow or partial piece
            return 'empty', 0.3