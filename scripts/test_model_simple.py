#!/usr/bin/env python3
"""
Simple test of the trained model
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import cv2

# Define the same CNN architecture
class ChessPieceCNN(nn.Module):
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

def test_model():
    """Test the model on some sample images"""
    
    # Load model
    checkpoint = torch.load('models/chess_piece_model.pth', map_location='cpu', weights_only=False)
    model = ChessPieceCNN(num_classes=checkpoint['num_classes'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    label_encoder = checkpoint['label_encoder']
    print(f"Model classes: {label_encoder.classes_}")
    print(f"Best validation accuracy: {checkpoint['val_accuracy']:.2f}%")
    
    # Transform for test images
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Test on one image from each class
    base_dir = Path("training_data_clean/squares")
    
    # Create visualization
    fig, axes = plt.subplots(2, 7, figsize=(14, 5))
    axes = axes.flatten()
    
    # Define proper mapping
    dir_to_label = {
        'empty': 'empty',
        'P': 'P', 'N': 'N', 'B': 'B', 'R': 'R', 'Q': 'Q', 'K': 'K',
        'p_black': 'p', 'n_black': 'n', 'b_black': 'b', 
        'r_black': 'r', 'q_black': 'q', 'k_black': 'k'
    }
    
    test_dirs = ['P', 'N', 'B', 'R', 'Q', 'K', 'empty', 'p_black', 'n_black', 'b_black', 'r_black', 'q_black', 'k_black']
    
    for i, dir_name in enumerate(test_dirs[:14]):
        piece_dir = base_dir / dir_name
        if not piece_dir.exists():
            continue
            
        # Get first image from directory
        images = list(piece_dir.glob("*.png"))
        if not images:
            continue
            
        img_path = images[0]
        true_label = dir_to_label.get(dir_name, dir_name)
        
        # Load and display image
        img_display = cv2.imread(str(img_path))
        img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
        
        # Predict
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            pred_idx = output.argmax(dim=1).item()
            pred_label = label_encoder.inverse_transform([pred_idx])[0]
            confidence = probs[0, pred_idx].item()
        
        # Display
        axes[i].imshow(img_display)
        axes[i].set_title(f"True: {true_label}")
        axes[i].text(0.5, -0.1, f"Predicted", ha='center', transform=axes[i].transAxes)
        
        # Color code based on correctness
        color = 'green' if pred_label == true_label else 'red'
        axes[i].text(0.5, -0.25, f"{pred_label}\n{confidence:.2f}", 
                     ha='center', transform=axes[i].transAxes, 
                     fontsize=12, fontweight='bold', color=color)
        axes[i].axis('off')
    
    # Hide unused axes
    for i in range(len(test_dirs), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle("Model Predictions on Sample Squares", fontsize=16)
    plt.tight_layout()
    plt.savefig('prediction_visualization.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to prediction_visualization.png")
    
    # Print confusion statistics
    print("\nTesting on more samples...")
    all_true = []
    all_pred = []
    
    for dir_name in test_dirs:
        piece_dir = base_dir / dir_name
        if not piece_dir.exists():
            continue
            
        true_label = dir_to_label.get(dir_name, dir_name)
        images = list(piece_dir.glob("*.png"))[:10]  # Test 10 from each
        
        for img_path in images:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0)
            
            with torch.no_grad():
                output = model(img_tensor)
                pred_idx = output.argmax(dim=1).item()
                pred_label = label_encoder.inverse_transform([pred_idx])[0]
            
            all_true.append(true_label)
            all_pred.append(pred_label)
    
    # Calculate accuracy
    correct = sum(1 for t, p in zip(all_true, all_pred) if t == p)
    accuracy = 100 * correct / len(all_true)
    print(f"\nTest accuracy: {accuracy:.2f}% ({correct}/{len(all_true)})")
    
    # Print per-class accuracy
    from collections import defaultdict
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    for true, pred in zip(all_true, all_pred):
        class_total[true] += 1
        if true == pred:
            class_correct[true] += 1
    
    print("\nPer-class accuracy:")
    for piece in ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k', 'empty']:
        if piece in class_total:
            acc = 100 * class_correct[piece] / class_total[piece]
            print(f"  {piece}: {acc:.1f}% ({class_correct[piece]}/{class_total[piece]})")

if __name__ == "__main__":
    test_model()