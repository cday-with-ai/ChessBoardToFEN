#!/usr/bin/env python3
"""
Train a CNN model for chess piece classification
Uses PyTorch for better Python 3.13 compatibility
"""

import json
import numpy as np
from pathlib import Path
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns


class ChessPieceDataset(Dataset):
    """PyTorch Dataset for chess pieces"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label


class ChessPieceCNN(nn.Module):
    """Simple CNN for chess piece classification"""
    
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


def load_training_data():
    """Load all training images and labels"""
    print("Loading training data...")
    
    base_dir = Path("training_data_clean/squares")
    # Map directory names to piece labels (handle case-insensitive filesystem)
    dir_to_label = {
        'empty': 'empty',
        'P': 'P', 'N': 'N', 'B': 'B', 'R': 'R', 'Q': 'Q', 'K': 'K',
        'p_black': 'p', 'n_black': 'n', 'b_black': 'b', 
        'r_black': 'r', 'q_black': 'q', 'k_black': 'k'
    }
    
    images = []
    labels = []
    
    for dir_name, piece_label in dir_to_label.items():
        piece_dir = base_dir / dir_name
        if piece_dir.exists():
            for img_path in piece_dir.glob("*.png"):
                images.append(str(img_path))
                labels.append(piece_label)
    
    print(f"Loaded {len(images)} images")
    
    # Print distribution
    from collections import Counter
    label_counts = Counter(labels)
    print("\nLabel distribution:")
    pieces_order = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k', 'empty']
    for piece in pieces_order:
        if piece in label_counts:
            print(f"  {piece}: {label_counts[piece]}")
    
    return images, labels


def train_model(images, labels, epochs=30, batch_size=32):
    """Train the CNN model"""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        images, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
    )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Define transforms
    transform_train = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = ChessPieceDataset(X_train, y_train, transform=transform_train)
    val_dataset = ChessPieceDataset(X_val, y_val, transform=transform_val)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = ChessPieceCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training history
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    print("\nStarting training...")
    best_val_acc = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100. * correct / len(val_dataset)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save({
                'model_state_dict': model.state_dict(),
                'label_encoder': label_encoder,
                'num_classes': num_classes,
                'val_accuracy': val_accuracy
            }, 'models/chess_piece_model.pth')
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val Acc: {val_accuracy:.2f}%")
    
    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training History - Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("\nTraining history saved to training_history.png")
    
    return model, label_encoder, device


def evaluate_model(model, label_encoder, device):
    """Evaluate the model on test data"""
    print("\n=== Model Evaluation ===")
    
    # Load some test images
    base_dir = Path("training_data_clean/squares")
    test_images = []
    test_labels = []
    
    # Get a few examples from each class
    for piece in label_encoder.classes_:
        piece_dir = base_dir / piece
        if piece_dir.exists():
            images = list(piece_dir.glob("*.png"))[:5]  # Get 5 examples
            for img in images:
                test_images.append(str(img))
                test_labels.append(piece)
    
    # Transform for evaluation
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Make predictions
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for img_path in test_images:
            image = Image.open(img_path).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)
            output = model(image)
            pred = output.argmax(dim=1).item()
            predictions.append(label_encoder.inverse_transform([pred])[0])
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(test_labels, predictions))
    
    # Save model info
    model_info = {
        'classes': label_encoder.classes_.tolist(),
        'num_classes': len(label_encoder.classes_),
        'input_shape': (64, 64, 3),
        'architecture': 'Simple CNN with 3 conv layers'
    }
    
    with open('models/model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("\nModel saved to models/chess_piece_model.pth")
    print("Model info saved to models/model_info.json")


def main():
    # Create models directory
    Path("models").mkdir(exist_ok=True)
    
    # Check if PyTorch is available
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
    except ImportError:
        print("PyTorch not installed. Installing...")
        import subprocess
        subprocess.check_call([".venv/bin/pip", "install", "torch", "torchvision", "scikit-learn", "matplotlib", "seaborn"])
        print("Please run the script again after installation.")
        return
    
    # Load data
    images, labels = load_training_data()
    
    if len(images) < 100:
        print(f"\nWarning: Only {len(images)} images found. Consider adding more data for better results.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Train model
    model, label_encoder, device = train_model(images, labels)
    
    # Evaluate model
    evaluate_model(model, label_encoder, device)
    
    print("\n✅ Training complete!")
    print("\nNext steps:")
    print("1. Test the model with: python test_model.py")
    print("2. Use in API by restarting the server")
    print("3. Keep adding more training data for better accuracy")


if __name__ == "__main__":
    main()