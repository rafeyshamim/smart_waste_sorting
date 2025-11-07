"""
Training script for waste classification model using TrashNet dataset
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import platform
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json

# Class labels
CLASSES = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']
NUM_CLASSES = len(CLASSES)

class WasteDataset(Dataset):
    """Dataset class for waste images"""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def load_dataset(data_dir='data/trashnet/data/dataset-resized'):
    """Load all images and labels from dataset"""
    image_paths = []
    labels = []
    
    for class_idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            continue
        
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(class_dir, filename))
                labels.append(class_idx)
    
    return image_paths, labels

def create_model(num_classes=NUM_CLASSES, pretrained=True):
    """Create a ResNet18 model for waste classification"""
    try:
        # Try new torchvision API (0.13+)
        if pretrained:
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            model = models.resnet18(weights=None)
    except AttributeError:
        # Fallback to old API
        if pretrained:
            model = models.resnet18(pretrained=True)
        else:
            model = models.resnet18(pretrained=False)
    # Replace the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model

def train_model(epochs=10, batch_size=32, learning_rate=0.001):
    """Train the waste classification model"""
    # Check for GPU availability
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        print(f'✓ GPU detected: {gpu_name}')
        print(f'✓ CUDA version: {torch.version.cuda}')
        print(f'✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
        # Increase batch size for GPU (can handle more)
        if batch_size < 64:
            print(f'  Note: Consider using batch_size=64 or higher for better GPU utilization')
    else:
        device = torch.device('cpu')
        print('⚠ No GPU detected. Training on CPU (will be slower).')
        print('  To use GPU, install CUDA-enabled PyTorch:')
        print('  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121')
    
    print(f'Using device: {device}')
    
    # Load dataset
    print('Loading dataset...')
    image_paths, labels = load_dataset()
    print(f'Total images: {len(image_paths)}')
    
    # Split dataset
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, labels, test_size=0.3, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f'Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}')
    
    # Data transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets and dataloaders
    train_dataset = WasteDataset(X_train, y_train, transform=train_transform)
    val_dataset = WasteDataset(X_val, y_val, transform=val_transform)
    test_dataset = WasteDataset(X_test, y_test, transform=val_transform)
    
    # Set num_workers=0 on Windows to avoid multiprocessing issues
    # On Linux/Mac, you can use num_workers=2 or 4 for faster loading
    num_workers = 0 if platform.system() == 'Windows' else 2
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Create model
    model = create_model(num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Training loop
    best_val_acc = 0.0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_bar.set_postfix({'acc': f'{100*correct/total:.2f}%'})
        
        val_acc = 100 * correct / total
        val_accuracies.append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'models/waste_classifier.pth')
            print(f'New best model saved! Val Acc: {val_acc:.2f}%')
        
        scheduler.step()
    
    # Test phase
    print('\nEvaluating on test set...')
    model.load_state_dict(torch.load('models/waste_classifier.pth'))
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_acc = 100 * correct / total
    print(f'Test Accuracy: {test_acc:.2f}%')
    
    # Save class mapping
    os.makedirs('models', exist_ok=True)
    with open('models/class_mapping.json', 'w') as f:
        json.dump(CLASSES, f)
    
    print('\nTraining completed!')
    return model

if __name__ == '__main__':
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Train the model
    # Note: Reduce epochs for faster training, increase for better accuracy
    # For quick testing: epochs=3-5, For good results: epochs=10-15
    # Learning rate: 0.001 is standard. Can increase to 0.003-0.005 for faster convergence
    # but may be less stable. Batch size can be increased if you have more RAM/GPU memory.
    
    # Auto-detect optimal batch size based on device
    if torch.cuda.is_available():
        # GPU can handle larger batches
        batch_size = 64
        print("GPU detected - using batch_size=64 for better GPU utilization")
    else:
        # CPU works better with smaller batches
        batch_size = 32
        print("CPU detected - using batch_size=32")
    
    model = train_model(epochs=25, batch_size=batch_size, learning_rate=0.001)
    print('Model training completed!')

