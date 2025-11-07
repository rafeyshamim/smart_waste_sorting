"""
Prediction module for waste classification
"""
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import os

CLASSES = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']
NUM_CLASSES = len(CLASSES)

class WasteClassifier:
    """Waste classification model wrapper"""
    
    def __init__(self, model_path='models/waste_classifier.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.classes = CLASSES
    
    def _load_model(self, model_path):
        """Load the trained model"""
        # Create model architecture
        model = models.resnet18(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, NUM_CLASSES)
        
        # Load weights if model exists
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f'Model loaded from {model_path}')
        else:
            print(f'Warning: Model not found at {model_path}. Using untrained model.')
        
        model = model.to(self.device)
        model.eval()
        return model
    
    def predict(self, image_path):
        """
        Predict waste class for an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            dict: Prediction results with class name and confidence
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted = torch.max(probabilities, 0)
                predicted_class_idx = predicted.item()
                confidence_score = confidence.item()
            
            # Get class name
            predicted_class = self.classes[predicted_class_idx]
            
            # Get all class probabilities
            all_probabilities = probabilities.cpu().numpy()
            class_probabilities = {
                self.classes[i]: float(all_probabilities[i])
                for i in range(len(self.classes))
            }
            
            return {
                'class': predicted_class,
                'confidence': confidence_score,
                'all_probabilities': class_probabilities
            }
        
        except Exception as e:
            return {
                'error': str(e),
                'class': None,
                'confidence': 0.0
            }
    
    def predict_from_bytes(self, image_bytes):
        """
        Predict waste class from image bytes
        
        Args:
            image_bytes: Image file bytes
            
        Returns:
            dict: Prediction results with class name and confidence
        """
        try:
            # Load image from bytes
            from io import BytesIO
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted = torch.max(probabilities, 0)
                predicted_class_idx = predicted.item()
                confidence_score = confidence.item()
            
            # Get class name
            predicted_class = self.classes[predicted_class_idx]
            
            # Get all class probabilities
            all_probabilities = probabilities.cpu().numpy()
            class_probabilities = {
                self.classes[i]: float(all_probabilities[i])
                for i in range(len(self.classes))
            }
            
            return {
                'class': predicted_class,
                'confidence': confidence_score,
                'all_probabilities': class_probabilities
            }
        
        except Exception as e:
            return {
                'error': str(e),
                'class': None,
                'confidence': 0.0
            }

# Global classifier instance
classifier = None

def get_classifier():
    """Get or create classifier instance"""
    global classifier
    if classifier is None:
        classifier = WasteClassifier()
    return classifier

