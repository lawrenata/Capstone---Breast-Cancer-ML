import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet
import os

class BreastCancerClassifier:
    def __init__(self, model_name='efficientnet-b0', num_classes=3):
        """
        Initialize the breast cancer classifier
        model_name: which EfficientNet variant to use (b0-b7)
        num_classes: 3 classes (no_lesion, benign, malignant)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load pretrained EfficientNet
        self.model = EfficientNet.from_pretrained(model_name, num_classes=num_classes)
        self.model = self.model.to(self.device)
        
        # Define loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
    def get_data_transforms(self):
        """Define image preprocessing"""
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        return train_transform, val_transform
    
    def load_data(self, data_dir, batch_size=32):
        """Load training and validation data"""
        train_transform, val_transform = self.get_data_transforms()
        
        train_dataset = datasets.ImageFolder(
            os.path.join(data_dir, 'train'),
            transform=train_transform
        )
        
        val_dataset = datasets.ImageFolder(
            os.path.join(data_dir, 'val'),
            transform=val_transform
        )
        
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Classes: {train_dataset.classes}")
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        return val_loss, val_acc
    
    def train(self, num_epochs=10):
        """Train the model for multiple epochs"""
        print("\nStarting training...")
        best_acc = 0.0
        
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                self.save_model('best_model.pth')
                print(f"  Saved new best model (acc: {best_acc:.2f}%)")
        
        print(f"\nTraining complete! Best validation accuracy: {best_acc:.2f}%")
    
    def save_model(self, filepath):
        """Save model weights"""
        torch.save(self.model.state_dict(), filepath)
    
    def load_model(self, filepath):
        """Load model weights"""
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f"Model loaded from {filepath}")
    
    def predict(self, image_path):
        """Predict on a single image"""
        from PIL import Image
        
        _, val_transform = self.get_data_transforms()
        image = Image.open(image_path).convert('RGB')
        image = val_transform(image).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(image)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = output.argmax(1).item()
            confidence = probabilities[0][predicted_class].item()
        
        class_names = ['benign', 'malignant', 'no_lesion']  # Alphabetical order
        return class_names[predicted_class], confidence


# Example usage
if __name__ == "__main__":
    # Initialize classifier
    classifier = BreastCancerClassifier(model_name='efficientnet-b0', num_classes=3)
    
    # Load your data
    data_dir = '/arc/project/st-ilker-1/undergrad-capstone-2025/SBME-Capstone-2025-Breast-Lesion-MRI-Analysis/EfficientNet/processed_data'
    classifier.load_data(data_dir, batch_size=8)  # Smaller batch for limited data
    
    # Train the model
    classifier.train(num_epochs=20)
    
    # Predict on a new image (example)
    # result, confidence = classifier.predict('path/to/test/image.png')
    # print(f"Prediction: {result}, Confidence: {confidence:.2%}")