"""
Model architecture and related components for Pneumonia Detection
"""

import torch
import torch.nn as nn


class PneumoniaCNN(nn.Module):
    """Simple CNN for Pneumonia Classification"""
    
    def __init__(self, num_classes=2):
        super(PneumoniaCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        
        # Pooling and activation
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        # After 5 pooling layers: 224 -> 112 -> 56 -> 28 -> 14 -> 7
        self.fc1 = nn.Linear(512 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Conv block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Conv block 5
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x


def load_model(model_path, device='cpu', num_classes=2):
    """
    Load a trained PneumoniaCNN model from checkpoint.
    
    Args:
        model_path: Path to the model checkpoint (.pth file)
        device: Device to load the model on ('cpu' or 'cuda')
        num_classes: Number of output classes (default: 2)
    
    Returns:
        model: Loaded model in evaluation mode
        checkpoint: Full checkpoint dictionary with metadata
    """
    # Initialize model
    model = PneumoniaCNN(num_classes=num_classes).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set to evaluation mode
    model.eval()
    
    return model, checkpoint


def get_model_info(checkpoint):
    """
    Extract and display model information from checkpoint.
    
    Args:
        checkpoint: Checkpoint dictionary from load_model
    
    Returns:
        dict: Dictionary containing model metadata
    """
    info = {
        'epoch': checkpoint.get('epoch', 'Unknown'),
        'val_acc': checkpoint.get('val_acc', 'Unknown'),
        'val_f1': checkpoint.get('val_f1', 'Unknown')
    }
    
    return info


if __name__ == "__main__":
    # Test the model
    print("Testing PneumoniaCNN model...")
    
    # Create model
    model = PneumoniaCNN(num_classes=2)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Architecture: PneumoniaCNN")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("\n✓ Model test successful!")
