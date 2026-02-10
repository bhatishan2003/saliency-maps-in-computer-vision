import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import kagglehub
import os
from pathlib import Path

# Import model from model.py
from model import PneumoniaCNN


# ================================
# DATASET CLASS
# ================================
class PneumoniaDataset(Dataset):
    """Custom Dataset for Chest X-Ray Pneumonia Classification"""
    
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
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        
        return image, label


# ================================
# DATA LOADING FUNCTION
# ================================
def load_dataset_paths(dataset_path):
    """Load image paths and labels from the dataset directory"""
    
    train_dir = os.path.join(dataset_path, 'chest_xray', 'train')
    test_dir = os.path.join(dataset_path, 'chest_xray', 'test')
    val_dir = os.path.join(dataset_path, 'chest_xray', 'val')
    
    def get_image_paths_and_labels(directory):
        image_paths = []
        labels = []
        
        # NORMAL = 0, PNEUMONIA = 1
        normal_dir = os.path.join(directory, 'NORMAL')
        pneumonia_dir = os.path.join(directory, 'PNEUMONIA')
        
        # Load normal images
        if os.path.exists(normal_dir):
            for img_file in os.listdir(normal_dir):
                if img_file.endswith(('.jpeg', '.jpg', '.png')):
                    image_paths.append(os.path.join(normal_dir, img_file))
                    labels.append(0)
        
        # Load pneumonia images
        if os.path.exists(pneumonia_dir):
            for img_file in os.listdir(pneumonia_dir):
                if img_file.endswith(('.jpeg', '.jpg', '.png')):
                    image_paths.append(os.path.join(pneumonia_dir, img_file))
                    labels.append(1)
        
        return image_paths, labels
    
    train_paths, train_labels = get_image_paths_and_labels(train_dir)
    test_paths, test_labels = get_image_paths_and_labels(test_dir)
    val_paths, val_labels = get_image_paths_and_labels(val_dir)
    
    # Combine train and val (val set is very small in this dataset)
    train_paths.extend(val_paths)
    train_labels.extend(val_labels)
    
    print(f"Training samples: {len(train_paths)}")
    print(f"Test samples: {len(test_paths)}")
    print(f"Normal (train): {train_labels.count(0)}, Pneumonia (train): {train_labels.count(1)}")
    print(f"Normal (test): {test_labels.count(0)}, Pneumonia (test): {test_labels.count(1)}")
    
    return train_paths, train_labels, test_paths, test_labels


# ================================
# TRANSFORMS
# ================================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])


# ================================
# TRAIN FUNCTION
# ================================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0, 0, 0

    pbar = tqdm(loader, desc="Training")

    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{100 * correct / total:.2f}%"
        )

    return running_loss / len(loader), 100 * correct / total


# ================================
# VALIDATE FUNCTION
# ================================
def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0, 0, 0
    
    # For calculating precision, recall, F1
    true_positives, false_positives, false_negatives = 0, 0, 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation")

        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Calculate TP, FP, FN for pneumonia class (label=1)
            true_positives += ((predicted == 1) & (labels == 1)).sum().item()
            false_positives += ((predicted == 1) & (labels == 0)).sum().item()
            false_negatives += ((predicted == 0) & (labels == 1)).sum().item()

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100 * correct / total:.2f}%"
            )
    
    # Calculate metrics
    accuracy = 100 * correct / total
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return running_loss / len(loader), accuracy, precision, recall, f1


# ================================
# MAIN
# ================================
if __name__ == "__main__":
    
    # Download latest version
    print("Downloading dataset...")
    path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    print("Path to dataset files:", path)

    # ================================
    # REPRODUCIBILITY
    # ================================
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ================================
    # DEVICE
    # ================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\n" + "="*50)
    print("LOADING DATASET")
    print("="*50)
    
    # Load dataset paths and labels
    train_paths, train_labels, test_paths, test_labels = load_dataset_paths(path)
    
    # Create datasets
    train_dataset = PneumoniaDataset(train_paths, train_labels, train_transform)
    test_dataset = PneumoniaDataset(test_paths, test_labels, test_transform)
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"\nBatch size: {batch_size}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    print("\n" + "="*50)
    print("BUILDING MODEL")
    print("="*50)
    
    # Model
    num_classes = 2  # NORMAL, PNEUMONIA
    model = PneumoniaCNN(num_classes).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer (weighted loss for class imbalance)
    class_counts = [train_labels.count(0), train_labels.count(1)]
    class_weights = torch.tensor([1.0/c for c in class_counts], dtype=torch.float32)
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

    print("\n" + "="*50)
    print("STARTING TRAINING")
    print("="*50)
    
    # Training loop
    num_epochs = 25
    best_val_acc = 0
    best_f1 = 0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    val_f1_scores = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc, val_precision, val_recall, val_f1 = validate(
            model, test_loader, criterion, device
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        val_f1_scores.append(val_f1)

        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")
        print(f"Val Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | F1: {val_f1:.4f}")

        # Save best model based on F1 score (better for imbalanced datasets)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
            }, "best_pneumonia_model.pth")
            print("✓ Best model saved!")

    # ======================
    # PLOTS
    # ======================
    print("\n" + "="*50)
    print("GENERATING PLOTS")
    print("="*50)
    
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label="Train", marker='o')
    plt.plot(val_losses, label="Validation", marker='s')
    plt.title("Loss over Epochs", fontsize=14, fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label="Train", marker='o')
    plt.plot(val_accs, label="Validation", marker='s')
    plt.title("Accuracy over Epochs", fontsize=14, fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(val_f1_scores, label="Validation F1", marker='d', color='green')
    plt.title("F1 Score over Epochs", fontsize=14, fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=300)
    print("Training plots saved as 'training_history.png'")

    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Best F1 Score: {best_f1:.4f}")
    print(f"Model saved as: best_pneumonia_model.pth")