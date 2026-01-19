import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import sys
import os
from pathlib import Path

from src.load.read_produce_dataset import load_yaml_classes

MODEL_DIR = Path(__file__).parents[2] / 'saved_models'

if not MODEL_DIR.exists():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 4. Define 6-Layer Network (same as before)
class DeepNet(nn.Module):
    def __init__(self, input_size, num_classes=63):
        super(DeepNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),            
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)


def save_model(model, epoch, optimizer=None, loss=None, filename='model.pth'):
    """Save model checkpoint"""
    save_path = os.path.join(MODEL_DIR, filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'loss': loss,
    }, save_path)
    print(f"Model saved to {save_path}")

##
def train_model(force=False):
    print(f'Starting the training process, force={force}')
    # check if the model already exists
    if not force and os.path.exists(os.path.join(MODEL_DIR, 'produce_classifier_iter_10000.pth')):
        print("Model already exists, skipping training")
        return
        
    print('model does not exist, continuing...')

    # Add the src directory to the path so we can import from load
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
    from src.load.read_produce_dataset import process_dataset, scale_data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load and prepare the dataset
    print("Loading dataset...")

    project_root = os.getcwd()
    images_root = os.path.join(project_root, "assets/produce_dataset/LVIS_Fruits_And_Vegetables")
    yaml_path = os.path.join(project_root, "assets/produce_dataset/LVIS_Fruits_And_Vegetables/data.yaml")

    data = load_yaml_classes(yaml_path)

    train_images = os.path.join(images_root, data["train"])
    train_labels = train_images.replace("images", "labels")

    X, Y = process_dataset(train_images, train_labels, target_size=64)
    X = scale_data(X, method='minmax')  # Scale to [0, 1]
    X = X.T  # Transpose to (m, 12288) for PyTorch


    # 2. Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(Y, dtype=torch.float32)

    # the number of classes 
    num_classes = len(data["names"])

    print(f"Number of classes: {num_classes}")

    # 3. Create DataLoader for Mini-batching
    batch_size = 1000
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = DeepNet(X.shape[1], num_classes).to(device)

    # 5. Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 6. Training Loop
    max_iterations = 1024
    iterations = 0
    start_time = time.time()
    total_loss = 0.0
    model.train()

    for epoch in range(max_iterations // len(dataloader) + 1):
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            if iterations >= max_iterations:
                break
                
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            epoch_loss += loss.item()
            iterations += 1
            
            if iterations % 10 == 0:
                print(f"Iteration [{iterations}/{max_iterations}], Loss: {loss.item():.4f}")
                
            if iterations >= max_iterations:
                break
        # Calculate average loss for the epoch
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f'Epoch [{epoch+1}], Average Loss: {avg_epoch_loss:.4f}')
    
    save_model(
        model,
        iterations,
        optimizer,
        avg_epoch_loss,
        'produce_classifier_final.pth'
    )
    end_time = time.time()
    print(f"\nTraining Complete in: {(end_time - start_time) / 60:.2f} minutes")

if __name__ == "__main__":
    import argparse
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train the model with force option')
    parser.add_argument('--force', type=bool, default=False,
                      help='Force retraining even if model already exists')
    args = parser.parse_args()
    train_model(force=args.force)