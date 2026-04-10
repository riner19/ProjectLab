import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from model import BiLSTMStrikeClassifier

# --- Configuration & Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = BASE_DIR if "HAR" in os.path.basename(BASE_DIR) else os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "datasets", "Skeleton_data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

X_PATH = os.path.join(DATA_DIR, "X_data.npy")
y_PATH = os.path.join(DATA_DIR, "y_data.npy")

# --- Hyperparameters ---
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
INPUT_DIM = 59  # UPDATED: 51 raw coords + 8 physics features
HIDDEN_DIM = 64
NUM_LAYERS = 2
NUM_CLASSES = 6


# --- Dataset Wrapper ---
class StrikeSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# --- Main Training Pipeline ---
def main():
    print("Initializing Physics-Enhanced Ref-Brain Training Pipeline...")

    if not os.path.exists(X_PATH) or not os.path.exists(y_PATH):
        print(f"Error: Data not found at {DATA_DIR}. Run extraction first.")
        return

    # 1. Load Data
    X = np.load(X_PATH)
    y = np.load(y_PATH)

    print(f"Data loaded successfully. Total sequences: {len(X)}")
    print(f"Mathematical Shape of X: {X.shape}")

    # 2. Hardware/Data Verification (The Clean Slate Check)
    assert X.shape[
               2] == INPUT_DIM, f"CRITICAL ERROR: Expected {INPUT_DIM} dimensions, got {X.shape[2]}. The old data is still loaded!"

    # 3. Train/Validation Split (80/20)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {len(X_train)} | Validation set: {len(X_val)}")

    # 4. Compute Class Weights
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights = torch.tensor(weights, dtype=torch.float32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_weights = class_weights.to(device)
    print(f"Hardware target: {device}")

    # 5. Initialize DataLoaders
    train_dataset = StrikeSequenceDataset(X_train, y_train)
    val_dataset = StrikeSequenceDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 6. Initialize Model
    # Note: We pass INPUT_DIM=59 here, overriding the default in model.py
    model = BiLSTMStrikeClassifier(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0
    best_model_path = os.path.join(MODELS_DIR, "best_referee_bilstm.pth")

    print("\nStarting Training Phase...")

    # 7. Execution Loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * sequences.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * sequences.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # Calculate metrics
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = (train_correct / train_total) * 100
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = (val_correct / val_total) * 100

        print(f"Epoch [{epoch + 1:02d}/{EPOCHS}]  "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%  ||  "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

    print(f"\nTraining Complete. Peak Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model successfully saved to {best_model_path}")


if __name__ == "__main__":
    main()