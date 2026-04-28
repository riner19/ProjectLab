import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from model import BoxingLSTM
from sklearn.utils.class_weight import compute_class_weight

# --- Configuration ---
DATA_DIR = "datasets/Skeleton_data_1"
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.0005
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data():
    X = np.load(os.path.join(DATA_DIR, "X_data.npy"))
    y = np.load(os.path.join(DATA_DIR, "y_data.npy"))
    groups = np.load(os.path.join(DATA_DIR, "groups_data.npy"))
    return X, y, groups


def main():
    print(f"Using device: {DEVICE}")

    # 1. Load Data
    X, y, groups = load_data()
    print(f"Loaded Dataset Shape -> X: {X.shape}, y: {y.shape}")


    # 2. Manual Group Split (Testing on different camera angles)
    # V1, V4, V6 bring the rare punches. V9, V10 teach it the front-facing angle.
    train_video_names = ['V1', 'V2', 'V5', 'V4', 'V9', 'V6', 'V7', 'V8']

    # V9 and V10 contain a balanced mix of every punch to act as the perfect test
    val_video_names = ['V3', 'V10']

    # Create boolean masks
    train_mask = np.isin(groups, train_video_names)
    val_mask = np.isin(groups, val_video_names)

    # Apply the masks to X and y
    X_train, X_val = X[train_mask], X[val_mask]
    y_train, y_val = y[train_mask], y[val_mask]

    print(f"Training sequences: {len(X_train)} | Validation sequences: {len(X_val)}")


    # 3. Convert to PyTorch Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    # 4. Create DataLoaders
    train_data = TensorDataset(X_train_t, y_train_t)
    val_data = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    # 5. Initialize Model, Loss, and Optimizer
    model = BoxingLSTM(input_size=62, num_classes=8).to(DEVICE)



    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 6. Training Loop
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        train_acc = 100 * correct / total

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()

        val_acc = 100 * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch + 1}/{EPOCHS}] "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        scheduler.step(avg_val_loss)

        # Save the absolute best model weights
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_boxing_model_1.pth")
            print(f"  --> New Best Model Saved! (Acc: {best_val_acc:.2f}%)")

    print("\nTraining Complete! Ready for inference.")


if __name__ == "__main__":
    main()