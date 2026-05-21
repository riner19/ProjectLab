import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import TensorDataset, DataLoader

# Import your trained architecture directly from your training script
from train_tcn import EdgeBoxingTCN

# --- Configuration ---
MODEL_WEIGHTS = 'edge_boxing_tcn.pth'

# Strike Data Paths
X_STRIKE_PATH = 'datasets/processed_X_features.pt'
Y_STRIKE_PATH = 'datasets/processed_y_labels.pt'

# Idle Data Paths
X_IDLE_PATH = 'datasets/processed_X_idle.pt'
Y_IDLE_PATH = 'datasets/processed_y_idle.pt'

CLASSES = [
    'Idle/Guard',
    'Jab',
    'Cross',
    'Lead Hook',
    'Rear Hook',
    'Lead Uppercut',
    'Rear Uppercut'
]


def generate_confusion_matrix():
    # --- Device Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Running evaluation on: {device.type.upper()}")

    # --- Load Model ---
    print("[*] Loading Trained EdgeBoxingTCN...")
    model = EdgeBoxingTCN(input_channels=34, num_classes=7, hidden_dim=64).to(device)

    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
        model.eval()  # Disable dropout for deterministic evaluation
    except Exception as e:
        print(f"[!] Failed to load model weights. Error: {e}")
        return

    # --- Load Data & Merge ---
    print("[*] Loading Tensor Data...")
    try:
        # Load Strikes
        X_strikes = torch.load(X_STRIKE_PATH)
        y_strikes = torch.load(Y_STRIKE_PATH)

        # Load Idle/Guard
        X_idle = torch.load(X_IDLE_PATH)
        y_idle = torch.load(Y_IDLE_PATH)

        # Concatenate along the batch dimension (dim=0)
        X = torch.cat([X_idle, X_strikes], dim=0)
        y = torch.cat([y_idle, y_strikes], dim=0)

        print(f"[*] Successfully loaded and merged datasets. Total sequences: {len(y)}")
    except Exception as e:
        print(f"[!] Failed to load data tensors. Ensure both scripts generated the .pt files. Error: {e}")
        return

    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    split_generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=split_generator
    )

    print(f"[*] Isolated Pure Validation Set: {len(val_dataset)} sequences.")

    # CRITICAL CHANGE: We only pass `val_dataset` to the dataloader now!
    dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    # ==========================================

    all_preds = []
    all_targets = []

    print("[*] Running Forward Passes...")
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)

            # Extract the class with the highest probability
            _, predicted = outputs.max(dim=1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.numpy())

    # --- Generate Metrics & Plot ---
    print("[*] Generating Confusion Matrix...")
    # Explicitly pass the labels [0, 1, 2, 3, 4, 5, 6] so it knows about all 7 classes
    labels_list = list(range(len(CLASSES)))

    cm = confusion_matrix(all_targets, all_preds, labels=labels_list)

    # Print text-based classification report to the console
    print("\n" + "=" * 50)
    print(" DETAILED CLASSIFICATION REPORT")
    print("=" * 50)
    print(classification_report(all_targets, all_preds, labels=labels_list, target_names=CLASSES, zero_division=0))

    # Plot visual heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES,
                cbar_kws={'label': 'Number of Predictions'})

    plt.title('EdgeBoxingTCN - Action Confusion Matrix', fontsize=16, pad=20)
    plt.ylabel('True / Ground Truth Action', fontsize=12)
    plt.xlabel('Predicted Action', fontsize=12)

    # Rotate x-axis labels so they don't overlap
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save a high-res PNG to your folder
    plt.savefig('confusion_matrix_results.png', dpi=300)
    print("[*] Heatmap saved as 'confusion_matrix_results.png'.")

    # Display the plot window
    plt.show()


if __name__ == "__main__":
    generate_confusion_matrix()