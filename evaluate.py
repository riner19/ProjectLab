import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from model import BiLSTMStrikeClassifier

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = BASE_DIR if "HAR" in os.path.basename(BASE_DIR) else os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "datasets", "Skeleton_data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

X_PATH = os.path.join(DATA_DIR, "X_data.npy")
y_PATH = os.path.join(DATA_DIR, "y_data.npy")
MODEL_PATH = os.path.join(MODELS_DIR, "best_referee_bilstm.pth")

# Updated 8-Class Structure
CLASS_NAMES = [
    "Cross", "Jab", "Lead Hook", "Lead Uppercut",
    "Rear Hook", "Rear Uppercut", "Guard", "Idle"
]

INPUT_DIM = 62
HIDDEN_DIM = 64
NUM_LAYERS = 2
NUM_CLASSES = 8
BATCH_SIZE = 64


def main():
    print("Loading Validation Data for 8-Class Evaluation...")
    if not os.path.exists(X_PATH) or not os.path.exists(y_PATH):
        print(f"Error: Data not found at {DATA_DIR}. Run extraction first.")
        return

    X = np.load(X_PATH)
    y = np.load(y_PATH)

    # Recreate the exact same split used in train.py (Random state 42 is critical here)
    _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading Model Weights from {MODEL_PATH}...")
    model = BiLSTMStrikeClassifier(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES
    ).to(device)

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except Exception as e:
        print(f"Failed to load model weights: {e}")
        print("Ensure you have trained the 8-class 62-D model first.")
        return

    model.eval()

    print("Running Inference on Validation Set...")
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)

    all_preds = []

    # Process in batches to prevent VRAM overflow
    with torch.no_grad():
        for i in range(0, len(X_val_tensor), BATCH_SIZE):
            batch = X_val_tensor[i:i + BATCH_SIZE]
            outputs = model(batch)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())

    # --- Metrics Output ---
    print("\n" + "=" * 60)
    print(" COMBAT AI REFEREE - 8-CLASS CLASSIFICATION REPORT ")
    print("=" * 60)

    report = classification_report(y_val, all_preds, target_names=CLASS_NAMES, zero_division=0)
    print(report)

    print("\n--- CONFUSION MATRIX ---")
    cm = confusion_matrix(y_val, all_preds)

    # Print dynamically formatted confusion matrix
    header = f"{'':<15} " + " ".join([f"{name[:4]:<5}" for name in CLASS_NAMES])
    print(header)
    for i, row in enumerate(cm):
        row_str = " ".join([f"{val:<5}" for val in row])
        print(f"{CLASS_NAMES[i]:<15} {row_str}")


if __name__ == "__main__":
    main()