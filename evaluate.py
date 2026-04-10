import os
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from model import BiLSTMStrikeClassifier

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = BASE_DIR if "HAR" in os.path.basename(BASE_DIR) else os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "datasets", "Skeleton_data")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_referee_bilstm.pth")

X_PATH = os.path.join(DATA_DIR, "X_data.npy")
y_PATH = os.path.join(DATA_DIR, "y_data.npy")

# Must match your training configuration
INPUT_DIM = 51
HIDDEN_DIM = 64
NUM_LAYERS = 2
NUM_CLASSES = 6

CLASS_NAMES = ["Cross", "Jab", "Lead Hook", "Lead Uppercut", "Rear Hook", "Rear Uppercut"]


def evaluate_model():
    print("Loading data for evaluation...")
    X = np.load(X_PATH)
    y = np.load(y_PATH)

    # Recreate the exact split used in training
    _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading weights from {MODEL_PATH}")
    model = BiLSTMStrikeClassifier(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    print("Running inference on validation set...")
    with torch.no_grad():
        X_val_tensor = X_val_tensor.to(device)
        outputs = model(X_val_tensor)
        _, predicted = torch.max(outputs, 1)

    y_true = y_val_tensor.cpu().numpy()
    y_pred = predicted.cpu().numpy()

    # Generate Reports
    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    print("\n" + "=" * 50)
    print("CONFUSION MATRIX")
    print("=" * 50)
    cm = confusion_matrix(y_true, y_pred)

    # Format nicely with Pandas
    df_cm = pd.DataFrame(cm, index=[f"True {name}" for name in CLASS_NAMES],
                         columns=[f"Pred {name}" for name in CLASS_NAMES])
    print(df_cm)


if __name__ == "__main__":
    evaluate_model()