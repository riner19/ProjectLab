import os
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from model import BoxingLSTM

# --- Configuration ---
DATA_DIR = "datasets/Skeleton_data_1"
X_PATH = os.path.join(DATA_DIR, "X_data.npy")
y_PATH = os.path.join(DATA_DIR, "y_data.npy")
GROUPS_PATH = os.path.join(DATA_DIR, "groups_data.npy")
MODEL_PATH = "best_boxing_model_1.pth"

# Updated for 6-Class Trigger Architecture
CLASS_NAMES = [
    "Cross", "Jab", "Lead Hook", "Lead Uppercut", "Rear Hook", "Rear Uppercut"
]

INPUT_DIM = 64
HIDDEN_DIM = 64 # Make sure this matches whatever you set in model.py!
NUM_LAYERS = 1
NUM_CLASSES = 6
BATCH_SIZE = 64

def main():
    print("Loading Validation Data for 6-Class Evaluation...")
    X = np.load(X_PATH)
    y = np.load(y_PATH)
    groups = np.load(GROUPS_PATH)

    # Recreate the exact same validation split
    val_video_names = ['V3', 'V10']
    val_mask = np.isin(groups, val_video_names)
    X_val, y_val = X[val_mask], y[val_mask]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading Model Weights from {MODEL_PATH}...")
    # NOTE: Ensure these params match the model that saved the weights!
    # If you are evaluating the old Epoch 3 weights, change HIDDEN_DIM back to 256 for this run!
    model = BoxingLSTM(
        input_size=INPUT_DIM,
        hidden_size=64, # Set to 256 to test the model you JUST trained
        num_layers=1,    # Set to 2 to test the model you JUST trained
        num_classes=NUM_CLASSES
    ).to(device)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    all_preds = []

    with torch.no_grad():
        for i in range(0, len(X_val_tensor), BATCH_SIZE):
            batch = X_val_tensor[i:i + BATCH_SIZE]
            outputs = model(batch)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())

    print("\n" + "=" * 60)
    print(" COMBAT AI REFEREE - 6-CLASS CLASSIFICATION REPORT ")
    print("=" * 60)

    report = classification_report(y_val, all_preds, target_names=CLASS_NAMES, zero_division=0)
    print(report)

    print("\n--- CONFUSION MATRIX ---")
    cm = confusion_matrix(y_val, all_preds)

    header = f"{'':<15} " + " ".join([f"{name[:4]:<5}" for name in CLASS_NAMES])
    print(header)
    for i, row in enumerate(cm):
        row_str = " ".join([f"{val:<5}" for val in row])
        print(f"{CLASS_NAMES[i]:<15} {row_str}")

if __name__ == "__main__":
    main()