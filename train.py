import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import shutil
import random
from tqdm.auto import tqdm
from timeit import default_timer as timer

from pose_dataset import PoseActionDataset
from model import PoseLSTM

def print_train_time(start: float, end: float, device: torch.device):
    total_time = end - start
    print(f"\nTrain time on {device}: {total_time:.3f} seconds")
    return total_time


# --- 1. VIDEO-LEVEL SPLIT FUNCTION ---
def prepare_train_val_folders(source_dir, train_dir="train_data", val_dir="val_data", split_ratio=0.8):
    """Sorts whole videos (CSVs) into train and val folders to prevent data leakage."""
    if os.path.exists(train_dir) and os.path.exists(val_dir):
        print(f"Using existing split folders: '{train_dir}' and '{val_dir}'")
        return

    print("Splitting dataset into separate Train and Validation folders...")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    all_csvs = [f for f in os.listdir(source_dir) if f.endswith('.csv')]

    # Group by label to maintain balance
    idle_csvs = [f for f in all_csvs if f.startswith('idle')]
    punch_csvs = [f for f in all_csvs if f.startswith('punch')]

    random.seed(42)  # For reproducibility
    random.shuffle(idle_csvs)
    random.shuffle(punch_csvs)

    # Copy files to respective folders
    for csv_list in [idle_csvs, punch_csvs]:
        train_count = int(len(csv_list) * split_ratio)
        for i, f in enumerate(csv_list):
            src = os.path.join(source_dir, f)
            dst = os.path.join(train_dir if i < train_count else val_dir, f)
            shutil.copy(src, dst)


if __name__ == '__main__':
    # --- 2. HYPERPARAMETERS ---
    EPOCHS = 100
    LEARNING_RATE = 0.001
    BATCH_SIZE = 8
    SOURCE_CSV_DIR = "custom_dataset_new"

    # --- 3. DEVICE AGNOSTIC SETUP ---
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Initializing Training Pipeline...")
    print(f"Hardware detected: {device}")

    # --- 4. LOAD DATA SAFELY ---
    prepare_train_val_folders(source_dir=SOURCE_CSV_DIR)

    train_dataset = PoseActionDataset(csv_dir="train_data")
    val_dataset = PoseActionDataset(csv_dir="val_data")

    if len(train_dataset) == 0:
        print("Error: No training sequences found. Check your extraction.")
        exit()

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Loaded {len(train_dataset)} Train sequences and {len(val_dataset)} Val sequences.")

    # --- 5. SETUP THE BRAIN ---
    model = PoseLSTM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_accuracy = 0.0

    print("\nStarting Training...\n")
    train_time_start_on_gpu = timer()

    # --- 6. THE TRAINING LOOP ---
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        # Wrap train_loader with tqdm
        train_loop = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{EPOCHS}] Train", leave=False)

        for features, labels in train_loop:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions, labels)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Update the progress bar with the current loss
            train_loop.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)

        # --- 7. VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        correct_predictions = 0

        # Wrap val_loader with tqdm
        val_loop = tqdm(val_loader, desc=f"Epoch [{epoch + 1}/{EPOCHS}] Val  ", leave=False)

        with torch.no_grad():
            for features, labels in val_loop:
                features, labels = features.to(device), labels.to(device)

                predictions = model(features)
                val_loss += criterion(predictions, labels).item()

                predicted_class = torch.argmax(predictions, dim=1)
                correct_predictions += (predicted_class == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = (correct_predictions / len(val_dataset)) * 100

        # Print the final summary for the epoch so it stays on the screen
        print(f"Epoch [{epoch + 1}/{EPOCHS}] -> Train Loss: {avg_train_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_pose_model.pth')
            print(f"   [*] New best model saved! ({best_val_accuracy:.2f}%)")

    train_time_end_on_gpu = timer()
    print_train_time(start=train_time_start_on_gpu,
                     end=train_time_end_on_gpu,
                     device=device)

    print(f"\nTraining complete. Best validation accuracy: {best_val_accuracy:.2f}%")
    print("Best weights saved to 'best_pose_model.pth'.")