import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import shutil
import random
from tqdm.auto import tqdm
from timeit import default_timer as timer

# Import your custom modules
from pose_dataset import PoseActionDataset
from model import PoseLSTM


def print_train_time(start: float, end: float, device: torch.device):
    total_time = end - start
    print(f"\nTrain time on {device}: {total_time:.3f} seconds")
    return total_time


# --- 1. SMART VIDEO-LEVEL SPLIT FUNCTION ---
def prepare_train_val_folders(source_dir, train_dir="train_data", val_dir="val_data", split_ratio=0.8):
    """Sorts CSVs into train/val folders by reading the actual labels inside the files."""
    print("Rebuilding Train and Validation folders with new mixed data...")

    # Delete the old folders so we don't train on stale data
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    if os.path.exists(val_dir):
        shutil.rmtree(val_dir)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    all_csvs = [f for f in os.listdir(source_dir) if f.endswith('.csv')]

    idle_csvs = []
    punch_csvs = []

    # Look INSIDE the CSV to find the true label
    for f in all_csvs:
        filepath = os.path.join(source_dir, f)
        try:
            with open(filepath, 'r') as file:
                header = file.readline()  # Skip header
                first_data_row = file.readline().strip().split(',')

                # Column index 1 is 'label'
                if len(first_data_row) > 1:
                    label = int(first_data_row[1])
                    if label == 1:
                        punch_csvs.append(f)
                    else:
                        idle_csvs.append(f)
        except Exception as e:
            pass  # Skip broken or empty files invisibly

    print(f"✅ Found {len(punch_csvs)} Punch sequences and {len(idle_csvs)} Idle sequences.")

    # Shuffle for a truly random mix of SBU, UTD, and KTH
    random.seed(42)
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
    EPOCHS = 100  # Increased for larger dataset
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32  # Increased for smoother learning curve
    SOURCE_CSV_DIR = "custom_dataset_final"

    # --- 3. DEVICE AGNOSTIC SETUP ---
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Initializing Training Pipeline...")
    print(f"Hardware detected: {device}")
    print(f"Source Folder: {SOURCE_CSV_DIR}")

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

    # --- 5. AUTOMATIC CLASS WEIGHTS (Anti-Lazy Math) ---
    print("\nCalculating Dataset Balance...")
    idle_count = sum(1 for _, label in train_dataset if label == 0)
    punch_count = sum(1 for _, label in train_dataset if label == 1)
    total_samples = idle_count + punch_count

    weight_idle = total_samples / (2.0 * idle_count) if idle_count > 0 else 1.0
    weight_punch = total_samples / (2.0 * punch_count) if punch_count > 0 else 1.0

    print(f"Weights -> Idle Multiplier: {weight_idle:.2f}x | Punch Multiplier: {weight_punch:.2f}x")
    class_weights = torch.tensor([weight_idle, weight_punch], dtype=torch.float32).to(device)

    # --- 6. SETUP THE BRAIN ---
    model = PoseLSTM().to(device)

    # Loss function now uses the calculated weights!
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # The Scheduler: Halves the LR if accuracy gets stuck for 10 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

    best_val_accuracy = 0.0

    print("\nStarting Training...\n")
    train_time_start_on_gpu = timer()

    # --- 7. THE TRAINING LOOP ---
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        train_loop = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{EPOCHS}] Train", leave=False)

        for features, labels in train_loop:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions, labels)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            train_loop.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)

        # --- 8. VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        correct_predictions = 0

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

        # Step the scheduler based on validation accuracy
        scheduler.step(val_accuracy)

        # Print the summary
        print(f"Epoch [{epoch + 1}/{EPOCHS}] -> Train Loss: {avg_train_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_pose_model.pth')
            print(f"   [*] New best model saved! ({best_val_accuracy:.2f}%)")

    train_time_end_on_gpu = timer()
    print_train_time(start=train_time_start_on_gpu, end=train_time_end_on_gpu, device=device)

    print(f"\nTraining complete. Best validation accuracy: {best_val_accuracy:.2f}%")
    print("Best weights safely secured in 'best_pose_model.pth'.")