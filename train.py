import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from Dataset import PoseActionDataset
from model import PoseLSTM

# --- 1. HYPERPARAMETERS ---
EPOCHS = 100
LEARNING_RATE = 0.001
BATCH_SIZE = 8

# --- 2. DEVICE AGNOSTIC SETUP ---
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f"Initializing Training Pipeline...")
print(f"Hardware detected: {device}")

# --- 3. LOAD DATA ---
# Pointing directly to your newly extracted custom dataset!
dataset = PoseActionDataset(csv_dir="custom_dataset_lastsuka")
total_files = len(dataset)

# Quick safety check to make sure the extraction worked
if total_files == 0:
    print("Error: No CSV files found in 'custom_dataset'. Please check your extraction.")
    exit()

print(f"Loaded {total_files} total sequences for training.")

train_size = int(0.8 * total_files)
val_size = total_files - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 4. SETUP THE BRAIN ---
model = PoseLSTM().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_val_accuracy = 0.0

# --- 5. THE TRAINING LOOP ---
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        predictions = model(features)
        loss = criterion(predictions, labels)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # --- 6. VALIDATION PHASE ---
    model.eval()
    val_loss = 0.0
    correct_predictions = 0

    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)

            predictions = model(features)
            val_loss += criterion(predictions, labels).item()

            predicted_class = torch.argmax(predictions, dim=1)
            correct_predictions += (predicted_class == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = (correct_predictions / val_size) * 100

    print(f"Epoch [{epoch + 1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), 'best_pose_model.pth')

print(f"\n Training complete. Best validation accuracy: {best_val_accuracy:.2f}%")
print(" Best weights saved to 'best_pose_model.pth'.")