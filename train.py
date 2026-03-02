import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import the two files we just built!
from Dataset import PoseActionDataset
from model import PoseLSTM

# --- 1. SETUP THE DATA ---
print("Loading dataset...")
dataset = PoseActionDataset(csv_dir="dataset_csvs")

# 4 videos at a time
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# --- 2. SETUP THE BRAIN ---
print("Initializing model...")
model = PoseLSTM()

# The "Grader" (CrossEntropy is standard for classification tasks)
criterion = nn.CrossEntropyLoss()

# The "Student"
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- 3. THE TRAINING LOOP (Draft) ---
epochs = 5
print("Starting Training (Scratch Draft)...")

for epoch in range(epochs):
    # Loop through the dataset in batches of 4
    for batch_idx, (features, labels) in enumerate(dataloader):
        # Step A: Clear out the old memories from the last batch
        optimizer.zero_grad()

        # Step B: Forward Pass (The model makes a guess: Punch or Walk?)
        predictions = model(features)

        # Step C: Calculate the Loss (The Grader says how wrong the guess was)
        loss = criterion(predictions, labels)

        # Step D: Backward Pass (Calculate the mathematical adjustments needed)
        loss.backward()

        # Step E: Optimizer Step (Actually apply the adjustments to the model's brain)
        optimizer.step()

    # Print the progress at the end of each epoch
    print(f"Epoch [{epoch + 1}/{epochs}] | Latest Batch Loss: {loss.item():.4f}")

print(" Training script rough draft finished!")