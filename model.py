import torch
import torch.nn as nn

class PoseLSTM(nn.Module):
    def __init__(self, input_size=51, hidden_size=64, num_layers=2, num_classes=2, dropout_rate=0.5):
        super(PoseLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 1. The Sequence Brain (Bidirectional LSTM with Dropout)
        # batch_first=True ensures it expects tensors of shape (Batch, Sequence, Features)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
            bidirectional=True
        )

        # 2. Regularization (Prevents overfitting to specific training punches)
        self.dropout = nn.Dropout(dropout_rate)

        # 3. The Classifier
        # We multiply hidden_size by 2 because bidirectional=True outputs both a forward and backward pass
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # Pass the sequence through the LSTM
        out, _ = self.lstm(x)

        # Grab the network's conclusion at the very end of the 30-frame sequence
        last_out = out[:, -1, :]

        # Pass through dropout and then the final fully connected layer
        out = self.fc(self.dropout(last_out))

        return out

# --- QUICK TEST BLOCK ---
# You can run this file directly just to make sure the tensor math works!
if __name__ == "__main__":
    print("Initializing the PoseLSTM Model...")
    model = PoseLSTM()

    # Create a fake batch of data to test the architecture
    # Shape: (Batch Size of 4 videos, 30 frames per video, 51 keypoint features)
    dummy_input = torch.randn(4, 30, 51)

    # Pass the fake data through the model
    predictions = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {predictions.shape} -> (Batch Size, Num Classes)")
    print("Model architecture is ready to go!")