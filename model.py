import torch
import torch.nn as nn


class PoseLSTM(nn.Module):
    def __init__(self, input_size=51, hidden_size=64, num_layers=2, num_classes=2):
        super(PoseLSTM, self).__init__()

        # 1. Store the parameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 2. The Sequence Brain (LSTM)
        # batch_first=True means our tensor shape will be (batch, sequence, features)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # 3. The Classifier (Fully Connected Layer)
        # This takes the final thought of the LSTM and maps it to our 2 classes (Punch or Idle)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Initialize hidden and cell states with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Pass the whole sequence of frames through the LSTM
        # 'out' holds the outputs for EVERY frame.
        out, _ = self.lstm(x, (h0, c0))

        # We only care about the network's conclusion at the very end of the video.
        # out[:, -1, :] grabs the hidden state from the LAST frame of the sequence.
        out = self.fc(out[:, -1, :])

        return out


# QUICK TEST BLOCK
if __name__ == "__main__":
    print("Initializing the PoseLSTM Model...")
    model = PoseLSTM()

    # Create a fake batch of data to test if the math works
    # Shape: (Batch Size of 4 videos, 30 frames per video, 51 keypoint features)
    dummy_input = torch.randn(4, 30, 51)

    # Pass the fake data through the model
    predictions = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {predictions.shape} -> (Batch Size, Num Classes)")