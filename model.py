import torch
import torch.nn as nn


class BiLSTMStrikeClassifier(nn.Module):
    def __init__(self, input_dim=59, hidden_dim=64, num_layers=2, num_classes=6):
        """
        Physics-Enhanced Skeletal Action Recognition (SAR) Model.
        Processes sequential 59-D vectors: 51 raw coords + 8 physics features.
        """
        super(BiLSTMStrikeClassifier, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Bi-LSTM to capture forward and backward motion mechanics
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # Fully connected layers for classification
        self.fc1 = nn.Linear(hidden_dim * 2, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        # x shape: (batch_size, SEQ_LENGTH, input_dim)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)

        out, _ = self.lstm(x, (h0, c0))

        # Decode only the final temporal step for the sequence classification
        out = out[:, -1, :]

        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


if __name__ == "__main__":
    # Diagnostic block: Run this file directly to verify tensor shapes
    print("Testing Physics-Enhanced BiLSTM Architecture...")

    # Simulating: Batch Size 32, 30 Frames, 59 Features (51 coords + 8 physics)
    dummy_input = torch.randn(32, 30, 59)

    model = BiLSTMStrikeClassifier()
    output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape} (Expected: 32, 6)")
    print("Architecture verified.")