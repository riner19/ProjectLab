import torch
import torch.nn as nn

class BoxingLSTM(nn.Module):
    # CHANGED: hidden_size to 64, num_layers to 1
    def __init__(self, input_size=64, hidden_size=64, num_layers=1, num_classes=6, dropout=0.6):
        super(BoxingLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            # Dropout in LSTM only works if num_layers > 1, so we handle it below
            bidirectional=True
        )

        # CHANGED: BiLSTM outputs hidden_size * 2 (64 * 2 = 128)
        self.fc1 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        # Increased dropout to prevent memorization
        self.dropout = nn.Dropout(0.7)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]

        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out