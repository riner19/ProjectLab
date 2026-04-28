import torch
import torch.nn as nn


class BoxingLSTM(nn.Module):
    def __init__(self, input_size=62, hidden_size=256, num_layers=2, num_classes=8, dropout=0.6):
        super(BoxingLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # The LSTM layer processing our (Batch, 30, 62) tensors
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Fully connected network
        # CHANGED: FC1 size from 64 to 128 to match the larger LSTM
        self.fc1 = nn.Linear(hidden_size, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Initialize hidden and cell states with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate through LSTM
        out, _ = self.lstm(x, (h0, c0))

        # We only care about the hidden state of the FINAL time step (frame 30)
        out = out[:, -1, :]

        # Pass through the linear layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out