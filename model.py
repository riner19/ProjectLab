import torch
import torch.nn as nn

class BiLSTMStrikeClassifier(nn.Module):
    def __init__(self, input_dim=62, hidden_dim=64, num_layers=2, num_classes=8):
        super(BiLSTMStrikeClassifier, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.layer_norm = nn.LayerNorm(input_dim)

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        self.fc1 = nn.Linear(hidden_dim * 2, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        # Explicitly allocate hidden states on the same device as input
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim, device=x.device)

        x = self.layer_norm(x)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out