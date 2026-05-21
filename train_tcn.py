import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

# Defining the model architecture
class KinematicFeatureExpansion(nn.Module):
    def __init__(self):
        super(KinematicFeatureExpansion, self).__init__()

    def forward(self, x):
        x_pad1 = F.pad(x, (1, 0), mode='replicate')
        velocity = x - x_pad1[:, :, :-1]

        v_pad1 = F.pad(velocity, (1, 0), mode='replicate')
        acceleration = velocity - v_pad1[:, :, :-1]

        return torch.cat([x, velocity, acceleration], dim=1)


class CausalDepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(CausalDepthwiseSeparableConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation

        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            padding=0, dilation=dilation,
            groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = F.pad(x, (self.padding, 0))
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class TemporalResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(TemporalResidualBlock, self).__init__()
        self.conv1 = CausalDepthwiseSeparableConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = CausalDepthwiseSeparableConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu_out = nn.ReLU()

    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)
        out = self.drop1(self.relu1(self.bn1(self.conv1(x))))
        out = self.drop2(self.relu2(self.bn2(self.conv2(out))))
        return self.relu_out(out + res)


class EdgeBoxingTCN(nn.Module):
    def __init__(self, input_channels=34, num_classes=7, hidden_dim=64):
        super(EdgeBoxingTCN, self).__init__()
        self.kinematic_extractor = KinematicFeatureExpansion()
        in_channels = input_channels * 3

        dilations = [1, 2, 4, 8]
        layers = []
        for d in dilations:
            layers.append(TemporalResidualBlock(in_channels, hidden_dim, kernel_size=3, dilation=d))
            in_channels = hidden_dim

        self.tcn_network = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.kinematic_extractor(x)
        x = self.tcn_network(x)
        x = self.gap(x).squeeze(-1)
        return self.classifier(x)


# 2. Optimization and loss
class MultiClassFocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super(MultiClassFocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        log_prob = F.log_softmax(logits, dim=-1)
        prob = torch.exp(log_prob)
        focal_weight = (1 - prob) ** self.gamma
        focal_log_prob = focal_weight * log_prob
        return F.nll_loss(focal_log_prob, targets)


# 3. training loop and sampling logic
def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Training on device: {device.type.upper()}")

    # 1. Loading both datasets
    print("[*] Loading extracted tensors...")
    try:
        # Load the strikes
        X_strikes = torch.load('datasets/processed_X_features.pt')
        y_strikes = torch.load('datasets/processed_y_labels.pt')

        # Load idle dataset
        X_idle = torch.load('datasets/processed_X_idle.pt')
        y_idle = torch.load('datasets/processed_y_idle.pt')
    except FileNotFoundError as e:
        print(f"[!] Missing tensor files! Ensure both data_extractor and idle_miner have been run. {e}")
        return

    # 2. Concatenate into a unified Master Dataset
    X = torch.cat([X_idle, X_strikes], dim=0)
    y = torch.cat([y_idle, y_strikes], dim=0)

    print(f"[*] Master Dataset Assembled. Total Sequences: {len(y)}")

    # 3. Train/Validation Split
    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    split_generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=split_generator
    )

    # 4. Generating Class Weights for the Sampler
    # Only balace training dataset. The validation set should represent the real-world (imbalanced) distribution.
    print("[*] Calculating Class Weights for the Sampler...")

    # extracting labels
    train_labels = [dataset[i][1].item() for i in train_dataset.indices]
    class_counts = np.bincount(train_labels, minlength=7)

    print(f"    Class Frequencies in Training Set: {class_counts}")

    # To prevent division by zero.
    class_weights = []
    for count in class_counts:
        if count > 0:
            class_weights.append(1.0 / count)
        else:
            class_weights.append(0.0)

    # Assign the calculated weight to every individual sample in the training set
    sample_weights = [class_weights[label] for label in train_labels]

    # Sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # 5. DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Model, loss, optimizer
    model = EdgeBoxingTCN().to(device)
    criterion = MultiClassFocalLoss(gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=50)

    epochs = 50
    print("\n[*] Starting Training Loop...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()

        scheduler.step()
        train_acc = 100. * correct / total

        # Validation pass
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()

        val_acc = 100. * val_correct / val_total
        print(
            f"Epoch {epoch + 1:02d}/{epochs} | Train Loss: {train_loss / len(train_loader):.4f} | Train Acc: {train_acc:05.2f}% | Val Acc: {val_acc:05.2f}%")

    # Saving up the model
    torch.save(model.state_dict(), 'edge_boxing_tcn.pth')
    print("\n[*] Training Complete. Model weights saved to 'edge_boxing_tcn.pth'.")


if __name__ == "__main__":
    train_model()