import torch
import torch.nn as nn
import torch.nn.functional as F


class HandCraftedFeaturesExtractor:
    def __init__(self, sampling_rate=20, num_bins=10, device="cpu"):
        """
        Args:
            sampling_rate (int): Sampling frequency in Hz
            num_bins (int): Number of bins per axis
            device (str): "cpu" or "cuda"
        """
        self.sampling_rate = sampling_rate
        self.num_bins = num_bins
        self.device = device

    def __call__(self, batch):
        """
        Args:
            batch: torch.Tensor of shape (B, 200, 3)
        Returns:
            features: torch.Tensor of shape (B, 43)
        """
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch, dtype=torch.float32, device=self.device)

        features = [self._extract_features(sample) for sample in batch]
        return torch.stack(features, dim=0)

    def _extract_features(self, sample):
        """
        Extract features from one sample (200,3)
        Returns:
            torch.Tensor of shape (43,)
        """
        feats = []
        x, y, z = sample[:, 0], sample[:, 1], sample[:, 2]

        for axis in [x, y, z]:
            feats.append(torch.mean(axis))              # Average
            feats.append(torch.std(axis))               # Std
            feats.append(torch.mean(torch.abs(axis - torch.mean(axis))))  # Avg Abs Diff

        # Average Resultant Acceleration
        resultant = torch.sqrt(x**2 + y**2 + z**2)
        feats.append(torch.mean(resultant))

        # Time Between Peaks (simple local maxima)
        for axis in [x, y, z]:
            diffs = axis[1:-1]
            peaks = (diffs > axis[:-2]) & (diffs > axis[2:])
            peak_indices = torch.where(peaks)[0] + 1

            if len(peak_indices) > 1:
                avg_dist = torch.mean(torch.diff(peak_indices.float()))
                time_ms = (avg_dist / self.sampling_rate) * 1000.0
            else:
                time_ms = torch.tensor(0.0, device=sample.device)
            feats.append(time_ms)

        # Binned Distribution (10 bins per axis → 30)
        for axis in [x, y, z]:
            min_v = torch.min(axis).item()
            max_v = torch.max(axis).item()
            if max_v <= min_v:
                max_v = min_v + 1e-6  # avoid degenerate hist range
            hist = torch.histc(axis, bins=self.num_bins, min=min_v, max=max_v)
            hist = hist / len(axis)  # normalize
            feats.extend(hist)

        return torch.stack(feats)


class DepthwiseSeparableConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        # Depthwise
        self.dw = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size,
                            stride=1, padding=padding, dilation=dilation,
                            groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.pw = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.dw(x)
        x = F.relu6(self.bn1(x))
        x = self.pw(x)
        x = F.relu6(self.bn2(x))
        return x


class SimpleCNN1D(nn.Module):
    """Simple 1D CNN for sequences of length N (raw=3*W, feat=43).

    Blocks (×3): DWConv1D(k=9, dilation=1/2/4) → BN → ReLU6 → PWConv(16→32→64) → BN → ReLU6
    Head: GlobalAvgPool → Dense(32) → Dropout(0.1) → Dense(4)
    """
    def __init__(self, num_classes=4):
        super().__init__()
        # Input is (B, 1, L)
        self.block1 = DepthwiseSeparableConv1D(1, 16, kernel_size=9, dilation=1)
        self.block2 = DepthwiseSeparableConv1D(16, 32, kernel_size=9, dilation=2)
        self.block3 = DepthwiseSeparableConv1D(32, 64, kernel_size=9, dilation=4)

        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        # x: (B, L) or (B, 1, L)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # Global average pooling over time dimension
        x = x.mean(dim=-1)  # (B, 64)
        x = self.dropout(F.relu6(self.fc1(x)))
        x = self.fc2(x)  # logits
        return x

# Backward-compat alias expected by main.py
HandCraftedFeaturesExtractorTorch = HandCraftedFeaturesExtractor
