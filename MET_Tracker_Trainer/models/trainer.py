import os
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

# Progress bar (tqdm). Gracefully degrade if unavailable.
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x


class Trainer:
    def __init__(self, model: nn.Module, device: str = "cpu", lr: float = 1e-3, epochs: int = 10):
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self, train_loader):
        self.model.train()
        epoch_bar = tqdm(range(1, self.epochs + 1), desc="Epoch", leave=False)
        for epoch in epoch_bar:
            total_loss = 0.0
            total_samples = 0

            for X, y in train_loader:
                # X: (B, L) already on device by collate; ensure shape (B, 1, L) in model
                # y: (B) on device
                self.optimizer.zero_grad()
                logits = self.model(X)
                loss = self.criterion(logits, y)
                loss.backward()
                self.optimizer.step()

                batch_size = X.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

            # Report average training loss only (no accuracy during training)
            avg_loss = total_loss / max(total_samples, 1)
            try:
                epoch_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
            except Exception:
                pass
            print(f"Epoch {epoch}/{self.epochs} - loss: {avg_loss:.4f}")

    @torch.no_grad()
    def evaluate(self, data_loader):
        self.model.eval()
        total = 0
        correct = 0
        for X, y in data_loader:
            logits = self.model(X)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += X.size(0)
        acc = correct / max(total, 1)
        print(f"Test accuracy: {acc:.4f}")
        return acc

    def export_onnx(self, sample_length: int, out_path: str):
        """Export model to ONNX with softmax head for deployment.

        Adds a Softmax layer in export graph for probabilities.
        """
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        class Wrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                self.softmax = nn.Softmax(dim=1)

            def forward(self, x):
                logits = self.model(x)
                return self.softmax(logits)

        wrapped = Wrapper(self.model).to(self.device)
        wrapped.eval()
        dummy = torch.randn(1, sample_length, device=self.device)

        torch.onnx.export(
            wrapped,
            dummy,
            out_path,
            input_names=["input"],
            output_names=["prob"],
            dynamic_axes={"input": {0: "batch", 1: "length"}, "prob": {0: "batch"}},
            opset_version=13,
        )
        print(f"Saved ONNX: {out_path}")
