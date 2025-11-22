import torch
from torch import nn
from typing import Any, Iterable, List

class BasicBlock(nn.Module):
    """small residual block."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.conv(x))

class GeoResNetLite(nn.Module):
    def __init__(self):
        super().__init__()

        # ---- Stem ----
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),  # 112×112
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                           # 56×56
        )

        # ---- Stages (like ResNet) ----
        self.stage1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 28×28
            nn.BatchNorm2d(64),
            nn.ReLU(),
            BasicBlock(64),
            BasicBlock(64),
        )

        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 14×14
            nn.BatchNorm2d(128),
            nn.ReLU(),
            BasicBlock(128),
            BasicBlock(128),
        )

        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1), # 7×7
            nn.BatchNorm2d(256),
            nn.ReLU(),
            BasicBlock(256),
            BasicBlock(256),
        )

        # ---- Head ----
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        return self.mlp(x)

# class SmallCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         self.features = nn.Sequential(
#             # ---- Block 1 ----
#             nn.Conv2d(3, 32, kernel_size=3, padding=1),   # (32, 224, 224)
#             nn.ReLU(),
#             nn.MaxPool2d(2),                              # (32, 112, 112)
#
#             # ---- Block 2 ----
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (64, 112, 112)
#             nn.ReLU(),
#             nn.MaxPool2d(2),                              # (64, 56, 56)
#
#             # ---- Block 3 ----
#             nn.Conv2d(64, 128, kernel_size=3, padding=1), # (128, 56, 56)
#             nn.ReLU(),
#             nn.MaxPool2d(2),                              # (128, 28, 28)
#
#             # ---- Block 4 ----
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),# (256, 28, 28)
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d((1, 1)),                 # (256, 1, 1)
#         )
#
#         self.regressor = nn.Linear(256, 2)
#
#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)  # flatten
#         return self.regressor(x)   # normalized lat/lon

class Model(nn.Module):
    """
    Template model for the leaderboard.

    Requirements:
    - Must be instantiable with no arguments (called by the evaluator).
    - Must implement `predict(batch)` which receives an iterable of inputs and
      returns a list of predictions (labels).
    - Must implement `eval()` to place the model in evaluation mode.
    - If you use PyTorch, submit a state_dict to be loaded via `load_state_dict`
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Initialize your model here
        # self.model = SmallCNN()
        self.model = GeoResNetLite()
        # Normalization stats
        self.register_buffer("lat_mean", torch.tensor(0.0))
        self.register_buffer("lat_std", torch.tensor(1.0))
        self.register_buffer("lon_mean", torch.tensor(0.0))
        self.register_buffer("lon_std", torch.tensor(1.0))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")
        self.to(self.device)

    def eval(self) -> None:
        # Optional: set your model to evaluation mode
        super().eval()

    def _denormalize(self, out: torch.Tensor) -> torch.Tensor:
        """
        out: (B, 2) normalized [lat_norm, lon_norm]
        returns: (B, 2) in degrees
        """
        mean = torch.stack([self.lat_mean, self.lon_mean]).to(out.device)
        std = torch.stack([self.lat_std, self.lon_std]).to(out.device)
        return out * std + mean

    def predict(self, batch: Iterable[Any]) -> List[Any]:
        """
        Implement your inference here.
        Inputs:
            batch: Iterable of preprocessed inputs (as produced by your preprocess.py)
        Returns:
            A list of predictions with the same length as `batch`.
        """
        self.eval()
        if torch.is_tensor(batch):
            x_tensor = batch
        else:
            xs = []
            for x in batch:
                if not torch.is_tensor(x):
                    x = torch.tensor(x, dtype=torch.float32)
                xs.append(x)

            if len(xs) == 0:
                return []

            x_tensor = torch.stack(xs, dim=0)

        x_tensor = x_tensor.to(self.device)

        with torch.no_grad():
            out = self.model(x_tensor)
            out = self._denormalize(out)
            out = out.cpu().tolist()

        return out

def get_model() -> Model:
    """
    Factory function required by the evaluator.
    Returns an uninitialized model instance. The evaluator may optionally load
    weights (if provided) before calling predict(...).
    """
    return Model()

