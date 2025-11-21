import torch
from torch import nn
from typing import Any, Iterable, List

from torchvision import models
import torch.nn.functional as F

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
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(512, 2)

        # Normalization stats
        self.register_buffer("lat_mean", torch.tensor(0.0))
        self.register_buffer("lat_std", torch.tensor(1.0))
        self.register_buffer("lon_mean", torch.tensor(0.0))
        self.register_buffer("lon_std", torch.tensor(1.0))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        preds: List[Any] = []

        with torch.no_grad():
            for x in batch:
                # convert to tensor if needed
                if not torch.is_tensor(x):
                    x = torch.tensor(x)

                x = x.unsqueeze(0).to(self.device)  # (1, C, H, W)
                out = self.model(x)  # normalized coords
                out = self._denormalize(out)  # convert to degrees
                out = out.squeeze(0).cpu().tolist()  # [lat_deg, lon_deg]

                preds.append(out)

        return preds

def get_model() -> Model:
    """
    Factory function required by the evaluator.
    Returns an uninitialized model instance. The evaluator may optionally load
    weights (if provided) before calling predict(...).
    """
    return Model()

