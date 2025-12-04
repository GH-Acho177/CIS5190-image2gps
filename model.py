import torch
from PIL import Image
from torch import nn
from typing import Any, Iterable, List
from torchvision.models import convnext_tiny
from torchvision import transforms as T


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

_transform = T.Compose([
    T.Resize((256,256)),
    T.CenterCrop(224),                # consistent framing
    #T.ColorJitter(0.1,0.1,0.1),       # mild lighting noise
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

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

    def __init__(self):
        super().__init__()
        # Initialize your model here

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.register_buffer("lat_center_buf", torch.tensor([0.0]))
        self.register_buffer("lon_center_buf", torch.tensor([0.0]))
        self.register_buffer("scale_buf", torch.tensor([1.0]))

        # Load resnet18
        # backbone = resnet18(weights=None)
        # in_features = backbone.fc.in_features
        # backbone.fc = nn.Linear(in_features, 2)
        # self.backbone = backbone.to(self.device)

        # Load ConvNeXt-Tiny
        from torchvision.models import ConvNeXt_Tiny_Weights
        backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        in_features = backbone.classifier[2].in_features
        #backbone.classifier[2] = nn.Linear(in_features, 2)
        backbone.classifier[2] = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 2)
        )

        # from torchvision.models import convnext_base, convnext_large
        # from torchvision.models import ConvNeXt_Base_Weights, ConvNeXt_Large_Weights
        # weights = ConvNeXt_Base_Weights.IMAGENET1K_V1
        # backbone = convnext_base(weights=weights)
        #
        # in_features = backbone.classifier[2].in_features
        #
        # backbone.classifier[2] = nn.Sequential(
        #     nn.Linear(in_features, 512),
        #     nn.GELU(),
        #     nn.Linear(512, 128),
        #     nn.GELU(),
        #     nn.Linear(128, 2)
        # )

        self.backbone = backbone.to(self.device)

    def eval(self) -> None:
        super().eval()
        self.backbone.eval()

    def predict(self, batch: Iterable[Any]) -> List[Any]:
        """
        Implement your inference here.
        Inputs:
            batch: Iterable of preprocessed inputs (as produced by your preprocess.py)
        Returns:
            A list of predictions with the same length as `batch`.
        """
        batch_list = list(batch)

        images = []
        for item in batch_list:
            if isinstance(item, str):
                img = Image.open(item)
                img = img.convert("RGB")
                img = _transform(img)
                images.append(img)
            else:
                images.append(item)

        X = torch.stack(images, dim=0).to(self.device)

        with torch.no_grad():
            preds = self.backbone(X)  # shape (B, 2)

        # Retrieve constants
        latc = self.lat_center_buf.item()
        lonc = self.lon_center_buf.item()
        scale = self.scale_buf.item()

        # If scale != 1.0, assume model outputs normalized coords
        preds_deg = preds.clone()
        preds_deg[:, 0] = latc + preds[:, 0] * scale
        preds_deg[:, 1] = lonc + preds[:, 1] * scale
        preds = preds_deg


        preds_list = preds.cpu().tolist()
        return preds_list

    def load_state_dict(self, sd, strict=False):
        # Load normalization constants into buffers
        if "lat_center_buf" in sd:
            self.lat_center_buf.copy_(sd["lat_center_buf"])
            sd.pop("lat_center_buf")
        if "lon_center_buf" in sd:
            self.lon_center_buf.copy_(sd["lon_center_buf"])
            sd.pop("lon_center_buf")
        if "scale_buf" in sd:
            self.scale_buf.copy_(sd["scale_buf"])
            sd.pop("scale_buf")

        return super().load_state_dict(sd, strict)


def get_model() -> Model:
    """
    Factory function required by the evaluator.
    Returns an uninitialized model instance. The evaluator may optionally load
    weights (if provided) before calling predict(...).
    """
    return Model()

