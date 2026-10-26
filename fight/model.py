import torch
from torch import nn


class VideoClassifier(nn.Module):
    def __init__(self, architecture="x3d_s", pretrained=True):
        super().__init__()
        self.architecture = architecture
        if architecture == "s3d":
            from torchvision.models.video import S3D_Weights, s3d

            original = s3d(weights=S3D_Weights.KINETICS400_V1 if pretrained else None)
            self.backbone = original.features
            self.head = nn.Linear(1024, 1)
        elif architecture == "x3d_s":
            from pytorchvideo.models.hub import x3d_s

            original = x3d_s(pretrained=pretrained)
            projection = original.blocks[-1].proj
            original.blocks[-1].proj = nn.Identity()
            original.blocks[-1].activation = nn.Identity()
            self.backbone = original
            self.head = nn.Linear(projection.in_features, 1)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

    def features(self, tensor):
        features = self.backbone(tensor)
        if features.ndim == 5:
            features = features.mean(dim=(2, 3, 4))
        return features

    def forward(self, tensor):
        return self.head(self.features(tensor)).flatten()

    def set_stage(self, stage):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        if stage == "finetune":
            blocks = (
                list(self.backbone.blocks)
                if hasattr(self.backbone, "blocks")
                else list(self.backbone.children())
            )
            for block in blocks[-2:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
        self.head.requires_grad_(True)
