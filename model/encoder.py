"""
Backbone encoder via segmentation_models_pytorch.
Full fine-tuning. Outputs 4-scale feature pyramid projected to d_model.

The paper uses 'tu-convnext_base' (ImageNet-pretrained); any smp encoder name
with a 5-stage feature pyramid works (e.g. 'tu-convnext_small', 'resnet50').
"""
import torch.nn as nn
import segmentation_models_pytorch as smp


class Encoder(nn.Module):
    def __init__(self, backbone_name='tu-convnext_base', d_model=256, pretrained=True):
        super().__init__()
        weights = 'imagenet' if pretrained else None
        self.backbone = smp.encoders.get_encoder(backbone_name, weights=weights, depth=5)

        # Project each of the last 4 stages to d_model
        channels = self.backbone.out_channels  # e.g. (3, 96, 96, 192, 384, 768)
        self.feat_channels = channels[-4:]     # last 4 = 1/4, 1/8, 1/16, 1/32

        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, d_model, 1, bias=False),
                nn.BatchNorm2d(d_model),
                nn.ReLU(inplace=True),
            ) for c in self.feat_channels
        ])

    def forward(self, x):
        """
        Returns:
            list of 4 feature maps: [feat_4(1/4), feat_8(1/8), feat_16(1/16), feat_32(1/32)]
        """
        features = self.backbone(x)
        features = features[-4:]
        return [self.proj[i](features[i]) for i in range(4)]
