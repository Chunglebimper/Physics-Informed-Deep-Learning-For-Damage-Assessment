import torch
import torch.nn as nn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

# Enhanced Damage Model: combines pre and post-disaster images to predict damage
class EnhancedDamageModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # Backbone from ResNet50 with FPN (Feature Pyramid Network)
        self.backbone = resnet_fpn_backbone('resnet50', pretrained=True, trainable_layers=3)

        # Fusion layer to combine pre/post features
        self.fusion = nn.Sequential(
            nn.Conv2d(256 * 2, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        # Segmentation head to predict class labels for each pixel
        self.seg_damage = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        )

    def forward(self, pre, post):
        # Extract features from both pre- and post-disaster images
        feat_pre = self.backbone(pre)['0']
        feat_post = self.backbone(post)['0']

        # Concatenate and fuse features
        fused = self.fusion(torch.cat([feat_pre, feat_post], dim=1))

        # Predict segmentation mask
        return self.seg_damage(fused)
