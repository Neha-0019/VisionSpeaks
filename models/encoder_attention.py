"""
CNN Encoder that outputs both a global vector and spatial feature maps for attention.
Use this with DecoderRNNAttention; requires retraining.
"""
import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNNWithAttention(nn.Module):
    """
    ResNet-50 encoder that returns:
    - global_feat: (batch_size, embed_size) for decoder initialization
    - spatial_feat: (batch_size, embed_size, H, W) for attention over image regions
    """

    def __init__(self, embed_size):
        super(EncoderCNNWithAttention, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for param in resnet.parameters():
            param.requires_grad_(False)

        # Use everything except avgpool and fc so we keep spatial dimensions
        # ResNet: conv1,bn1,relu,maxpool,layer1,layer2,layer3,layer4,avgpool,fc
        # Output of layer4: (B, 2048, 7, 7)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.channel_size = 2048
        self.spatial_size = 7  # 7x7 feature map

        # Project spatial features to embed_size for attention
        self.spatial_embed = nn.Conv2d(self.channel_size, embed_size, kernel_size=1)
        # Global vector (avg pool 2048 then linear)
        self.embed = nn.Linear(self.channel_size, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        """
        Args:
            images: (batch_size, 3, H, W)
        Returns:
            global_feat: (batch_size, embed_size)
            spatial_feat: (batch_size, embed_size, 7, 7)
        """
        # Removed torch.no_grad() to allow fine-tuning of the resnet backbone
        spatial = self.resnet(images)  # (B, 2048, 7, 7)
        spatial_proj = self.spatial_embed(spatial)  # (B, embed_size, 7, 7)
        global_pooled = spatial.mean(dim=[2, 3])   # (B, 2048)
        global_feat = self.bn(self.embed(global_pooled))
        return global_feat, spatial_proj
