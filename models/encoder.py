import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    """
    CNN Encoder using a pretrained ResNet-50.
    """

    def __init__(self, embed_size):
        """
        Args:
            embed_size (int): The size of the embedding vector.
        """
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Freeze all layers in the network
        for param in resnet.parameters():
            param.requires_grad_(False)

        # Remove the final classification layer (fc)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        # Add a linear layer to map the features to the embedding size
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        """
        Forward pass of the encoder.

        Args:
            images (torch.Tensor): Input images of shape (batch_size, 3, H, W).

        Returns:
            torch.Tensor: Feature vectors of shape (batch_size, embed_size).
        """
        # Extract features from the pretrained ResNet
        with torch.no_grad():
            features = self.resnet(images)

        # Reshape features to (batch_size, -1)
        features = features.view(features.size(0), -1)

        # Embed the features to the desired size
        embedded_features = self.embed(features)
        embedded_features = self.bn(embedded_features)
        return embedded_features
