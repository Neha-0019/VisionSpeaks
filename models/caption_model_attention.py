"""
Caption model that uses EncoderCNNWithAttention + DecoderRNNAttention.
Use this for training with attention; requires retraining from scratch.
"""
import torch
import torch.nn as nn
from .encoder_attention import EncoderCNNWithAttention
from .decoder_attention import DecoderRNNAttention


class CaptionModelAttention(nn.Module):
    """Encoder-decoder with visual attention over spatial CNN features."""

    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(CaptionModelAttention, self).__init__()
        self.encoder = EncoderCNNWithAttention(embed_size)
        self.decoder = DecoderRNNAttention(
            embed_size, hidden_size, vocab_size, num_layers, encoder_size=embed_size
        )

    def forward(self, images, captions):
        global_feat, spatial_feat = self.encoder(images)
        return self.decoder(global_feat, spatial_feat, captions)
