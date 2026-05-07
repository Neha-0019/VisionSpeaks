import torch
import torch.nn as nn
from .encoder import EncoderCNN
from .decoder import DecoderRNN


class CaptionModel(nn.Module):
    """
    A wrapper for the Encoder-Decoder architecture.
    """

    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        """
        Args:
            embed_size (int): The size of the embedding vector.
            hidden_size (int): The number of features in the hidden state of the LSTM.
            vocab_size (int): The size of the vocabulary.
            num_layers (int): The number of recurrent layers in the LSTM.
        """
        super(CaptionModel, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        """
        The forward pass for training.

        Args:
            images (torch.Tensor): The input images.
            captions (torch.Tensor): The ground truth captions.

        Returns:
            torch.Tensor: The predicted logits for the captions.
        """
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def generate_caption(self, image, vocab, max_len=20):
        """
        Generates a caption for a single image.

        Args:
            image (torch.Tensor): The input image tensor.
            vocab (Vocabulary): The vocabulary object.
            max_len (int): The maximum length of the generated caption.

        Returns:
            str: The generated caption.
        """
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            features = self.encoder(image).unsqueeze(1)
            sampled_ids = self.decoder.sample(features, max_len=max_len)

        # Convert word IDs to words
        sampled_caption = []
        for word_id in sampled_ids:
            word = vocab.idx2word[word_id]
            if word == '<start>':
                continue
            if word == '<end>':
                break
            sampled_caption.append(word)

        return ' '.join(sampled_caption)
