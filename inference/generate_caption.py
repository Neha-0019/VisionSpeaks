import torch
from torchvision import transforms
from PIL import Image
import pickle
import os
import sys
import argparse
import json
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.caption_model import CaptionModel
from models.caption_model_attention import CaptionModelAttention
from training.build_vocab import Vocabulary

def load_image(image_path, transform=None):
    """Load an image and apply transformations."""
    image = Image.open(image_path).convert('RGB')
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

def generate_caption(image_path, model_path, vocab_path, embed_size, hidden_size, beam_size=5, max_len=20, length_penalty=0.8, temperature=0.0, return_all=False, repetition_penalty=1.2):
    """
    Generates a caption for a single image.

    Args:
        image_path: Path to the image file.
        model_path: Path to the trained model weights.
        vocab_path: Path to the vocabulary pickle file.
        embed_size: Size of the embedding layer.
        hidden_size: Size of the hidden layer in the LSTM.
        beam_size: Beam size for beam search (1 = greedy). Use 7–10 for better captions.
        max_len: Maximum caption length in words.
        length_penalty: For beam search, score = log_prob / (length ** length_penalty).
        temperature: If > 0, use temperature sampling instead of beam (e.g. 1.2–1.5 for more diverse captions).
        return_all: If True, returns list of (caption, confidence) for all beams.
        repetition_penalty: Penalty for repeating words. > 1.0 reduces repetition.

    Returns:
        tuple: (Generated caption string, confidence score) if return_all=False
        list: [(caption, confidence), ...] if return_all=True
    """
    # --- Hyperparameters --- #
    num_layers = 1 # As defined in train.py

    # --- Device Configuration --- #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Image Transformation --- #
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # --- Load Vocabulary --- #
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)
    end_idx = vocab.word2idx['<end>']
    start_idx = vocab.word2idx.get('<start>', 1)
    use_attention = 'attention' in os.path.basename(model_path)

    if use_attention:
        model = CaptionModelAttention(embed_size, hidden_size, vocab_size, num_layers).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        model = CaptionModel(embed_size, hidden_size, vocab_size, num_layers).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # --- Prepare Image --- #
    image = load_image(image_path, transform)
    image_tensor = image.to(device)

    # --- Generate Caption --- #
    results = []
    with torch.no_grad():
        if use_attention:
            global_feat, spatial_feat = model.encoder(image_tensor)
            if beam_size > 1:
                raw_results = model.decoder.sample_beam(
                    global_feat, spatial_feat, beam_size=beam_size, max_len=max_len,
                    end_idx=end_idx, length_penalty=length_penalty, return_all=return_all,
                    repetition_penalty=repetition_penalty
                )
                if not return_all:
                    results = [raw_results]
                else:
                    results = raw_results
            else:
                sampled_ids, confidence = model.decoder.sample(global_feat, spatial_feat, max_len=max_len, end_idx=end_idx)
                results = [(sampled_ids, confidence)]
        else:
            features = model.encoder(image_tensor)
            inputs = features.unsqueeze(1)
            if temperature > 0:
                sampled_ids, confidence = model.decoder.sample_temperature(
                    inputs, temperature=temperature, max_len=max_len, end_idx=end_idx
                )
                results = [(sampled_ids, confidence)]
            elif beam_size <= 1:
                sampled_ids, confidence = model.decoder.sample(inputs, max_len=max_len)
                results = [(sampled_ids, confidence)]
            else:
                raw_results = model.decoder.sample_beam(
                    inputs, beam_size=beam_size, max_len=max_len, end_idx=end_idx, length_penalty=length_penalty, 
                    return_all=return_all, repetition_penalty=repetition_penalty
                )
                if not return_all:
                    results = [raw_results]
                else:
                    results = raw_results

    # --- Convert Word IDs to Words --- #
    final_captions = []
    for sampled_ids, confidence in results:
        caption = []
        for word_id in sampled_ids:
            word = vocab.idx2word.get(word_id, vocab.idx2word.get(vocab.word2idx['<unk>'], '<unk>'))
            if word == '<start>':
                continue
            if word == '<end>':
                break
            caption.append(word)
        final_captions.append((' '.join(caption), confidence))

    if not return_all:
        return final_captions[0]
    return final_captions

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a caption for an image.')
    parser.add_argument('--image', type=str, required=True, help='Path to the image file.')
    parser.add_argument('--model_path', type=str, default='../models/weights/caption-model-5.pth', help='Path to the trained model.')
    parser.add_argument('--vocab_path', type=str, default='../data/processed/vocab.pkl', help='Path to the vocabulary file.')
    parser.add_argument('--beam_size', type=int, default=5, help='Beam size for beam search (1=greedy).')
    parser.add_argument('--max_len', type=int, default=20, help='Max caption length.')
    args = parser.parse_args()

    caption = generate_caption(args.image, args.model_path, args.vocab_path, beam_size=args.beam_size, max_len=args.max_len)
    print("Generated Caption:")
    print(caption)
