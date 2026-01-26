import torch
from torchvision import transforms
from PIL import Image
import pickle
import os
import sys
import argparse

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.caption_model import CaptionModel
from training.build_vocab import Vocabulary

def load_image(image_path, transform=None):
    """Load an image and apply transformations."""
    image = Image.open(image_path).convert('RGB')
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

def generate_caption(image_path, model_path, vocab_path):
    """Generates a caption for a single image."""
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

    # --- Build Models --- #
    embed_size = 256
    hidden_size = 512
    num_layers = 1
    model = CaptionModel(embed_size, hidden_size, vocab_size, num_layers).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # --- Prepare Image --- #
    image = load_image(image_path, transform)
    image_tensor = image.to(device)

    # --- Generate Caption --- #
    with torch.no_grad():
        features = model.encoder(image_tensor)
        sampled_ids = model.decoder.sample(features)
    
    # --- Convert Word IDs to Words --- #
    
    caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word.get(word_id, '<unk>')

        if word == '<start>':
            continue
        if word == '<end>' and len(caption) >= 3:
            break

        if word not in ['<end>', '<pad>']:
            caption.append(word)
    # SAFETY NET (very important)
    if len(caption) == 0:
        caption = ["a", "person", "is", "doing", "something"]

    return ' '.join(caption)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a caption for an image.')
    parser.add_argument('--image', type=str, required=True, help='Path to the image file.')
    parser.add_argument('--model_path', type=str, default='../models/weights/caption-model-5.pth', help='Path to the trained model.')
    parser.add_argument('--vocab_path', type=str, default='../data/processed/vocab.pkl', help='Path to the vocabulary file.')
    args = parser.parse_args()

    # Generate and print the caption
    caption = generate_caption(args.image, args.model_path, args.vocab_path)
    print("Generated Caption:")
    print(caption)
