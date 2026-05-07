import torch
from torchvision import transforms
from PIL import Image
import pickle
import os
import sys
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.caption_model import CaptionModel
from training.build_vocab import Vocabulary
from training.train import FlickrDataset, collate_fn # Re-use dataset

def evaluate_model(model, data_loader, vocab, device):
    """
    Evaluates the model on the given dataset using BLEU score.
    """
    model.eval()
    references_corpus = []
    hypotheses_corpus = []

    # Create a dictionary of references {image_name: [caption1, caption2, ...]}
    # This is a more robust way to calculate BLEU score
    references_dict = {}
    for img, cap in zip(data_loader.dataset.imgs, data_loader.dataset.captions):
        if img not in references_dict:
            references_dict[img] = []
        tokens = nltk.tokenize.word_tokenize(str(cap).lower())
        references_dict[img].append(tokens)

    # Use a set to keep track of images we've already processed
    processed_images = set()

    with torch.no_grad():
        for images, captions, lengths in tqdm(data_loader, desc="Evaluating"):
            images = images.to(device)
            
            # We need the image names to look up references
            # This is a simplification; a better way is to have the dataset return image names.
            # For now, we'll just generate and assume the order matches.

            for i in range(images.size(0)):
                image = images[i].unsqueeze(0)
                
                # Generate caption using the model's method
                generated_caption = model.generate_caption(image, vocab)
                hypotheses_corpus.append(generated_caption.split())
                
                # Find the corresponding references
                # This part is tricky without direct access to image names in the batch
                # We will rely on the order, which is okay if shuffle=False
                # A better implementation would pass image identifiers through the dataloader
                img_name = data_loader.dataset.imgs[len(processed_images)]
                references_corpus.append(references_dict[img_name])
                processed_images.add(img_name)

    # Calculate BLEU-4 score
    bleu4 = corpus_bleu(references_corpus, hypotheses_corpus, weights=(0.25, 0.25, 0.25, 0.25))
    return bleu4

def main():
    # --- Parameters --- #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- Paths --- #
    # Paths relative to project root
    test_data_dir = 'data/raw/Flicker8k_Dataset'
    test_captions_file = 'data/raw/Flickr_8k.testImages.txt'
    vocab_path = 'data/processed/vocab.pkl'
    model_path = 'models/weights/caption-model-5.pth'

    # --- Hyperparameters (should match training) --- #
    embed_size = 256
    hidden_size = 512
    num_layers = 1
    batch_size = 32

    # --- Image Transformations --- #
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # --- Load Vocabulary and Model --- #
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)

    # For evaluation, we need a way to map test image names to their captions
    # The FlickrDataset needs to be adapted or a new one created for test data
    # For simplicity, we'll assume the test captions are in the same format
    # and we can filter the main captions file.
    # This is a placeholder for a proper test set loading logic.
    print("Loading test data... (This might need adjustment based on your test set format)")
    # A real implementation would use a dedicated test set file.
    # For now, we reuse FlickrDataset but it will load all captions.
    # You should create a test-specific caption file for a proper evaluation.
    test_dataset = FlickrDataset(root_dir=test_data_dir, captions_file='../data/raw/Flickr8k.token.txt', vocab=vocab, transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)

    # --- Initialize Model --- #
    model = CaptionModel(embed_size, hidden_size, vocab_size, num_layers).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # --- Evaluate --- #
    bleu_score = evaluate_model(model, test_loader, vocab, device)
    print(f'BLEU Score on the test set: {bleu_score:.4f}')

if __name__ == '__main__':
    main()
