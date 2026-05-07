import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence
from PIL import Image
import os
import pickle
import nltk
from tqdm import tqdm

import sys
# Add project root to path to allow direct imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.build_vocab import Vocabulary
from models.caption_model import CaptionModel

# --- Dataset Class --- #
class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, vocab, transform=None):
        self.root_dir = root_dir
        self.df = self._load_captions(captions_file)
        self.vocab = vocab
        self.transform = transform
        self.imgs = self.df['image']
        self.captions = self.df['caption']

    def _load_captions(self, captions_file):
        import pandas as pd
        df = pd.read_csv(captions_file, sep='\t', header=None, names=['image', 'caption'])
        df['image'] = df['image'].apply(lambda x: x.split('#')[0])
        # For training, we might use a subset of the data
        # For this project, we'll use the full dataset as specified by the file
        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        img_name = self.imgs[idx]
        img_path = os.path.join(self.root_dir, img_name)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError):
            # Handle missing or corrupt files by loading the next item
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            image = self.transform(image)

        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption_vec = [self.vocab('<start>')] + [self.vocab(token) for token in tokens] + [self.vocab('<end>')]
        target = torch.Tensor(caption_vec)
        return image, target

# --- Collate Function --- #
def collate_fn(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)
    images = torch.stack(images, 0)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths

# --- Main Training Function --- #
def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # Paths
    data_dir = 'data/raw/Flicker8k_Dataset'  # Path relative to project root
    captions_file = 'data/raw/Flickr8k.token.txt' # Path relative to project root
    vocab_path = 'data/processed/vocab.pkl' # Path relative to project root
    model_save_path = 'models/weights/' # Path relative to project root

    # Create model directory if it doesn't exist
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    # Hyperparameters
    embed_size = 256
    hidden_size = 512
    num_layers = 1
    num_epochs = 5
    batch_size = 128
    learning_rate = 0.001

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Load vocabulary
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)

    # Build data loader
    dataset = FlickrDataset(root_dir=data_dir, captions_file=captions_file, vocab=vocab, transform=transform)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)

    # Build the models
    model = CaptionModel(embed_size, hidden_size, vocab_size, num_layers).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # We only train the decoder and the new embedding layer of the encoder
    params = list(model.decoder.parameters()) + list(model.encoder.embed.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    # Training loop
    total_step = len(data_loader)
    for epoch in range(num_epochs):
        for i, (images, captions, lengths) in enumerate(tqdm(data_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")):
            
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            
            # Forward, backward and optimize
            outputs = model(images, captions)
            packed_outputs = pack_padded_sequence(outputs, lengths, batch_first=True)[0]
            loss = criterion(packed_outputs, targets)
            
            model.zero_grad()
            loss.backward()
            optimizer.step()

            # Print log info
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}, Perplexity: {torch.exp(loss):.4f}')

        # Save the model checkpoints
        torch.save(model.state_dict(), os.path.join(model_save_path, f'caption-model-{epoch+1}.pth'))
        print(f'Saved model checkpoint to {model_save_path}')

    print("--- Training Completed ---")

if __name__ == '__main__':
    main()
