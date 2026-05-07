"""
Train the attention-based caption model (EncoderCNNWithAttention + DecoderRNNAttention).
Run this to get more accurate, image-specific captions. Saves as caption-attention-model-{epoch}.pth.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
import os
import pickle
import sys
from tqdm import tqdm

# Project root (VisionSpeak folder) - so paths work no matter where you run from
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)  # run from project root so relative paths in script are correct

# So pickle can load vocab.pkl (saved when build_vocab was run as __main__)
import __main__
from training.build_vocab import Vocabulary
__main__.Vocabulary = Vocabulary

from training.train import collate_fn
from models.caption_model_attention import CaptionModelAttention
from PIL import Image
import nltk
import pandas as pd


class FlickrDatasetCSV(torch.utils.data.Dataset):
    """Like FlickrDataset but for comma-separated captions.txt with header (image,caption)."""
    def __init__(self, root_dir, captions_file, vocab, transform=None):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file, sep=',', header=0)
        self.vocab = vocab
        self.transform = transform
        self.imgs = self.df['image']
        self.captions = self.df['caption']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        img_name = self.imgs[idx]
        img_path = os.path.join(self.root_dir, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError):
            return self.__getitem__((idx + 1) % len(self))
        if self.transform:
            image = self.transform(image)
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption_vec = [self.vocab('<start>')] + [self.vocab(token) for token in tokens] + [self.vocab('<end>')]
        return image, torch.tensor(caption_vec, dtype=torch.long)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training ATTENTION model on device: {device}")

    # Your layout: images in data/raw/flickr8k/Images, captions in data/raw/flickr8k/captions.txt
    data_dir = os.path.join(_ROOT, 'data', 'raw', 'flickr8k', 'Images')
    captions_file = os.path.join(_ROOT, 'data', 'raw', 'flickr8k', 'captions.txt')
    vocab_path = os.path.join(_ROOT, 'data', 'processed', 'vocab.pkl')
    model_save_path = os.path.join(_ROOT, 'models', 'weights')

    os.makedirs(model_save_path, exist_ok=True)

    embed_size = 256
    hidden_size = 512
    num_layers = 1
    num_epochs = 15  # Full training for maximum accuracy
    batch_size = 32  # Reduced to 32 to lower CPU and RAM load on laptop
    learning_rate = 0.0001  # Standard LR for starting fresh with fine-tuning
    dropout = 0.5   # Dropout to prevent overfitting and improve generalization

    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        # RandomResizedCrop helps the model focus on different parts of the image (like faces/hair)
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), 
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), 
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)

    dataset = FlickrDatasetCSV(root_dir=data_dir, captions_file=captions_file, vocab=vocab, transform=transform)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)

    model = CaptionModelAttention(embed_size, hidden_size, vocab_size, num_layers).to(device)

    # Starting fresh, no need to load old weights
    criterion = nn.CrossEntropyLoss()
    # Build the models
    # model = CaptionModelAttention(embed_size, hidden_size, vocab_size, num_layers).to(device)  # Removed duplicate initialization

    # Enable fine-tuning of the encoder for much higher accuracy
    # Unfreeze all layers of ResNet-50 to allow full adaptation to the captioning task
    for param in model.encoder.resnet.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    
    # Differential learning rates: encoder should learn slower than the decoder
    # This helps preserve pre-trained features while adapting to new data
    params = [
        {'params': model.decoder.parameters(), 'lr': learning_rate},
        {'params': model.encoder.spatial_embed.parameters(), 'lr': learning_rate},
        {'params': model.encoder.embed.parameters(), 'lr': learning_rate},
        {'params': model.encoder.bn.parameters(), 'lr': learning_rate},
        {'params': model.encoder.resnet.parameters(), 'lr': learning_rate * 0.1} # 10x slower
    ]
    optimizer = torch.optim.Adam(params)
    
    # Add learning rate scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    total_step = len(data_loader)
    
    # --- RESUME LOGIC --- #
    # To pause at epoch 5 and resume later, set resume_epoch = 5
    # The script will automatically look for 'caption-attention-model-5.pth'
    resume_epoch = 9 
    
    if resume_epoch > 0:
        weights_to_load = os.path.join(model_save_path, f'caption-attention-model-{resume_epoch}.pth')
        if os.path.exists(weights_to_load):
            print(f"Resuming training from Epoch {resume_epoch} using {weights_to_load}...")
            model.load_state_dict(torch.load(weights_to_load, map_location=device), strict=False)
            start_epoch = resume_epoch
        else:
            print(f"Warning: Could not find weights for Epoch {resume_epoch}. Starting from scratch.")
            start_epoch = 0
    else:
        start_epoch = 0 # Starting from scratch

    for epoch in range(start_epoch, num_epochs):
        model.train()
        for i, (images, captions, lengths) in enumerate(tqdm(data_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")):
            # ... (rest of loop remains same)
            images = images.to(device)
            captions = captions.to(device)

            outputs = model(images, captions)
            lengths_out = [max(1, l - 1) for l in lengths]
            targets = captions[:, 1:].long()
            packed_outputs = pack_padded_sequence(outputs, lengths_out, batch_first=True)[0]
            packed_targets = pack_padded_sequence(targets, lengths_out, batch_first=True)[0]
            loss = criterion(packed_outputs, packed_targets)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')
        
        # Step the scheduler
        scheduler.step()
        print(f"Learning rate adjusted to: {scheduler.get_last_lr()[0]}")

        torch.save(model.state_dict(), os.path.join(model_save_path, f'caption-attention-model-{epoch+1}.pth'))
        print(f'Saved attention model to {model_save_path}')

    print("--- Attention model training completed ---")


if __name__ == '__main__':
    main()
