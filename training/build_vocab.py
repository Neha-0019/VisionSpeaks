import nltk
import pickle
import os
from collections import Counter

class Vocabulary:
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(caption_file, threshold):
    """Build a vocabulary from a caption file."""
    # Read the captions
    with open(caption_file, 'r') as f:
        lines = f.readlines()

    counter = Counter()
    for i, line in enumerate(lines):
        # Image name and caption are separated by a tab
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue
        caption = parts[1]
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

        if (i+1) % 1000 == 0:
            print(f"[{i+1}/{len(lines)}] Tokenized the captions.")

    # Filter words with frequency below the threshold
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add the special tokens
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary
    for word in words:
        vocab.add_word(word)
    return vocab

def main():
    # For the purpose of this example, we assume the dataset is in data/raw
    # and the captions file is named 'captions.txt'
    # You will need to download the Flickr8k dataset and place it accordingly.
    caption_path = 'data/raw/Flickr8k.token.txt' # Path relative to project root
    vocab_path = 'data/processed/vocab.pkl'
    threshold = 4

    # Create processed data directory if it doesn't exist
    if not os.path.exists(os.path.dirname(vocab_path)):
        os.makedirs(os.path.dirname(vocab_path))

    print("Building vocabulary...")
    vocab = build_vocab(caption_file=caption_path, threshold=threshold)
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    
    print(f"Total vocabulary size: {len(vocab)}")
    print(f"Saved the vocabulary to '{vocab_path}'")

if __name__ == '__main__':
    # You may need to download the NLTK tokenizer model first
    # nltk.download('punkt')
    main()
