import pandas as pd

def load_caption_map(captions_file):
    df = pd.read_csv(captions_file)
    caption_map = {}

    for _, row in df.iterrows():
        img = row['image']
        caption = row['caption']
        caption_map.setdefault(img, []).append(caption)

    return caption_map
