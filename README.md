# AI Image Caption Generator

A production-quality, minimal, and clean implementation of an Image Caption Generation system using PyTorch. This project is designed to be a comprehensive yet easy-to-understand resource for a college final-year project.

---

## 🎯 Project Overview

This project is an AI system that automatically generates a natural language description (a "caption") for a given image. It combines Computer Vision (to understand the image) and Natural Language Processing (to create the sentence).

The core of the system is a Deep Learning model based on a **CNN-LSTM architecture**. The CNN (Convolutional Neural Network) acts as an **encoder** to "see" the image and extract its important features, while the LSTM (Long Short-Term Memory) network acts as a **decoder** to generate a meaningful sentence based on those features.

## 💡 Real-World Applications

Image captioning technology has several practical uses:

-   **Accessibility**: Assisting visually impaired users by describing images on web pages or apps.
-   **Content Discovery**: Improving image search engines by allowing searches based on image content.
-   **Social Media**: Automatically generating alt-text for images, improving SEO and accessibility.
-   **Digital Asset Management**: Cataloging and organizing large collections of images with descriptive tags and captions.

## 🏗️ System Architecture

The model follows a classic Encoder-Decoder architecture, which is common for sequence-to-sequence tasks like machine translation and, in our case, image captioning.

### 1. The Encoder (CNN)

-   **What it is**: A pretrained **ResNet-50**, which is a powerful CNN that has already been trained on the massive ImageNet dataset.
-   **What it does**: It processes the input image and converts it into a compact feature vector (an "embedding"). This vector is a numerical representation of the image's visual content.
-   **How it works**: We remove the final classification layer of the ResNet-50. Instead of predicting an image class (like "cat" or "dog"), the network outputs a rich feature map that captures the objects, scenes, and attributes in the image.

### 2. The Decoder (LSTM)

-   **What it is**: A **Long Short-Term Memory (LSTM)** network, which is a type of Recurrent Neural Network (RNN) excellent at handling sequential data like text.
-   **What it does**: It takes the feature vector from the encoder and generates the caption one word at a time.
-   **How it works**: 
    1.  The image feature vector is fed into the LSTM as its initial state. This "primes" the decoder with the context of the image.
    2.  A special `<start>` token is fed as the first input to begin the generation process.
    3.  The LSTM predicts the next word. This predicted word is then fed back into the LSTM in the next time step to predict the subsequent word.
    4.  This process continues until the LSTM predicts a special `<end>` token or a maximum length is reached.

  
*A simplified view of the Encoder-Decoder pipeline.*

---

## 📊 Dataset

This project uses the **Flickr8k dataset**, a popular benchmark dataset for image captioning.

-   **Content**: 8,000 images.
-   **Captions**: Each image is paired with 5 different human-generated captions.
-   **Splits**: The dataset is typically split into training, validation, and testing sets to ensure the model generalizes well to new, unseen images.

Before training, all captions are preprocessed: they are converted to lowercase, and a **vocabulary** is built to map each unique word to a numerical index.

---

## ⚙️ How It Works

### Training (`training/train.py`)

The goal of training is to teach the model to predict the correct sequence of words for a given image.

1.  **Input**: The model receives a batch of (image, caption) pairs.
2.  **Encoder Pass**: The image is passed through the CNN encoder to get its feature vector.
3.  **Decoder Pass (Teacher Forcing)**: 
    -   The image feature vector initializes the LSTM's state.
    -   During training, instead of feeding the model's *own* prediction back into it, we use the **ground-truth caption** from the dataset. This technique, called **Teacher Forcing**, stabilizes training and helps the model learn faster.
4.  **Loss Calculation**: The model's output is a probability distribution over the entire vocabulary for each word in the caption. We use **Cross-Entropy Loss** to measure the difference between the model's predictions and the actual words in the ground-truth caption.
5.  **Optimization**: The **Adam optimizer** adjusts the model's weights to minimize this loss, making its predictions more accurate over time.

### Caption Generation (`inference/generate_caption.py`)

Once the model is trained, it can generate captions for new images.

1.  **Input**: A single image.
2.  **Encoder Pass**: The image is passed through the CNN to get its feature vector.
3.  **Decoder Pass (Greedy Decoding)**:
    -   The image feature vector initializes the LSTM's state.
    -   The process starts with the `<start>` token.
    -   The LSTM predicts the most likely next word. This word is then used as the input for the next time step.
    -   This is repeated until an `<end>` token is generated or the caption reaches its maximum length.

---

## 📁 Folder Structure

```
image-caption-generator/
│
├── data/
│   ├── raw/                # Dataset location (e.g., Flickr8k images and text files)
│   └── processed/          # Vocabulary file (vocab.pkl)
│
├── models/
│   ├── encoder.py          # CNN Encoder (ResNet-50)
│   ├── decoder.py          # LSTM Decoder
│   └── caption_model.py    # Main model that wraps the encoder and decoder
│
├── training/
│   ├── build_vocab.py      # Script to create the vocabulary from captions
│   ├── train.py            # The main training script
│   └── evaluate.py         # Script to evaluate the model with BLEU score
│
├── inference/
│   └── generate_caption.py # Logic to generate a caption for a single image
│
├── app.py                  # The Streamlit web application
├── requirements.txt        # Project dependencies
├── README.md               # This file
└── .gitignore              # Files and folders to ignore in git
```

---

## 🚀 How to Run the Project

Follow these steps to set up and run the project on your local machine.

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/image-caption-generator.git
cd image-caption-generator
```

### Step 2: Set Up the Environment

It's recommended to use a virtual environment.

```bash
# Create a virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install the required libraries
pip install -r requirements.txt
```

### Step 3: Download the Dataset

1.  Download the Flickr8k dataset. You can find it on Kaggle or other academic sites.
2.  Extract the contents and place the images in `data/raw/Flicker8k_Dataset/` and the token file (`Flickr8k.token.txt`) in `data/raw/`.

*Note: The paths are configured in `training/train.py`. You may need to adjust them based on where you store the data.*

### Step 4: Build the Vocabulary

Before training, you need to create the vocabulary file.

```bash
# NLTK's tokenizer might need to be downloaded first
python -m nltk.downloader punkt

# Run the script
python training/build_vocab.py
```

This will create `data/processed/vocab.pkl`.

### Step 5: Train the Model

Now you can start training the model.

```bash
python training/train.py
```

Training can take a long time, depending on your hardware. The script will save model checkpoints (`.pth` files) in the `models/weights/` directory after each epoch.

### Step 6: Run the Interactive Demo

Once the model is trained, you can launch the Streamlit app to see it in action.

```bash
streamlit run app.py
```

This will open a web browser where you can upload an image and generate a caption.

---

## 🖼️ Sample Outputs

*(You can add screenshots of your app's output here.)*

**Example 1:**

-   **Image**: A dog catching a frisbee.
-   **Generated Caption**: "A dog is jumping to catch a red frisbee in a grassy field."

**Example 2:**

-   **Image**: A group of people at a beach.
-   **Generated Caption**: "A group of people are standing on a sandy beach near the ocean."

---

## 🧠 Viva Preparation Q&A

Here are some common questions you might face in a viva or project defense.

#### Q1: What is Image Captioning?

**Answer**: Image captioning is the process of generating a descriptive, human-like sentence for an image. It's a task that requires both understanding the visual content of the image (Computer Vision) and generating fluent, natural language (NLP).

#### Q2: Why did you choose a CNN + LSTM architecture?

**Answer**: This is a standard and effective architecture for this problem. 
-   The **CNN (like ResNet-50)** is excellent at image feature extraction. Because it's pretrained on a large dataset like ImageNet, it already knows how to recognize a wide variety of objects, textures, and patterns.
-   The **LSTM** is designed to handle sequential data. It can remember context over long sequences, which is perfect for generating grammatically correct and coherent sentences.

The CNN acts as the "eyes" of the model, and the LSTM acts as the "language brain."

#### Q3: What is the BLEU Score?

**Answer**: The **BLEU (Bilingual Evaluation Understudy)** score is a metric used to evaluate the quality of a machine-generated sentence by comparing it to one or more human-generated reference sentences. 

-   It measures how many words and phrases from the generated sentence appear in the reference sentences.
-   It's a measure of **precision**: how much of the generated caption is relevant and correct. 
-   A score closer to 1.0 indicates a better match, while a score closer to 0 indicates a poor match.

#### Q4: What is the purpose of the Cross-Entropy Loss function here?

**Answer**: Cross-Entropy Loss is used to train classification models, and we can think of caption generation as a series of classification problems. At each step, the model must "classify" the next word from the entire vocabulary.

-   The model outputs a probability distribution for all possible words in the vocabulary.
-   Cross-Entropy Loss measures how different the model's predicted probability distribution is from the actual target (the one-hot encoded ground-truth word).
-   By minimizing this loss, we train the model to assign a higher probability to the correct word at each step, making the generated captions more accurate.
