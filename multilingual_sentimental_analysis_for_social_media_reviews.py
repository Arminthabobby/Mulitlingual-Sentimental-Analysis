


# Install required libraries (uncomment the next three lines if these libraries are not installed)
# !pip install transformers
# !pip install sentencepiece
# !pip install torch

import torch
from transformers import XLMRobertaTokenizer, XLMRobertaModel
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np

# Define the XLM-R model and tokenizer
model_name = "xlm-roberta-base"
model = XLMRobertaModel.from_pretrained(model_name)
tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)

# Initialize the SVM classifier
svm_classifier = SVC(probability=True)
scaler = StandardScaler()  # For normalizing the features

# Corrected training data and labels
train_texts = [
    "I don't love this product!",  # Negative
    "This is a great product.",    # Positive
    "The product is amazing",      # Positive
    "Je suis très heureux d'apprendre cela.",  # Positive
    "Esto es fantástico y emocionante.",       # Positive
    "This is the worst thing I've ever used.", # Negative
    "Terrible experience, will never buy again.", # Negative
]

train_labels = [0, 1, 1, 1, 1, 0, 0]  # 0 for negative, 1 for positive

# Tokenize and encode the training data
train_inputs = tokenizer(train_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)

# Extract hidden states from the XLM-R model
with torch.no_grad():
    train_hidden_states = model(**train_inputs).last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)

# Average pooling to get sentence-level embeddings
train_embeddings = train_hidden_states.mean(dim=1).numpy()  # Shape: (batch_size, hidden_size)

# Normalize the embeddings
train_embeddings = scaler.fit_transform(train_embeddings)

# Fit the SVM classifier with normalized embeddings
svm_classifier.fit(train_embeddings, train_labels)

# Multilingual sentiment analysis function
def multilingual_sentiment_analysis(texts):
    # Tokenize and encode the input text
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)

    # Extract hidden states from the XLM-R model
    with torch.no_grad():
        hidden_states = model(**inputs).last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)

    # Average pooling to get sentence-level embeddings
    embeddings = hidden_states.mean(dim=1).numpy()  # Shape: (batch_size, hidden_size)

    # Normalize the embeddings
    embeddings = scaler.transform(embeddings)

    # Classify the sentiment using the pre-trained SVM classifier
    svm_predictions = svm_classifier.predict(embeddings)

    return svm_predictions

# Example statements for sentiment analysis
statements = [
    "I don't love this product!",
    "The product is amazing",
    "Je suis très heureux d'apprendre cela.",
    "Esto es fantástico y emocionante.",
    "This is the worst thing I've ever used.",
    "Terrible experience, will never buy again.",
]

# Perform sentiment analysis
sentiment_predictions = multilingual_sentiment_analysis(statements)

# Display the sentiment predictions (0 for negative, 1 for positive)
for statement, sentiment in zip(statements, sentiment_predictions):
    print(f"Statement: {statement}\nSentiment: {'Positive' if sentiment == 1 else 'Negative'}\n")