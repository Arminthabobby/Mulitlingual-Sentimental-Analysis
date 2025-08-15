!pip install datasets
import pandas as pd
import torch
import re
import numpy as np
from datasets import Dataset
from transformers import XLMRobertaTokenizer, XLMRobertaModel, XLMRobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from nltk.corpus import wordnet
import random
from torch.cuda.amp import autocast
import nltk
nltk.download('wordnet')
# -------------------------------
# 1️⃣ LOAD DATASET & PREPROCESSING
# -------------------------------
file_path = "/content/data.csv"
df = pd.read_csv(file_path, encoding="utf-8").dropna()

# Convert sentiment labels to numeric categories
df["sentiment"] = df["sentiment"].map({"1 star": 0, "2 stars": 0, "3 stars": 1, "4 stars": 2, "5 stars": 2})

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z0-9À-ÿ\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

df["tweet"] = df["tweet"].apply(clean_text)

# -------------------------------
# 2️⃣ DATA AUGMENTATION (Synonym Replacement)
# -------------------------------
def synonym_replacement(sentence, num_words=2):
    words = sentence.split()
    new_words = words.copy()
    random_words = random.sample(words, min(num_words, len(words)))
    for word in random_words:
        synonyms = wordnet.synsets(word)
        if synonyms:
            synonym = synonyms[0].lemmas()[0].name()
            new_words = [synonym if w == word else w for w in new_words]
    return " ".join(new_words)

df["tweet_augmented"] = df["tweet"].apply(synonym_replacement)
df_augmented = df[["tweet_augmented", "sentiment"]].rename(columns={"tweet_augmented": "tweet"})
df = pd.concat([df, df_augmented], ignore_index=True)

# Split data into train and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(df["tweet"].tolist(), df["sentiment"].tolist(), test_size=0.2, random_state=42)

# -------------------------------
# 3️⃣ TRAIN XLM-RoBERTa CLASSIFIER
# -------------------------------
model_name = "xlm-roberta-large"
tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
model_xlmr = XLMRobertaForSequenceClassification.from_pretrained(model_name, num_labels=3).to("cuda")

# Tokenization
def tokenize_function(examples):
    return tokenizer(examples, padding=True, truncation=True, max_length=256)

train_encodings = tokenize_function(train_texts)
test_encodings = tokenize_function(test_texts)

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_dataset = SentimentDataset(train_encodings, train_labels)
test_dataset = SentimentDataset(test_encodings, test_labels)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=10,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model_xlmr,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

trainer.train()
eval_results = trainer.evaluate()
accuracy_xlmr = eval_results.get("eval_accuracy", 0) * 100

# -------------------------------
# 4️⃣ TOKENIZATION & EMBEDDING EXTRACTION (Optimized)
# -------------------------------
model = XLMRobertaModel.from_pretrained(model_name).to("cuda")
model = torch.compile(model)

def get_embeddings(texts, batch_size=16):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=256).to("cuda")
        with torch.no_grad(), autocast():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

X_train = get_embeddings(train_texts, batch_size=16)
X_test = get_embeddings(test_texts, batch_size=16)

# -------------------------------
# 5️⃣ TRAIN XGBOOST CLASSIFIER
# -------------------------------
xgb = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=8, colsample_bytree=0.8, early_stopping_rounds=50)
xgb.fit(X_train, train_labels, eval_set=[(X_test, test_labels)], verbose=True)
y_pred_xgb = xgb.predict(X_test)
accuracy_xgb = accuracy_score(test_labels, y_pred_xgb) * 100

# -------------------------------
# 6️⃣ TRAIN LIGHTGBM CLASSIFIER
# -------------------------------
lgbm = LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=8)
lgbm.fit(X_train, train_labels)
y_pred_lgbm = lgbm.predict(X_test)
accuracy_lgbm = accuracy_score(test_labels, y_pred_lgbm) * 100

# -------------------------------
# 7️⃣ SAVE MODELS & PRINT ACCURACY
# -------------------------------
print(f"✅ Model fine-tuned and saved successfully!")
print(f"🎯 XLM-RoBERTa Accuracy: {accuracy_xlmr:.2f}%")
print(f"🎯 XGBoost Accuracy: {accuracy_xgb:.2f}%")
print(f"🎯 LightGBM Accuracy: {accuracy_lgbm:.2f}%")
final_accuracy = (accuracy_xgb + accuracy_lgbm ) / 2
print(f"🚀 Final Model Accuracy ( XGBoost + LightGBM): {final_accuracy:.2f}%")
