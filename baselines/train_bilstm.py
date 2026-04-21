import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import re
import os
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s\.\,\!\?\'\-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

def tokenize(text):
    return text.split()

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=128):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.vocab = vocab
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        tokens = tokenize(self.texts[idx])[:self.max_len]
        ids = [self.vocab.get(t, 1) for t in tokens]
        if len(ids) < self.max_len:
            ids += [0] * (self.max_len - len(ids))
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes, num_layers=2, dropout=0.3):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)
    def forward(self, input_ids):
        embedded = self.dropout(self.embedding(input_ids))
        lstm_out, (hidden, _) = self.lstm(embedded)
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        hidden_cat = self.dropout(hidden_cat)
        return self.classifier(hidden_cat)

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    df = pd.read_csv("mental_health.csv")
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 10].reset_index(drop=True)
    train_val, test = train_test_split(df, test_size=0.15, random_state=42, stratify=df["label"])
    train, val = train_test_split(train_val, test_size=0.176, random_state=42, stratify=train_val["label"])
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
