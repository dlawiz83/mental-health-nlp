import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def clean_text(text):
    """Clean raw Reddit post text."""
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s\.\,\!\?]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.lower()
    return text


def build_dataset(csv_path="mental_health.csv"):
    """Build dataset from real Reddit mental health corpus.
    
    Dataset: https://www.kaggle.com/datasets/reihanenamdari/mental-health-corpus
    27,924 real Reddit posts (after cleaning), binary classification:
        0 = neutral
        1 = mental_health
    Split: 70% train / 15% val / 15% test (stratified)
    """
    np.random.seed(42)

    df = pd.read_csv(csv_path)
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 10].reset_index(drop=True)

    train_val, test = train_test_split(
        df, test_size=0.15, random_state=42, stratify=df["label"]
    )
    train, val = train_test_split(
        train_val, test_size=0.176, random_state=42, stratify=train_val["label"]
    )

    return train, val, test


def filter_short_posts(df, min_words=5):
    """Remove posts shorter than min_words words."""
    df = df[df["clean_text"].apply(lambda x: len(x.split()) >= min_words)]
    return df.reset_index(drop=True)


if __name__ == "__main__":
    train, val, test = build_dataset()
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print(f"Label distribution:\n{train['label'].value_counts()}")
