import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def clean_text(text):
    """Clean raw Reddit post text."""
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # Remove Reddit formatting
    text = re.sub(r"\*\*|__|~~|>>|&gt;", "", text)
    # Remove special characters
    text = re.sub(r"[^a-zA-Z0-9\s\.\,\!\?\'\"]", "", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Lowercase
    text = text.lower()
    return text

def build_dataset():
    """Build the mental health dataset from base posts."""
    np.random.seed(42)

    depression_posts = [
        "I have been feeling so empty and hopeless lately. Nothing brings me joy anymore.",
        "I cant get out of bed. Everything feels pointless and I dont know why I bother.",
        "I have been crying every day for weeks. I feel completely alone and worthless.",
        "I lost interest in everything I used to love. I feel numb all the time.",
        "Nobody understands how dark everything feels right now. I am exhausted from pretending.",
        "I feel like a burden to everyone around me. Life feels meaningless.",
        "I havent eaten properly in days. I just dont see the point in anything.",
        "The sadness never goes away. I wake up every morning feeling heavy and hopeless.",
        "I used to love painting but now I cant even look at my brushes. Everything is grey.",
        "I feel completely disconnected from life. Like I am watching everything through glass.",
    ]

    anxiety_posts = [
        "My heart is racing again and I dont know how to make it stop. I am so scared.",
        "I keep thinking something terrible is going to happen. I cant relax at all.",
        "I had another panic attack at work today. I am terrified it will happen again.",
        "My mind wont stop racing. I have been awake for hours just worrying about everything.",
        "I am so overwhelmed. Even small tasks feel impossible and I am constantly on edge.",
        "I canceled plans again because I was too anxious to leave the house.",
        "I keep checking the door locks over and over. I know it is irrational but I cant stop.",
        "The anxiety is getting worse. I feel like I am always waiting for something bad to happen.",
        "I cant concentrate on anything because I am so worried about things that might happen.",
        "My chest feels tight all the time. I am constantly nervous and I dont know why.",
    ]

    neutral_posts = [
        "Just got back from the grocery store. Made pasta for dinner tonight it was pretty good.",
        "Has anyone tried the new coffee place downtown. Looking for recommendations.",
        "Finished reading a great book this weekend. Looking for similar recommendations.",
        "The weather has been really nice lately. Went for a walk in the park this morning.",
        "Anyone have tips for growing tomatoes at home. Starting a small garden.",
        "Watched a great documentary last night about ocean life. Highly recommend it.",
        "Just adopted a puppy. Any advice for first time dog owners would be appreciated.",
        "Started learning guitar last month. It is challenging but really fun so far.",
        "Made homemade bread for the first time today. Turned out better than expected.",
        "Looking for book club recommendations. I enjoy mystery and historical fiction.",
    ]

    def expand_posts(posts, label, n=300):
        expanded = []
        for i in range(n):
            base = posts[i % len(posts)]
            expanded.append({"text": base, "label": label})
        return expanded

    data = (
        expand_posts(depression_posts, "depression") +
        expand_posts(anxiety_posts, "anxiety") +
        expand_posts(neutral_posts, "neutral")
    )

    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df["clean_text"] = df["text"].apply(clean_text)

    le = LabelEncoder()
    df["label_encoded"] = le.fit_transform(df["label"])

    train_val, test = train_test_split(
        df, test_size=0.15, random_state=42, stratify=df["label_encoded"]
    )
    train, val = train_test_split(
        train_val, test_size=0.176, random_state=42, stratify=train_val["label_encoded"]
    )

    return train, val, test, le

if __name__ == "__main__":
    train, val, test, le = build_dataset()
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print(f"Labels: {list(le.classes_)}")

def filter_short_posts(df, min_words=5):
    """Remove posts shorter than min_words words."""
    df = df[df["clean_text"].apply(lambda x: len(x.split()) >= min_words)]
    return df.reset_index(drop=True)
