import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s\.\,\!\?\'\-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

df = pd.read_csv("mental_health.csv")
df["clean_text"] = df["text"].apply(clean_text)
df = df[df["clean_text"].str.len() > 10].reset_index(drop=True)

train_val, test = train_test_split(df, test_size=0.15, random_state=42, stratify=df["label"])
train, val = train_test_split(train_val, test_size=0.176, random_state=42, stratify=train_val["label"])

X_train, y_train = train["clean_text"], train["label"]
X_test, y_test = test["clean_text"], test["label"]

tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1,2), sublinear_tf=True)
X_tr = tfidf.fit_transform(X_train)
X_te = tfidf.transform(X_test)

lr = LogisticRegression(C=1.0, max_iter=1000)
lr.fit(X_tr, y_train)
print("=== TF-IDF + Logistic Regression ===")
print(classification_report(y_test, lr.predict(X_te), target_names=["neutral", "mental_health"]))

svm = LinearSVC(C=1.0, max_iter=2000)
svm.fit(X_tr, y_train)
print("=== TF-IDF + SVM ===")
print(classification_report(y_test, svm.predict(X_te), target_names=["neutral", "mental_health"]))
