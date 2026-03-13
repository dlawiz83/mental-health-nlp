import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
import wandb
import numpy as np
import argparse

from model import MentalHealthClassifier
from preprocess import build_dataset

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


class MentalHealthDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }


def train_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss, all_preds, all_labels = 0, [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / len(loader), f1_score(all_labels, all_preds, average="macro")


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / len(loader), f1_score(all_labels, all_preds, average="macro")


def main(lr=2e-5, epochs=4, batch_size=16, dropout=0.3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project="mental-health-nlp",
        config={"lr": lr, "epochs": epochs,
                "batch_size": batch_size, "dropout": dropout,
                "dataset": "real-reddit-27924-samples",
                "num_classes": 2}
    )

    train, val, test = build_dataset()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_loader = DataLoader(
        MentalHealthDataset(train["clean_text"], train["label"], tokenizer),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        MentalHealthDataset(val["clean_text"], val["label"], tokenizer),
        batch_size=batch_size, shuffle=False
    )

    model = MentalHealthClassifier(num_classes=2, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(train_loader) * epochs // 10,
        num_training_steps=len(train_loader) * epochs
    )

    best_val_f1 = 0
    patience_counter = 0
    patience = 3

    for epoch in range(epochs):
        train_loss, train_f1 = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        val_loss, val_f1 = eval_epoch(model, val_loader, criterion, device)

        wandb.log({"epoch": epoch+1, "train_loss": train_loss,
                   "train_f1": train_f1, "val_loss": val_loss, "val_f1": val_f1})

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | "
              f"Train F1: {train_f1:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            checkpoint = f"checkpoint_epoch{epoch+1}_valf1_{val_f1:.4f}.pt"
            torch.save(model.state_dict(), checkpoint)
            print(f"Saved: {checkpoint}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.3)
    args = parser.parse_args()
    main(args.lr, args.epochs, args.batch_size, args.dropout)
