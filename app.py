import gradio as gr
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from huggingface_hub import hf_hub_download
import re

class MentalHealthClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3):
        super(MentalHealthClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.pooler_output
        cls_output = self.dropout(cls_output)
        return self.classifier(cls_output)

def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s\.\,\!\?\'\-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

print("Loading model...")
device = torch.device("cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = MentalHealthClassifier(num_classes=2, dropout=0.3)
checkpoint_path = hf_hub_download(
    repo_id="Delaviz/mental-health-bert",
    filename="checkpoint_real_epoch4_valf1_0.9641.pt"
)
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
model.eval()
print("Model loaded!")

def predict(text):
    if not text.strip():
        return {"Neutral": 0.0, "Mental Health Signal": 0.0}
    cleaned = clean_text(text)
    encoding = tokenizer(
        cleaned,
        max_length=128,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(
            encoding["input_ids"].to(device),
            encoding["attention_mask"].to(device)
        )
    probs = torch.softmax(logits, dim=1).squeeze().tolist()
    return {"Neutral": round(probs[0], 4), "Mental Health Signal": round(probs[1], 4)}

examples = [
    "I have been feeling so empty and hopeless lately. Nothing brings me joy anymore.",
    "Just got back from the grocery store. Made pasta for dinner, it was pretty good.",
    "I cant stop worrying about everything. My chest feels tight all the time.",
    "Started learning guitar last month. It is challenging but really fun so far.",
    "I feel like a burden to everyone around me. Life feels completely meaningless.",
]

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        lines=5,
        placeholder="Type or paste a Reddit-style post here...",
        label="Input Text"
    ),
    outputs=gr.Label(
        num_top_classes=2,
        label="Prediction"
    ),
    title="Mental Health Signal Detection",
    description="""Fine-tuned BERT model trained on 27,924 real Reddit posts to detect mental health signals in text.

Not a clinical tool. For research purposes only.

Model achieves 96% macro F1 on held-out test set of 4,189 posts.

Built by Ayesha Dawodi | GitHub: https://github.com/dlawiz83/mental-health-nlp""",
    examples=examples,
    theme=gr.themes.Soft()
)

demo.launch()
