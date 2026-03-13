import streamlit as st
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from huggingface_hub import hf_hub_download
import re

# Page config
st.set_page_config(
    page_title="Mental Health Signal Detection",
    page_icon="",
    layout="centered"
)

# Model definition
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
    text = re.sub(r"[^a-zA-Z0-9\s\.\,\!\?]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

@st.cache_resource
def load_model():
    device = torch.device("cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = MentalHealthClassifier(num_classes=2, dropout=0.3)
    checkpoint_path = hf_hub_download(
        repo_id="Delaviz/mental-health-bert",
        filename="checkpoint_real_epoch4_valf1_0.9641.pt"
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model, tokenizer, device

def predict(text, model, tokenizer, device):
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
    probs = torch.softmax(logits, dim=1).squeeze()
    pred = torch.argmax(probs).item()
    return pred, probs[pred].item(), probs.tolist()

# UI
st.title("Mental Health Signal Detection")
st.markdown(
    "This tool uses a fine-tuned BERT model trained on 27,924 real Reddit posts "
    "to detect potential mental health signals in text. "
    "**Not a clinical tool. For research purposes only.**"
)

st.divider()

example_texts = [
    "I have been feeling so empty and hopeless lately. Nothing brings me joy anymore.",
    "Just got back from the grocery store. Made pasta for dinner, it was pretty good.",
    "I cant stop worrying about everything. My chest feels tight all the time.",
]

st.markdown("**Try an example or type your own:**")
col1, col2, col3 = st.columns(3)
if col1.button("Example 1"):
    st.session_state.input_text = example_texts[0]
if col2.button("Example 2"):
    st.session_state.input_text = example_texts[1]
if col3.button("Example 3"):
    st.session_state.input_text = example_texts[2]

user_input = st.text_area(
    "Enter text to analyze:",
    value=st.session_state.get("input_text", ""),
    height=150,
    placeholder="Type or paste a Reddit-style post here..."
)

if st.button("Analyze", type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Loading model and analyzing..."):
            model, tokenizer, device = load_model()
            pred, confidence, probs = predict(user_input, model, tokenizer, device)

        labels = ["Neutral", "Mental Health Signal"]
        colors = ["#2ecc71", "#e74c3c"]
        label = labels[pred]
        color = colors[pred]

        st.divider()
        st.markdown(f"### Result: :{color}[{label}]")

        col1, col2 = st.columns(2)
        col1.metric("Prediction", label)
        col2.metric("Confidence", f"{confidence*100:.1f}%")

        st.markdown("**Confidence scores:**")
        for i, (lbl, prob) in enumerate(zip(labels, probs)):
            st.progress(prob, text=f"{lbl}: {prob*100:.1f}%")

        st.divider()
        st.caption(
            "This model was trained on English Reddit posts and may not generalize "
            "to all demographics or writing styles. It should never be used for "
            "clinical diagnosis or intervention."
        )

st.divider()
st.markdown(
    "[GitHub](https://github.com/dlawiz83/mental-health-nlp) | "
    "[Model on HF](https://huggingface.co/Delaviz/mental-health-bert)"
)
