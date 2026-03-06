import torch
import torch.nn as nn
from transformers import BertModel

class MentalHealthClassifier(nn.Module):
    """BERT-based classifier for mental health signal detection."""

    def __init__(self, num_classes=3, dropout=0.3):
        super(MentalHealthClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(dropout)
        # BERT hidden size is 768, map to num_classes
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Use [CLS] token representation
        cls_output = outputs.pooler_output
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits

# Loss: CrossEntropyLoss | Regularization: Dropout(p=0.3)

# num_classes=3: anxiety, depression, neutral

# dropout=0.3 reduces overfitting on small datasets
