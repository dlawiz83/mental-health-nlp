# Mental Health Signal Detection from Reddit Posts

A BERT-based NLP classifier that detects mental health signals (depression, anxiety, neutral) in text posts.

## Project Structure
```
mental-health-nlp/
├── src/
│   ├── preprocess.py    # Data cleaning and dataset building
│   ├── model.py         # BERT classifier architecture
│   └── train.py         # Training loop with W&B tracking
├── tests/
│   └── test_preprocess.py  # Unit tests
├── requirements.txt
└── README.md
```

## Setup
```bash
pip install -r requirements.txt
```

## Training
```bash
python src/train.py --lr 2e-5 --epochs 4 --batch_size 16 --dropout 0.3
```

## Results
| Split | Loss   | Macro F1 |
|-------|--------|----------|
| Train | 0.0069 | 1.0000   |
| Val   | 0.0045 | 1.0000   |
| Test  | 0.0045 | 1.0000   |

## Model
- **Architecture:** BERT-base-uncased + linear classification head
- **Loss:** Weighted Cross-Entropy
- **Optimizer:** AdamW (lr=2e-5)
- **Regularization:** Dropout (p=0.3)

## Dataset
- 900 samples across 3 classes (depression, anxiety, neutral)
- 70/15/15 train/val/test split (stratified)
- Source: Constructed from representative Reddit-style posts

## Experiment Tracking
Tracked with Weights & Biases:
https://wandb.ai/ayeshadawodi83/mental-health-nlp

## Ethical Considerations
- Dataset is limited to English, Western social media style
- Model should NOT be used for clinical diagnosis
- Bias: short posts harder to classify correctly
- License: MIT (compatible with Apache 2.0 BERT license)

## Author
Ayesha Dawodi
