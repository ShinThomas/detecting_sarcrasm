# Sarcasm Detection in News Headlines

## Overview
This project investigates sarcasm detection in news headlines using both classical machine learning models and transformer-based models. We compare how different text representations -- including bag-of-words, TF-IDF, engineered features, and contextual embeddings -- impact model performance.

The goal is to evaluate which approaches are most effective at capturing the subtle semantic and contextual cues required for sarcasm detection.

---

## Dataset
We use a publicly available dataset of news headlines labeled as sarcastic or non-sarcastic.

- Source: Kaggle (News Headlines Dataset for Sarcasm Detection)
- Format: JSON → converted to CSV for processing
- Size: ~26,000 headlines

---

## Project Structure
detecting_sarcasm/
│
├── data/
│ ├── raw/ # Original dataset (JSON)
│ ├── processed/ # Cleaned datasets and model outputs
│ │ ├── sarcasm_cleaned.csv
│ │ ├── sarcasm_raw.csv
│ │ ├── bert_eval_results.csv
│ │ ├── bert_predictions.csv
│ │
│ └── splits/ # Train/test splits
│ ├── X_train.csv
│ ├── X_test.csv
│ ├── y_train.csv
│ ├── y_test.csv
│
├── notebooks/
│ ├── eda.ipynb # Data cleaning and exploratory analysis
│ ├── bert_model.ipynb # DistilBERT training and evaluation
│ ├── bert_model.ipynb - Colab.pdf
│ ├── model_comparison.ipynb # Results, visualization, and error analysis
│
└── README.md


---

## Models Used

### Classical Models
- Naive Bayes (CountVectorizer)
- Logistic Regression (TF-IDF)
- Logistic Regression (TF-IDF + engineered features)

### Engineered Features
- Sentiment scores (VADER)
- Headline length
- Punctuation usage

### Transformer Model
- DistilBERT (fine-tuned for binary classification)

---

## How to Run the Project

The easiest way to view the project is through the notebooks:

1. All notebooks are able to run on their own except bert_model.ipynb which must be ran in Google Colab with GPU support.


(Locally)
#### Requirements
Install dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn transformers torch
