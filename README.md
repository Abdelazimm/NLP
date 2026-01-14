# SMS Spam Detection – NLP Classification Project

## Project Overview
This project focuses on SMS spam detection using Natural Language Processing (NLP).
The objective is to evaluate whether modern transformer-based models are necessary for
short-text classification, or if traditional statistical models can still deliver strong performance.

The project compares:
- A baseline majority-class classifier
- TF-IDF with Multinomial Naive Bayes
- A fine-tuned DistilBERT transformer model

The comparison emphasizes classification performance, recall on spam messages,
and computational efficiency.

---

## Objectives
- Compare traditional NLP methods with transformer-based models
- Analyze trade-offs between performance and computational cost
- Evaluate model behavior on an imbalanced real-world dataset
- Provide practical insights for deployment scenarios

---

## Dataset
- Dataset: UCI SMS Spam Collection  
- Total samples: 5,572 SMS messages  
  - Ham: 4,825 (≈ 87%)
  - Spam: 747 (≈ 13%)
- Language: English  
- Task: Binary classification (Spam vs Ham)

---

## Methodology

### Baseline Model
- Majority-class classifier using DummyClassifier
- Demonstrates why accuracy alone is misleading for imbalanced datasets

### Statistical Model
- TF-IDF vectorization (unigrams + bigrams, max 5000 features)
- Multinomial Naive Bayes classifier
- Preprocessing steps:
  - Lowercasing
  - Stopword removal
  - Punctuation removal
  - Stemming

### Deep Learning Model
- DistilBERT fine-tuned for sequence classification
- Raw text used without preprocessing
- Implemented using PyTorch with GPU acceleration

---

## Evaluation Metrics
- Precision
- Recall
- F1-score (primary metric)
- Confusion Matrix
- Accuracy (reported but not relied upon alone)

---

## Results Summary

| Model | Accuracy | Precision | Recall | F1-score |
|------|----------|-----------|--------|----------|
| Baseline (Majority Class) | 0.866 | 0.000 | 0.000 | 0.000 |
| Naive Bayes + TF-IDF | 0.972 | 1.000 | 0.792 | 0.884 |
| DistilBERT | 0.990 | 0.986 | 0.940 | 0.962 |

---

## Key Insights
- DistilBERT achieved the best overall performance, especially in recall.
- Naive Bayes performed strongly despite its simplicity.
- Accuracy alone is unreliable for imbalanced datasets.
- Computational cost matters:
  - Naive Bayes trains in milliseconds
  - DistilBERT requires GPU acceleration

---

## How to Run

```bash
pip install numpy pandas scikit-learn nltk torch transformers


Open and run:

SMS_Spam_Filtering.ipynb


Pretrained models are downloaded automatically during runtime
and are not stored in the repository.

References

Almeida et al. (2011) – SMS Spam Collection Dataset

Devlin et al. (2018) – BERT

Sanh et al. (2019) – DistilBERT

Scikit-learn Documentation

UCI Machine Learning Repository

Author

Aly Abdelazim
NLP and Machine Learning Projec
```bash
pip install numpy pandas scikit-learn nltk torch transformers
