#  Sentiment Analysis Multilingue (FR + EN)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange)
![Gradio](https://img.shields.io/badge/Demo-Gradio-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A multilingual sentiment analysis system fine-tuned on French and English reviews using DistilBERT. The model classifies text as positive or negative and is deployed as an interactive web application.

 **[Live Demo on HuggingFace Spaces](https://huggingface.co/spaces/alioubguindo/sentiment-analysis-demo)**

---

##  Project Overview

This project compares two approaches to sentiment analysis:

- **Baseline**: TF-IDF + Logistic Regression (classical NLP)
- **Advanced**: Fine-tuned DistilBERT multilingual (deep learning)

The goal is to show that transformer-based models understand linguistic nuances like negation ("not bad" = positive) that bag-of-words models cannot capture.

---

##  Results

| Model | AUC-ROC | Accuracy |
|-------|---------|----------|
| TF-IDF + Logistic Regression | 0.897 | 0.805 |
| DistilBERT multilingual | 0.883 | 0.804 |

> **Note**: BERT was trained on only 2,000 examples due to Colab GPU constraints. With more data, BERT consistently outperforms TF-IDF — especially on nuanced sentences.
''' sentiment-analysis-multilingual/
├── notebooks/
│   ├── 01_EDA_Sentiment.ipynb        # Exploratory Data Analysis
│   └── 02_Modeling.ipynb             # TF-IDF baseline + BERT fine-tuning
├── app/
│   └── app.py                        # Gradio web application
├── requirements.txt
└── README.md
'''
---

## 🗂️ Dataset

- **English**: [Stanford SST-2](https://huggingface.co/datasets/stanfordnlp/sst2) — 5,000 movie review sentences
- **French**: [Allociné](https://huggingface.co/datasets/allocine) — 5,000 French movie reviews
- **Total**: 10,000 samples | 52% positive / 48% negative

---

## 🧠 Model

- **Architecture**: `distilbert-base-multilingual-cased`
- **Fine-tuning**: 3 epochs, batch size 16, learning rate warmup
- **Languages**: French + English (104 languages supported)
- **Hosted on**: [HuggingFace Hub](https://huggingface.co/alioubguindo/sentiment-multilingual-distilbert)

---

## 🚀 Run Locally

```bash
git clone https://github.com/alioubguindo/sentiment-analysis-multilingual
cd sentiment-analysis-multilingual
pip install -r requirements.txt
python app/app.py
```

---

## 💡 Key Learnings

- TF-IDF treats text as a bag of words — it cannot understand context or negation
- BERT reads the full sentence bidirectionally, capturing nuance
- Fine-tuning a pre-trained model requires far less data than training from scratch
- Multilingual models enable cross-language applications with a single model

---

## 🛠️ Tech Stack

`Python` `HuggingFace Transformers` `PyTorch` `scikit-learn` `Gradio` `Pandas` `Google Colab`

---

## 👤 Author

**Badara Aliou Guindo** — Master's student in Data Science & AI  
[GitHub](https://github.com/badaraaliouguindo)
• [HuggingFace](https://huggingface.co/alioubguindo)

---

## 📁 Project Structure
