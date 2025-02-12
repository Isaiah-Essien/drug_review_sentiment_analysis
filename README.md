# Sentiment Analysis of Drug Reviews
![Predictions on LSTM](src\image.png)
## Overview

This project performs **sentiment analysis** on a dataset of drug reviews. The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/mohamedabdelwahabali/drugreview) and contains reviews of different drugs, along with user ratings and sentiments.

The notebook follows a structured pipeline, including:
- **Data Cleaning**
- **Exploratory Data Analysis (EDA)**
- **Feature Engineering**
- **Model Training** using:
  - Support Vector Machine (SVM)
  - Long Short-Term Memory (LSTM)
  - Transformer-based models (e.g., BERT)

## Dataset

The dataset is downloaded from **Google Drive** for ease of access. It contains the following key fields:
- `review` - The text review of the drug.
- `rating` - The numerical rating (1-10) given by users.
- `sentiment` - The derived sentiment from the rating (positive, neutral, or negative).
- `drug_name` - Name of the drug.

---

## Project Workflow

### 1. Data Cleaning

Before performing analysis, the dataset is **cleaned and preprocessed** by:
- Removing missing values.
- Handling duplicate reviews.
- Correcting common misspellings.
- Converting ratings into sentiment categories.

### 2. Exploratory Data Analysis (EDA)

EDA includes **statistical and visual analysis** of the dataset:

#### - **Language Analysis**
- Unique vocabulary count.
- Term frequency analysis.
- Most frequent **n-grams** (single words, bigrams, trigrams).
- Class balance (distribution of positive, neutral, and negative sentiments).

#### - **Review Counts & Length Analysis**
- Distribution of review lengths.
- Analysis of average review length across sentiment classes.

#### - **Missing Values Handling**
- Checking for missing values and handling them appropriately.

#### - **Unique Ratings Analysis**
- Distribution of unique ratings before transforming them into sentiments.

### 3. Model Training

Three different models are used to classify sentiment:

#### 1️⃣ **Support Vector Machine (SVM)**
- Traditional machine learning approach.
- Requires text vectorization (TF-IDF or Word Embeddings).
- Trained using Scikit-learn.

#### 2️⃣ **Long Short-Term Memory (LSTM)**
- A type of recurrent neural network (RNN).
- Can capture **sequential dependencies** in text.
- Implemented using **TensorFlow/Keras**.

#### 3️⃣ **Transformer (e.g., BERT)**
- Pretrained **Hugging Face Transformer** is fine-tuned on the dataset.
- Converts text to embeddings using tokenization.
- Achieves high accuracy due to context-aware learning.

---

## Implementation Steps

### Data Preparation
- Convert dataset from **Pandas DataFrame** to **Hugging Face Dataset**.
- Tokenize review text using **Transformers tokenizer**.

### Training & Evaluation
![Test performace of LSTM](src\test_per_lstm.png)
- Train models on the **preprocessed dataset**.
- Evaluate performance using **accuracy, precision, recall, and F1-score**.

---

## Results & Performance Comparison

- The **Transformer-based model** (e.g., BERT) performs **best** with the highest accuracy.
- LSTM performs **moderately well** but requires more computational power.
- SVM, while effective, struggles with **complex linguistic patterns**.

### Performance Metrics:

| Model | Accuracy | Precision | Recall | F1-Score |
|--------|----------|----------|--------|----------|
| SVM | ~78% | Moderate | Moderate | Moderate |
| LSTM | ~84% | High | High | High |
| Transformer | ~90%+ | Very High | Very High | Very High |

---

## How to Use

### Installation

Ensure you have the necessary libraries installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow torch transformers datasets
