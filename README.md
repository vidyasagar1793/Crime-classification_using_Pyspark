# Crime Classification System using PySpark

## Overview
This project develops an automated crime classification system to categorize crime descriptions into 39 predefined categories. Using the **San Francisco Crime Dataset** from Kaggle, the system leverages PySpark for efficient data processing and machine learning model training. The final model assists law enforcement by enabling resource allocation based on the nature of the crime.

---

## Features
- **Preprocessing:**
  - Tokenization using `RegexTokenizer`.
  - Stopword removal using `StopWordsRemover`.
  - Feature extraction via `CountVectorizer`, `TF-IDF`, and `Word2Vec`.

- **Modeling:**
  - Supervised machine learning models: Logistic Regression and Naive Bayes.
  - Multi-class text classification with 39 predefined categories.

- **Performance:**
  - Achieved 99.5% accuracy using Naive Bayes with TF-IDF features.
  - Comparison of feature extraction techniques: CountVectorizer, TF-IDF, Word2Vec.

---

## Dataset
- **Source:** [Kaggle San Francisco Crime Dataset](https://www.kaggle.com/c/sf-crime).
- **Data Description:** The dataset contains crime descriptions, their categories, and additional metadata.

---

## Preprocessing Steps
1. **Tokenization:** Used `RegexTokenizer` to split crime descriptions into tokens based on a regex pattern.
2. **Stopword Removal:** Removed common words (e.g., "the", "and") using `StopWordsRemover`.
3. **Feature Extraction:**
   - `CountVectorizer`: Generates a sparse representation of word counts.
   - `TF-IDF`: Calculates term frequency-inverse document frequency scores.
   - `Word2Vec`: Captures word semantics as dense vectors.
4. **Category Encoding:** Encoded crime categories as numeric labels using `StringIndexer`.

---

## Model Building and Evaluation
1. **Baseline Model:**
   - Logistic Regression with `CountVectorizer` features (Accuracy: 97.2%).

2. **Advanced Models:**
   - Naive Bayes with `CountVectorizer` (Accuracy: 99.3%).
   - Naive Bayes with `TF-IDF` (Accuracy: 99.5%).
   - Logistic Regression and Word2Vec (Accuracy: 90.7%).
