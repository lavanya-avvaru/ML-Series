# Spam Detection using NLP & Naive Bayes 📩

## 📌 Objective
Build a machine learning model to classify messages as **Spam** or **Ham (Not Spam)** using Multinomial Naive Bayes.

---

## 📊 Dataset
- SMS Spam Collection dataset (CSV)
- Columns:
  - `message`: Text message content
  - `label`: Spam or Ham
- Labels encoded as:
  - 0 → Ham
  - 1 → Spam

---

## 🔍 Steps Performed
- Loaded dataset from CSV file
- Performed data preprocessing
- Encoded target labels
- Split data into training and testing sets
- Converted text data into numerical form using TF-IDF
- Trained model using Multinomial Naive Bayes
- Evaluated model performance
- Tested model with custom input messages

---

## ⚙️ Tools & Libraries
- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorizer (NLP)

---

## 📈 Model Evaluation
- Accuracy Score
- Classification Report

---

## 💬 Sample Predictions
- "You won a free prize" → **Spam**
- "Let's meet tomorrow" → **Ham**

---

## 🚀 Conclusion
The model successfully classifies spam messages using NLP techniques. TF-IDF combined with Naive Bayes provides efficient and accurate text classification.

---
