# 📩 SMS Spam Detector

A web-based application built with **Streamlit** that uses a trained machine learning model to detect whether an SMS message is **Spam** or **Not Spam**.

&#x20;

---

## 🚀 Features

- 🔍 Classifies SMS messages as **Spam** or **Not Spam**
- ✅ Displays confidence scores for predictions
- 📆 Built using `scikit-learn`, `Streamlit`, and `TF-IDF Vectorization`
- 💡 Easy-to-use web interface

---

## 🧠 Model Overview

The model was trained on a dataset of labeled SMS messages using a pipeline consisting of:

- Text Preprocessing (cleaning, stopword removal, stemming)
- TF-IDF Vectorization
- Classification using algorithms like Naive Bayes / Logistic Regression (adjustable)

All preprocessing and model training steps can be found in the [Jupyter Notebook](sms_spam.ipynb).

---

## 📂 Project Structure

```
.
├── app.py                    # Streamlit web app
├── sms_spam.ipynb           # Model training and evaluation notebook
├── spam_detector_model.pkl  # Trained ML model (pickle)
├── tfidf_vectorizer.pkl     # TF-IDF vectorizer (pickle)
├── requirements.txt         # Project dependencies
└── README.md                # You're here
```

---

## 💻 How to Run Locally

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/sms-spam-detector.git
cd sms-spam-detector
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

---

## 💪 Dependencies

Make sure you have the following installed (see `requirements.txt`):

- `streamlit`
- `pandas`
- `scikit-learn`
- `nltk`

To install NLTK stopwords and stemmers:

```python
import nltk
nltk.download('stopwords')
```

---

## 📊 Example

Enter a message like:

```
Congratulations! You’ve won a $1000 Walmart gift card. Go to http://bit.ly/123456 to claim now.
```

The model might return:

```
Prediction: Spam
Confidence: 98.34%
```

---

## 📌 Notes

- Ensure that `spam_detector_model.pkl` and `tfidf_vectorizer.pkl` are in the same directory as `app.py`.
- You can retrain the model or test different algorithms using the notebook.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

