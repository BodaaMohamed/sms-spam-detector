import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import sys

# Load the saved model and vectorizer
def load_model():
    with open('spam_detector_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

def load_vectorizer():
    with open('tfidf_vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
    return vectorizer

def predict_spam(message, model, vectorizer):
    """
    Predict if a message is spam or not
    
    Args:
        message (str): Input message to classify
        model: Trained machine learning model
        vectorizer: TF-IDF vectorizer
        
    Returns:
        tuple: (prediction, confidence_score)
    """
    if not isinstance(message, str):
        print(f"Error: Message is not a string. Type: {type(message)}", file=sys.stderr)
        return ("Error", None)
    
    try:
        # Transform the message using the original vectorizer
        input_transformed = vectorizer.transform([message])
        
        # Get the classifier from the model
        classifier = model.named_steps['clf']
        
        # Make prediction
        prediction = classifier.predict(input_transformed)[0]
        
        # Calculate confidence score
        confidence_score = None
        
        # Try to get probabilities if available
        if hasattr(classifier, 'predict_proba'):
            proba = classifier.predict_proba(input_transformed)[0]
            confidence_score = max(proba)
        
        # For binary classifiers that don't have predict_proba
        elif hasattr(classifier, 'decision_function'):
            decision = classifier.decision_function(input_transformed)[0]
            confidence_score = abs(decision) / (1 + abs(decision))  # Normalize to [0,1]
        
        return ('Spam' if prediction == 1 else 'Not Spam', confidence_score)
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}", file=sys.stderr)
        return ("Error", None)

# Streamlit App
st.title("SMS Spam Detector")

# Load model and vectorizer
model = load_model()
vectorizer = load_vectorizer()

# User Input
user_input = st.text_area("Enter an SMS message:", "")

if st.button("Predict"):
    if user_input.strip():
        # Make prediction
        result, confidence = predict_spam(user_input, model, vectorizer)
        
        if result == "Error":
            st.error("An error occurred during prediction. Please try again.")
        else:
            # Display result
            st.write(f"### Prediction: {result}")
            
            # Display confidence score if available
            if confidence is not None:
                st.write(f"Confidence: {confidence:.2%}")
            else:
                st.write("Confidence score not available for this model")
           
    else:
        st.warning("Please enter a message.")