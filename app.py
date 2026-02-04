import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load model and vectorizer
with open("models/sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# NLTK setup
nltk.download("stopwords")
nltk.download("wordnet")
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# UI
st.set_page_config(page_title="Product Sentiment Analyzer", layout="centered")

st.title("📦 Product Sentiment Analyzer")
st.write("Enter your product feedback to calculate customer satisfaction percentage.")

review = st.text_area("Product Review:")

if st.button("Calculate Score"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        clean_review = clean_text(review)
        X = vectorizer.transform([clean_review])

        # Probability prediction
        prob = model.predict_proba(X)[0][1]   # Positive class probability
        score = round(prob * 100, 2)

        st.subheader(f"Customer Satisfaction: {score}%")
        st.progress(int(score))

        # Feedback message
        if score >= 75:
            st.success("Highly Positive Feedback!")
        elif score >= 50:
            st.info("Moderately Positive Feedback")
        elif score >= 35:
            st.warning("Neutral / Mixed Feedback")
        else:
            st.error("Negative Feedback")
