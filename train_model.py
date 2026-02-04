import pandas as pd
import re
import nltk
import pickle
import os

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report

# -----------------------------
# NLTK setup
# -----------------------------
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# Load dataset

# Try different separators
df=pd.read_csv("data/data.csv")



# Clean column names (VERY IMPORTANT)
df.columns = df.columns.str.strip().str.lower()

print("Available columns:", df.columns.tolist())

# Auto-detect columns
# Possible rating column names
rating_candidates = ['rating', 'review_rating', 'ratings', 'star_rating']
# Possible review text column names
text_candidates = ['review text', 'review_text', 'review', 'comment', 'comments', 'text']

rating_col = None
text_col = None

for col in rating_candidates:
    if col in df.columns:
        rating_col = col
        break

for col in text_candidates:
    if col in df.columns:
        text_col = col
        break

if rating_col is None or text_col is None:
    raise ValueError(
        f"Could not find required columns.\n"
        f"Found columns: {df.columns.tolist()}"
    )

print(f"Using rating column: {rating_col}")
print(f"Using review text column: {text_col}")

# Create sentiment label

df = df[df[rating_col] != 3]  # drop neutral reviews
df['sentiment'] = df[rating_col].apply(lambda x: 1 if x >= 4 else 0)


# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

df['clean_text'] = df[text_col].apply(clean_text)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'],
    df['sentiment'],
    test_size=0.2,
    random_state=42,
    stratify=df['sentiment']
)

# TF-IDF Vectorization

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)


# Model Training

 
model = LogisticRegression(max_iter=1000)

model.fit(X_train_vec, y_train)


# Evaluation
y_pred = model.predict(X_test_vec)
f1 = f1_score(y_test, y_pred)

print("\nF1 Score:", f1)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# Save model artifacts

os.makedirs("models", exist_ok=True)

with open("models/sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("\n✅ Model and vectorizer saved successfully!")
