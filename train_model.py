import pandas as pd
import re
import string
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ---------------- LOAD DATA ----------------
df = pd.read_csv("D:/Project/tweets.csv")

# ---------------- CLEANING FUNCTION ----------------
def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

df["tweet"] = df["tweet"].apply(clean_text)

# ---------------- SPLIT DATA ----------------
X_train, X_test, y_train, y_test = train_test_split(
    df["tweet"], df["sentiment"], test_size=0.2, random_state=42
)

# ---------------- VECTORIZER ----------------
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---------------- MODEL ----------------
model = LogisticRegression(max_iter=300)
model.fit(X_train_vec, y_train)

# ---------------- EVALUATION ----------------
y_pred = model.predict(X_test_vec)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))

# ---------------- SAVE MODEL ----------------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\nModel trained and saved successfully!")

# ==================================================
# 🔥 USER INPUT TESTING SECTION
# ==================================================

def predict_sentiment(text):
    text = clean_text(text)
    vector = vectorizer.transform([text])
    return model.predict(vector)[0]

print("\n🔍 You can now test your model!")

while True:
    user_input = input("\nEnter a tweet (or type 'exit' to quit): ")

    if user_input.lower() == 'exit':
        print("Exiting...")
        break

    prediction = predict_sentiment(user_input)
    print("Predicted Sentiment:", prediction)