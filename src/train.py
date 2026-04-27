import os
import pandas as pd
import pickle
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.preprocess import clean_text

nltk.download('stopwords')

def train_model():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(BASE_DIR, 'data', 'dataset.csv')
    model_dir = os.path.join(BASE_DIR, 'models')

    # Load data
    df = pd.read_csv(data_path)

    # Fix issues (important for real datasets)
    df['text'] = df['text'].astype(str)
    df = df.dropna(subset=['text', 'sentiment'])

    # Preprocess
    df['clean_text'] = df['text'].apply(clean_text)

    X = df['clean_text']
    y = df['sentiment']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Vectorize
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train_vec, y_train)

    # Evaluate
    y_pred = model.predict(X_test_vec)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save models
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, 'model.pkl')
    vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')

    pickle.dump(model, open(model_path, 'wb'))
    pickle.dump(vectorizer, open(vectorizer_path, 'wb'))

    print("Model saved successfully!")

    return model, vectorizer


# Run manually
if __name__ == "__main__":
    train_model()