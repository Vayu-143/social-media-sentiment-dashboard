import sys
import os

# Fix import issue for Streamlit Cloud
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocess import clean_text
from src.train import train_model

# ---------------- PATH SETUP ---------------- #
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = os.path.join(BASE_DIR, 'data', 'dataset.csv')
model_path = os.path.join(BASE_DIR, 'models', 'model.pkl')
vectorizer_path = os.path.join(BASE_DIR, 'models', 'vectorizer.pkl')

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="Sentiment Dashboard", layout="wide")

st.title("📊 Social Media Sentiment Analysis Dashboard")

# ---------------- LOAD MODEL (CACHED) ---------------- #
@st.cache_resource
def load_model():
    try:
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            st.warning("⚠️ Model not found. Training model... (first run only)")
            return train_model()
        else:
            model = pickle.load(open(model_path, 'rb'))
            vectorizer = pickle.load(open(vectorizer_path, 'rb'))
            return model, vectorizer
    except Exception as e:
        st.error(f"Error loading/training model: {e}")
        st.stop()

model, vectorizer = load_model()

# ---------------- LOAD DATA ---------------- #
@st.cache_data
def load_data():
    df = pd.read_csv(data_path)
    df['text'] = df['text'].astype(str)
    df = df.dropna(subset=['text', 'sentiment'])
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# ---------------- INPUT ---------------- #
st.subheader("Analyze New Post")

user_input = st.text_area("Enter text")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]

        if prediction == "positive":
            st.success(f"Prediction: {prediction}")
        elif prediction == "negative":
            st.error(f"Prediction: {prediction}")
        else:
            st.warning(f"Prediction: {prediction}")

# ---------------- DATA PREVIEW ---------------- #
st.subheader("Dataset Preview")
st.dataframe(df.head())

# ---------------- SENTIMENT DISTRIBUTION ---------------- #
st.subheader("Sentiment Distribution")

fig, ax = plt.subplots()
sns.countplot(x='sentiment', data=df, ax=ax)
ax.set_title("Sentiment Count")
st.pyplot(fig)

# ---------------- TEXT LENGTH INSIGHT ---------------- #
st.subheader("Text Length Insight")

df['length'] = df['text'].apply(lambda x: len(str(x)))

fig2, ax2 = plt.subplots()
ax2.plot(df['length'].values[:1000])
ax2.set_title("Text Length Trend (Sample)")
st.pyplot(fig2)