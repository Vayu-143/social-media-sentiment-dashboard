import streamlit as st
import pickle
import pandas as pd
import os

# =========================
# PATH SETUP
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = os.path.join(BASE_DIR, 'data', 'dataset.csv')
model_path = os.path.join(BASE_DIR, 'models', 'model.pkl')
vectorizer_path = os.path.join(BASE_DIR, 'models', 'vectorizer.pkl')
output_path = os.path.join(BASE_DIR, 'outputs', 'predictions.csv')

# =========================
# LOAD MODEL
# =========================

model = pickle.load(open(model_path, 'rb'))
vectorizer = pickle.load(open(vectorizer_path, 'rb'))

# =========================
# UI
# =========================

st.set_page_config(page_title="Sentiment Dashboard", layout="wide")

st.title("📊 Social Media Sentiment Analysis Dashboard")

# =========================
# INPUT
# =========================

st.subheader("Analyze New Post")

user_input = st.text_area("Enter text")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        vec = vectorizer.transform([user_input])
        result = model.predict(vec)[0]

        if result == "positive":
            st.success("😊 Positive")
        elif result == "negative":
            st.error("😡 Negative")
        else:
            st.warning("😐 Neutral")

        # Save prediction log
        os.makedirs('outputs', exist_ok=True)
        new_data = pd.DataFrame([[user_input, result]], columns=['text', 'sentiment'])
        new_data.to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False)

# =========================
# DATASET
# =========================

df = pd.read_csv(data_path)

df['text'] = df['text'].fillna('')
df['sentiment'] = df['sentiment'].fillna('neutral')

col1, col2 = st.columns(2)

with col1:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(20))

with col2:
    st.subheader("Sentiment Distribution")
    st.bar_chart(df['sentiment'].value_counts())

# =========================
# TEXT LENGTH GRAPH
# =========================

st.subheader("Text Length Insight (Sample View)")

df['length'] = df['text'].apply(lambda x: len(str(x)))

sample_df = df.sample(500)

st.line_chart(sample_df['length'].reset_index(drop=True))