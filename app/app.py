import streamlit as st
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Sentiment Dashboard", layout="wide")

# -----------------------------
# PATH SETUP (WORKS LOCAL + CLOUD)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = os.path.join(BASE_DIR, 'data', 'dataset.csv')
model_path = os.path.join(BASE_DIR, 'models', 'model.pkl')
vectorizer_path = os.path.join(BASE_DIR, 'models', 'vectorizer.pkl')

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(data_path)

# -----------------------------
# LOAD / TRAIN MODEL
# -----------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(model_path):
        st.warning("⚠ Model not found. Training model... (first run only)")
        from src.train import train_model
        return train_model()
    else:
        model = pickle.load(open(model_path, 'rb'))
        vectorizer = pickle.load(open(vectorizer_path, 'rb'))
        return model, vectorizer

model, vectorizer = load_model()

# -----------------------------
# TITLE
# -----------------------------
st.title("📊 Social Media Sentiment Analysis Dashboard")

# -----------------------------
# METRICS
# -----------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Total Rows", len(df))
col2.metric("Positive", (df['sentiment'] == "positive").sum())
col3.metric("Negative", (df['sentiment'] == "negative").sum())

# -----------------------------
# INPUT SECTION
# -----------------------------
st.subheader("Analyze New Post")

user_input = st.text_area("Enter text")

if st.button("Analyze"):
    if user_input:
        transformed = vectorizer.transform([user_input])
        prediction = model.predict(transformed)[0]

        # Better UI Output
        if prediction == "positive":
            st.success("😊 Positive Sentiment")
        elif prediction == "negative":
            st.error("😡 Negative Sentiment")
        else:
            st.warning("😐 Neutral Sentiment")

# -----------------------------
# FILTER + DATA PREVIEW
# -----------------------------
st.subheader("Dataset Preview")

sentiment_filter = st.selectbox(
    "Filter by Sentiment",
    ["All", "positive", "neutral", "negative"]
)

if sentiment_filter != "All":
    filtered_df = df[df['sentiment'] == sentiment_filter]
else:
    filtered_df = df

st.dataframe(filtered_df.head(10))

# -----------------------------
# SENTIMENT DISTRIBUTION
# -----------------------------
st.subheader("Sentiment Distribution")

sentiment_counts = df['sentiment'].value_counts()
st.bar_chart(sentiment_counts)

# -----------------------------
# WORDCLOUD (FULL FIXED VERSION)
# -----------------------------
st.subheader("WordCloud")

try:
    text_series = df['text'].dropna().astype(str)
    text_series = text_series[text_series.str.strip() != ""]

    text_data = " ".join(text_series.tolist())

    if len(text_data) > 0:
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black'
        ).generate(text_data)

        fig, ax = plt.subplots()
        ax.imshow(wordcloud)
        ax.axis("off")

        st.pyplot(fig)
    else:
        st.warning("No valid text available for WordCloud")

except Exception as e:
    st.error(f"WordCloud error: {e}")

# -----------------------------
# TEXT LENGTH INSIGHT
# -----------------------------
st.subheader("Text Length Insight")

df['length'] = df['text'].astype(str).apply(len)
st.line_chart(df['length'])

# -----------------------------
# DOWNLOAD BUTTON
# -----------------------------
st.download_button(
    label="Download Dataset",
    data=df.to_csv(index=False),
    file_name='dataset.csv',
    mime='text/csv'
)