import streamlit as st
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Sentiment Dashboard", layout="wide")

# ==============================
# PATH SETUP
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = os.path.join(BASE_DIR, 'data', 'dataset.csv')
model_path = os.path.join(BASE_DIR, 'models', 'model.pkl')
vectorizer_path = os.path.join(BASE_DIR, 'models', 'vectorizer.pkl')

# ==============================
# SIDEBAR
# ==============================
st.sidebar.title("About Project")
st.sidebar.info("""
📊 Sentiment Analysis Dashboard  
👨‍💻 Built with ML + NLP  
🚀 Deployed using Streamlit  

Author: Vayunandan Mishra
""")

# ==============================
# LOAD DATA
# ==============================
@st.cache_data
def load_data():
    return pd.read_csv(data_path)

df = load_data()

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    if not os.path.exists(model_path):
        st.warning("Model not found. Please train the model first.")
        return None, None
    else:
        model = pickle.load(open(model_path, 'rb'))
        vectorizer = pickle.load(open(vectorizer_path, 'rb'))
        return model, vectorizer

model, vectorizer = load_model()

# ==============================
# TITLE
# ==============================
st.title("📊 Social Media Sentiment Analysis Dashboard")

st.markdown("""
### 🔍 About
This dashboard analyzes social media text and predicts sentiment using Machine Learning.

- Model: Logistic Regression / LinearSVC  
- Technique: TF-IDF  
- Classes: Positive, Neutral, Negative  
""")

# ==============================
# METRICS
# ==============================
col1, col2, col3 = st.columns(3)

col1.metric("Total Rows", len(df))
col2.metric("Positive", (df['sentiment'] == "positive").sum())
col3.metric("Negative", (df['sentiment'] == "negative").sum())

# ==============================
# INPUT TEXT
# ==============================
st.subheader("Analyze New Post")

user_input = st.text_area("Enter text")

if st.button("Analyze"):
    if model is None:
        st.error("Model not loaded")
    elif user_input.strip() == "":
        st.warning("Please enter text")
    else:
        vect_text = vectorizer.transform([user_input])
        prediction = model.predict(vect_text)[0]

        if prediction == "positive":
            st.success("😊 Positive Sentiment")
        elif prediction == "negative":
            st.error("😡 Negative Sentiment")
        else:
            st.warning("😐 Neutral Sentiment")

# ==============================
# FILTER
# ==============================
st.subheader("Dataset Preview")

sentiment_filter = st.selectbox("Filter by Sentiment", ["All", "positive", "neutral", "negative"])

if sentiment_filter != "All":
    filtered_df = df[df['sentiment'] == sentiment_filter]
else:
    filtered_df = df

st.dataframe(filtered_df.head(10))

# ==============================
# DOWNLOAD BUTTON
# ==============================
st.download_button(
    label="Download Dataset",
    data=df.to_csv(index=False),
    file_name='dataset.csv',
    mime='text/csv'
)

# ==============================
# SENTIMENT DISTRIBUTION
# ==============================
st.subheader("Sentiment Distribution")

sentiment_counts = df['sentiment'].value_counts()

fig, ax = plt.subplots()
ax.bar(sentiment_counts.index, sentiment_counts.values)
st.pyplot(fig)

# ==============================
# WORDCLOUDS (FIXED VERSION)
# ==============================
st.subheader("Sentiment WordClouds")

col1, col2, col3 = st.columns(3)

def generate_wordcloud(data, title):
    text = " ".join(data.dropna().astype(str))

    if text.strip() == "":
        st.write(f"No data for {title}")
        return None

    wc = WordCloud(width=400, height=300, background_color='black').generate(text)

    fig, ax = plt.subplots()
    ax.imshow(wc)
    ax.axis("off")
    ax.set_title(title)
    return fig

with col1:
    fig1 = generate_wordcloud(df[df['sentiment']=="positive"]['text'], "Positive")
    if fig1:
        st.pyplot(fig1)

with col2:
    fig2 = generate_wordcloud(df[df['sentiment']=="neutral"]['text'], "Neutral")
    if fig2:
        st.pyplot(fig2)

with col3:
    fig3 = generate_wordcloud(df[df['sentiment']=="negative"]['text'], "Negative")
    if fig3:
        st.pyplot(fig3)

# ==============================
# CONFUSION MATRIX
# ==============================
st.subheader("Model Performance")

conf_path = os.path.join(BASE_DIR, "outputs", "confusion_matrix.png")

if os.path.exists(conf_path):
    st.image(conf_path, caption="Confusion Matrix")
else:
    st.warning("Confusion matrix not found")

# ==============================
# TEXT LENGTH INSIGHT (FIXED)
# ==============================
st.subheader("Text Length Insight")

# FIX: clean text column safely
df['text'] = df['text'].fillna("").astype(str)

# Now compute length
df['length'] = df['text'].apply(lambda x: len(x))

fig, ax = plt.subplots()
ax.plot(df['length'])
st.pyplot(fig)