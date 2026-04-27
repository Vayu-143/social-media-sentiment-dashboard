import streamlit as st
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Sentiment Dashboard", layout="wide")

# ==============================
# UI STYLE
# ==============================
st.markdown("""
<style>
.big-font {
    font-size:20px !important;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# PATHS
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = os.path.join(BASE_DIR, 'data', 'dataset.csv')
model_path = os.path.join(BASE_DIR, 'models', 'model.pkl')
vectorizer_path = os.path.join(BASE_DIR, 'models', 'vectorizer.pkl')
feedback_path = os.path.join(BASE_DIR, 'outputs', 'feedback.csv')

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
    df = pd.read_csv(data_path)
    df['text'] = df['text'].fillna("").astype(str)
    return df

df = load_data()

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    if not os.path.exists(model_path):
        st.warning("Model not found. Train model first.")
        return None, None
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
col2.metric("Positive", (df['sentiment']=="positive").sum())
col3.metric("Negative", (df['sentiment']=="negative").sum())

# ==============================
# INPUT + PREDICTION
# ==============================
st.subheader("Analyze New Post")

user_input = st.text_area("Enter text")

prediction = None

if st.button("Analyze"):
    if model is None:
        st.error("Model not loaded")
    elif user_input.strip() == "":
        st.warning("Enter some text")
    else:
        vect_text = vectorizer.transform([user_input])
        prediction = model.predict(vect_text)[0]

        # confidence (if supported)
        try:
            probs = model.predict_proba(vect_text)
            confidence = max(probs[0])
            st.write(f"Confidence: {confidence:.2f}")
        except:
            st.write("Confidence not available for this model")

        if prediction == "positive":
            st.success("😊 Positive Sentiment")
        elif prediction == "negative":
            st.error("😡 Negative Sentiment")
        else:
            st.warning("😐 Neutral Sentiment")

# ==============================
# FEEDBACK SYSTEM
# ==============================
if prediction is not None:
    feedback = st.radio("Was this prediction correct?", ["Yes", "No"])

    if st.button("Submit Feedback"):
        feedback_data = {
            "text": user_input,
            "prediction": prediction,
            "feedback": feedback
        }

        feedback_df = pd.DataFrame([feedback_data])

        if os.path.exists(feedback_path):
            feedback_df.to_csv(feedback_path, mode='a', header=False, index=False)
        else:
            feedback_df.to_csv(feedback_path, index=False)

        st.success("Feedback saved!")

# ==============================
# FILTER + TABLE
# ==============================
st.subheader("Dataset Preview")

sentiment_filter = st.selectbox("Filter by Sentiment", ["All", "positive", "neutral", "negative"])

if sentiment_filter != "All":
    filtered_df = df[df['sentiment'] == sentiment_filter]
else:
    filtered_df = df

st.dataframe(filtered_df.head(10))

# ==============================
# DOWNLOAD
# ==============================
st.download_button(
    label="Download Dataset",
    data=df.to_csv(index=False),
    file_name="dataset.csv"
)

# ==============================
# SENTIMENT DISTRIBUTION
# ==============================
st.subheader("Sentiment Distribution")

counts = df['sentiment'].value_counts()

fig, ax = plt.subplots()
ax.bar(counts.index, counts.values)
st.pyplot(fig)

# ==============================
# WORDCLOUDS
# ==============================
st.subheader("Sentiment WordClouds")

col1, col2, col3 = st.columns(3)

def generate_wordcloud(data, title):
    text = " ".join(data.dropna().astype(str))

    if text.strip() == "":
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
# TOP WORDS
# ==============================
st.subheader("Top Words")

words = " ".join(df['text']).split()
common_words = Counter(words).most_common(10)

words_df = pd.DataFrame(common_words, columns=['word', 'count'])

fig, ax = plt.subplots()
ax.bar(words_df['word'], words_df['count'])
st.pyplot(fig)

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
# TEXT LENGTH (SAFE)
# ==============================
st.subheader("Text Length Insight")

df['length'] = df['text'].apply(lambda x: len(str(x)))

fig, ax = plt.subplots()
ax.plot(df['length'])
st.pyplot(fig)