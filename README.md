# 📊 Social Media Sentiment Analysis Dashboard

## 🚀 Overview
This project performs **sentiment analysis on social media text** using Natural Language Processing (NLP) and Machine Learning. It classifies text into **positive, negative, or neutral sentiment** and provides an interactive dashboard for visualization and real-time predictions.

---

## 🎯 Problem Statement
Organizations need to understand public opinion from large volumes of social media data. This project builds an **end-to-end pipeline** to analyze and visualize sentiment automatically.

---

## 🛠 Tech Stack
- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, NLTK  
- **NLP Technique:** TF-IDF Vectorization  
- **Model:** Logistic Regression  
- **Visualization:** Matplotlib, Seaborn  
- **Frontend:** Streamlit  

---

## ⚙️ Features
- ✅ Text preprocessing (cleaning, stopword removal)
- ✅ Sentiment classification (positive, negative, neutral)
- ✅ Confusion matrix visualization
- ✅ Interactive dashboard using Streamlit
- ✅ Real-time sentiment prediction
- ✅ Prediction logging system (stores user inputs)

---

## 📂 Project Structure
```
Social-Media-Sentiment-Analysis-Dashboard/
│
├── data/                # Dataset
│   └── dataset.csv
│
├── src/                 # ML pipeline
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
│
├── models/              # Saved models
│   ├── model.pkl
│   └── vectorizer.pkl
│
├── app/                 # Streamlit dashboard
│   └── app.py
│
├── outputs/             # Results & logs
│   ├── confusion_matrix.png
│   └── predictions.csv
│
├── images/              # Screenshots
│   ├── dashboard.png
│   ├── prediction.png
│   ├── confusion_matrix.png
│   └── model_results.png
│
├── requirements.txt
├── README.md
└── main.py
```

---

## ▶️ How to Run

### 1️⃣ Install Dependencies
```
pip install -r requirements.txt
```

### 2️⃣ Train the Model
```
python src/train.py
```

### 3️⃣ Run the Dashboard
```
streamlit run app/app.py
```

---

## 📊 Results
- ✔ Accuracy: ~77% – 82%
- ✔ Balanced performance across all sentiment classes
- ✔ Tested on real-world Twitter dataset

---

## 📸 Screenshots

### 🔹 Dashboard
![Dashboard](images/dashboard.png)

### 🔹 Prediction
![Prediction](images/prediction.png)

### 🔹 Confusion Matrix
![Confusion Matrix](images/confusion_matrix.png)

### 🔹 Model Results
![Results](images/model_results.png)

---

## 📌 Learning Outcomes
- NLP preprocessing techniques (cleaning, tokenization, stopwords)
- Feature extraction using TF-IDF
- Machine Learning model training & evaluation
- Building interactive dashboards with Streamlit
- Handling real-world dataset issues (missing values, noise)

---

## 💼 Resume Description
Built an end-to-end NLP pipeline to classify social media text into positive, negative, and neutral sentiment. Applied TF-IDF vectorization and Logistic Regression, and developed an interactive Streamlit dashboard for real-time predictions. Achieved ~80% accuracy on real-world Twitter dataset.

---

## 🔮 Future Improvements
- Deploy the dashboard online (Streamlit Cloud)
- Upgrade to Deep Learning models (LSTM/BERT)
- Add WordCloud visualization
- Perform hyperparameter tuning

---

## 👨‍💻 Author
Vayunandan Mishra

---

## ⭐ If you found this project useful, consider giving it a star!