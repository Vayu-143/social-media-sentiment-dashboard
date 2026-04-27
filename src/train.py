import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

from preprocess import clean_text

# Load dataset
df = pd.read_csv('data/dataset.csv')

# Clean data
df['text'] = df['text'].fillna('')
df['clean_text'] = df['text'].apply(clean_text)

# Features & labels
X = df['clean_text']
y = df['sentiment']

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

# Model
model = LogisticRegression()
model.fit(X_vec, y)

# Predictions
y_pred = model.predict(X_vec)

# Evaluation
print("Accuracy:", accuracy_score(y, y_pred))
print("\nClassification Report:\n", classification_report(y, y_pred))

# Confusion Matrix
cm = confusion_matrix(y, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")

# Save image
os.makedirs('outputs', exist_ok=True)
plt.savefig('outputs/confusion_matrix.png')
plt.show()

# Save model
os.makedirs('models', exist_ok=True)

pickle.dump(model, open('models/model.pkl', 'wb'))
pickle.dump(vectorizer, open('models/vectorizer.pkl', 'wb'))

print("Model saved successfully!")