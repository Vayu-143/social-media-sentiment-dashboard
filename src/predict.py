import pickle

model = pickle.load(open('models/model.pkl', 'rb'))
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

text = input("Enter text: ")

vec = vectorizer.transform([text])
result = model.predict(vec)[0]

print("Prediction:", result)