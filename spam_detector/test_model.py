import joblib
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
print('Model loaded')

import string
import nltk
from nltk.corpus import stopwords

def preprocess(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    words = text.split()
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords', quiet=True)
        stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

msg = 'This is a long test message to see if the model can handle it properly and predict correctly without any errors or slowdowns that might cause network disconnect issues in the web application interface when users input text.'
p = preprocess(msg)
v = vectorizer.transform([p])
pred = model.predict(v)
print(pred)