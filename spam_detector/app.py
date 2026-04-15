from flask import Flask, request, jsonify, render_template
import joblib
import string
import nltk
from nltk.corpus import stopwords
import os

app = Flask(__name__)

# Load models safely
MODEL_PATH = 'model.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'

if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
else:
    model = None
    vectorizer = None

def preprocess_text(text):
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

@app.route('/')
def index():
    accuracy = "N/A"
    if os.path.exists('accuracy.txt'):
        with open('accuracy.txt', 'r') as f:
            accuracy = f.read().strip()
    return render_template('index.html', accuracy=accuracy)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or vectorizer is None:
            return jsonify({'error': 'Model not trained yet.'}), 500
        
        data = request.get_json(silent=True)
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided.'}), 400
            
        message = data['message']
        if not message.strip():
            return jsonify({'result': 'Not Spam'})
            
        processed_message = preprocess_text(message)
        vectorized_message = vectorizer.transform([processed_message])
        prediction = model.predict(vectorized_message)[0]
        
        # In the dataset, labels are 'ham' and 'spam'
        result = "Spam" if prediction == 'spam' else "Not Spam"
        
        return jsonify({'result': result})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)


"""what is ann
mention the layers of ann 
what is difference the model of machine learning and deep learning
what is the meanning of sequential adn dense
what is the role of activation function
what is the role of optimizer
what is tensorflow
what are the components of tensorflow
what is the difference between variable and placeholder
what is deep learning

"""