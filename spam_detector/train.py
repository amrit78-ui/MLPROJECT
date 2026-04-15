import os
import urllib.request
import zipfile
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

# Ensure stopwords are downloaded
nltk.download('stopwords')

def download_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    zip_path = "smsspamcollection.zip"
    extract_dir = "data"
    
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)
        
    if not os.path.exists(os.path.join(extract_dir, "SMSSpamCollection")):
        print("Downloading dataset...")
        urllib.request.urlretrieve(url, zip_path)
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        if os.path.exists(zip_path):
            os.remove(zip_path)

def preprocess_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

def main():
    download_data()
    
    print("Loading data...")
    # The file is tab-separated without a header
    data_path = os.path.join("data", "SMSSpamCollection")
    df = pd.read_csv(data_path, sep='\t', header=None, names=['label', 'message'])
    
    print("Preprocessing text...")
    df['processed_message'] = df['message'].apply(preprocess_text)
    
    print("Vectorizing...")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['processed_message'])
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training model...")
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    
    print("Saving model and vectorizer...")
    joblib.dump(model, 'model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    
    # Save accuracy to a file for the frontend to show
    with open('accuracy.txt', 'w') as f:
        f.write(f"{accuracy:.4f}")
    
    print("Done!")

if __name__ == "__main__":
    main()
