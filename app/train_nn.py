# app/train_nn.py
import json
import pandas as pd
import os
import re
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure NLTK data is available
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def clean_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove all non-word characters
    text = re.sub(r'\s+', ' ', text)  # Remove all extra spaces
    text = text.lower()  # Convert to lowercase
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

def load_data(file_path):
    logging.info(f"Loading data from {file_path}")
    with open(file_path, 'r') as file:
        data = json.load(file)
    return pd.DataFrame(data['root'])

def preprocess_data(df):
    logging.info("Preprocessing data")
    df = df.dropna(subset=['category', 'headline', 'short_description'])
    df['text'] = df['headline'] + ' ' + df['short_description']
    df['text'] = df['text'].apply(clean_text)
    return df

def build_model(input_dim, num_classes):
    model = Sequential()
    model.add(Dense(512, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    return model

def train_model(df):
    logging.info("Starting model training")
    start_time = time.time()
    
    X = df['text']
    y = df['category']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=5000)
    logging.info("Fitting the vectorizer")
    X_train_transformed = vectorizer.fit_transform(X_train).toarray()
    X_test_transformed = vectorizer.transform(X_test).toarray()
    
    logging.info(f"Vectorization completed in {time.time() - start_time:.2f} seconds")
    
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    num_classes = len(label_encoder.classes_)
    input_dim = X_train_transformed.shape[1]
    
    model = build_model(input_dim, num_classes)
    
    logging.info("Training the neural network")
    model.fit(X_train_transformed, y_train_encoded, epochs=10, batch_size=32, validation_split=0.1, verbose=1)
    
    y_pred_encoded = model.predict_classes(X_test_transformed)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    
    logging.info(f"Total training time: {time.time() - start_time:.2f} seconds")

    # Save predictions and actual values for evaluation
    results = pd.DataFrame({"text": X_test, "actual": y_test, "predicted": y_pred})
    results.to_csv('predictions.csv', index=False)
    
    # Ensure the models directory exists
    os.makedirs('app/models', exist_ok=True)
    
    # Save the vectorizer, label encoder, and model
    logging.info("Saving the vectorizer, label encoder, and model")
    joblib.dump(vectorizer, 'app/models/vectorizer.joblib')
    joblib.dump(label_encoder, 'app/models/label_encoder.joblib')
    model.save('app/models/nn_model.h5')

    logging.info("Model training complete")

if __name__ == '__main__':
    df = load_data('data/articles_valid.json')
    df = preprocess_data(df)
    train_model(df)