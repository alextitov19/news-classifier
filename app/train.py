# app/train.py
import json
import pandas as pd
import os
import re
import time
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
import joblib
from tqdm import tqdm
import numpy as np
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

def train_model(df):
    logging.info("Starting model training")
    start_time = time.time()
    
    X = df['text']
    y = df['category']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    logging.info("Fitting the vectorizer")
    X_train_transformed = vectorizer.fit_transform(X_train)
    X_test_transformed = vectorizer.transform(X_test)
    
    logging.info(f"Vectorization completed in {time.time() - start_time:.2f} seconds")

    param_grid = {
        'alpha': [1e-4, 1e-3, 1e-2, 1e-1],
        'penalty': ['l2', 'elasticnet'],
        'max_iter': [1000, 2000],
        'loss': ['hinge', 'log_loss', 'modified_huber']
    }

    classifier = SGDClassifier(tol=1e-3)
    grid_search = GridSearchCV(classifier, param_grid, cv=5, n_jobs=-1)

    logging.info("Performing grid search")
    grid_search.fit(X_train_transformed, y_train)
    logging.info(f"Grid search completed in {time.time() - start_time:.2f} seconds")

    best_classifier = grid_search.best_estimator_

    # Cross-validation scores
    logging.info("Performing cross-validation")
    scores = cross_val_score(best_classifier, X_train_transformed, y_train, cv=5)
    logging.info(f"Cross-validation scores: {scores}")
    logging.info(f"Best parameters found: {grid_search.best_params_}")

    logging.info("Fitting the best classifier")
    best_classifier.fit(X_train_transformed, y_train)
    y_pred = best_classifier.predict(X_test_transformed)
    
    logging.info(f"Total training time: {time.time() - start_time:.2f} seconds")

    # Save predictions and actual values for evaluation
    results = pd.DataFrame({"text": X_test, "actual": y_test, "predicted": y_pred})
    results.to_csv('predictions.csv', index=False)
    
    # Ensure the models directory exists
    os.makedirs('app/models', exist_ok=True)
    
    # Save the vectorizer and classifier
    logging.info("Saving the vectorizer and classifier")
    joblib.dump(vectorizer, 'app/models/vectorizer.joblib')
    joblib.dump(best_classifier, 'app/models/classifier.joblib')

    logging.info("Model training complete")

if __name__ == '__main__':
    df = load_data('data/articles_valid.json')
    df = preprocess_data(df)
    train_model(df)