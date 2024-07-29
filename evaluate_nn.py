# evaluate_nn.py
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf

def evaluate_predictions(file_path):
    df = pd.read_csv(file_path)
    y_true = df['actual']
    y_pred = df['predicted']
    
    # Classification report
    report = classification_report(y_true, y_pred)
    print("Classification Report:")
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def load_and_evaluate():
    # Load vectorizer, label encoder, and model
    vectorizer = joblib.load('app/models/vectorizer.joblib')
    label_encoder = joblib.load('app/models/label_encoder.joblib')
    model = tf.keras.models.load_model('app/models/nn_model.h5')

    # Load test data
    df = pd.read_csv('predictions.csv')
    X_test = df['text']
    y_test = df['actual']
    
    # Transform test data
    X_test_transformed = vectorizer.transform(X_test).toarray()
    y_test_encoded = label_encoder.transform(y_test)
    
    # Make predictions
    y_pred_encoded = model.predict(X_test_transformed)
    y_pred = label_encoder.inverse_transform(y_pred_encoded.argmax(axis=1))
    
    # Save the results to CSV
    results = pd.DataFrame({"text": X_test, "actual": y_test, "predicted": y_pred})
    results.to_csv('predictions_nn.csv', index=False)

    # Evaluate the predictions
    evaluate_predictions('predictions_nn.csv')

if __name__ == '__main__':
    load_and_evaluate()