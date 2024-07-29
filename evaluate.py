# evaluate.py
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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

if __name__ == '__main__':
    evaluate_predictions('predictions.csv')