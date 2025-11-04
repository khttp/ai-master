import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
)

def load_data(filepath: str) -> pd.DataFrame:

    """Load dataset from a CSV file."""

    return pd.read_csv(filepath)
def encode_gender_column(data:pd.DataFrame,gender_col="Gender"):
    if gender_col in data.columns:
        data[gender_col] = data[gender_col].map({"Male": 1, "Female": 0})
    return data


def preprocess_data(data: pd.DataFrame):

    """Split data into features and target."""

    X =pd.DataFrame( data[['Gender','Age','EstimatedSalary']])

    y = data['Purchased']

    return X, y

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """Split dataset into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Apply feature scaling for better model performance."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def train_model(X_train, y_train) -> LogisticRegression:
    """Train the logistic regression model."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: LogisticRegression, X_test, y_test):
    """Evaluate the model performance."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, cm, report, y_pred

def visualize_results(model: LogisticRegression, X_test, y_test):
    """Plot ROC curve if possible (for binary classification)."""
    try:
        RocCurveDisplay.from_estimator(model, X_test, y_test)
        plt.title("ROC Curve")
        plt.show()
    except Exception as e:
        print(f"Skipping ROC curve plot: {e}")

def start():

    """Entry point for Poetry script."""

    data = load_data('datasets/Social_Network_Ads.csv')
    data = encode_gender_column(data)
    X, y = preprocess_data(data)
     
    print(X,y)
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(X,y)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    model = train_model(X_train_scaled, y_train)
    accuracy, cm, report, _ = evaluate_model(model, X_test_scaled, y_test)
    # Display results
    print(f"Model Accuracy: {accuracy * 100:.2f}%\n")
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)

    visualize_results(model, X_test_scaled, y_test)


if __name__ == "__main__":

    start()