import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

def start():
    # Load the dataset
    data = pd.read_csv("datasets/Social_Network_Ads.csv")

    # Encode Gender (Male=1, Female=0)
    gender_encoder = LabelEncoder()
    data["Gender"] = gender_encoder.fit_transform(data["Gender"])

    # Features and target
    X = data[["Gender", "Age", "EstimatedSalary"]]
    y = data["Purchased"]

    # Scale features (important for SVM)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # SVM Classifier
    model = SVC(
        kernel="linear", 
        probability=True,   # required for ROC curve
        random_state=42
    )

    # Train the model
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluation
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Purchased", "Purchased"])
    disp.plot()
    plt.title("SVM - Confusion Matrix")
    plt.show()


if __name__ == "__main__":
    start()
