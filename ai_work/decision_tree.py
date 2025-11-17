import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay

def start():
    # Load the dataset
    data = pd.read_csv("datasets/Social_Network_Ads.csv")

    # Encode Gender (Male=1, Female=0)
    gender_encoder = LabelEncoder()
    data["Gender"] = gender_encoder.fit_transform(data["Gender"])
    # Male -> 1, Female -> 0

    # Features and target
    X = data[["Gender", "Age", "EstimatedSalary"]]
    y = data["Purchased"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Decision Tree Classifier
    model = DecisionTreeClassifier(
        criterion="entropy",   # or "gini"
        max_depth=4,           # avoid overfitting
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

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0", "1"])
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()


    # Visualize the decision tree
    plt.figure(figsize=(14, 8))
    plot_tree(
        model,
        feature_names=X.columns,
        class_names=["Not Purchased", "Purchased"],
        filled=True,
        rounded=True
    )
    plt.show()


if __name__ == "__main__":
    start()

