import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score





def load_data(filepath: str) -> pd.DataFrame:

    """Load dataset from a CSV file."""

    return pd.read_csv(filepath)





def preprocess_data(data: pd.DataFrame):

    """Split data into features and target."""

    X = data[['YearsExperience']]

    y = data['Salary']

    return X, y





def split_data(X, y, test_size: float = 0.2, random_state: int = 42):

    """Split dataset into training and testing sets."""

    return train_test_split(X, y, test_size=test_size, random_state=random_state)





def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:

    """Train the linear regression model."""

    model = LinearRegression()

    model.fit(X_train, y_train)

    return model





def evaluate_model(model: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series):

    """Evaluate the model using MSE and R² score."""

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)

    r2 = r2_score(y_test, y_pred)

    return mse, r2, y_pred





def visualize_results(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, y_pred: np.ndarray):

    """Visualize actual vs predicted data."""

    plt.figure(figsize=(8, 6))

    plt.scatter(X, y, color='blue', label='Actual Data')

    plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')

    plt.xlabel('Years of Experience')

    plt.ylabel('Salary')

    plt.title('Linear Regression: Salary vs Years of Experience')

    plt.legend()

    plt.show()





def start():

    """Entry point for Poetry script."""

    data = load_data('datasets/Salary_dataset.csv')

    X, y = preprocess_data(data)

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)

    mse, r2, y_pred = evaluate_model(model, X_test, y_test)



    print(f"Mean Squared Error: {mse:.2f}")

    print(f"R² Score: {r2:.2f}")



    visualize_results(X, y, X_test, y_pred)





if __name__ == "__main__":

    start()
