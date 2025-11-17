import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def start():
    # Load the dataset
    data = pd.read_csv("datasets/Mall_Customers.csv")  # change to your filename

    # Encode Genre (Male=1, Female=0)
    genre_encoder = LabelEncoder()
    data["Genre"] = genre_encoder.fit_transform(data["Genre"])

    # Feature selection
    # We DO NOT include CustomerID — it is not a useful feature
    X = data[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]

    # Feature scaling (important for KMeans)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow Method (find the optimal k)
    inertia_values = []
    k_range = range(1, 11)

    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42)
        model.fit(X_scaled)
        inertia_values.append(model.inertia_)

    # Plot Elbow curve
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertia_values, marker="o")
    plt.title("Elbow Method: Optimal Number of Clusters")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.grid(True)
    plt.show()

    # Fit final KMeans model
    # NOTE: You should pick k from the elbow plot
    optimal_k = 5   # common choice, change based on elbow result
    final_model = KMeans(n_clusters=optimal_k, random_state=42)

    # Train model & assign clusters
    clusters = final_model.fit_predict(X_scaled)
    data["Cluster"] = clusters

    # Cluster visualization
    # (Income vs Spending Score)
    plt.figure(figsize=(10, 6))
    plt.scatter(
        data["Annual Income (k$)"],
        data["Spending Score (1-100)"],
        c=data["Cluster"],
        cmap="viridis",
        s=60
    )
    plt.title("Customer Segmentation (K-Means Clustering)")
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.colorbar(label="Cluster")
    plt.show()

    # Print sample data with clusters assigned
    print("\nFirst 10 rows with assigned clusters:")
    print(data.head(10))


if __name__ == "__main__":
    start()
