# Import necessary libraries
from palmerpenguins import load_penguins
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

def main():
    # Load the dataset
    print("Loading the dataset...")
    penguins = load_penguins()

    if penguins is None or penguins.empty:
        print("Error: The dataset could not be loaded.")
        return

    print("Dataset loaded:\n", penguins.head())

    # Clean the data: Remove rows with missing values
    print("\nCleaning the dataset...")
    penguins_cleaned = penguins.dropna()
    print("Remaining entries after cleaning:", len(penguins_cleaned))

    # Select two features and two species
    print("\nSelecting features and reducing to two species...")
    features = ["flipper_length_mm", "body_mass_g"]
    species = ["Adelie", "Chinstrap"]  # Two species for binary classification
    penguins_binary = penguins_cleaned[penguins_cleaned["species"].isin(species)]

    X = penguins_binary[features].values
    y = penguins_binary["species"].apply(lambda s: 0 if s == "Adelie" else 1).values

    print(f"Selected features: {features}")
    print(f"Number of samples: {len(y)}")

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    print("\nStandardizing the features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # (b) Train a Linear SVM for soft margin classification
    print("\nTraining a Linear SVM classifier...")
    svm = SVC(kernel="linear", C=1.0)  # Soft margin with default regularization (C=1.0)
    svm.fit(X_train_scaled, y_train)

    # Visualize the decision boundary
    print("\nVisualizing the decision boundary...")
    visualize_decision_boundary(svm, X_train_scaled, y_train, X_test_scaled, y_test, features)

def visualize_decision_boundary(model, X_train, y_train, X_test, y_test, feature_names):
    # Create a grid for plotting
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    # Predict over the grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.8, cmap="coolwarm")
    sns.scatterplot(x=X_train[:, 0], y=X_train[:, 1], hue=y_train, style=y_train, palette="deep", legend="full")
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title("Linear SVM Decision Boundary (Train Data)")
    plt.show()

if __name__ == "__main__":
    main()
