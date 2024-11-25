# Import necessary libraries
from palmerpenguins import load_penguins
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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

    # Select two features and two species for binary classification
    print("\nSelecting features and reducing to two species...")
    features = ["flipper_length_mm", "body_mass_g"]
    species = ["Adelie", "Chinstrap"]  # Two species for classification
    penguins_binary = penguins_cleaned[penguins_cleaned["species"].isin(species)]

    X = penguins_binary[features].values
    y = penguins_binary["species"].apply(lambda s: 0 if s == "Adelie" else 1).values

    print(f"Selected features: {features}")
    print(f"Number of samples: {len(y)}")

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # (a) Train a polynomial SVM
    print("\nTraining a polynomial SVM...")
    for degree in [2, 3, 4]:  # Try different polynomial degrees
        print(f"\nPolynomial SVM with degree {degree}:")
        polynomial_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='poly', degree=degree, C=1.0))
        ])
        polynomial_pipeline.fit(X_train, y_train)
        visualize_decision_boundary(
            polynomial_pipeline, X_train, y_train, X_test, y_test, features, title=f"Polynomial SVM (degree {degree})"
        )

    # (b) Train a Gaussian RBF SVM
    print("\nTraining a Gaussian RBF SVM...")
    rbf_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', gamma=0.1, C=1.0))
    ])
    rbf_pipeline.fit(X_train, y_train)
    visualize_decision_boundary(
        rbf_pipeline, X_train, y_train, X_test, y_test, features, title="Gaussian RBF SVM"
    )

def visualize_decision_boundary(model, X_train, y_train, X_test, y_test, feature_names, title="Decision Boundary"):
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
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    main()
