# Import necessary libraries
from palmerpenguins import load_penguins
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    # (a) Load the dataset
    print("Loading the dataset...")
    penguins = load_penguins()
    
    if penguins is None or penguins.empty:
        print("Error: The dataset could not be loaded.")
        return
    
    print("Dataset loaded:\n", penguins.head())

    # (b) Check for missing values and clean the data
    print("\nChecking for missing values...")
    print(penguins.isnull().sum())

    print("\nRemoving incomplete entries...")
    penguins_cleaned = penguins.dropna()
    print("Remaining entries after cleaning:", len(penguins_cleaned))

    # (c) List the available features
    print("\nAvailable features:")
    print(penguins_cleaned.columns.tolist())

    # (d) Visualize feature combinations
    print("\nCreating pair plots for feature combinations...")
    sns.pairplot(
        penguins_cleaned,
        hue="species",
        vars=["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    )
    plt.title("Feature combinations by species")
    plt.show()

    # (e) Select two suitable features
    print("\nSelected features: 'flipper_length_mm' and 'body_mass_g'.")
    sns.scatterplot(
        data=penguins_cleaned,
        x="flipper_length_mm",
        y="body_mass_g",
        hue="species",
        palette="deep"
    )
    plt.title("Scatterplot: Flipper length vs. Body mass")
    plt.xlabel("Flipper length (mm)")
    plt.ylabel("Body mass (g)")
    plt.show()

    print("\nProgram completed!")

if __name__ == "__main__":
    main()
