"""
Classification Models for Flower Species Prediction
CS 4407 - Machine Learning

Builds and compares six classification models (Decision Tree, Rule-Based,
Naive Bayes, Logistic Regression, KNN, and SVM) to predict flower species
from the Iris dataset.
"""

import pandas as pd
from sklearn.datasets import load_iris
from tabulate import tabulate

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

# ---------------------------------------------------------------------------
# Step 1 / Question 1a: Load the Iris dataset
# ---------------------------------------------------------------------------

print("=" * 70)
print("QUESTION 1a: LOAD THE IRIS DATASET")
print("=" * 70)

iris_data = load_iris()
df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
species_names = dict(enumerate(iris_data.target_names))
df["species"] = iris_data.target
df["species_name"] = df["species"].map(species_names)

print("\nDataset preview:")
print(
    tabulate(
        df.head(),
        headers="keys",
        tablefmt="fancy_grid",
        stralign="center",
        numalign="center",
    )
)

print("\nDataset shape:", df.shape)

# ---------------------------------------------------------------------------
# Step 2 / Question 1b: Describe dataset characteristics
# ---------------------------------------------------------------------------
# Iris is a classic benchmark dataset: balanced classes (50/50/50) make accuracy
# a reasonably fair metric, unlike on an imbalanced dataset.
 
print("\n" + "=" * 70)
print("QUESTION 1b: DESCRIBE DATASET CHARACTERISTICS")
print("=" * 70)
 
print(f"\nNumber of samples: {df.shape[0]}")
print(f"Number of features: {len(iris_data.feature_names)}")
print(f"Feature names: {iris_data.feature_names}")
print(f"Target classes: {[str(name) for name in iris_data.target_names]}")
 
print("\nClass distribution:")
print(
    tabulate(
        df["species_name"].value_counts().rename_axis("species").reset_index(name="count"),
        headers="keys",
        tablefmt="fancy_grid",
        showindex=False,
    )
)
 
print("\nFeature summary statistics:")
print(
    tabulate(
        df[iris_data.feature_names].describe(),
        headers="keys",
        tablefmt="fancy_grid",
    )
)
