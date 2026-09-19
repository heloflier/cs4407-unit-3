"""
Classification Models for Flower Species Prediction
CS 4407 - Machine Learning

Builds and compares six classification models (Decision Tree, Rule-Based,
Naive Bayes, Logistic Regression, KNN, and SVM) to predict flower species
from the Iris dataset.
"""

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
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

# ---------------------------------------------------------------------------
# Step 3 / Question 1c: Train/test split
# ---------------------------------------------------------------------------
# Stratifying keeps the 50/50/50 class balance in both splits.

print("\n" + "=" * 70)
print("QUESTION 1c: TRAIN/TEST SPLIT")
print("=" * 70)

X = df[iris_data.feature_names]  # features
y = df["species"]  # target (numeric species label)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTraining set size: {len(X_train)} rows")
print(f"Test set size: {len(X_test)} rows")
print("\nTraining set class distribution:")
print(y_train.map(species_names).value_counts())
print("\nTest set class distribution:")
print(y_test.map(species_names).value_counts())

# ---------------------------------------------------------------------------
# Step 4 / Question 2A.a: Decision Tree classifier
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("QUESTION 2A.a: DECISION TREE CLASSIFIER")
print("=" * 70)

decision_tree_model = DecisionTreeClassifier(random_state=42)  # fixed seed for reproducible results
decision_tree_model.fit(X_train, y_train)
decision_tree_predictions = decision_tree_model.predict(X_test)  # test on held-out data

print(f"\nTest accuracy: {accuracy_score(y_test, decision_tree_predictions):.4f}")

# ---------------------------------------------------------------------------
# Step 5 / Question 2A.b: Rule-based (if-else) classifier
# ---------------------------------------------------------------------------
# Thresholds derived from training data: petal length cleanly separates
# setosa (max 1.9) from the other two (min 3.0); petal width splits
# versicolor/virginica near the midpoint of their class means (1.31, 2.07).

print("\n" + "=" * 70)
print("QUESTION 2A.b: RULE-BASED (IF-ELSE) CLASSIFIER")
print("=" * 70)

def classify_by_rules(row):
    if row["petal length (cm)"] < 2.5:
        return 0  # setosa
    elif row["petal width (cm)"] < 1.7:
        return 1  # versicolor
    else:
        return 2  # virginica

rule_based_predictions = X_test.apply(classify_by_rules, axis=1)
print(f"\nTest accuracy: {accuracy_score(y_test, rule_based_predictions):.4f}")

# ---------------------------------------------------------------------------
# Step 6 / Question 2A.c: Naive Bayes classifier
# ---------------------------------------------------------------------------
# GaussianNB, since all four features are continuous measurements.

print("\n" + "=" * 70)
print("QUESTION 2A.c: NAIVE BAYES CLASSIFIER")
print("=" * 70)

naive_bayes_model = GaussianNB()
naive_bayes_model.fit(X_train, y_train)
naive_bayes_predictions = naive_bayes_model.predict(X_test)

print(f"\nTest accuracy: {accuracy_score(y_test, naive_bayes_predictions):.4f}")

# ---------------------------------------------------------------------------
# Step 7 / Question 2B.a: Logistic Regression classifier
# ---------------------------------------------------------------------------
# Logistic Regression, KNN, and SVM are all scale-sensitive; unscaled,
# Logistic Regression failed to converge within the default iteration
# limit. The added scaler is reused for KNN and SVM as well.

print("\n" + "=" * 70)
print("QUESTION 2B.a: LOGISTIC REGRESSION CLASSIFIER")
print("=" * 70)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logistic_regression_model = LogisticRegression(random_state=42)  # fixed seed for reproducible results
logistic_regression_model.fit(X_train_scaled, y_train)
logistic_regression_predictions = logistic_regression_model.predict(X_test_scaled)

print(f"\nTest accuracy: {accuracy_score(y_test, logistic_regression_predictions):.4f}")

# ---------------------------------------------------------------------------
# Step 8 / Question 2B.b: K-Nearest Neighbors classifier
# ---------------------------------------------------------------------------
# Reuses the scaler fit in Question 2B.a, since KNN's distance calculation
# is scale-sensitive too.

print("\n" + "=" * 70)
print("QUESTION 2B.b: K-NEAREST NEIGHBORS CLASSIFIER")
print("=" * 70)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
knn_predictions = knn_model.predict(X_test_scaled)

print(f"\nTest accuracy: {accuracy_score(y_test, knn_predictions):.4f}")
