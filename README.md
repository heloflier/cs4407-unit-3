# Classification Models for Flower Species Prediction

Programming assignment: building and comparing six classification models
(Decision Tree, Rule-Based, Naive Bayes, Logistic Regression, KNN, and
SVM) to predict flower species from the Iris dataset, using
`scikit-learn`.

## Contents

- `iris_classification.py` — main analysis script (data loading, model
  building, evaluation)
- `requirements.txt` — Python dependencies

## Workflow

The script is built incrementally, corresponding to the assignment's
four questions:

1. Data generation and preparation (load dataset, describe
   characteristics, train/test split)
2. Model development:
   - 2A: Rule-based and probabilistic models (Decision Tree, Rule-Based
     Classifier, Naive Bayes)
   - 2B: Distance and optimization-based models (Logistic Regression,
     KNN, SVM)
3. Model evaluation (confusion matrix, precision/recall, F1-score) for
   all six models
4. Analysis and comparison across all models

## Running

```bash
pip install -r requirements.txt
python iris_classification.py
```
