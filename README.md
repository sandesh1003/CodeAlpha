# Iris Flower Classification

Classify iris flowers into **setosa**, **versicolor** or **virginica** from four measurements: sepal length, sepal width, petal length and petal width. Built with Scikit-learn.

## How to run

```bash
pip install scikit-learn pandas matplotlib seaborn
python iris_classification.py
```

The script prints all results and saves `pairplot.png`, `model_comparison.png`, `confusion_matrix.png`, `feature_importance.png` and `model_comparison.csv`.

## What the script does

1. **Load data**: `load_iris()` from Scikit-learn (150 samples, 4 features, 3 balanced classes, no missing values).
2. **Explore**: summary statistics, class counts and a pairplot to see how well the species separate.
3. **Split**: 80% train / 20% test (120 / 30), `stratify=y` so each species keeps equal proportion in both sets.
4. **Train five classifiers**: Logistic Regression, K-Nearest Neighbors, SVM (RBF), Decision Tree, Random Forest. Models that depend on feature scale are wrapped in a pipeline with `StandardScaler`, so the scaler only learns from training data.
5. **Evaluate** on the unseen test set: accuracy, precision, recall, F1 and a confusion matrix.
6. **Cross-validate** (5-fold) for a more stable estimate than a single 30-sample test set.
7. **Predict** the species of three new flowers.

## Results

| Model | Test accuracy | 5-fold CV accuracy | CV std |
|---|---|---|---|
| Decision Tree (depth 3) | 0.967 | 0.973 | 0.025 |
| SVM (RBF) | 0.967 | 0.967 | 0.021 |
| Random Forest | 0.900 | 0.967 | 0.021 |
| Logistic Regression | 0.933 | 0.960 | 0.039 |
| K-Nearest Neighbors | 0.933 | 0.960 | 0.025 |

- All models reach about 96-97% cross-validated accuracy. The differences between them are smaller than the fold-to-fold variation (std about 2-4%), so no model is clearly the best on this dataset.
- The script reports the Decision Tree in detail because it ranked first on CV accuracy. It got 29/30 test samples right; the one error was a versicolor predicted as virginica.
- **Setosa** was classified perfectly by every model. **Versicolor and virginica** overlap slightly, which is where nearly all mistakes happen.
- Petal length and petal width are the most informative features (about 45% and 41% of Random Forest importance); sepal width contributes very little.
- The test set is only 30 flowers, so one mistake changes accuracy by about 3.3 points. This is why the Random Forest's 0.900 test score should not be read as it being worse than the others (its CV score is 0.967).

## Basic concepts covered

- **Supervised classification**: learning from labelled examples to predict a category.
- **Features and target**: the four measurements are inputs (X), the species is the label (y).
- **Train/test split**: evaluate on data the model has never seen, to measure generalization instead of memorization.
- **Stratification**: keeps class balance in both splits.
- **Feature scaling**: needed for distance- and gradient-based models (KNN, SVM, Logistic Regression), not for tree-based ones.
- **Accuracy, precision, recall, F1**: accuracy is the share of correct predictions; precision is how many predicted-X were truly X; recall is how many true X were found; F1 combines both.
- **Confusion matrix**: shows which classes get confused with which.
- **Cross-validation**: repeats train/test over several splits for a more reliable estimate.
- **Overfitting**: a model that fits training data too closely and does worse on new data (the Decision Tree is limited to depth 3 for this reason).
