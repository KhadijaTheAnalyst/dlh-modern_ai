# Tree-Based Machine Learning Models

A hands-on project covering the full lifecycle of tree-based classifiers with Scikit-learn — from building and training a single decision tree, through pruning strategies to fight overfitting, up to ensemble methods (Random Forest and Boosting).

**Repository:** `dlh-modern_ai`
**Directory:** `machine_learning/tree_models`
**Dataset used throughout examples:** `sklearn.datasets.load_wine`

---

## Project Overview

This project builds a reusable, composable pipeline for tree-based classification:

```
build → train → inspect → predict → evaluate → tune (pre-pruning / post-pruning) → ensemble (bagging / boosting)
```

Every task is a small, single-purpose function. Later tasks reuse earlier ones directly (e.g. `train_tree`, `generate_predictions`, and `evaluate` are used unchanged all the way from a single Decision Tree through to Random Forests and Boosting), which is possible because every Scikit-learn estimator shares the same `.fit()` / `.predict()` interface.

---

## Requirements

- Python 3
- `scikit-learn`
- `numpy`
- `matplotlib` (for plots in the main test files)
- `xgboost`
- `lightgbm`

Style: `pycodestyle` compliant.

---

## File Structure

| Task | File | Function | Description |
|---|---|---|---|
| 0 | `0-build.py` | `build_decision_tree(min_samples_leaf, min_samples_split, random_state)` | Builds an untrained `DecisionTreeClassifier` (Gini criterion, unlimited depth) |
| 1 | `1-train.py` | `train_tree(clf, X, y)` | Trains any classifier in place via `.fit()`. Returns `None` |
| 2 | `2-draw.py` | `draw(clf, feature_names, class_names)` | Prints the text structure of a trained tree's decision rules |
| 3 | `3-generate_predictions.py` | `generate_predictions(clf, X)` | Returns predicted labels for a set of samples |
| 4 | `4-evaluate.py` | `evaluate(true_labels, predicted_labels, class_names)` | Returns a precision/recall/F1 classification report |
| 5 | `5-pre_pruning.py` | `prepruning(X, y, clf)` | Grid-searches pre-pruning hyperparameters (`criterion`, `max_depth`, `min_samples_leaf`, `min_samples_split`) |
| 6 | `6-pruning_path.py` | `get_pruning_path(clf, X, y)` | Returns the cost-complexity pruning path (`ccp_alphas`, `impurities`) |
| 7 | `7-prune_decision_tree.py` | `prune_and_evaluate_trees(...)` | Trains one tree per `ccp_alpha` and records train/test accuracy |
| 8 | `8-best_ccp_alpha.py` | `get_best_alpha(clfs, train_scores, test_scores, ccp_alphas)` | Selects the best-pruned tree using a 3-step tie-break rule |
| 9 | `9-random_forest.py` | `random_forest(n_estimators, random_state)` | Builds an untrained `RandomForestClassifier` |
| 10 | `10-feature_importance.py` | `feature_importance(rf)` | Returns feature importance scores, sorted ascending by importance |
| 11 | `11-boosting.py` | `compare_boosting_classifiers(name, n_estimators, random_state)` | Factory for AdaBoost / GradientBoosting / XGBoost / LightGBM classifiers |

---

## Concepts Covered

### 1. Decision Trees (Tasks 0–4)
A decision tree predicts by asking a sequence of yes/no questions about feature values, splitting data at each node to minimize **Gini impurity** until leaves are pure (or a stopping condition is met). Since `max_depth` is left unrestricted, `min_samples_leaf` and `min_samples_split` act as the main defense against overfitting.

- **Build vs. Train**: building (Task 0) only configures the model; training (Task 1, via `.fit()`) is what actually grows the tree structure from data, mutating the classifier in place.
- **Inspection**: `tree.export_text()` (Task 2) prints the learned if/else rules in human-readable form.
- **Prediction vs. Training**: `.predict()` (Task 3) is a read-only operation, unlike `.fit()` — it doesn't mutate the model.
- **Evaluation**: `metrics.classification_report()` (Task 4) reports **precision** (of predicted class X, how much was correct), **recall** (of actual class X, how much was caught), and **F1-score** (their harmonic mean) per class.

### 2. Pre-Pruning (Task 5)
Restricts tree growth *during* training via hyperparameters (`max_depth`, `min_samples_leaf`, `min_samples_split`). `GridSearchCV` automates the search across all combinations of candidate values, using cross-validation to select the best-performing set — replacing hand-guessed values with a systematic search.

### 3. Post-Pruning / Cost-Complexity Pruning (Tasks 6–8)
Lets a tree grow fully, then trims it back:

- **`ccp_alpha`** is a "cost per branch" threshold. `alpha = 0` keeps the full tree; increasing alpha prunes more aggressively into a simpler tree.
- `cost_complexity_pruning_path()` (Task 6) computes every alpha breakpoint where the optimal pruning changes, along with the resulting total leaf impurity at each stage.
- Training one tree per candidate alpha (Task 7) reveals the classic **bias-variance tradeoff**: training accuracy monotonically decreases as pruning increases, while test accuracy rises, peaks, then collapses as over-pruning removes real signal, not just noise.
- The best alpha (Task 8) is chosen by: (1) highest test accuracy, (2) smallest train/test accuracy gap as a tiebreaker (favors generalization), (3) largest `ccp_alpha` as a final tiebreaker (favors simpler trees — Occam's razor).

### 4. Random Forest / Bagging (Tasks 9–10)
An ensemble of many trees trained independently and combined by majority vote. Diversity between trees comes from:
1. **Bootstrap sampling** — each tree trains on a random sample of rows (with replacement).
2. **Random feature subsets** — each split only considers a random subset of features (default `max_features='sqrt'`).

Averaging many diverse trees cancels out individual overfitting errors, generally outperforming any single pruned tree. Trained forests also expose **feature importances** (Task 10) — a measure of how much each feature contributed to impurity reduction across all trees, giving interpretability as a free side effect of training.

### 5. Boosting (Task 11)
Unlike bagging, boosting builds trees **sequentially**, where each new tree specifically corrects the errors of the trees before it, and outputs are combined into a weighted prediction. This is generally more accurate but harder to parallelize than bagging.

- **AdaBoost**: reweights misclassified samples so later trees focus on them.
- **GradientBoosting**: each tree fits the residual errors (loss gradient) of the current ensemble.
- **XGBoost / LightGBM**: optimized, regularized gradient boosting implementations built for speed and performance at scale.

---

## Typical Usage Pattern

```python
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

build_decision_tree = __import__('0-build').build_decision_tree
train_tree = __import__('1-train').train_tree
generate_predictions = __import__('3-generate_predictions').generate_predictions
evaluate = __import__('4-evaluate').evaluate

wine = load_wine()
X, y = wine.data, wine.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=4000
)

clf = build_decision_tree(min_samples_leaf=2, min_samples_split=3, random_state=4000)
train_tree(clf, X_train, y_train)
y_pred = generate_predictions(clf, X_test)
print(evaluate(y_test, y_pred, wine.target_names))
```

This same five-step shape (build → train → predict → evaluate) is reused, unchanged, for Random Forests and every Boosting classifier in the project — only the model-construction step (Task 0, 9, or 11) changes.

---

## Results Summary (Wine dataset, seed = 4000)

| Model | Accuracy |
|---|---|
| Base Decision Tree (Task 0–4) | 0.75 |
| Pre-Pruned Decision Tree (Task 5) | 0.83 |
| Post-Pruned Decision Tree (Task 8) | 0.83 |
| Random Forest, 20 trees (Task 9) | 0.94 |
| Boosting (AdaBoost / GradientBoosting / XGBoost / LightGBM) (Task 11) | compared by accuracy and training time |

The progression illustrates a core ML lesson: a single tree overfits, pruning (either strategy) meaningfully improves generalization, and ensembling (bagging or boosting) improves it further still — at the cost of interpretability and, for boosting, training time.

---

## Author

Khadija Mustafa (KhadijaTheAnalyst) — AI Academy, Digital Learning Hub Luxembourg