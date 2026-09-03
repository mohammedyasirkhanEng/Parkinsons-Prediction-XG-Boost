"""
Train and evaluate an XGBoost classifier to predict Parkinson's disease
from voice-measurement data.

Usage:
    python train.py
"""

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from data_preprocessing import load_raw_data, preprocess


def train_model(X_train, y_train):
    """Trains an XGBoost classifier with a small grid search for tuning."""
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
    }

    base_model = XGBClassifier(
        eval_metric="logloss",
        random_state=42,
    )

    grid_search = GridSearchCV(
        base_model, param_grid, cv=5, scoring="f1", n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n=== Model Evaluation ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Healthy", "Parkinson's"]))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks([0, 1], ["Healthy", "Parkinson's"])
    plt.yticks([0, 1], ["Healthy", "Parkinson's"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("Saved confusion matrix to confusion_matrix.png")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    sorted_idx = importances.argsort()[::-1]

    plt.figure(figsize=(8, 6))
    plt.barh(
        [feature_names[i] for i in sorted_idx[:10][::-1]],
        importances[sorted_idx[:10][::-1]],
    )
    plt.title("Top 10 Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    print("Saved feature importance plot to feature_importance.png")


def main():
    print("Loading and preprocessing data...")
    df = load_raw_data()
    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess(df)

    print("Training XGBoost model (grid search)...")
    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)
    plot_feature_importance(model, feature_names)

    model.save_model("parkinsons_xgboost_model.json")
    print("Model saved to parkinsons_xgboost_model.json")


if __name__ == "__main__":
    main()
