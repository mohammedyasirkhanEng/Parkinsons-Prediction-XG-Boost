"""
Load and preprocess the UCI Parkinson's Disease dataset.

Dataset source: https://archive.ics.uci.edu/dataset/174/parkinsons
It contains 195 voice recordings from 31 people (23 with Parkinson's disease),
with 22 biomedical voice measurement features per recording.

The dataset downloads automatically from the UCI repository the first time
you run this script.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"


def load_raw_data():
    """Downloads and returns the raw Parkinson's dataset as a DataFrame."""
    df = pd.read_csv(DATA_URL)
    return df


def preprocess(df, test_size=0.2, random_state=42):
    """
    Splits features/target, scales features, and returns train/test sets.

    The 'status' column is the target (1 = Parkinson's, 0 = healthy).
    The 'name' column is a subject identifier and is dropped.
    """
    X = df.drop(columns=["name", "status"])
    y = df["status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist(), scaler


if __name__ == "__main__":
    df = load_raw_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:\n{df['status'].value_counts()}")

    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess(df)
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Features used ({len(feature_names)}): {feature_names}")
