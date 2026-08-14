"""Train the classifier from SQL Server landmark samples.

By default, samples are loaded from ASLRecognizer.dbo.hand_landmarks.
Use ``--source csv`` only when a database is unavailable.

Usage:
    python train.py
    python train.py --source csv
"""
import argparse
import os
import numpy as np
import pandas as pd
import pyodbc
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

from model_def import LandmarkClassifier, get_device

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "landmarks.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "classifier.pt")

EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-3

SERVER = r"LAPTOP-PUUSOUD1\SQLEXPRESS01"
DATABASE = "ASLRecognizer"
TABLE = "dbo.hand_landmarks"
LANDMARK_COLUMNS = [f"{axis}{index}" for index in range(21) for axis in ("x", "y", "z")]
CONNECTION_STRING = (
    "Driver={ODBC Driver 18 for SQL Server};"
    f"Server={SERVER};"
    f"Database={DATABASE};"
    "Trusted_Connection=yes;"
    "Encrypt=no;"
    "TrustServerCertificate=yes;"
    "Connection Timeout=10;"
)


def load_dataset(source: str) -> pd.DataFrame:
    if source == "csv":
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"No CSV dataset found at {DATA_PATH}.")
        return pd.read_csv(DATA_PATH)

    columns = ["label", *LANDMARK_COLUMNS]
    query = f"SELECT {', '.join(f'[{column}]' for column in columns)} FROM {TABLE}"
    try:
        with pyodbc.connect(CONNECTION_STRING) as connection:
            cursor = connection.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            field_names = [column[0] for column in cursor.description]
    except pyodbc.Error as error:
        raise RuntimeError(f"Could not load training data from SQL Server: {error}") from error

    if not rows:
        raise RuntimeError(f"No samples found in {DATABASE}.{TABLE}.")
    return pd.DataFrame.from_records(rows, columns=field_names)


def main():
    parser = argparse.ArgumentParser(description="Train the ASL classifier from landmark samples.")
    parser.add_argument(
        "--source",
        choices=("sql", "csv"),
        default="sql",
        help="Training dataset source (default: sql)",
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Training on device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    df = load_dataset(args.source)
    print(f"Loaded {len(df):,} samples from {args.source.upper()}.")
    if len(df) < 50:
        print(f"Warning: only {len(df)} samples found. Collect more for a robust model.")

    X = df.drop(columns=["label"]).values.astype(np.float32)
    y_raw = df["label"].astype(str).values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw).astype(np.int64)
    num_classes = len(encoder.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
    )

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    X_test_t = torch.from_numpy(X_test).to(device)
    y_test_t = torch.from_numpy(y_test).to(device)

    model = LandmarkClassifier(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        if epoch % 10 == 0 or epoch == EPOCHS:
            model.eval()
            with torch.no_grad():
                preds = model(X_test_t).argmax(dim=1)
                acc = (preds == y_test_t).float().mean().item() if len(X_test) > 0 else 0.0
            best_acc = max(best_acc, acc)
            print(f"Epoch {epoch:3d} | loss {total_loss/len(train_ds):.4f} | val_acc {acc:.3f}")

    if len(X_test) > 0:
        model.eval()
        with torch.no_grad():
            preds = model(X_test_t).argmax(dim=1).cpu().numpy()
        print(classification_report(y_test, preds, target_names=[str(c) for c in encoder.classes_], zero_division=0))

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "labels": list(encoder.classes_),
        "num_classes": num_classes,
        "input_dim": X.shape[1],
    }, MODEL_PATH)

    print(f"Model saved to {MODEL_PATH}")
    print(f"Classes: {list(encoder.classes_)}")


if __name__ == "__main__":
    main()
