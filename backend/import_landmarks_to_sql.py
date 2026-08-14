"""Import converted ASL hand landmarks from CSV into local SQL Server.

Usage (from backend):
    python import_landmarks_to_sql.py
    python import_landmarks_to_sql.py --csv C:\\path\\to\\your-landmarks.csv
    python import_landmarks_to_sql.py --create-table
    python import_landmarks_to_sql.py --clear --create-table

The default is safe to re-run: it appends samples and never deletes rows.
"""

import argparse
import csv
import os
import sys

import pyodbc


SERVER = r"LAPTOP-PUUSOUD1\SQLEXPRESS01"
DATABASE = "ASLRecognizer"
TABLE = "dbo.hand_landmarks"
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "landmarks.csv")
LANDMARK_COLUMNS = [f"{axis}{index}" for index in range(21) for axis in ("x", "y", "z")]
EXPECTED_COLUMNS = ["label", *LANDMARK_COLUMNS]

CONNECTION_STRING = (
    "Driver={ODBC Driver 18 for SQL Server};"
    f"Server={SERVER};"
    f"Database={DATABASE};"
    "Trusted_Connection=yes;"
    "Encrypt=no;"
    "TrustServerCertificate=yes;"
    "Connection Timeout=10;"
)


CREATE_TABLE_SQL = f"""
IF OBJECT_ID(N'{TABLE}', N'U') IS NULL
BEGIN
    CREATE TABLE {TABLE} (
        id BIGINT IDENTITY(1,1) NOT NULL PRIMARY KEY,
        label CHAR(1) NOT NULL,
        {', '.join(f'[{column}] FLOAT NOT NULL' for column in LANDMARK_COLUMNS)},
        imported_at DATETIME2 NOT NULL CONSTRAINT DF_hand_landmarks_imported_at DEFAULT SYSUTCDATETIME()
    );
END
"""


def table_columns(cursor):
    cursor.execute(
        """
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
        """,
        "dbo",
        "hand_landmarks",
    )
    return {row[0].lower() for row in cursor.fetchall()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Import landmarks.csv into SQL Server.")
    parser.add_argument("--csv", default=DATA_PATH, help="CSV path (default: backend/data/landmarks.csv)")
    parser.add_argument("--create-table", action="store_true", help="Create dbo.hand_landmarks only if it is missing")
    parser.add_argument("--clear", action="store_true", help="Delete existing rows before importing (use deliberately)")
    parser.add_argument("--batch-size", type=int, default=500, help="Rows per insert batch (default: 500)")
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"CSV file not found: {args.csv}", file=sys.stderr)
        return 1
    if args.batch_size < 1:
        print("--batch-size must be at least 1", file=sys.stderr)
        return 1

    try:
        with open(args.csv, newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames != EXPECTED_COLUMNS:
                print("CSV does not have the expected label + 63 landmark columns.", file=sys.stderr)
                return 1
            rows = [
                tuple([row["label"], *[float(row[column]) for column in LANDMARK_COLUMNS]])
                for row in reader
            ]
    except (OSError, ValueError) as error:
        print(f"Could not read CSV: {error}", file=sys.stderr)
        return 1

    placeholders = ", ".join("?" for _ in EXPECTED_COLUMNS)
    column_list = ", ".join(f"[{column}]" for column in EXPECTED_COLUMNS)
    insert_sql = f"INSERT INTO {TABLE} ({column_list}) VALUES ({placeholders})"

    try:
        with pyodbc.connect(CONNECTION_STRING) as connection:
            cursor = connection.cursor()
            if args.create_table:
                cursor.execute(CREATE_TABLE_SQL)

            missing = set(EXPECTED_COLUMNS) - table_columns(cursor)
            if missing:
                print(
                    f"{TABLE} is missing required columns: {', '.join(sorted(missing))}. "
                    "Run with --create-table only if the table is absent, or adjust the existing table schema.",
                    file=sys.stderr,
                )
                return 1

            if args.clear:
                cursor.execute(f"DELETE FROM {TABLE}")
                print("Existing rows deleted.")

            cursor.fast_executemany = True
            for start in range(0, len(rows), args.batch_size):
                cursor.executemany(insert_sql, rows[start : start + args.batch_size])
            connection.commit()

            total = cursor.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchval()
        print(f"Imported {len(rows):,} samples into {TABLE}. Table now has {total:,} rows.")
        return 0
    except pyodbc.Error as error:
        print(f"SQL Server import failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
