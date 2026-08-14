"""Verify the local Microsoft SQL Server connection used by ASL Recognizer."""

import sys

import pyodbc


CONNECTION_STRING = (
    "Driver={ODBC Driver 18 for SQL Server};"
    # This is the instance verified in SQL Server Management Studio.
    "Server=LAPTOP-PUUSOUD1\\SQLEXPRESS01;"
    "Database=ASLRecognizer;"
    "Trusted_Connection=yes;"
    "Encrypt=no;"
    "TrustServerCertificate=yes;"
    "Connection Timeout=5;"
)


def main() -> int:
    try:
        with pyodbc.connect(CONNECTION_STRING) as connection:
            cursor = connection.cursor()
            cursor.execute("SELECT COUNT(*) FROM dbo.hand_landmarks")
            count = cursor.fetchone()[0]
        print(f"Connected successfully. dbo.hand_landmarks has {count} rows.")
        return 0
    except pyodbc.Error as error:
        print(f"SQL Server connection failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
