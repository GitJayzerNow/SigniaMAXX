"""Collect your own ASL hand-sign samples from a webcam directly into SQL Server.

This is a standalone collection program; it does not start the recognizer UI.

Controls:
  A-Z or 0-9  Select the label you are signing.
  Hold SPACE  Record samples while your sign is visible.
  C           Clear the selected label / stop recording.
  Q           Quit and save all collected samples to SQL Server.

Usage (from backend):
    python collect_to_sql.py
"""

import sys

import cv2
import mediapipe as mp
import pyodbc


SERVER = r"LAPTOP-PUUSOUD1\SQLEXPRESS01"
DATABASE = "ASLRecognizer"
TABLE = "dbo.hand_landmarks"
LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
LANDMARK_COLUMNS = [f"{axis}{index}" for index in range(21) for axis in ("x", "y", "z")]
EXPECTED_COLUMNS = {"label", *LANDMARK_COLUMNS}
CONNECTION_STRING = (
    "Driver={ODBC Driver 18 for SQL Server};"
    f"Server={SERVER};"
    f"Database={DATABASE};"
    "Trusted_Connection=yes;"
    "Encrypt=no;"
    "TrustServerCertificate=yes;"
    "Connection Timeout=10;"
)
INSERT_SQL = (
    f"INSERT INTO {TABLE} ([label], {', '.join(f'[{column}]' for column in LANDMARK_COLUMNS)}) "
    f"VALUES ({', '.join('?' for _ in range(64))})"
)


def normalize(landmarks):
    """Return the same normalized 63-coordinate feature vector used by training."""
    points = [(landmark.x, landmark.y, landmark.z) for landmark in landmarks.landmark]
    wrist_x, wrist_y, wrist_z = points[0]
    points = [(x - wrist_x, y - wrist_y, z - wrist_z) for x, y, z in points]
    middle_x, middle_y, middle_z = points[9]
    scale = (middle_x**2 + middle_y**2 + middle_z**2) ** 0.5 or 1.0
    return [coordinate / scale for point in points for coordinate in point]


def verify_table(cursor):
    cursor.execute(
        "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?",
        "dbo",
        "hand_landmarks",
    )
    columns = {row[0].lower() for row in cursor.fetchall()}
    missing = EXPECTED_COLUMNS - columns
    if missing:
        raise RuntimeError(f"{TABLE} is missing columns: {', '.join(sorted(missing))}")


def main() -> int:
    try:
        connection = pyodbc.connect(CONNECTION_STRING)
        cursor = connection.cursor()
        verify_table(cursor)
    except (pyodbc.Error, RuntimeError) as error:
        print(f"SQL Server setup failed: {error}", file=sys.stderr)
        return 1

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        connection.close()
        print("Could not open the default camera.", file=sys.stderr)
        return 1

    hands_api = mp.solutions.hands
    drawing = mp.solutions.drawing_utils
    styles = mp.solutions.drawing_styles
    selected_label = None
    recording = False
    pending_rows = []
    session_counts = {label: 0 for label in LABELS}

    print("SQL collection started. Select A-Z/0-9, hold SPACE to record, and press Q to save and quit.")
    try:
        with hands_api.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
            while camera.isOpened():
                ok, frame = camera.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                if results.multi_hand_landmarks:
                    hand = results.multi_hand_landmarks[0]
                    drawing.draw_landmarks(frame, hand, hands_api.HAND_CONNECTIONS,
                                           styles.get_default_hand_landmarks_style(),
                                           styles.get_default_hand_connections_style())
                    if recording and selected_label:
                        pending_rows.append((selected_label, *normalize(hand)))
                        session_counts[selected_label] += 1

                selected_count = session_counts.get(selected_label, 0) if selected_label else 0
                state = "RECORDING" if recording and selected_label else "READY"
                color = (0, 210, 255) if state == "RECORDING" else (255, 220, 0)
                cv2.putText(frame, f"{state}  Label: {selected_label or '-'}  Session: {selected_count}",
                            (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2)
                cv2.putText(frame, "A-Z/0-9 select | Hold SPACE record | C clear | Q save & quit",
                            (12, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (255, 255, 255), 1)
                cv2.imshow("ASL SQL Dataset Collector", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("c"):
                    selected_label, recording = None, False
                elif key == 32:
                    recording = True
                elif key != 255:
                    character = chr(key).upper()
                    if character in LABELS:
                        selected_label, recording = character, False
                if key != 32:
                    recording = False

        if pending_rows:
            cursor.fast_executemany = True
            cursor.executemany(INSERT_SQL, pending_rows)
            connection.commit()
        print(f"Saved {len(pending_rows):,} new samples to {DATABASE}.{TABLE}.")
        print("Session counts:", {label: count for label, count in session_counts.items() if count})
        return 0
    except (pyodbc.Error, cv2.error) as error:
        connection.rollback()
        print(f"Collection failed; no new samples were saved: {error}", file=sys.stderr)
        return 1
    finally:
        camera.release()
        cv2.destroyAllWindows()
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
