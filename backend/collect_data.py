"""
Landmark data collection tool.

Shows your webcam feed, extracts 21 hand landmarks with MediaPipe,
and lets you record labeled samples for each letter/number by pressing
its key on the keyboard. Saves everything to data/landmarks.csv.

Controls:
  - Press any key A-Z or 0-9  -> sets the "current label" to record
  - Hold SPACE                -> records a sample every frame while held
  - Press 'c'                 -> clear current label (stop recording)
  - Press 'q'                 -> quit and save

Usage:
    python collect_data.py

Aim for at least 150-300 samples per class, varying hand angle,
distance from camera, and slight rotation for a more robust model.
"""
import csv
import os
import cv2
import mediapipe as mp

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CSV_PATH = os.path.join(DATA_DIR, "landmarks.csv")
os.makedirs(DATA_DIR, exist_ok=True)

LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def normalize(landmarks):
    pts = [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]
    wx, wy, wz = pts[0]
    pts = [(x - wx, y - wy, z - wz) for x, y, z in pts]
    mx, my, mz = pts[9]
    scale = (mx**2 + my**2 + mz**2) ** 0.5 or 1.0
    pts = [(x / scale, y / scale, z / scale) for x, y, z in pts]
    flat = []
    for x, y, z in pts:
        flat.extend([x, y, z])
    return flat


def main():
    file_exists = os.path.exists(CSV_PATH)
    csv_file = open(CSV_PATH, "a", newline="")
    writer = csv.writer(csv_file)
    if not file_exists:
        header = ["label"] + [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
        writer.writerow(header)

    cap = cv2.VideoCapture(0)
    current_label = None
    recording = False
    sample_count = {label: 0 for label in LABELS}

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ) as hands:
        print("Data collection started. Press a letter/number key to select label, "
              "hold SPACE to record, 'q' to quit.")

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )
                    if recording and current_label:
                        row = [current_label] + normalize(hand_landmarks)
                        writer.writerow(row)
                        sample_count[current_label] += 1

            status = f"Label: {current_label or '-'}  Samples: {sample_count.get(current_label, 0) if current_label else 0}"
            color = (0, 0, 255) if recording else (0, 255, 0)
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, "SPACE=record  c=clear  q=quit", (10, 460),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow("ASL Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                current_label = None
                recording = False
            elif key == 32:  # space
                recording = True
            elif key != 255:
                ch = chr(key).upper()
                if ch in LABELS:
                    current_label = ch
                    recording = False

            if key != 32:
                recording = False

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()
    print(f"Saved dataset to {CSV_PATH}")
    print("Sample counts:", sample_count)


if __name__ == "__main__":
    main()
