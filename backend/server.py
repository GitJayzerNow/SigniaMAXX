"""
ASL Fingerspelling Recognition Server
Receives hand landmark vectors from the Next.js frontend over WebSocket,
runs them through a PyTorch classifier (GPU-accelerated via CUDA when
available, e.g. RTX 5050), and returns the predicted letter (A-Z) or
digit (0-9).

Fully offline: no external API calls, model loaded from local disk.
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO

from model_def import LandmarkClassifier, get_device

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "classifier.pt")

app = Flask(__name__)
app.config["SECRET_KEY"] = "asl-local-dev"
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

_device = get_device()
_model = None
_labels = None


def load_model():
    global _model, _labels
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"No trained model found at {MODEL_PATH}. "
                "Run `python train.py` after collecting data with `python collect_data.py`."
            )
        checkpoint = torch.load(MODEL_PATH, map_location=_device)
        model = LandmarkClassifier(
            num_classes=checkpoint["num_classes"],
            input_dim=checkpoint.get("input_dim", 63),
        ).to(_device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        _model = model
        _labels = checkpoint["labels"]
        print(f"Model loaded on {_device}" + (f" ({torch.cuda.get_device_name(0)})" if _device.type == "cuda" else ""))
    return _model, _labels


def landmarks_to_features(landmarks):
    """
    landmarks: list of 21 {x, y, z} dicts (MediaPipe hand landmark format).
    Normalizes relative to the wrist (landmark 0) and scales by hand size,
    so the model is invariant to hand position/distance from camera.
    """
    pts = np.array([[p["x"], p["y"], p["z"]] for p in landmarks], dtype=np.float32)
    wrist = pts[0].copy()
    pts -= wrist
    scale = np.linalg.norm(pts[9])  # middle finger MCP as scale reference
    if scale < 1e-6:
        scale = 1.0
    pts /= scale
    return pts.flatten()


@app.route("/health")
def health():
    try:
        load_model()
        return {"status": "ok", "model_loaded": True, "device": str(_device)}
    except FileNotFoundError as e:
        return {"status": "no_model", "detail": str(e), "device": str(_device)}, 200


@socketio.on("predict")
def handle_predict(data):
    """
    Expects: { landmarks: [{x,y,z}, ...21 items] }
    Emits: { label: "A", confidence: 0.93 }
    """
    try:
        model, labels = load_model()
    except FileNotFoundError as e:
        socketio.emit("prediction_error", {"error": str(e)})
        return

    landmarks = data.get("landmarks")
    if not landmarks or len(landmarks) != 21:
        socketio.emit("prediction_error", {"error": "Expected 21 landmarks"})
        return

    features = landmarks_to_features(landmarks)
    x = torch.from_numpy(features).unsqueeze(0).to(_device)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]
        idx = int(torch.argmax(probs).item())
        confidence = float(probs[idx].item())

    label = labels[idx]
    socketio.emit("prediction", {"label": label, "confidence": round(confidence, 3)})


if __name__ == "__main__":
    print(f"Starting ASL recognition server on http://localhost:5000 (device: {_device})")
    socketio.run(app, host="0.0.0.0", port=5000, debug=True, allow_unsafe_werkzeug=True)
