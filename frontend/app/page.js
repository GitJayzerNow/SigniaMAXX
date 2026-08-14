"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { io } from "socket.io-client";
import HandCanvas from "../components/HandCanvas";

const SOCKET_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:5000";

export default function Home() {
  const videoRef = useRef(null);
  const socketRef = useRef(null);
  const [connected, setConnected] = useState(false);
  const [handsReady, setHandsReady] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [confidence, setConfidence] = useState(0);
  const [error, setError] = useState(null);

  useEffect(() => {
    const socket = io(SOCKET_URL, { transports: ["websocket"] });
    socketRef.current = socket;
    socket.on("connect", () => setConnected(true));
    socket.on("disconnect", () => setConnected(false));
    socket.on("prediction", ({ label, confidence: score }) => {
      setPrediction(label);
      setConfidence(score);
      setError(null);
    });
    socket.on("prediction_error", (data) => setError(data.error));
    return () => socket.disconnect();
  }, []);

  const handleLandmarks = useCallback((landmarks) => {
    if (landmarks && socketRef.current?.connected) {
      socketRef.current.emit("predict", { landmarks });
    } else if (!landmarks) {
      setPrediction(null);
      setConfidence(0);
    }
  }, []);

  const status = error ? "Recognition unavailable" : prediction ? "Recognizing" : handsReady ? "Show a sign" : "Starting camera";

  return (
    <main className="signia-app">
      <video ref={videoRef} className="camera-feed" autoPlay playsInline muted />
      <HandCanvas videoRef={videoRef} onLandmarks={handleLandmarks} onReady={() => setHandsReady(true)} />

      <div className="hand-placement-guide" aria-hidden="true">
        <span>Place your hand here</span>
      </div>

      <header className="brand">
        <span className="brand-pulse" aria-hidden="true" />
        <span>Signia<span>MAXX</span></span>
      </header>

      {!handsReady && (
        <div className="startup-message"><span className="loading-ring" /> Preparing hand tracking</div>
      )}

      <section className="recognition-overlay" aria-live="polite" aria-label="Recognition result">
        <div className="result-label">Recognition result</div>
        <div className="result-value">{prediction || "—"}</div>
        <div className="result-status">
          <span className={`status-dot ${connected ? "online" : ""}`} />
          {status}{prediction && ` · ${Math.round(confidence * 100)}%`}
        </div>
      </section>

      {error && <div className="error-message">{error}</div>}
    </main>
  );
}
