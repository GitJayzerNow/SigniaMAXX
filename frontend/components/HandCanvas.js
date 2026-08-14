"use client";

import { useEffect, useRef } from "react";
import { Hands, HAND_CONNECTIONS } from "@mediapipe/hands";
import { Camera } from "@mediapipe/camera_utils";
import { drawConnectors, drawLandmarks } from "@mediapipe/drawing_utils";

/**
 * Runs MediaPipe Hands entirely client-side (offline once model files are
 * cached by the browser/service worker) on the given <video> element,
 * draws a skeleton overlay on an internal canvas, and reports normalized
 * 21-point landmarks via onLandmarks.
 */
export default function HandCanvas({ videoRef, onLandmarks, onReady }) {
  const canvasRef = useRef(null);
  const handsRef = useRef(null);
  const cameraRef = useRef(null);

  useEffect(() => {
    if (!videoRef.current) return;

    const hands = new Hands({
      locateFile: (file) => `/mediapipe/hands/${file}`,
    });
    hands.setOptions({
      maxNumHands: 1,
      modelComplexity: 1,
      minDetectionConfidence: 0.7,
      minTrackingConfidence: 0.7,
    });

    hands.onResults((results) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      canvas.width = results.image.width;
      canvas.height = results.image.height;
      ctx.save();
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        const lms = results.multiHandLandmarks[0];
        drawConnectors(ctx, lms, HAND_CONNECTIONS, { color: "#2dd4d1", lineWidth: 3 });
        drawLandmarks(ctx, lms, { color: "#f8ce57", lineWidth: 1, radius: 3 });
        onLandmarks(lms.map((p) => ({ x: p.x, y: p.y, z: p.z })));
      } else {
        onLandmarks(null);
      }
      ctx.restore();
    });

    handsRef.current = hands;

    const camera = new Camera(videoRef.current, {
      onFrame: async () => {
        if (handsRef.current) {
          await handsRef.current.send({ image: videoRef.current });
        }
      },
      // Use a 16:9 source to better match the full-screen stage and reduce
      // crop-related drift between the rendered video and landmark canvas.
      width: 1280,
      height: 720,
    });
    cameraRef.current = camera;
    camera.start().then(() => onReady && onReady());

    return () => {
      camera.stop();
      hands.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return <canvas ref={canvasRef} />;
}
