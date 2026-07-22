"""Low-latency local WebSocket service for browser-originated JPEG frames."""
import asyncio
import json
import logging
import os
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from rppg.input.processor import TrustedInputPipeline
from rppg.input.types import FramePacket
from runtime.rppg_pipeline import RPPGPipeline

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(os.getenv("RPPG_CONFIG", ROOT / "configs/inference_config.yaml"))
PREVIEW_INTERVAL_NS = int(1e9 / float(os.getenv("RPPG_PREVIEW_FPS", "5")))
JPEG_QUALITY = int(os.getenv("RPPG_PREVIEW_JPEG_QUALITY", "72"))

app = FastAPI(title="rPPG Inference Service", version="0.1.0")
logger = logging.getLogger(__name__)


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as source:
        return yaml.safe_load(source)


class MeasurementWorker:
    """Owns one stateful pipeline; one worker must not be shared by simultaneous sessions."""

    def __init__(self):
        config = load_config()
        self.pipeline = RPPGPipeline(config)
        self.input_pipeline = TrustedInputPipeline(config, detector=self.pipeline.detector)
        self.frame_id = 0
        self.last_preview_ns = 0
        self.last_input_status = None

    def process_jpeg(self, payload: bytes):
        frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("invalid JPEG frame")
        timestamp_ns = time.monotonic_ns()
        self.frame_id += 1
        if self.frame_id == 1:
            logger.info("Received first browser frame: %dx%d, %d bytes", frame.shape[1], frame.shape[0], len(payload))
        items = self.input_pipeline.process(FramePacket(frame, timestamp_ns, self.frame_id, source="web"))
        if not items:
            return None
        item = items[-1]
        result = self.pipeline.process_input(item)
        message = {
            "type": "MEASUREMENT_UPDATED",
            "timestamp": int(time.time() * 1000),
            "heartRate": result.get("hr"),
            "status": result.get("measurement_state", "waiting"),
            "inputStatus": result.get("input_status", "unknown"),
            "confidence": self._confidence(result),
            "signalQuality": self._signal_quality(item),
            "framesProcessed": self.frame_id,
        }
        if item.status.value != self.last_input_status:
            logger.info("Input status changed: session frame=%d status=%s", self.frame_id, item.status.value)
            self.last_input_status = item.status.value
        previews = []
        if timestamp_ns - self.last_preview_ns >= PREVIEW_INTERVAL_NS:
            self.last_preview_ns = timestamp_ns
            pos = self._encode_preview(self._draw_pos(item))
            efficient = self._encode_preview(item.model_input_face)
            if pos is not None: previews.append((1, pos))
            if efficient is not None: previews.append((2, efficient))
        return message, previews

    @staticmethod
    def _confidence(result):
        candidates = (result.get("detail") or {}).get("candidates") or []
        scores = [candidate.get("sqi") for candidate in candidates if candidate.get("sqi") is not None]
        return max(scores, default=0.0)

    @staticmethod
    def _signal_quality(item):
        if item.quality is None:
            return 0.0
        return max(0.0, min(1.0, float(item.quality.skin_ratio)))

    @staticmethod
    def _draw_pos(item):
        if item.aligned_face is None:
            return None
        display = item.aligned_face.copy()
        styles = {"forehead": (0, 255, 0), "cheek_l": (0, 165, 255), "cheek_r": (0, 165, 255)}
        for name, color in styles.items():
            region = item.roi_boxes.get(name)
            if region is None:
                continue
            if isinstance(region, np.ndarray):
                cv2.polylines(display, [region.astype(np.int32)], True, color, 1)
            else:
                x1, y1, x2, y2 = region
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 1)
        return display

    @staticmethod
    def _encode_preview(image):
        if image is None:
            return None
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return encoded.tobytes() if ok else None


def preview_packet(kind: int, payload: bytes) -> bytes:
    # RPPG + version + preview kind; the remaining bytes are a complete JPEG.
    return b"RPPG" + bytes((1, kind)) + payload


@app.get("/health")
def health():
    return {"status": "UP"}


@app.websocket("/ws/inference/{session_id}")
async def inference(websocket: WebSocket, session_id: str):
    await websocket.accept()
    try:
        worker = await asyncio.to_thread(MeasurementWorker)
    except Exception as error:
        logger.exception("Failed to initialize inference worker")
        await websocket.send_json({"type": "MEASUREMENT_ERROR", "sessionId": session_id, "message": str(error)})
        await websocket.close(code=1011)
        return

    latest_frame: asyncio.Queue[bytes] = asyncio.Queue(maxsize=1)

    async def receive_frames():
        received = 0
        while True:
            payload = await websocket.receive_bytes()
            received += 1
            if received == 1 or received % 100 == 0:
                logger.info("Received inference frame bytes: session=%s count=%d size=%d", session_id, received, len(payload))
            if latest_frame.full():
                latest_frame.get_nowait()
            latest_frame.put_nowait(payload)

    async def process_frames():
        while True:
            payload = await latest_frame.get()
            try:
                logger.debug("Processing inference frame: session=%s size=%d", session_id, len(payload))
                result = await asyncio.to_thread(worker.process_jpeg, payload)
                if result is not None:
                    measurement, previews = result
                    measurement["sessionId"] = session_id
                    await websocket.send_text(json.dumps(measurement, separators=(",", ":")))
                    for kind, preview in previews:
                        await websocket.send_bytes(preview_packet(kind, preview))
            except Exception as error:
                logger.exception("Frame processing failed for session %s", session_id)
                await websocket.send_json({"type": "MEASUREMENT_ERROR", "sessionId": session_id, "timestamp": int(time.time() * 1000), "status": "FAILED", "message": str(error)})
                raise

    receiver = asyncio.create_task(receive_frames())
    processor = asyncio.create_task(process_frames())
    try:
        await asyncio.gather(receiver, processor)
    except WebSocketDisconnect:
        pass
    finally:
        receiver.cancel()
        processor.cancel()
