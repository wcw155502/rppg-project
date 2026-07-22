# rPPG WebSocket inference adapter

Run from the Python repository root:

```powershell
uvicorn services.rppg_api.app:app --host 127.0.0.1 --port 8090
```

The first WebSocket connection loads SCRFD and EfficientPhys. Each binary message is one JPEG frame.
The server keeps only the newest unprocessed frame to prevent latency growth.
