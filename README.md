# sentinelCam Worker Development

The worker is the processing backend of the sentinelCam stack.

It captures a video source, runs object detection and optional pose inference with YOLO, and exposes the result through a small HTTP API for the browser UI in [`sentinelCam-web`](https://github.com/okixk/sentinelCam-web).

## What this repo does

- captures video from:
  - a local webcam
  - a device path
  - a file
  - a remote stream URL
- runs YOLO-based detection and optional pose inference
- supports runtime model switching
- supports pose, overlay, and inference toggles at runtime
- serves:
  - `GET /api/state`
  - `POST /api/cmd`
  - `GET /health`
  - `POST /api/webrtc/offer` when WebRTC is enabled
  - `GET /stream.mjpg` as MJPEG output and fallback
  - `GET /frame.jpg` for the latest snapshot (with YOLO overlay)
  - `GET /frame-raw.jpg` for the latest snapshot without YOLO overlay

## Where this repo fits

Typical flow:

`camera -> worker -> web browser`

Future flow with edge nodes:

`camera -> sentinelCam-edge -> sentinelCam-worker -> sentinelCam-web`

## Related repositories

- **Web UI:** [`sentinelCam-web`](https://github.com/okixk/sentinelCam-web)  
  Static browser frontend plus optional local reverse proxy for controlling and viewing the worker.

- **Edge capture node:** [`sentinelCam-edge`](https://github.com/okixk/sentinelCam-edge)  
  Planned lightweight camera-side component for forwarding streams into the worker.

## Quick start

### Linux / macOS

```bash
./run.sh
```

### Windows

```bat
run.bat
```

The launcher can:

- create a local virtual environment in `.runtime`
- install Python dependencies
- ask for a camera index or stream source
- ask whether to bind to localhost or all interfaces when `--host` is not passed
- start `webcam.py`

By default, the worker binds to `127.0.0.1` and uses `DEFAULT_STREAM_MODE=auto`, which prefers WebRTC when available and keeps MJPEG as fallback.

## Stream modes

The worker supports three web stream modes:

- `--stream auto`  
  Preferred mode. Uses WebRTC when the WebRTC dependencies are available, with MJPEG fallback still exposed.

- `--stream webrtc`  
  Forces WebRTC mode. The worker serves:
  - `POST /api/webrtc/offer`
  - `GET /webrtc-test`
  - `GET /api/state`
  - `POST /api/cmd`
  - `GET /health`
  - `GET /stream.mjpg` as fallback
  - `GET /frame.jpg` latest snapshot with YOLO overlay
  - `GET /frame-raw.jpg` latest snapshot without YOLO overlay

- `--stream mjpeg`  
  Forces MJPEG-only mode and serves:
  - `GET /stream.mjpg`
  - `GET /frame.jpg`
  - `GET /frame-raw.jpg`
  - `GET /api/state`
  - `POST /api/cmd`
  - `GET /health`

## Connect the web UI

Open the UI from [`sentinelCam-web`](https://github.com/okixk/sentinelCam-web), then connect it to:

```text
http://127.0.0.1:8080
```

or, for LAN access:

```text
http://WORKER_IP:8080
```

The current web UI prefers WebRTC and falls back to MJPEG automatically when the worker does not expose WebRTC.

## Common usage

Use a local webcam:

```bash
./run.sh --source 0
```

Use a remote stream:

```bash
./run.sh --source http://HOST:PORT/stream.mjpg
```

Run the worker on the LAN:

```bash
./run.sh --host 0.0.0.0 --port 8080
```

Force WebRTC:

```bash
./run.sh --stream webrtc
```

Use a higher WebRTC bitrate:

```bash
./run.sh --webrtc-bitrate 2500
```

Force MJPEG only:

```bash
./run.sh --stream mjpeg
```

Use a higher stream quality preset:

```bash
./run.sh --stream-quality ultra
```

Window only:

```bash
./run.sh --no-web --window
```

Force H.264/VP8 codec for WebRTC (VP9/AV1 require aiortc support):

```bash
./run.bat --stream webrtc --webrtc-codec h264
```

Restrict WebRTC media to a specific UDP port range (useful for firewalls):

```bash
./run.sh --webrtc-port-min 50000 --webrtc-port-max 51000
```

Force CPU encoding (disable GPU auto-detection):

```bash
./run.sh --webrtc-gpu 0
```

Disable frame sharing (separate encoder per client):

```bash
./run.sh --webrtc-frame-sharing 0
```

## Controls

The worker supports runtime controls such as:
- next / previous model
- pose on / off
- overlay on / off
- inference on / off
- quit

These can be triggered from the web UI or from local hotkeys when a preview window is enabled.

## Configuration

Shared launcher defaults live in `webcam.properties`.

Important defaults:

- `DEFAULT_WEB_HOST=127.0.0.1`
- `DEFAULT_STREAM_MODE=auto`
- `DEFAULT_WEBRTC_BITRATE_KBPS=2500`
- `DEFAULT_WEBRTC_PORT_MIN=50000` / `DEFAULT_WEBRTC_PORT_MAX=51000`
- `DEFAULT_WEBRTC_GPU=1` (auto-detect GPU encoder; `0` = force CPU)
- `DEFAULT_WEBRTC_FRAME_SHARING=1` (encode once, share to all clients)
- `DEFAULT_STREAM_QUALITY=high`
- `DEFAULT_JPEG_QUALITY=88`

Useful security-related settings:

- `WEB_AUTH_TOKEN=...`
- `WEB_ALLOWED_ORIGINS=http://127.0.0.1:3000,http://localhost:3000`
- `WEB_MAX_CMD_BYTES=8192`

## Project structure

- `webcam.py` - main processing app
- `stream_server.py` - MJPEG stream + control API
- `webrtc_server.py` - worker-side WebRTC signaling and media server
- `run.sh` - Linux/macOS launcher
- `run.bat` - Windows launcher
- `webcam.properties` - shared launcher defaults
- `Dockerfile` - multi-stage Docker build (Python 3.12-slim)
- `docker-compose.worker.yml` - Docker Compose config
- `docker-compose.linux.yml` - optional Linux host-network override for loopback-only workers
- `docker-compose.worker-cam.yml` - optional Linux webcam passthrough override
- `.dockerignore` - excludes `.git`, caches, and non-runtime files from the image
- `requirements.txt` - Python dependencies for Docker and pip installs

## Docker

The image uses a multi-stage build based on `python:3.12-slim`, runs as a non-root user (`sentinelcam`), and includes a built-in healthcheck on `/health`.

Default container entrypoint arguments:

```
--host 0.0.0.0 --port 8080 --no-window --stream auto
```

### Build

```bash
docker build -t sentinelcam-worker .
```

### Run (with webcam, Linux only)

Webcam passthrough via `--device` requires Linux. It does **not** work on Windows or macOS Docker Desktop.

```bash
docker run --rm -p 8080:8080 --device /dev/video0 sentinelcam-worker --source 0
```

To keep the worker accessible only on localhost (recommended for local use):

```bash
docker run --rm -p 127.0.0.1:8080:8080 --device /dev/video0 sentinelcam-worker --source 0
```

### Run (with remote stream)

This works on all platforms (Linux, macOS, Windows) because no device passthrough is needed.

```bash
docker run --rm -p 127.0.0.1:8080:8080 sentinelcam-worker --source http://HOST:PORT/stream.mjpg
```

### WebRTC UDP ports

To use WebRTC, expose the ICE UDP port range:

```bash
docker run --rm -p 127.0.0.1:8080:8080 -p 50000-51000:50000-51000/udp sentinelcam-worker --source 0
```

### Environment Variables

- `WEB_AUTH_TOKEN` — Bearer token for API authentication
- `WEB_ALLOWED_ORIGINS` — Comma-separated CORS origin whitelist

Example:

```bash
docker run --rm -p 127.0.0.1:8080:8080 \
  -e WEB_AUTH_TOKEN=mytoken \
  -e WEB_ALLOWED_ORIGINS=http://localhost:3000 \
  sentinelcam-worker --source http://HOST:PORT/stream.mjpg
```

### Docker Compose

```bash
docker compose -f docker-compose.worker.yml up
```

The base compose file is now cross-platform:

- default source: `testsrc`
- works on Linux, macOS, and Windows
- maps port `8080` and the WebRTC UDP range
- supports `WORKER_TOKEN` as an environment variable for auth

This gives you the same smoke-test startup path everywhere:

```bash
docker compose -f docker-compose.worker.yml up -d --build
```

Use a remote stream on any platform:

```bash
WORKER_SOURCE=rtsp://HOST:PORT/stream docker compose -f docker-compose.worker.yml up -d --build
```

For a loopback-only worker on Linux (useful with the legacy `sentinelCam-web`
`docker-compose.linux.yml` override that proxies to `http://127.0.0.1:8080`):

```bash
docker compose -f docker-compose.worker.yml -f docker-compose.linux.yml up -d --build
```

Use a real webcam on Linux only:

```bash
docker compose -f docker-compose.worker.yml -f docker-compose.worker-cam.yml up -d --build
```

Use a real webcam on Linux with loopback-only host networking:

```bash
WORKER_SOURCE=0 docker compose -f docker-compose.worker.yml -f docker-compose.worker-cam.yml -f docker-compose.linux.yml up -d --build
```

Keep `docker-compose.linux.yml` last so its loopback-only bind overrides the webcam compose command.

Use a different Linux camera device:

```bash
WORKER_VIDEO_DEVICE=/dev/video2 docker compose -f docker-compose.worker.yml -f docker-compose.worker-cam.yml up -d --build
```

## Notes

- The worker binds to localhost by default for safer local use.
- If `sentinelCam-web` runs in Docker on Linux, the default web compose expects the worker to be reachable through `host.docker.internal:8080`. A locally started worker should use `./run.sh --host 0.0.0.0`. If the worker stays on `127.0.0.1`, use the legacy `docker-compose.linux.yml` override in the web repo, and this repo's `docker-compose.linux.yml` too when the worker itself also runs in Docker.
- For remote browser access, prefer a reverse proxy or the `sentinelCam-web` local web server instead of exposing the worker directly.
- If you enable worker auth, the browser UI should usually connect through the proxy/web server, which injects the worker token server-side.
- WebRTC support requires `aiohttp`, `aiortc`, and `av`.
- GPU H.264 encoding is auto-detected at startup (NVIDIA NVENC → AMD AMF → Intel QSV → CPU libx264). Disable with `--webrtc-gpu 0` if needed.
- Frame sharing encodes each video frame once and distributes the encoded packets to all connected WebRTC clients, significantly reducing CPU/GPU load with multiple viewers.
- WebRTC ICE UDP ports default to 50000–51000 for easier firewall configuration. Set both to `0` for OS-assigned ports.

## Status

Active core backend of the sentinelCam stack.
