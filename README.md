# sentinelCam Worker

`sentinelCam-worker` is the capture and inference backend of the sentinelCam stack.

It opens a video source, runs YOLO detection and optional pose inference, exposes a small HTTP API, and streams video to `sentinelCam-web` through WebRTC with MJPEG fallback.

## What This Repo Does

- captures video from:
  - local webcam index such as `0`
  - device path
  - file
  - remote stream URL
  - generated test source via `synthetic` / `testsrc`
- runs YOLO detection and optional pose inference
- supports runtime model switching and worker commands
- serves:
  - `GET /health`
  - `GET /api/state`
  - `POST /api/cmd`
  - `POST /api/webrtc/offer` when WebRTC is enabled
  - `GET /stream.mjpg`
  - `GET /frame.jpg`
  - `GET /frame-raw.jpg`

## Where It Fits

```text
camera/source -> sentinelCam-worker -> sentinelCam-web -> browser
```

## Platform Matrix

| Platform | Recommended start | Docker support |
|---|---|---|
| Windows | local `run.bat` | Docker only for synthetic/testsrc or remote streams |
| Linux | local `run.sh` or Docker | full Docker including webcam passthrough |
| macOS | local `run.sh` | Docker only for synthetic/testsrc or remote streams |

Important:

- For real webcams, Windows and macOS should run the worker locally.
- Linux can run the worker locally or in Docker.
- For first-time smoke tests on any platform, `--source synthetic` is the easiest path.

## 1. Prerequisites

Windows:

- Python 3 with `py` launcher
- PowerShell or Command Prompt

Linux:

- Python 3 with `venv`
- Bash

macOS:

- Python 3 with `venv`
- Bash or zsh

Optional for web integration:

- `sentinelCam-web`
- shared worker token

Optional for context detection:

- Ollama running locally
- a vision model such as `gemma3:4b`, `llava`, `moondream`, or `llama3.2-vision`

## 2. Quick Smoke Test Without A Camera

Use this first if you only want to confirm that the worker starts and responds.

### Windows

```powershell
cd C:\path\to\sentinelCam-worker
.\run.bat --source synthetic --host 0.0.0.0 --no-window --stream auto
```

### Linux

```bash
cd ~/sentinelCam-worker
bash ./run.sh --source synthetic --host 0.0.0.0 --no-window --stream auto
```

## Context Detection With Ollama

The worker can ask a local Ollama vision model for a short "what is currently happening" sentence. This is opt-in. It runs in a background thread after you enable it, publishes the latest result in `GET /api/state` under `context`, and can draw the sentence on the stream overlay.

Context profiles choose the default Ollama model when `--context-model auto` is used:

- `low` - CPU-only machines, pulls `moondream`.
- `mid` - small GPU / Apple Silicon, pulls `gemma3:4b`.
- `high` - stronger GPU, pulls `llama3.2-vision:11b`.
- `max` - powerful GPU, pulls `llava:13b`.
- `auto` - chooses one of the above from the local hardware.

Enable context detection. The launcher installs Ollama natively for the current platform and pulls the selected model unless `DEFAULT_CONTEXT_SETUP_OLLAMA=0` is set:

```bash
bash ./run.sh --context --context-profile mid --context-model auto --source synthetic --no-window
```

Useful controls:

```bash
curl http://127.0.0.1:8080/api/state
curl -X POST http://127.0.0.1:8080/api/cmd -H 'Content-Type: application/json' -d '{"cmd":"context_on"}'
curl -X POST http://127.0.0.1:8080/api/cmd -H 'Content-Type: application/json' -d '{"cmd":"context_off"}'
curl -X POST http://127.0.0.1:8080/api/cmd -H 'Content-Type: application/json' -d '{"cmd":"context_analyze"}'
curl -X POST http://127.0.0.1:8080/api/cmd -H 'Content-Type: application/json' -d '{"cmd":"context_emergency_stop"}'
curl -X POST http://127.0.0.1:8080/api/cmd -H 'Content-Type: application/json' -d '{"cmd":"context_config","enabled":true,"trigger":"person_appears","cooldown":30,"profile":"mid","model":"auto"}'
```

Trigger modes:

- `interval` - analyze periodically.
- `person_appears` - analyze when YOLO first sees a person after none were present.
- `person_present` - analyze while a person is visible, limited by cooldown.
- `manual` - only analyze after `context_analyze`.

`context_analyze` is a one-shot request; it does not enable interval analysis. Use `context_emergency_stop` or the web UI's Stop AI button to disable AI context detection, clear queued frames, and ignore late in-flight results.
On the web stream page, press `c` for one-shot context analysis, `i` for inference, `p` for pose, `o` for overlay, `m` / `n` for model cycling, and `q` to stop the worker.

Tune with `--context-profile`, `--context-model`, `--context-trigger`, `--context-cooldown`, `--context-interval`, `--context-timeout`, `--context-image-width`, `--context-overlay`, `--context-host`, and `--context-prompt`.
Automatic context runs are skipped while YOLO FPS is below `--context-min-yolo-fps` so Ollama does not steal the whole GPU budget from object detection. Pose is also adaptive by default: it only runs when a person is present and pauses while detection FPS is under `--pose-min-yolo-fps`.

Ollama is expected to run natively on the host, not inside Docker. That matters on Windows and macOS because Docker Desktop runs Linux containers inside a VM and generally cannot use the host GPU for Ollama. Install Ollama for the host OS, pull a vision model such as `gemma3:4b`, and point the worker at it with `OLLAMA_HOST` or `--context-host`.

### macOS

```bash
cd ~/sentinelCam-worker
bash ./run.sh --source synthetic --host 0.0.0.0 --no-window --stream auto
```

Then verify:

```text
http://127.0.0.1:8080/health
```

## 3. Start On Windows

Recommended path:

- run locally with `run.bat`
- expose `0.0.0.0` when the web app is in Docker

### Start With A Real Camera

```powershell
cd C:\path\to\sentinelCam-worker
.\run.bat --source 0 --host 0.0.0.0 --no-window --stream auto
```

### Start With A Remote Stream

```powershell
.\run.bat --source http://HOST:PORT/stream.mjpg --host 0.0.0.0 --no-window --stream auto
```

### Start For Use With sentinelCam-web

Set the same token that `sentinelCam-web` uses as `WORKER_TOKEN`:

```powershell
$env:WEB_AUTH_TOKEN = "<same token as sentinelCam-web/.env -> WORKER_TOKEN>"
$env:WEB_ALLOWED_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000"
.\run.bat --source 0 --host 0.0.0.0 --no-window --stream auto
```

Notes:

- If you omit `--source`, `run.bat` will prompt you for a camera or URL.
- If you omit `--host`, `run.bat` will ask whether to use `127.0.0.1` or `0.0.0.0`.
- `--source synthetic` is the quickest no-camera test.

## 4. Start On Linux

### Start With A Real Camera

```bash
cd ~/sentinelCam-worker
bash ./run.sh --source 0 --host 0.0.0.0 --no-window --stream auto
```

### Start With A Remote Stream

```bash
bash ./run.sh --source rtsp://HOST:PORT/stream --host 0.0.0.0 --no-window --stream auto
```

### Start For Use With sentinelCam-web

```bash
export WEB_AUTH_TOKEN="<same token as sentinelCam-web/.env -> WORKER_TOKEN>"
export WEB_ALLOWED_ORIGINS="http://localhost:3000,http://127.0.0.1:3000"
bash ./run.sh --source 0 --host 0.0.0.0 --no-window --stream auto
```

Notes:

- If you prefer localhost-only access and the web app is also local, use `--host 127.0.0.1`.
- If the web app runs in Docker, use `--host 0.0.0.0` so the container can reach the worker.
- If `run.sh` is not executable, `bash ./run.sh ...` is fine.

## 5. Start On macOS

### Start With A Real Camera

```bash
cd ~/sentinelCam-worker
bash ./run.sh --source 0 --host 0.0.0.0 --no-window --stream auto
```

### Start With A Remote Stream

```bash
bash ./run.sh --source http://HOST:PORT/stream.mjpg --host 0.0.0.0 --no-window --stream auto
```

### Start For Use With sentinelCam-web

```bash
export WEB_AUTH_TOKEN="<same token as sentinelCam-web/.env -> WORKER_TOKEN>"
export WEB_ALLOWED_ORIGINS="http://localhost:3000,http://127.0.0.1:3000"
bash ./run.sh --source 0 --host 0.0.0.0 --no-window --stream auto
```

macOS notes:

- On first camera use, macOS may ask you to allow camera access for Terminal, iTerm, or Python.
- `DEFAULT_DEVICE=auto` prefers Apple `mps` when available.

## 6. Connect sentinelCam-web

The web UI usually connects through its proxy, not directly from the browser.

If the web app runs locally outside Docker, the default worker URL is usually:

```text
http://127.0.0.1:8080
```

If the web app runs in Docker on Windows, Linux, or macOS:

- keep the worker on the host
- start the worker with `--host 0.0.0.0`
- let the web container reach it through `host.docker.internal:8080`

Open the web UI at:

```text
http://localhost:3000
```

## 7. Verify The Worker

### Windows PowerShell

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8080/health | Select-Object -ExpandProperty Content
```

### Linux / macOS

```bash
curl http://127.0.0.1:8080/health
```

If you configured `WEB_AUTH_TOKEN`, authenticated API checks look like this:

### Windows PowerShell

```powershell
Invoke-WebRequest -UseBasicParsing `
  -Headers @{ Authorization = "Bearer YOUR_TOKEN" } `
  http://127.0.0.1:8080/api/state | Select-Object -ExpandProperty Content
```

### Linux / macOS

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" http://127.0.0.1:8080/api/state
```

## 8. Stop The Worker

If you started the worker locally:

- press `Ctrl+C` in the worker terminal

If you started it with Docker:

```bash
docker compose -f docker-compose.worker.yml down
```

## Common Commands

Use a local webcam:

```bash
bash ./run.sh --source 0 --host 0.0.0.0 --no-window --stream auto
```

Use generated frames:

```bash
bash ./run.sh --source synthetic --host 0.0.0.0 --no-window --stream auto
```

Use a remote stream:

```bash
bash ./run.sh --source rtsp://HOST:PORT/stream --host 0.0.0.0 --no-window --stream auto
```

Force MJPEG only:

```bash
bash ./run.sh --source 0 --host 0.0.0.0 --no-window --stream mjpeg
```

Force WebRTC:

```bash
bash ./run.sh --source 0 --host 0.0.0.0 --no-window --stream webrtc
```

Raise WebRTC bitrate:

```bash
bash ./run.sh --source 0 --host 0.0.0.0 --no-window --stream auto --webrtc-bitrate 8000
```

Request higher camera FPS:

```bash
bash ./run.sh --source 0 --host 0.0.0.0 --no-window --stream auto --camera-fps 60
```

Show the OpenCV preview window:

```bash
bash ./run.sh --source 0 --host 0.0.0.0 --window --stream auto
```

## Stream Modes

- `--stream auto`
  Preferred mode. Uses WebRTC when available and still exposes MJPEG fallback.
- `--stream webrtc`
  Forces WebRTC mode.
- `--stream mjpeg`
  Forces MJPEG-only mode.

## Worker Security Settings

Shared launcher defaults live in `webcam.properties`.

Important security-related settings:

- `WEB_AUTH_TOKEN`
- `WEB_ALLOWED_ORIGINS`
- `WEB_MAX_CMD_BYTES`
- `DEFAULT_WEB_HOST`

Important runtime defaults:

- `DEFAULT_STREAM_MODE=auto`
- `DEFAULT_PERFORMANCE_PROFILE=auto`
- `DEFAULT_WEBRTC_CODEC=auto`
- `DEFAULT_WEBRTC_BITRATE_KBPS=-1`
- `DEFAULT_WEBRTC_FPS=0`
- `DEFAULT_STREAM_QUALITY=auto`
- `DEFAULT_JPEG_QUALITY=88`
- `DEFAULT_TRACKER_MODE=simple`
- `DEFAULT_YOLO_HALF=1`
- `DEFAULT_CONTEXT_MIN_YOLO_FPS=15.0`
- `DEFAULT_ADAPTIVE_POSE=1`
- `DEFAULT_POSE_MIN_YOLO_FPS=12.0`

Performance notes:

- The default `yolo` preset now scales by VRAM instead of forcing the heaviest GPU model on every CUDA card.
- `DEFAULT_TRACKER_MODE=simple` avoids Ultralytics tracker overhead for live streams. Use `--tracker-mode ultralytics` if you specifically want ByteTrack.
- `DEFAULT_PRESET_ACCEL=yolov8m` keeps the accelerated default responsive on 8-16 GB GPUs; select `yolov8l`, `yolov8x`, or `yolo26x` manually when you want quality over FPS.
- `DEFAULT_YOLO_HALF=1` enables FP16 inference on CUDA.

## Docker

### Cross-Platform Docker Smoke Test

This works on Windows, Linux, and macOS because the default source is `testsrc`:

```bash
docker compose -f docker-compose.worker.yml up -d --build
```

Then open:

```text
http://127.0.0.1:8080/health
```

### Cross-Platform Docker With Remote Stream

Windows PowerShell:

```powershell
$env:WORKER_SOURCE = "http://HOST:PORT/stream.mjpg"
docker compose -f docker-compose.worker.yml up -d --build
```

Linux / macOS:

```bash
WORKER_SOURCE=http://HOST:PORT/stream.mjpg docker compose -f docker-compose.worker.yml up -d --build
```

### Linux Docker With Real Webcam

```bash
WORKER_SOURCE=0 docker compose -f docker-compose.worker.yml -f docker-compose.worker-cam.yml up -d --build
```

Use a different Linux camera device:

```bash
WORKER_VIDEO_DEVICE=/dev/video2 docker compose -f docker-compose.worker.yml -f docker-compose.worker-cam.yml up -d --build
```

Loopback-only Linux Docker worker:

```bash
docker compose -f docker-compose.worker.yml -f docker-compose.linux.yml up -d --build
```

Notes:

- Webcam passthrough in Docker is a Linux-only path.
- On Windows and macOS Docker Desktop, use Docker only for `testsrc` / `synthetic`-style smoke tests or remote streams.
- Keep Ollama native on the host. For Linux Docker workers using host networking, the default `OLLAMA_HOST=http://127.0.0.1:11434` reaches the host Ollama service; otherwise set `OLLAMA_HOST=http://host.docker.internal:11434` or the LAN URL of the Ollama host.

## Project Structure

- `webcam.py` - main processing app
- `stream_server.py` - MJPEG stream and control API
- `webrtc_server.py` - WebRTC signaling and media path
- `run.sh` - Linux/macOS launcher
- `run.bat` - Windows launcher
- `webcam.properties` - shared launcher defaults
- `docker-compose.worker.yml` - standalone worker compose
- `docker-compose.worker-cam.yml` - Linux webcam passthrough override
- `docker-compose.linux.yml` - Linux host-network override
- `requirements.txt` - Python dependencies

## Related Repos

- Web UI: `../sentinelCam-web`
- Edge capture node: `sentinelCam-edge`
