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
- `DEFAULT_WEBRTC_BITRATE_KBPS=-1`
- `DEFAULT_WEBRTC_FPS=0`
- `DEFAULT_STREAM_QUALITY=auto`
- `DEFAULT_JPEG_QUALITY=88`

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
