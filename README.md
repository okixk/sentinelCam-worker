# sentinelCam Worker

The worker is the processing backend of the sentinelCam stack.

It captures a video source, runs object detection / pose inference with YOLO, and exposes the processed result as an MJPEG stream plus a small HTTP control API. The browser UI in [`sentinelCam-web`](https://github.com/okixk/sentinelCam-web) connects to this repo.

## What this repo does

- captures video from:
  - a local webcam
  - a device path
  - a file
  - a remote stream URL
- runs YOLO-based inference
- supports runtime model switching
- supports pose toggling, overlay toggling, and inference toggling
- serves:
  - `GET /stream.mjpg`
  - `GET /api/state`
  - `POST /api/cmd`
  - `GET /health`

## Where this repo fits

Typical flow:

`camera -> worker -> web browser`

Future flow with edge nodes:

`camera -> sentinelCam-edge -> sentinelCam-worker -> sentinelCam-web`

## Related repositories

- **Web UI:** [`sentinelCam-web`](https://github.com/okixk/sentinelCam-web)  
  Static browser frontend that displays the worker stream and sends control commands.

- **Edge capture node:** [`sentinelCam-edge`](https://github.com/okixk/sentinelCam-edge)  
  Planned lightweight camera-side component that will forward a stream into the worker.

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
- create a local virtual environment
- install Python dependencies
- ask for a camera or stream source
- start the worker

By default, the worker serves its stream on port `8080`.

Example output target:

```text
http://127.0.0.1:8080/stream.mjpg
```

## Connect the web UI

Open the UI from [`sentinelCam-web`](https://github.com/okixk/sentinelCam-web), then connect it to:

```text
http://WORKER_IP:8080
```

The web UI uses:
- `/stream.mjpg` for the live stream
- `/api/state` for status
- `/api/cmd` for controls

## Common usage

Use a local webcam:

```bash
./run.sh --source 0
```

Use a remote stream:

```bash
./run.sh --source http://HOST:PORT/stream.mjpg
```

Run headless web mode:

```bash
./run.sh --web --host 0.0.0.0 --port 8080
```

Window only:

```bash
./run.sh --no-web --window
```

## Controls

The worker supports runtime controls such as:
- next / previous model
- pose on / off
- overlay on / off
- inference on / off
- quit

These can be triggered from the web UI or from local hotkeys.

## Project structure

- `webcam.py` — main processing app
- `stream_server.py` — MJPEG stream + control API
- `run.sh` — Linux/macOS launcher
- `run.bat` — Windows launcher
- `webcam.properties` — shared defaults for the launchers

## Notes

- There is no built-in auth layer in the worker HTTP server.
- For real deployments, put it behind a reverse proxy, VPN, or private network.
- The worker can already run without `sentinelCam-edge`. The edge repo is planned for distributed camera-side capture later.

## Status

Active core backend of the sentinelCam stack.
