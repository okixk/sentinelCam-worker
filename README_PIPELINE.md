# sentinelCam-worker — web-streaming pipeline mode

This branch (`feature/web-streaming-pipeline`) introduces a new operating
mode where the worker dials outbound to a public `sentinelCam-web`
instance, receives raw JPEG frames from the camera (a Raspberry Pi on
the other side of the web server), and pushes processed frames with the
detection overlay back to the same web server.

The web server is the **only** publicly reachable endpoint. The Pi and
this worker both reach it over the VPN (UDP/1194 WireGuard) or the
public internet.

## Topology

```
                                public internet
[ Raspberry Pi ]                                            [ Browser ]
       |  wss /api/ingest/{cam_id}                               ^
       |  bearer sc-cam-<id>-...                                 |  WebRTC (planned)
       v                                                         |  MJPEG (today)
                  +---------------------------------------+      |
                  |  sentinelCam-web (Apache + FastAPI)   |------+
                  |   FrameHub: raw + processed lanes     |
                  +-------------+-------------------------+
                                ^
                                |  wss /api/worker/connect
                                |  bearer sc-wrk-<id>-...
                                |  bidirectional binary frames + JSON heartbeats
                                v
                       [ sentinelCam-worker (this repo) ]
                            YOLO + pose + overlay
```

## Quick start (skeleton mode)

The stub processor draws a "PROCESSING" banner so you can verify the
pipeline end-to-end before plugging YOLO in.

1. In the web admin UI, issue a worker token (`sc-wrk-<id>-<32_hex>`). The
   token is shown exactly once — copy it.
2. Set environment variables on the worker host:

   ```bash
   export WEB_URL="https://<web-server>"          # public or VPN URL
   export WEB_TOKEN="sc-wrk-1-deadbeef..."        # from step 1
   export WORKER_NAME="lab-h200"                  # informational
   export JPEG_QUALITY=88
   ```

3. Install dependencies and run:

   ```bash
   pip install -r requirements-pipeline.txt
   python -m web_pipeline.run
   ```

You should see `worker connected; awaiting frames` and the admin status
panel on the web server should flip the worker to **online**.

## Wire protocol

Web <-> worker frames share a 17-byte binary envelope:

```
[1 byte : message type]
[8 bytes: camera id, unsigned big-endian]
[8 bytes: capture timestamp in milliseconds since epoch, unsigned big-endian]
[N bytes: payload (JPEG bitstream for frame messages)]
```

| Type | Direction         | Meaning                                |
|------|-------------------|----------------------------------------|
| 0x01 | web -> worker     | Raw JPEG straight from the Pi          |
| 0x02 | worker -> web     | JPEG with overlay (return path)        |
| 0x03 | web -> worker     | Hint to emit a fresh keyframe (future) |

Heartbeats / status / settings updates are JSON text frames on the same
socket:

```json
{"type": "heartbeat", "status": {"uptime_s": 42, "frames_in": 1234, ...}}
```

The web server stores `last_status` into the `workers` table and surfaces
it on the admin status panel.

## Plan for real inference

The current `web_pipeline.client.stub_overlay` is a placeholder. Real
inference will plug in by replacing it with a callable that:

1. Decodes the incoming JPEG (already done in `client.py`).
2. Runs YOLO detection + pose (this repo already has the model loading
   code in `webcam.py`; the heavy module-level state needs to be pulled
   into a small `Inference` class).
3. Draws the overlay (existing code path in `webcam.py`).
4. Encodes the result. **On the H200 + libnvjpeg this should be NVENC
   JPEG so we avoid the CPU JPEG re-encode.**

The follow-up commit on this branch will:

- factor the existing YOLO + pose pipeline out of `webcam.py` into
  `web_pipeline/inference.py`,
- add NVENC-accelerated JPEG encode where available (libjpeg-turbo
  fallback),
- expose runtime settings (model variant, conf threshold, pose
  on/off) that read from `app_settings` in the web DB and apply on
  the next frame.

## Legacy modes

The old `webcam.py` / `stream_server.py` / `webrtc_server.py` are still
present and unchanged on this branch — they handle the case where the
worker has its own camera and serves browsers directly. Both modes will
coexist on this branch until the new pipeline is fully fleshed out.
