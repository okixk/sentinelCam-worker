# sentinelCam Worker — Setup-Anleitung

Diese Anleitung erklärt die zwei Betriebsmodi des Workers, wie du sie
startest und konfigurierst:

1. **Pipeline-Modus** (Branch `feature/web-streaming-pipeline`):
   Der Worker dialt outbound per WebSocket zu einer öffentlichen
   `sentinelCam-web`-Instanz, holt rohe JPEGs von einer Pi-Kamera und
   schickt verarbeitete Frames (mit YOLO/Pose-Overlay) zurück.
2. **Standalone-Modus** (Default-Branch):
   Der Worker hat seine eigene Kamera (oder eine Test-Source) und bedient
   Browser direkt via WebRTC/MJPEG/HTTP.

Beide Modi koexistieren im Repo. Wähle den, der zu deinem Setup passt.

> Für die Architekturdetails siehe [README.md](README.md) und
> [README_PIPELINE.md](README_PIPELINE.md).

---

## 0. Welcher Modus passt zu mir?

| Setup | Modus | Warum |
|---|---|---|
| Pi nimmt auf, separater GPU-Host (z. B. H200) rechnet, `sentinelCam-web` ist öffentlich erreichbar | **Pipeline** | Web-Server ist das einzige öffentliche Tor; alle Komponenten dialen outbound. |
| Worker hat eigene Kamera (USB/CSI) und soll Browsern direkt Stream liefern (LAN oder VPN) | **Standalone** | Kein zusätzlicher Web-Service nötig. |
| Demo / Smoke-Test ohne echte Kamera | **Standalone** mit `--source synthetic` | Erzeugt Testbilder im Worker selbst. |

---

## 1. Voraussetzungen

- Python 3.11+ (3.12/3.13 getestet)
- `pip` + `venv`
- Linux, macOS oder Windows
- Optional: NVIDIA-GPU mit CUDA für schnellere YOLO-Inferenz
- Optional: Docker, wenn du nicht lokal installieren willst

Bei echter Webcam:

- Linux: `v4l2`-Device, User in `video`-Gruppe
- Windows/macOS: Lokal starten — Webcam-Passthrough in Docker ist auf
  diesen Plattformen unzuverlässig.

---

## 2. Pipeline-Modus (mit sentinelCam-web)

### 2.1 Worker-Token im Web-UI ausstellen

Auf `https://<dein-web-host>/admin` einloggen → **Workers → New worker**.
Der angezeigte Token (`sc-wrk-<id>-<32_hex>`) wird **nur einmal** angezeigt.
Sofort kopieren.

### 2.2 Branch auschecken & Abhängigkeiten installieren

```bash
git checkout feature/web-streaming-pipeline

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-pipeline.txt
```

### 2.3 Umgebungsvariablen setzen

```bash
export WEB_URL="https://sentinelcam.example.com"   # öffentlich oder VPN-URL
export WEB_TOKEN="sc-wrk-1-deadbeef..."            # aus Schritt 2.1
export WORKER_NAME="lab-h200"                       # informativ
export JPEG_QUALITY=88                              # 70..95, Default 88
# Optional:
export LOG_LEVEL=INFO                               # DEBUG für mehr Detail
```

Tipp: in eine `.env` schreiben und `set -a; source .env; set +a` ausführen.

### 2.4 Starten

```bash
python -m web_pipeline.run
```

Erwartete Logs:

```
... starting worker name=lab-h200 -> https://sentinelcam.example.com
... worker connected; awaiting frames
```

Im Admin-UI sollte der Worker auf **online** flippen und Heartbeats
schicken. Sobald die Pi-Kamera (`sentinelCam-web`-Ingest) Frames liefert,
laufen sie durch den Worker und du siehst das Overlay im Browser.

### 2.5 Echtes YOLO statt Stub-Overlay einbauen

Aktuell läuft `web_pipeline.client.stub_overlay` als Platzhalter und zeichnet
nur ein „PROCESSING"-Banner. Wechsel auf echte Inferenz:

1. Eigene Funktion `def my_overlay(jpeg_bytes, capture_ms) -> bytes` schreiben
   (siehe `webcam.py` für YOLO-Modell-Loading-Code).
2. In `web_pipeline/run.py` `processor=stub_overlay` durch deine Funktion
   ersetzen.
3. Auf NVIDIA-GPU: NVENC-JPEG-Encode aktivieren, sonst frisst der CPU-Encode
   spürbar Latenz.

---

## 3. Standalone-Modus (Worker bedient Browser direkt)

### 3.1 Lokal starten (Linux/macOS)

```bash
./run.sh
```

`run.sh` legt automatisch `.runtime/venv` an, installiert Abhängigkeiten und
startet `webcam.py`. Optionen:

```bash
./run.sh --source synthetic        # Testbild ohne Kamera
./run.sh --source 0                # /dev/video0 oder Webcam-Index 0
./run.sh --source /dev/video2      # konkretes Device
./run.sh --host 0.0.0.0 --port 8080 # auf LAN binden
./run.sh --stream webrtc           # nur WebRTC
./run.sh --stream mjpeg            # nur MJPEG
./run.sh --no-window               # ohne OpenCV-Preview
./run.sh --help-web                # alle Flags
```

### 3.2 Windows

```powershell
cd C:\path\to\sentinelCam-worker
.\run.bat
```

`run.bat` macht dasselbe wie `run.sh` (venv, deps, start).

### 3.3 Docker (nur Linux empfohlen)

```bash
docker compose -f docker-compose.worker-cam.yml up --build
```

Webcam-Passthrough mounts `/dev/video0` ins Container. Für reines
Test/Synthetic:

```bash
docker compose -f docker-compose.worker.yml up --build
```

### 3.4 Konfiguration via `webcam.properties`

Defaults setzt du in `webcam.properties` im Repo-Root. Wichtigste Keys:

```
DEFAULT_SOURCE=0
DEFAULT_WEB_HOST=127.0.0.1
DEFAULT_WEB_PORT=8080
DEFAULT_STREAM_MODE=auto           # auto | webrtc | mjpeg
DEFAULT_WEBRTC_CODEC=auto          # auto | h264 | vp8 | vp9
DEFAULT_WEBRTC_BITRATE_KBPS=-1     # -1 = auto
DEFAULT_WEBRTC_FPS=0               # 0 = source fps
DEFAULT_JPEG_QUALITY=88
DEFAULT_TRACKER_MODE=simple
DEFAULT_YOLO_HALF=1                # FP16 wenn GPU es kann
DEFAULT_CONTEXT_ENABLED=0          # Ollama-Kontext aus

# Sicherheit
WEB_AUTH_TOKEN=langer-zufallswert  # Bearer für /api/cmd, /stream, ...
WEB_ALLOWED_ORIGINS=http://127.0.0.1:3000,http://localhost:3000
```

> Wenn `WEB_AUTH_TOKEN` gesetzt ist, müssen Browser/CLI-Clients den Header
> `Authorization: Bearer <token>` schicken. Ohne Token bleibt der Server
> offen — nur für strikt vertraute Netze geeignet.

### 3.4 Endpoints (Standalone)

| URL | Zweck |
|---|---|
| `GET /health` | Healthcheck |
| `GET /api/state` | aktueller Worker-Zustand (Modelle, FPS, Source) |
| `POST /api/cmd` | Runtime-Befehle (Modell wechseln, Pose an/aus, …) |
| `POST /api/webrtc/offer` | WebRTC-SDP-Negotiation |
| `GET /stream.mjpg` | MJPEG-Fallback |
| `GET /frame.jpg` | letzter verarbeiteter Frame |
| `GET /frame-raw.jpg` | letzter roher Frame (vor Overlay) |

---

## 4. Optional: Ollama-Kontext-Detektion

Aktivieren via `webcam.properties`:

```
DEFAULT_CONTEXT_ENABLED=1
DEFAULT_CONTEXT_PROFILE=auto
DEFAULT_CONTEXT_MODEL=auto         # bspw. gemma3:4b, llava, moondream
DEFAULT_CONTEXT_INTERVAL=30.0      # Sekunden zwischen Inferenzläufen
```

Beim ersten Start wird `setup_ollama.py` gefragt, ob es Ollama-Modelle
ziehen darf (`DEFAULT_CONTEXT_SETUP_OLLAMA=1`). Bei langsamer Bandbreite
einmal manuell `ollama pull <model>` vorab laufen lassen.

---

## 5. Troubleshooting

| Symptom | Ursache / Fix |
|---|---|
| `worker connected` erscheint nicht | Token falsch, WSS-Zertifikat falsch, oder Web-Server unerreichbar. `curl -k $WEB_URL/healthz` testen. |
| Pipeline läuft, aber kein Bild im Browser | Pi-Ingest fehlt — im Web-Admin „Cameras" prüfen, ob `last_frame_at` aktualisiert wird. |
| Hohe CPU, niedrige FPS | `DEFAULT_YOLO_HALF=1` für GPU oder `DEFAULT_CPU_THREADS=<n>` setzen. Kleineres YOLO-Modell wählen. |
| WebRTC-Standalone bringt schwarzes Bild | Browser-Konsole — meist fehlende `WEB_ALLOWED_ORIGINS` oder kein TURN-Server hinter NAT. |
| Docker findet `/dev/video0` nicht | User ausserhalb `video`-Gruppe oder `docker-compose.worker-cam.yml` ohne Device-Mount. |

---

## 6. Updates

```bash
git pull
source .venv/bin/activate
pip install -U -r requirements.txt -r requirements-pipeline.txt
```

Bei Docker:

```bash
docker compose -f docker-compose.worker.yml up -d --build
```
