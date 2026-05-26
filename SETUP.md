# sentinelCam Worker — Setup-Anleitung

Diese Anleitung erklärt die zwei Betriebsmodi des Workers und wie du sie
**komplett über Docker** startest. Beide Modi sind so vorbereitet, dass
nach dem Ausfüllen einer einzigen `.env`-Datei ein einzelner Befehl
genügt.

1. **Pipeline-Modus** (Branch `feature/web-streaming-pipeline`):
   Der Worker dialt outbound per WebSocket zu einer öffentlichen
   `sentinelCam-web`-Instanz, holt rohe JPEGs von einer Pi-Kamera und
   schickt verarbeitete Frames (mit YOLO/Pose-Overlay) zurück.
2. **Standalone-Modus** (Default-Branch):
   Der Worker hat seine eigene Kamera (oder eine Test-Source) und bedient
   Browser direkt via WebRTC/MJPEG/HTTP.

> Für die Architekturdetails siehe [README.md](README.md) und
> [README_PIPELINE.md](README_PIPELINE.md).

---

## 0. Welcher Modus passt zu mir?

| Setup | Modus | Warum |
|---|---|---|
| Pi nimmt auf, separater GPU-Host rechnet, `sentinelCam-web` ist öffentlich erreichbar | **Pipeline** | Web-Server ist das einzige öffentliche Tor; alle Komponenten dialen outbound. |
| Worker hat eigene Kamera (USB/CSI) und soll Browsern direkt Stream liefern (LAN/VPN) | **Standalone** | Kein zusätzlicher Web-Service nötig. |
| Demo / Smoke-Test ohne echte Kamera | **Standalone** mit `WORKER_SOURCE=testsrc` | Erzeugt Testbilder im Worker selbst. |

---

## 1. Voraussetzungen (Host)

- Docker Engine ≥ 24 (`curl -fsSL https://get.docker.com | sudo sh`)
- `docker compose` v2 (in aktuellen Docker-Versionen enthalten)
- Optional, nur für Pipeline + GPU: NVIDIA-Treiber + **NVIDIA Container Toolkit**

NVIDIA-Setup auf Ubuntu/Debian:

```bash
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Wenn `nvidia-smi` im Test-Container erscheint, ist der Host bereit.

---

## 2. One-Shot: `./setup.sh`

Der einfachste Weg — interaktiv, idempotent, wählt automatisch das
richtige Compose-Set und erkennt GPU/Kamera:

```bash
./setup.sh
```

Beim ersten Lauf legt das Skript `.env` aus dem passenden Template an
(`.env.pipeline.example` oder `.env.standalone.example`) und stoppt.
Bearbeite `.env` mit Token + URL, dann `./setup.sh` erneut ausführen — es
baut das Image und startet den Container im Hintergrund.

---

## 3. Pipeline-Modus (manuell)

### 3.1 Worker-Token im Web-UI ausstellen

Auf `https://<dein-web-host>/admin` einloggen → **Workers → New worker**.
Der Token (`sc-wrk-<id>-<32_hex>`) wird **nur einmal** angezeigt — sofort
kopieren.

### 3.2 `.env` vorbereiten

```bash
cp .env.pipeline.example .env
chmod 600 .env
$EDITOR .env        # WEB_URL, WEB_TOKEN, WORKER_NAME setzen
```

Wichtige Variablen sind im Template kommentiert. Ohne `WORKER_YOLO_MODEL`
läuft nur der Stub-Overlay (keine Auto-Recording-Detections).

### 3.3 Starten

```bash
# Mit NVIDIA-GPU:
docker compose -f docker-compose.pipeline.yml up -d --build

# CPU-only Host:
docker compose -f docker-compose.pipeline.yml -f docker-compose.pipeline.cpu.yml up -d --build
```

Logs prüfen:

```bash
docker compose -f docker-compose.pipeline.yml logs -f
```

Erwartete Zeilen:
```
... starting worker name=<WORKER_NAME> -> https://<web>
... worker connected; awaiting frames
```

Im Admin-UI sollte der Worker auf **online** flippen.

### 3.4 Auto-Recording: Detection-Frames

Mit `WORKER_YOLO_MODEL` gesetzt schickt der Worker pro detektiertem Frame
einen Detection-JSON-Frame zum Web-Server (max. 1/s pro Kamera). Der
Web-Server hat zusätzlich seinen eigenen Cooldown (Default 30 s, im
Admin-Panel konfigurierbar).

---

## 4. Standalone-Modus (manuell)

### 4.1 `.env` vorbereiten

```bash
cp .env.standalone.example .env
chmod 600 .env
$EDITOR .env        # WORKER_TOKEN (langer Zufallswert!), WORKER_SOURCE setzen
```

### 4.2 Starten

```bash
# Test ohne Kamera (synthetic source):
docker compose -f docker-compose.worker.yml up -d --build

# Mit Linux-USB-Webcam (/dev/video0):
docker compose -f docker-compose.worker.yml -f docker-compose.worker-cam.yml up -d --build

# Host-Network (für legacy Reverse-Proxies auf 127.0.0.1):
docker compose -f docker-compose.worker.yml -f docker-compose.linux.yml up -d --build
```

Browser: `http://<host>:8080/health` → muss `OK` liefern. Streams und
API-Endpoints siehe Tabelle in Abschnitt 6.

### 4.3 Sicherheit Standalone

`WORKER_TOKEN` setzt den Bearer-Token für `/api/cmd`, `/stream`, etc.
Ohne Token ist der Server offen — nur in strikt vertrauten Netzen
sinnvoll. `WORKER_ALLOWED_ORIGINS` schränkt CORS auf die UIs ein, die
sich verbinden dürfen.

---

## 5. Updates

```bash
git pull
docker compose -f docker-compose.pipeline.yml build --pull   # bzw. worker.yml
docker compose -f docker-compose.pipeline.yml up -d
```

`docker-compose.pipeline.yml` baut das Image lokal. Bei Bedarf alte
Images aufräumen: `docker image prune`.

---

## 6. Endpoints (Standalone)

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

## 7. Optional: Ohne Docker (Entwicklung)

Pip/venv-Setup ist weiterhin möglich, aber **nicht** der empfohlene Pfad
für Produktion. Kurzversion:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-pipeline.txt
# Pipeline-Mode mit YOLO:
pip install ultralytics torch torchvision  # CUDA-Variante siehe pytorch.org
set -a; source .env; set +a
python -m web_pipeline.run
# Standalone:
./run.sh                 # Linux/macOS
.\run.bat                # Windows
```

---

## 8. Troubleshooting

| Symptom | Ursache / Fix |
|---|---|
| `nvidia-smi` läuft im Test-Container nicht | NVIDIA Container Toolkit fehlt — Abschnitt 1 wiederholen. |
| Worker bleibt offline im Admin-UI | Token falsch oder `WEB_URL` nicht erreichbar. `docker compose logs -f` prüfen. |
| `WEB_URL and WEB_TOKEN must be set` beim Start | `.env` nicht vorhanden oder fehlerhaft. `cat .env` prüfen. |
| Pipeline läuft, aber kein Bild im Browser | Pi-Ingest fehlt — im Web-Admin „Cameras" prüfen, ob `last_frame_at` aktualisiert wird. |
| Hohe CPU, niedrige FPS | `WORKER_YOLO_HALF=1` für GPU, kleineres Modell (yolov8n) wählen. |
| Docker findet `/dev/video0` nicht | User nicht in `video`-Gruppe oder Compose-Override fehlt. |
| `read_only`-Container kann nichts schreiben | Schreibpfade sind als tmpfs/Volume gemountet — neue Pfade ggf. in `docker-compose.pipeline.yml` ergänzen. |

---

## 9. Sicherheits-Checkliste (Produktion)

- [x] `.env` mit `chmod 600`; nicht ins Repo committen (`.gitignore` deckt das ab).
- [x] Pipeline-Modus braucht **keine** offenen Inbound-Ports — Firewall: `ufw default deny incoming`.
- [x] `WEB_URL` immer `https://` mit gültigem Zertifikat.
- [x] Pro Worker einen eigenen Token im Web-Admin; rotieren bei Personalwechsel.
- [x] Container läuft als non-root (UID 1000) mit `cap_drop: ALL`, `no-new-privileges`, `read_only: true` (Pipeline-Compose).
- [x] Log-Rotation aktiv (json-file, 10 MB × 5).
- [x] `restart: unless-stopped` sorgt für Auto-Start nach Reboot — Docker selbst per `systemctl enable docker` aktivieren.
