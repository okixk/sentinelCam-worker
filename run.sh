#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CFG_FILE="${CFG_FILE:-$SCRIPT_DIR/webcam.properties}"

# -----------------------------
# Web help (quick reference)
# -----------------------------
if [[ "${1:-}" == "--help-web" ]]; then
  cat <<'EOF'
sentinelCam Web-Startoptionen (run.sh)

Grundlage:
  ./run.sh --web [--stream webrtc|mjpeg|auto] [--host 127.0.0.1] [--port 8080]

Web / Streaming:
  --web                      Startet den Web-Server (statt OpenCV GUI)
  --stream auto|webrtc|mjpeg  auto=WebRTC wenn verfuegbar, sonst MJPEG
  --host HOST                Bind-Adresse (127.0.0.1 nur lokal, 0.0.0.0 im LAN)
  --port PORT                TCP-Port fuer Webseite/Signaling

WebRTC:
  --webrtc-codec auto|h264|vp8|vp9|av1  Codec-Praeferenz (auto bevorzugt h264)
  --advertise-ip IP           Erzwingt diese LAN-IP in ICE-Candidates (gegen VPN/falsche NIC)
  --rtc-min-port 50000        UDP-Port-Range fuer ICE/RTP (Firewall passend oeffnen)
  --rtc-max-port 60000

MJPEG:
  --jpeg-quality 10-95        JPEG-Qualitaet (niedriger = weniger Bandbreite)

Capture/Quelle (wichtig fuer echte FHD):
  --width W --height H        Versucht die Capture-Aufloesung zu setzen
  --source N|URL              Kamera-Index (0,1,2...) oder RTSP/URL

Tipp:
  ./run.sh --help             zeigt alle Optionen von webcam.py
EOF
  exit 0
fi

if [[ ! -f "$CFG_FILE" ]]; then
  echo "ERROR: Shared config not found: $CFG_FILE" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
. "$CFG_FILE"
set +a

# Everything generated goes here (easy to .gitignore):
RUNTIME_DIR="${RUNTIME_DIR:-.runtime}"
VENV_SUBDIR="${VENV_SUBDIR:-venv}"
ULTRA_CFG_SUBDIR="${ULTRA_CFG_SUBDIR:-ultralytics_config}"
WEIGHTS_SUBDIR="${WEIGHTS_SUBDIR:-weights}"
RUNS_SUBDIR="${RUNS_SUBDIR:-runs}"
DATASETS_SUBDIR="${DATASETS_SUBDIR:-datasets}"
PIP_CACHE_SUBDIR="${PIP_CACHE_SUBDIR:-pip-cache}"

VENV_DIR="$RUNTIME_DIR/$VENV_SUBDIR"
ULTRA_CFG_DIR="$RUNTIME_DIR/$ULTRA_CFG_SUBDIR"
WEIGHTS_DIR="$RUNTIME_DIR/$WEIGHTS_SUBDIR"
RUNS_DIR="$RUNTIME_DIR/$RUNS_SUBDIR"
DATASETS_DIR="$RUNTIME_DIR/$DATASETS_SUBDIR"
PIP_CACHE_DIR_LOCAL="$RUNTIME_DIR/$PIP_CACHE_SUBDIR"

OS_NAME="$(uname -s)"
ARCH_NAME="$(uname -m)"

# Build deps; safe even if not needed.
APT_PKGS=(python3 python3-pip python3-venv git ffmpeg v4l-utils libgl1 build-essential python3-dev)
BREW_PKGS=(python ffmpeg git)

# Optional vcam deps (only installed when you pass --vcam)
VCAM_APT_PKGS=(v4l2loopback-dkms dkms "linux-headers-$(uname -r)")
# Preinstall lap so Ultralytics doesn't try system pip (PEP 668)
PIP_PKGS=(ultralytics opencv-python numpy "lap>=0.5.12")

# Optional (installed only when you run with --web and want WebRTC):
PIP_PKGS_WEBRTC=(aiohttp aiortc av)

log()  { printf '%s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
need_cmd() { command -v "$1" >/dev/null 2>&1; }

prefer_brew_on_path() {
  if [[ "$OS_NAME" == "Darwin" ]] && need_cmd brew; then
    local brew_prefix=""
    brew_prefix="$(brew --prefix 2>/dev/null || true)"
    if [[ -n "$brew_prefix" && -d "$brew_prefix/bin" ]]; then
      export PATH="$brew_prefix/bin:$brew_prefix/sbin:$PATH"
      hash -r
    fi
  fi
}

ensure_apt_pkgs() {
  need_cmd apt-get || die "apt-get not found. Ubuntu/Debian only."
  local missing=()
  for p in "${APT_PKGS[@]}"; do
    dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p")
  done
  if ((${#missing[@]})); then
    warn "Installing missing apt packages: ${missing[*]}"
    sudo apt-get update -y
    sudo apt-get install -y "${missing[@]}"
  else
    log "apt packages: OK"
  fi
}

ensure_brew_pkgs() {
  need_cmd brew || die "Homebrew not found. Install Homebrew first (Apple Silicon usually uses /opt/homebrew, Intel uses /usr/local)."
  prefer_brew_on_path

  local missing=()
  for p in "${BREW_PKGS[@]}"; do
    brew list --formula "$p" >/dev/null 2>&1 || missing+=("$p")
  done

  if ((${#missing[@]})); then
    warn "Installing missing Homebrew packages: ${missing[*]}"
    brew update
    brew install "${missing[@]}"
  else
    log "Homebrew packages: OK"
  fi

  prefer_brew_on_path
}

ensure_system_pkgs() {
  case "$OS_NAME" in
    Linux) ensure_apt_pkgs ;;
    Darwin) ensure_brew_pkgs ;;
    *) die "Unsupported OS: $OS_NAME (Linux and macOS are supported)." ;;
  esac
}

ensure_runtime_dirs() {
  mkdir -p "$RUNTIME_DIR" "$ULTRA_CFG_DIR" "$WEIGHTS_DIR" "$RUNS_DIR" "$DATASETS_DIR" "$PIP_CACHE_DIR_LOCAL"
}

ensure_venv() {
  prefer_brew_on_path

  local py_bin="python3"
  if [[ "$OS_NAME" == "Darwin" ]]; then
    py_bin="$(command -v python3 || true)"
    [[ -n "$py_bin" ]] || die "python3 not found after Homebrew setup"
  fi

  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    log "Creating venv in: $VENV_DIR"
    "$py_bin" -m venv "$VENV_DIR"
  fi

  # Make venv "active" for subprocesses too (Ultralytics calls pip via subprocess)
  export VIRTUAL_ENV="$SCRIPT_DIR/$VENV_DIR"
  export PATH="$VIRTUAL_ENV/bin:$PATH"
  hash -r

  export PIP_CACHE_DIR="$SCRIPT_DIR/$PIP_CACHE_DIR_LOCAL"
  python -m pip install --upgrade pip >/dev/null
}

install_pip_deps() {
  warn "Ensuring pip deps in venv: ${PIP_PKGS[*]}"
  python -m pip install "${PIP_PKGS[@]}"
}

install_webrtc_deps_best_effort() {
  warn "Ensuring (optional) WebRTC deps: ${PIP_PKGS_WEBRTC[*]}"
  python -m pip install "${PIP_PKGS_WEBRTC[@]}" || warn "WebRTC deps install failed (MJPEG fallback still works)."
}

configure_ultralytics_dirs() {
  # Keep Ultralytics settings JSON inside .runtime (instead of ~/.config/Ultralytics)
  export YOLO_CONFIG_DIR="$SCRIPT_DIR/$ULTRA_CFG_DIR"

  WEIGHTS_ABS="$SCRIPT_DIR/$WEIGHTS_DIR"
  RUNS_ABS="$SCRIPT_DIR/$RUNS_DIR"
  DATASETS_ABS="$SCRIPT_DIR/$DATASETS_DIR"

  WEIGHTS_DIR="$WEIGHTS_ABS" RUNS_DIR="$RUNS_ABS" DATASETS_DIR="$DATASETS_ABS" python - <<'PY'
import os
from ultralytics import settings
settings.update({
    "weights_dir": os.environ["WEIGHTS_DIR"],
    "runs_dir": os.environ["RUNS_DIR"],
    "datasets_dir": os.environ["DATASETS_DIR"],
    "sync": False,
})
PY
}

cleanup_root_weights() {
  for f in "yolo26x.pt" "yolo26x-pose.pt"; do
    if [[ -f "$SCRIPT_DIR/$f" ]]; then
      warn "Moving $f into $WEIGHTS_DIR/"
      mv -n "$SCRIPT_DIR/$f" "$SCRIPT_DIR/$WEIGHTS_DIR/" || true
    fi
  done
}

check_camera() {
  log "Camera check:"

  if [[ "$OS_NAME" == "Darwin" ]]; then
    if need_cmd ffmpeg; then
      ffmpeg -hide_banner -f avfoundation -list_devices true -i "" </dev/null 2>&1 || true
    else
      warn "ffmpeg not found."
    fi
    return 0
  fi

  if need_cmd v4l2-ctl; then
    v4l2-ctl --list-devices || true
  else
    warn "v4l2-ctl not found."
  fi

  if compgen -G "/dev/video*" >/dev/null; then
    ls -l /dev/video* || true
  else
    warn "No /dev/video* devices found."
  fi
}

ensure_vcam_pkgs() {
  local missing=()
  for p in "${VCAM_APT_PKGS[@]}"; do
    # linux-headers-... might not exist on all systems; treat as best-effort.
    dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p")
  done
  if ((${#missing[@]})); then
    warn "Installing vcam packages: ${missing[*]}"
    sudo apt-get update -y
    sudo apt-get install -y "${missing[@]}" || warn "vcam apt install failed (continuing)"
  fi
}

has_arg() {
  local needle="$1"; shift
  for a in "$@"; do
    [[ "$a" == "$needle" ]] && return 0
  done
  return 1
}

prompt_yes_no() {
  # Usage: prompt_yes_no "Question" "Y"  (default Y or N)
  local prompt="$1"
  local def="${2:-Y}"
  local suffix="Y/n"
  [[ "$def" == "N" ]] && suffix="y/N"
  local reply=""

  while true; do
    read -r -p "${prompt} [${suffix}]: " reply || reply=""
    reply="${reply:-$def}"
    case "$reply" in
      Y|y) return 0 ;;
      N|n) return 1 ;;
      *) echo "Please answer y or n." ;;
    esac
  done
}

prompt_input() {
  # Usage: val="$(prompt_input "Prompt" "default")"
  local prompt="$1"
  local def="${2-}"
  local reply=""
  if [[ -n "$def" ]]; then
    read -r -p "${prompt} [${def}]: " reply || reply=""
    echo "${reply:-$def}"
  else
    read -r -p "${prompt}: " reply || reply=""
    echo "$reply"
  fi
}

start_vcam() {
  # Creates a v4l2loopback device (default /dev/video42) and feeds it with ffmpeg.
  [[ "$OS_NAME" == "Linux" ]] || die "--vcam is Linux-only (uses v4l2loopback). On macOS, use OBS Studio Virtual Camera or Continuity Camera."

  local video_nr="$1"
  local input="$2"   # "testsrc" or a file/url
  local size="$3"    # e.g. 1280x720
  local fps="$4"     # e.g. 30

  # ffmpeg scale filter expects 1280:720, while testsrc2 uses 1280x720
  local scale_size="${size/x/:}"

  ensure_vcam_pkgs

  if ! sudo -n true 2>/dev/null; then
    warn "vcam needs sudo to load v4l2loopback (you may be prompted)."
  fi

  warn "Loading v4l2loopback (/dev/video${video_nr})..."
  sudo modprobe v4l2loopback video_nr="$video_nr" card_label="vcam-yolo" exclusive_caps=1 || die "modprobe v4l2loopback failed"

  local dev="/dev/video${video_nr}"
  [[ -e "$dev" ]] || die "$dev not found after modprobe"

  warn "Starting ffmpeg feed -> $dev (stop script to quit)"

  if [[ "$input" == "testsrc" ]]; then
    ffmpeg -hide_banner -loglevel warning -re \
      -f lavfi -i "testsrc2=size=${size}:rate=${fps}" \
      -vcodec rawvideo -pix_fmt yuyv422 -f v4l2 "$dev" &
  else
    ffmpeg -hide_banner -loglevel warning -stream_loop -1 -re \
      -i "$input" -vf "scale=${scale_size}" \
      -vcodec rawvideo -pix_fmt yuyv422 -f v4l2 "$dev" &
  fi

  VCAM_FFMPEG_PID=$!
  trap '[[ -n "${VCAM_FFMPEG_PID:-}" ]] && kill "$VCAM_FFMPEG_PID" >/dev/null 2>&1 || true' EXIT
  echo "$dev"
}

select_ultralytics_device() {
  # Ultralytics device arg:
  #   - '0'   -> CUDA GPU #0
  #   - 'mps' -> Apple Silicon Metal backend
  #   - 'cpu' -> CPU fallback
  local detected
  detected="$(python - <<'PY'
try:
    import torch
except Exception:
    print("cpu")
    raise SystemExit

try:
    if torch.cuda.is_available():
        print("0")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        print("mps")
    else:
        print("cpu")
except Exception:
    print("cpu")
PY
)"
  echo "$detected"
}

main() {
  # ---------------------------
  # run.sh arguments
  #   -s / --silent          No interactive prompts, always use defaults
  #   --install              Force install step
  #   --no-install           Skip install step
  #   --vcam                 Create a virtual camera (Linux v4l2loopback) and feed a test pattern
  #   --vcam-input <path/url> Feed the vcam from a file or URL instead of a test pattern
  #   --vcam-nr <N>           vcam device number (default 42 -> /dev/video42)
  #   --vcam-size <WxH>       default 1280x720
  #   --vcam-fps <FPS>        default 30
  #   --help / --help-web     print help
  # Anything else is forwarded to webcam.py
  # ---------------------------
  local SILENT=0
  local INSTALL_MODE="ask"  # ask|force|skip

  local VCAM=0
  local VCAM_INPUT="${VCAM_DEFAULT_INPUT:-testsrc}"
  local VCAM_NR="${VCAM_DEFAULT_NR:-42}"
  local VCAM_SIZE="${VCAM_DEFAULT_SIZE:-1280x720}"
  local VCAM_FPS="${VCAM_DEFAULT_FPS:-30}"
  local -a FORWARD=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --vcam) VCAM=1; shift;;
      --vcam-input) shift; VCAM_INPUT="${1:-}"; shift;;
      --vcam-nr) shift; VCAM_NR="${1:-42}"; shift;;
      --vcam-size) shift; VCAM_SIZE="${1:-1280x720}"; shift;;
      --vcam-fps) shift; VCAM_FPS="${1:-30}"; shift;;
      -s|--silent) SILENT=1; shift;;
      --install) INSTALL_MODE="force"; shift;;
      --no-install) INSTALL_MODE="skip"; shift;;
      --help|-h)
        cat <<'EOF'
Usage:
  ./run.sh [-s|--silent] [--install|--no-install]
          [--vcam [--vcam-input <file/url>] [--vcam-nr 42] [--vcam-size 1280x720] [--vcam-fps 30]]
          [webcam.py args...]

  ./run.sh --help-web             zeigt Web/Streaming-spezifische Optionen

Examples:
  ./run.sh                       # interactive prompts (Enter = defaults)
  ./run.sh -s                    # silent: always defaults
  ./run.sh --no-install          # skip setup/install step
  ./run.sh --preset yolo26x       # force GPU preset (if CUDA)
  ./run.sh --vcam                 # virtual cam with test pattern on /dev/video42
  ./run.sh --vcam --vcam-input demo.mp4  # feeds demo.mp4 into /dev/video42 and uses it
  ./run.sh --web --stream webrtc --host 0.0.0.0 --port 8080

Tip (cross-platform): OBS Studio (open source) has a built-in Virtual Camera.
Note: On macOS, this script uses Homebrew for system deps and prefers Apple Silicon (MPS) when available.
EOF
        exit 0
        ;;
      *) FORWARD+=("$1"); shift;;
    esac
  done


# ---------------------------
# Interactive: setup/install?
# ---------------------------
local DO_INSTALL=1
if [[ "$INSTALL_MODE" == "skip" ]]; then
  DO_INSTALL=0
elif [[ "$INSTALL_MODE" == "force" ]]; then
  DO_INSTALL=1
elif [[ "$SILENT" == "1" ]]; then
  DO_INSTALL=1   # defaults
else
  if prompt_yes_no "Run setup/install step (system deps + venv + pip deps)?" "Y"; then
    DO_INSTALL=1
  else
    DO_INSTALL=0
  fi
fi

# In "no-install" mode, still try to run using an existing venv if present.
ensure_runtime_dirs
if [[ "$DO_INSTALL" == "1" ]]; then
  ensure_system_pkgs
  ensure_venv
  install_pip_deps

  # If the user asks for web output, try to install optional WebRTC deps.
  # This is best-effort: if it fails, webcam.py will fall back to MJPEG.
  if has_arg "--web" "${FORWARD[@]}"; then
    local SKIP_WEBRTC=0
    if has_arg "--stream" "${FORWARD[@]}"; then
      for ((i=0; i<${#FORWARD[@]}; i++)); do
        if [[ "${FORWARD[$i]}" == "--stream" && "${FORWARD[$((i+1))]:-}" == "mjpeg" ]]; then
          SKIP_WEBRTC=1
        fi
      done
    fi
    if [[ "$SKIP_WEBRTC" != "1" ]]; then
      install_webrtc_deps_best_effort
    fi
  fi
else
  warn "Skipping setup/install step (--no-install). Assuming python + deps already exist."
  prefer_brew_on_path
  if [[ -x "$VENV_DIR/bin/python" ]]; then
    export VIRTUAL_ENV="$SCRIPT_DIR/$VENV_DIR"
    export PATH="$VIRTUAL_ENV/bin:$PATH"
    export PIP_CACHE_DIR="$SCRIPT_DIR/$PIP_CACHE_DIR_LOCAL"
    hash -r
  fi
  # If macOS only has python3, make 'python' available for the rest of this script.
  if ! need_cmd python && need_cmd python3; then
    python() { python3 "$@"; }
  fi
fi

  [[ -f "webcam.py" ]] || die "webcam.py not found in: $SCRIPT_DIR"


# Configure Ultralytics settings to stay inside .runtime (best-effort if deps are present)
export YOLO_CONFIG_DIR="$SCRIPT_DIR/$ULTRA_CFG_DIR"
if python -c "import ultralytics" >/dev/null 2>&1; then
  configure_ultralytics_dirs
else
  warn "Ultralytics not importable yet; skipping Ultralytics settings update."
fi

  cleanup_root_weights
  check_camera

  # Move additional custom weights if they were downloaded into repo root
  for f in "sam32.pt" "sam32-pose.pt"; do
    if [[ -f "$SCRIPT_DIR/$f" ]]; then
      warn "Moving $f into $WEIGHTS_DIR/"
      mv -n "$SCRIPT_DIR/$f" "$SCRIPT_DIR/$WEIGHTS_DIR/" || true
    fi
  done

  # Default source is camera 0 unless user provided --source
  local SRC_DEFAULT="0"
  if [[ "$OS_NAME" == "Darwin" ]]; then
    SRC_DEFAULT="${DEFAULT_SOURCE_MACOS:-${DEFAULT_SOURCE_LINUX:-0}}"
  else
    SRC_DEFAULT="${DEFAULT_SOURCE_LINUX:-0}"
  fi

  # If --vcam, create /dev/videoNN and feed it. Unless user already set --source.
  if [[ "$VCAM" == "1" ]]; then
    if ((${#FORWARD[@]})) && has_arg "--source" "${FORWARD[@]}"; then
      warn "--vcam requested, but you already passed --source; not overriding source."
    else
      vcam_dev="$(start_vcam "$VCAM_NR" "$VCAM_INPUT" "$VCAM_SIZE" "$VCAM_FPS")"
      SRC_DEFAULT="$vcam_dev"
      warn "Using virtual camera source: $SRC_DEFAULT"
    fi
  fi

  ULTRA_DEVICE="$(select_ultralytics_device)"
  if [[ "$ULTRA_DEVICE" == "0" ]]; then
    log "Accelerator check: CUDA available (device=0)"
  elif [[ "$ULTRA_DEVICE" == "mps" ]]; then
    log "Accelerator check: Apple Metal available (device=mps)"
  else
    warn "Accelerator check: no GPU backend detected (device=cpu)"
  fi



# ---------------------------
# Interactive: camera + preset
# ---------------------------
local preset_cpu="${DEFAULT_PRESET_CPU:-yolov8n}"
local preset_accel="${DEFAULT_PRESET_ACCEL:-yolo26x}"
local PRESET_DEFAULT="$preset_cpu"
[[ "$ULTRA_DEVICE" != "cpu" ]] && PRESET_DEFAULT="$preset_accel"

# Camera source selection (only if user didn't pass --source)
if ! has_arg "--source" "${FORWARD[@]}"; then
  if [[ "$SILENT" != "1" ]]; then
    log ""
    log "Choose camera source (press Enter for default):"
    check_camera
    SRC_DEFAULT="$(prompt_input "Camera source (index like 0, /dev/video0, or RTSP URL)" "$SRC_DEFAULT")"
  fi
fi

# Model preset selection (only if user didn't pass --preset)
if ! has_arg "--preset" "${FORWARD[@]}"; then
  if [[ "$SILENT" == "1" ]]; then
    FORWARD=("--preset" "$PRESET_DEFAULT" "${FORWARD[@]}")
  else
    log ""
    log "Choose model preset (press Enter for default). Tip: 'yolo' = auto CPU/GPU default."
    python webcam.py --list-presets 2>/dev/null || true
    local chosen_preset
    chosen_preset="$(prompt_input "Preset" "$PRESET_DEFAULT")"
    FORWARD=("--preset" "$chosen_preset" "${FORWARD[@]}")
  fi
fi

# ---------------------------
# webcam.py defaults:
  #   --preset yolo   (CPU->yolov8n, GPU->yolo26x)
  #   --device auto   (will pick CUDA if available)
  # so we only pass the device + sensible defaults here.
  if ((${#FORWARD[@]} == 0)); then
    FORWARD=("--source" "$SRC_DEFAULT")
  elif ! has_arg "--source" "${FORWARD[@]}"; then
    FORWARD=("--source" "$SRC_DEFAULT" "${FORWARD[@]}")
  fi
  if ! has_arg "--device" "${FORWARD[@]}"; then
    local default_device="${DEFAULT_DEVICE:-auto}"
    if [[ "$default_device" == "auto" ]]; then
      default_device="$ULTRA_DEVICE"
    fi
    FORWARD=("--device" "$default_device" "${FORWARD[@]}")
  fi
  if ! has_arg "--max-fps" "${FORWARD[@]}"; then
    FORWARD+=("--max-fps" "${DEFAULT_MAX_FPS:-120}")
  fi
  # Keep pose on by default, unless user explicitly disables it via --no-pose
  if [[ "${DEFAULT_USE_POSE:-1}" == "1" ]] && ! has_arg "--no-pose" "${FORWARD[@]}" && ! has_arg "--use-pose" "${FORWARD[@]}"; then
    FORWARD+=("--use-pose")
  fi

  exec python webcam.py "${FORWARD[@]}"
}

main "$@"
