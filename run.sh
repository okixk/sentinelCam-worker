#!/usr/bin/env bash
set -euo pipefail

# Everything generated goes here (easy to .gitignore):
RUNTIME_DIR=".runtime"
VENV_DIR="$RUNTIME_DIR/venv"
ULTRA_CFG_DIR="$RUNTIME_DIR/ultralytics_config"
WEIGHTS_DIR="$RUNTIME_DIR/weights"
RUNS_DIR="$RUNTIME_DIR/runs"
DATASETS_DIR="$RUNTIME_DIR/datasets"
PIP_CACHE_DIR_LOCAL="$RUNTIME_DIR/pip-cache"

# Build deps; safe even if not needed.
APT_PKGS=(python3 python3-pip python3-venv git ffmpeg v4l-utils libgl1 build-essential python3-dev)

# Optional vcam deps (only installed when you pass --vcam)
VCAM_APT_PKGS=(v4l2loopback-dkms dkms "linux-headers-$(uname -r)")
# Preinstall lap so Ultralytics doesn't try system pip (PEP 668)
PIP_PKGS=(ultralytics opencv-python numpy "lap>=0.5.12")

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

log()  { printf '%s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
need_cmd() { command -v "$1" >/dev/null 2>&1; }

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

ensure_runtime_dirs() {
  mkdir -p "$RUNTIME_DIR" "$ULTRA_CFG_DIR" "$WEIGHTS_DIR" "$RUNS_DIR" "$DATASETS_DIR" "$PIP_CACHE_DIR_LOCAL"
}

ensure_venv() {
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    log "Creating venv in: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
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

start_vcam() {
  # Creates a v4l2loopback device (default /dev/video42) and feeds it with ffmpeg.
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
  # Ultralytics device arg: '0' for CUDA GPU #0, or 'cpu'
  # Fallback to CPU if CUDA isn't available.
  local use_cuda
  use_cuda="$(python - <<'PY'
import torch
print("1" if torch.cuda.is_available() else "0")
PY
)"
  if [[ "$use_cuda" == "1" ]]; then
    echo "0"
  else
    echo "cpu"
  fi
}

main() {
  # ---------------------------
  # run.sh arguments
  #   --vcam                 Create a virtual camera (Linux v4l2loopback) and feed a test pattern
  #   --vcam-input <path/url> Feed the vcam from a file or URL instead of a test pattern
  #   --vcam-nr <N>           vcam device number (default 42 -> /dev/video42)
  #   --vcam-size <WxH>       default 1280x720
  #   --vcam-fps <FPS>        default 30
  #   --help                  print help
  # Anything else is forwarded to webcam.py
  # ---------------------------
  local VCAM=0
  local VCAM_INPUT="testsrc"
  local VCAM_NR=42
  local VCAM_SIZE="1280x720"
  local VCAM_FPS=30
  local FORWARD=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --vcam) VCAM=1; shift;;
      --vcam-input) shift; VCAM_INPUT="${1:-}"; shift;;
      --vcam-nr) shift; VCAM_NR="${1:-42}"; shift;;
      --vcam-size) shift; VCAM_SIZE="${1:-1280x720}"; shift;;
      --vcam-fps) shift; VCAM_FPS="${1:-30}"; shift;;
      --help|-h)
        cat <<'EOF'
Usage:
  ./run.sh [--vcam [--vcam-input <file/url>] [--vcam-nr 42] [--vcam-size 1280x720] [--vcam-fps 30]] [webcam.py args...]

Examples:
  ./run.sh                       # physical camera 0
  ./run.sh --preset yolo26x       # force GPU preset (if CUDA)
  ./run.sh --vcam                 # virtual cam with test pattern on /dev/video42
  ./run.sh --vcam --vcam-input demo.mp4  # feeds demo.mp4 into /dev/video42 and uses it

Tip (cross-platform): OBS Studio (open source) has a built-in Virtual Camera.
EOF
        exit 0
        ;;
      *) FORWARD+=("$1"); shift;;
    esac
  done

  ensure_apt_pkgs
  ensure_runtime_dirs
  ensure_venv
  install_pip_deps

  [[ -f "webcam.py" ]] || die "webcam.py not found in: $SCRIPT_DIR"

  configure_ultralytics_dirs
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

  # If --vcam, create /dev/videoNN and feed it. Unless user already set --source.
  if [[ "$VCAM" == "1" ]]; then
    if has_arg "--source" "${FORWARD[@]}"; then
      warn "--vcam requested, but you already passed --source; not overriding source."
    else
      vcam_dev="$(start_vcam "$VCAM_NR" "$VCAM_INPUT" "$VCAM_SIZE" "$VCAM_FPS")"
      SRC_DEFAULT="$vcam_dev"
      warn "Using virtual camera source: $SRC_DEFAULT"
    fi
  fi

  ULTRA_DEVICE="$(select_ultralytics_device)"
  if [[ "$ULTRA_DEVICE" == "0" ]]; then
    log "torch.cuda.is_available(): True  (device=0)"
  else
    warn "torch.cuda.is_available(): False (device=cpu)"
  fi

  # webcam.py defaults:
  #   --preset yolo   (CPU->yolov8n, GPU->yolo26x)
  #   --device auto   (will pick CUDA if available)
  # so we only pass the device + sensible defaults here.
  if ! has_arg "--source" "${FORWARD[@]}"; then
    FORWARD=("--source" "$SRC_DEFAULT" "${FORWARD[@]}")
  fi
  if ! has_arg "--device" "${FORWARD[@]}"; then
    FORWARD=("--device" "auto" "${FORWARD[@]}")
  fi
  if ! has_arg "--max-fps" "${FORWARD[@]}"; then
    FORWARD+=("--max-fps" "120")
  fi
  # Keep pose on by default, unless user explicitly disables it via --no-pose
  if ! has_arg "--no-pose" "${FORWARD[@]}" && ! has_arg "--use-pose" "${FORWARD[@]}"; then
    FORWARD+=("--use-pose")
  fi

  exec python webcam.py "${FORWARD[@]}"
}

main "$@"
