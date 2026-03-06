#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CFG_FILE="${CFG_FILE:-$SCRIPT_DIR/webcam.properties}"

if [[ "${1:-}" == "--help-web" ]]; then
  cat <<'EOF'
sentinelCam-worker launcher (run.sh)

This script ONLY manages the worker repo:
  - creates/uses a venv in .runtime/venv
  - installs python deps via an inline pip list (NO requirements.txt)
  - starts webcam.py

Web streaming is provided by the worker itself (webstream.py in this repo).
The web repo simply displays http://WORKER_IP:8080/stream.mjpg.

Examples:
  ./run.sh
  ./run.sh --no-install
  ./run.sh --no-web                 # window-only
  ./run.sh --window                 # also show OpenCV preview window
  ./run.sh --host 0.0.0.0 --port 8080
EOF
  exit 0
fi

# Load shared defaults (KEY=VALUE)
if [[ -f "$CFG_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  . "$CFG_FILE"
  set +a
fi

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

log()  { printf '%s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

need_cmd() { command -v "$1" >/dev/null 2>&1; }

has_arg() {
  local needle="$1"; shift
  local a
  for a in "$@"; do
    [[ "$a" == "$needle" ]] && return 0
  done
  return 1
}

ensure_runtime_dirs() {
  mkdir -p "$RUNTIME_DIR" "$VENV_DIR" "$ULTRA_CFG_DIR" "$WEIGHTS_DIR" "$RUNS_DIR" "$DATASETS_DIR" "$PIP_CACHE_DIR_LOCAL"
}

ensure_system_pkgs_best_effort() {
  # Minimal best-effort; don't hard-fail if sudo isn't available.
  if [[ "$OS_NAME" == "Linux" ]] && need_cmd apt-get; then
    local pkgs=(python3 python3-pip python3-venv)
    local missing=()
    local p
    for p in "${pkgs[@]}"; do
      dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p")
    done
    if ((${#missing[@]})); then
      warn "Installing missing system packages (best-effort): ${missing[*]}"
      sudo apt-get update -y || true
      sudo apt-get install -y "${missing[@]}" || true
    fi
  fi
}

ensure_venv() {
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    need_cmd python3 || die "python3 not found"
    log "Creating venv: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
  fi
  export VIRTUAL_ENV="$SCRIPT_DIR/$VENV_DIR"
  export PATH="$VIRTUAL_ENV/bin:$PATH"
  export PIP_CACHE_DIR="$SCRIPT_DIR/$PIP_CACHE_DIR_LOCAL"
  hash -r
}

install_pip_deps() {
  python -m pip install --upgrade pip wheel setuptools >/dev/null
  # Inline deps (no requirements.txt)
  python -m pip install ultralytics opencv-python numpy "lap>=0.5.12"
}

# ---------------------------
# Parse launcher args
# ---------------------------
SILENT=0
INSTALL_MODE="ask"   # ask|force|skip
FORWARD=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--silent) SILENT=1; shift;;
    --install) INSTALL_MODE="force"; shift;;
    --no-install) INSTALL_MODE="skip"; shift;;
    --help|-h)
      cat <<'EOF'
Usage:
  ./run.sh [-s|--silent] [--install|--no-install] [webcam.py args...]

Notes:
  - web server is the default (webcam.py defaults web=True)
  - window is optional: add --window
  - disable web server: --no-web

See also:
  ./run.sh --help-web
EOF
      exit 0
      ;;
    *) FORWARD+=("$1"); shift;;
  esac
done

DO_INSTALL=1
if [[ "$INSTALL_MODE" == "skip" ]]; then
  DO_INSTALL=0
elif [[ "$INSTALL_MODE" == "force" ]]; then
  DO_INSTALL=1
elif [[ "$SILENT" == "1" ]]; then
  DO_INSTALL=1
else
  # default Yes
  read -r -p "Run setup/install step (venv + pip deps)? [Y/n]: " reply || true
  reply="${reply:-Y}"
  if [[ "${reply,,}" == "n" || "${reply,,}" == "no" ]]; then
    DO_INSTALL=0
  fi
fi

ensure_runtime_dirs

if [[ "$DO_INSTALL" == "1" ]]; then
  ensure_system_pkgs_best_effort
  ensure_venv
  install_pip_deps
else
  warn "Skipping setup/install step (--no-install). Assuming venv + deps already exist."
  ensure_venv
fi

[[ -f "webcam.py" ]] || die "webcam.py not found in: $SCRIPT_DIR"
[[ -f "webstream.py" ]] || warn "webstream.py missing. Web server mode will fail unless you add it to worker repo."

# Configure Ultralytics settings to stay inside .runtime (best-effort)
export YOLO_CONFIG_DIR="$SCRIPT_DIR/$ULTRA_CFG_DIR"
export SC_WEIGHTS_DIR="$SCRIPT_DIR/$WEIGHTS_DIR"
export SC_RUNS_DIR="$SCRIPT_DIR/$RUNS_DIR"
export SC_DATASETS_DIR="$SCRIPT_DIR/$DATASETS_DIR"

# Forward defaults from webcam.properties if user didn't pass them
if ! has_arg "--source" "${FORWARD[@]}" && ! has_arg "--cam" "${FORWARD[@]}"; then
  if [[ "$OS_NAME" == "Linux" ]]; then
    FORWARD=(--source "${DEFAULT_SOURCE_LINUX:-0}" "${FORWARD[@]}")
  else
    FORWARD=(--source "${DEFAULT_SOURCE_WINDOWS:-0}" "${FORWARD[@]}")
  fi
fi

if ! has_arg "--device" "${FORWARD[@]}"; then
  FORWARD=(--device "${DEFAULT_DEVICE:-auto}" "${FORWARD[@]}")
fi

if ! has_arg "--max-fps" "${FORWARD[@]}"; then
  FORWARD+=(--max-fps "${DEFAULT_MAX_FPS:-120}")
fi

if [[ "${DEFAULT_USE_POSE:-1}" == "1" ]] && ! has_arg "--no-pose" "${FORWARD[@]}" && ! has_arg "--use-pose" "${FORWARD[@]}"; then
  FORWARD+=(--use-pose)
fi

exec python webcam.py "${FORWARD[@]}"
