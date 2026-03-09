#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CFG_FILE="${CFG_FILE:-$SCRIPT_DIR/webcam.properties}"

if [[ "${1:-}" == "--help-web" ]]; then
  cat <<'HELP'
sentinelCam-worker launcher (run.sh)

This script ONLY manages the worker repo:
  - creates/uses a venv in .runtime/venv
  - installs python deps via an inline pip list (NO requirements.txt)
  - starts webcam.py

The web repo simply displays http://WORKER_IP:8080/stream.mjpg.
If --host is omitted, choose 1 for localhost or 2 for 0.0.0.0.
By default the worker binds only to 127.0.0.1. Change DEFAULT_WEB_HOST in webcam.properties
or pass --host 0.0.0.0 to expose it on the LAN.
Optional hardening in webcam.properties:
  WEB_AUTH_TOKEN=long-random-secret
  WEB_ALLOWED_ORIGINS=http://127.0.0.1:3000,http://localhost:3000

Examples:
  ./run.sh
  ./run.sh --no-install
  ./run.sh --no-web                 # window-only
  ./run.sh --window                 # also show OpenCV preview window
  ./run.sh --host 0.0.0.0 --port 8080
HELP
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
DEFAULT_WEB_HOST="${DEFAULT_WEB_HOST:-127.0.0.1}"

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

has_opt() {
  local name="$1"; shift
  local a
  for a in "$@"; do
    [[ "$a" == "$name" || "$a" == "$name="* ]] && return 0
  done
  return 1
}

default_host_choice() {
  if [[ "${DEFAULT_WEB_HOST:-127.0.0.1}" == "0.0.0.0" ]]; then
    printf '%s' "2"
  else
    printf '%s' "1"
  fi
}

resolve_host_choice() {
  case "$1" in
    1) printf '%s' "127.0.0.1" ;;
    2) printf '%s' "0.0.0.0" ;;
    *) printf '%s' "${DEFAULT_WEB_HOST:-127.0.0.1}" ;;
  esac
}

default_source_for_os() {
  case "$OS_NAME" in
    Linux)
      printf '%s' "${DEFAULT_SOURCE_LINUX:-0}"
      ;;
    Darwin)
      printf '%s' "${DEFAULT_SOURCE_MAC:-${DEFAULT_SOURCE_LINUX:-0}}"
      ;;
    *)
      printf '%s' "${DEFAULT_SOURCE_WINDOWS:-0}"
      ;;
  esac
}

validate_source() {
  local src="$1"
  python - "$src" <<'PY'
import sys
import cv2

src = sys.argv[1]
ok = False

try:
    if isinstance(src, str) and src.isdigit():
        idx = int(src)
        backends = [None]
        if sys.platform.startswith("win") and hasattr(cv2, "CAP_DSHOW"):
            backends = [cv2.CAP_DSHOW, None]
        elif sys.platform == "darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
            backends = [cv2.CAP_AVFOUNDATION, None]

        for backend in backends:
            cap = cv2.VideoCapture(idx, backend) if backend is not None else cv2.VideoCapture(idx)
            try:
                if cap is not None and cap.isOpened():
                    ret, _frame = cap.read()
                    if ret:
                        ok = True
                        break
            finally:
                try:
                    if cap is not None:
                        cap.release()
                except Exception:
                    pass
    else:
        cap = cv2.VideoCapture(src)
        try:
            if cap is not None and cap.isOpened():
                ret, _frame = cap.read()
                ok = bool(ret)
        finally:
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass
except Exception:
    ok = False

raise SystemExit(0 if ok else 1)
PY
}

prompt_for_source() {
  local default_source="$1"
  local reply=""
  read -r -p "Which cam index or stream URL/path should YOLO use? [${default_source}]: " reply || true
  reply="${reply:-$default_source}"
  printf '%s' "$reply"
}

prompt_for_host_choice() {
  local default_choice="$1"
  local reply=""
  while true; do
    read -r -p "Stream host waehlen: [1] localhost/127.0.0.1, [2] alle Interfaces/0.0.0.0 [${default_choice}]: " reply || true
    reply="${reply:-$default_choice}"
    case "$reply" in
      1|2)
        printf '%s' "$reply"
        return 0
        ;;
    esac
    warn "Bitte nur 1 oder 2 eingeben."
  done
}

ensure_runtime_dirs() {
  mkdir -p "$RUNTIME_DIR" "$VENV_DIR" "$ULTRA_CFG_DIR" "$WEIGHTS_DIR" "$RUNS_DIR" "$DATASETS_DIR" "$PIP_CACHE_DIR_LOCAL"
}

ensure_system_pkgs_best_effort() {
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
  python -m pip install ultralytics opencv-python numpy "lap>=0.5.12"
}

SILENT=0
INSTALL_MODE="ask"
FORWARD=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--silent) SILENT=1; shift;;
    --install) INSTALL_MODE="force"; shift;;
    --no-install) INSTALL_MODE="skip"; shift;;
    --help|-h)
      cat <<'HELP'
Usage:
  ./run.sh [-s|--silent] [--install|--no-install] [webcam.py args...]

Notes:
  - web server is the default (webcam.py defaults web=True)
  - default bind host is 127.0.0.1 (localhost only)
  - if --host is not passed, the script offers [1]=localhost and [2]=0.0.0.0
  - window is optional: add --window
  - disable web server: --no-web

See also:
  ./run.sh --help-web
HELP
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

export YOLO_CONFIG_DIR="$SCRIPT_DIR/$ULTRA_CFG_DIR"
export SC_WEIGHTS_DIR="$SCRIPT_DIR/$WEIGHTS_DIR"
export SC_RUNS_DIR="$SCRIPT_DIR/$RUNS_DIR"
export SC_DATASETS_DIR="$SCRIPT_DIR/$DATASETS_DIR"

if ! has_opt "--source" "${FORWARD[@]}" && ! has_opt "--cam" "${FORWARD[@]}"; then
  default_source="$(default_source_for_os)"
  if [[ "$SILENT" == "1" || ! -t 0 ]]; then
    selected_source="$default_source"
  else
    selected_source="$(prompt_for_source "$default_source")"
  fi
  FORWARD=(--source "$selected_source" "${FORWARD[@]}")
fi

if ! has_opt "--device" "${FORWARD[@]}"; then
  FORWARD=(--device "${DEFAULT_DEVICE:-auto}" "${FORWARD[@]}")
fi

if ! has_opt "--host" "${FORWARD[@]}"; then
  default_host_choice_value="$(default_host_choice)"
  if [[ "$SILENT" == "1" || ! -t 0 ]]; then
    selected_host="$(resolve_host_choice "$default_host_choice_value")"
  else
    selected_host_choice="$(prompt_for_host_choice "$default_host_choice_value")"
    selected_host="$(resolve_host_choice "$selected_host_choice")"
  fi
  FORWARD=(--host "$selected_host" "${FORWARD[@]}")
fi

if ! has_arg "--max-fps" "${FORWARD[@]}"; then
  FORWARD+=(--max-fps "${DEFAULT_MAX_FPS:-120}")
fi

if [[ "${DEFAULT_USE_POSE:-1}" == "1" ]] && ! has_opt "--no-pose" "${FORWARD[@]}" && ! has_opt "--use-pose" "${FORWARD[@]}"; then
  FORWARD+=(--use-pose)
fi

FINAL_SOURCE=""
for ((i=0; i<${#FORWARD[@]}; i++)); do
  case "${FORWARD[$i]}" in
    --source)
      if (( i + 1 < ${#FORWARD[@]} )); then
        FINAL_SOURCE="${FORWARD[$((i + 1))]}"
      fi
      ;;
    --source=*) FINAL_SOURCE="${FORWARD[$i]#--source=}" ;;
    --cam)
      if (( i + 1 < ${#FORWARD[@]} )); then
        FINAL_SOURCE="${FORWARD[$((i + 1))]}"
      fi
      ;;
    --cam=*) FINAL_SOURCE="${FORWARD[$i]#--cam=}" ;;
  esac
  [[ -n "$FINAL_SOURCE" ]] && break
done

if [[ -n "$FINAL_SOURCE" ]]; then
  if ! validate_source "$FINAL_SOURCE"; then
    if [[ "$FINAL_SOURCE" =~ ^[0-9]+$ ]]; then
      die "Selected camera '$FINAL_SOURCE' is not available."
    else
      die "Selected source '$FINAL_SOURCE' could not be opened."
    fi
  fi
fi

exec python webcam.py "${FORWARD[@]}"
