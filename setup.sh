#!/usr/bin/env bash
# Interactive one-shot setup for sentinelCam-worker.
# Asks which mode to run, prepares .env from the matching template, builds
# and starts the container. Idempotent — re-running just re-applies state.

set -euo pipefail

cd "$(dirname "$0")"

color() { printf "\033[%sm%s\033[0m\n" "$1" "$2"; }
info()  { color "1;34" "==> $*"; }
warn()  { color "1;33" "!!  $*"; }
die()   { color "1;31" "xx  $*" >&2; exit 1; }

command -v docker >/dev/null || die "docker is required. Install: https://get.docker.com"
docker compose version >/dev/null 2>&1 || die "docker compose v2 is required (try: docker compose version)"

if [[ "${1:-}" == "" ]]; then
  echo "Which mode do you want to run?"
  echo "  1) Pipeline   — outbound WSS client to sentinelCam-web (GPU recommended)"
  echo "  2) Standalone — worker serves browsers directly (WebRTC/MJPEG)"
  read -r -p "Choice [1/2]: " choice
else
  choice="$1"
fi

case "$choice" in
  1|pipeline)
    MODE=pipeline
    EXAMPLE=.env.pipeline.example
    COMPOSE_FILES=(-f docker-compose.pipeline.yml)
    if ! command -v nvidia-smi >/dev/null 2>&1; then
      warn "nvidia-smi not found — falling back to CPU compose override."
      COMPOSE_FILES+=(-f docker-compose.pipeline.cpu.yml)
    fi
    ;;
  2|standalone)
    MODE=standalone
    EXAMPLE=.env.standalone.example
    COMPOSE_FILES=(-f docker-compose.worker.yml)
    if [[ "$(uname -s)" == "Linux" ]] && [[ -e "/dev/video0" ]]; then
      read -r -p "Use host webcam /dev/video0? [y/N] " usecam
      if [[ "$usecam" =~ ^[Yy]$ ]]; then
        COMPOSE_FILES+=(-f docker-compose.worker-cam.yml)
      fi
    fi
    ;;
  *)
    die "Unknown choice: $choice (expected 1/pipeline or 2/standalone)"
    ;;
esac

info "Selected mode: $MODE"

if [[ ! -f .env ]]; then
  cp "$EXAMPLE" .env
  chmod 600 .env
  warn ".env created from $EXAMPLE — edit it now to set credentials, then re-run this script."
  warn "  \$EDITOR .env"
  exit 0
fi

chmod 600 .env || true

info "Building image..."
docker compose "${COMPOSE_FILES[@]}" build

info "Starting container..."
docker compose "${COMPOSE_FILES[@]}" up -d

info "Status:"
docker compose "${COMPOSE_FILES[@]}" ps

cat <<EOF

Done. Useful follow-ups:
  Logs:     docker compose ${COMPOSE_FILES[*]} logs -f
  Restart:  docker compose ${COMPOSE_FILES[*]} restart
  Stop:     docker compose ${COMPOSE_FILES[*]} down
EOF
