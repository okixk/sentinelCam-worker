# Multi-stage Build

# Stage 1: Builder
FROM python:3.12-slim AS builder
WORKDIR /build
# System-Dependencies für OpenCV + Build-Tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime
FROM python:3.12-slim
WORKDIR /app

# System-Dependencies für OpenCV Runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root User
RUN groupadd -g 1000 sentinelcam && \
    useradd -u 1000 -g sentinelcam -m sentinelcam

COPY --from=builder /install /usr/local
COPY --chown=sentinelcam:sentinelcam . .

USER sentinelcam

EXPOSE 8080
EXPOSE 50000-51000/udp

HEALTHCHECK --interval=10s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health', timeout=5)" || exit 1

ENTRYPOINT ["python", "webcam.py"]
CMD ["--host", "0.0.0.0", "--port", "8080", "--no-window", "--stream", "auto"]
