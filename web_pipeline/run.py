"""Entry point: ``python -m web_pipeline.run``.

Reads ``WEB_URL`` and ``WEB_TOKEN`` from the environment, picks a frame
processor based on ``WORKER_YOLO_MODEL`` (if set, load real YOLO via
:mod:`web_pipeline.inference`; otherwise fall back to the
:func:`stub_overlay` so the pipeline is still testable end-to-end), sets
up logging, and runs the WebSocket worker client until stopped.
"""
from __future__ import annotations

import asyncio
import logging
import os

from web_pipeline.client import WorkerConfig, run_worker, stub_overlay
from web_pipeline.inference import load_yolo_processor


def _setup_logging() -> None:
    level = (os.environ.get("LOG_LEVEL") or "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )


def main() -> int:
    _setup_logging()
    log = logging.getLogger("sentinelCam.worker.run")
    config = WorkerConfig.from_env()

    try:
        processor = load_yolo_processor()
    except RuntimeError as exc:
        log.error("%s", exc)
        return 2
    if processor is None:
        log.warning(
            "WORKER_YOLO_MODEL not set — using stub overlay (no detections will fire auto-recording)"
        )
        processor = stub_overlay
    else:
        log.info("YOLO inference active")

    log.info("starting worker name=%s -> %s", config.name, config.web_url)
    asyncio.run(run_worker(config, processor=processor))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
