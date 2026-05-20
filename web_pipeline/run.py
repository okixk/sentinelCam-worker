"""Entry point: ``python -m web_pipeline.run``.

Reads WEB_URL, WEB_TOKEN, WORKER_NAME, JPEG_QUALITY from the environment,
sets up logging, and runs the WebSocket worker client until stopped.
"""
from __future__ import annotations

import asyncio
import logging
import os

from web_pipeline.client import WorkerConfig, run_worker, stub_overlay


def _setup_logging() -> None:
    level = (os.environ.get("LOG_LEVEL") or "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )


def main() -> int:
    _setup_logging()
    config = WorkerConfig.from_env()
    log = logging.getLogger("sentinelCam.worker.run")
    log.info("starting worker name=%s -> %s", config.name, config.web_url)
    asyncio.run(run_worker(config, processor=stub_overlay))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
