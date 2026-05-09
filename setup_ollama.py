#!/usr/bin/env python3
"""Opt-in native Ollama setup for sentinelCam context detection."""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from typing import Optional


PROFILE_MODELS = {
    "low": "moondream",
    "mid": "gemma3:4b",
    "high": "llama3.2-vision:11b",
    "max": "llava:13b",
}


def _run(cmd: list[str], *, shell: bool = False) -> None:
    print("+ " + (" ".join(cmd) if not shell else cmd[0]), flush=True)
    subprocess.run(cmd if not shell else cmd[0], shell=shell, check=True)


def _capture(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _nvidia_vram_gb() -> float:
    out = _capture(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"])
    values = []
    for line in out.splitlines():
        try:
            values.append(float(line.strip()) / 1024.0)
        except Exception:
            pass
    return max(values) if values else 0.0


def resolve_profile(profile: str) -> str:
    requested = (profile or "auto").strip().lower()
    if requested in PROFILE_MODELS:
        return requested
    if requested not in ("", "auto"):
        raise SystemExit(f"Invalid context profile {profile!r}; use auto, low, mid, high, or max.")

    system = platform.system().lower()
    mem_gb = 0.0
    try:
        if system == "darwin":
            mem_gb = float(int(_capture(["sysctl", "-n", "hw.memsize"]).strip() or "0")) / (1024**3)
    except Exception:
        mem_gb = 0.0

    vram_gb = _nvidia_vram_gb()
    if vram_gb >= 16 or (system == "darwin" and mem_gb >= 32):
        return "max"
    if vram_gb >= 10 or (system == "darwin" and mem_gb >= 24):
        return "high"
    if vram_gb >= 4 or system == "darwin":
        return "mid"
    return "low"


def resolve_model(profile: str, model: Optional[str]) -> tuple[str, str]:
    selected_profile = resolve_profile(profile)
    requested_model = (model or "").strip()
    if requested_model and requested_model.lower() != "auto":
        return selected_profile, requested_model
    return selected_profile, PROFILE_MODELS[selected_profile]


def install_ollama_if_missing() -> None:
    if shutil.which("ollama"):
        print("Ollama CLI already installed.", flush=True)
        return

    system = platform.system().lower()
    if system in ("linux", "darwin"):
        if not shutil.which("curl"):
            raise SystemExit("curl is required to install Ollama automatically.")
        _run(["curl -fsSL https://ollama.com/install.sh | sh"], shell=True)
        return

    if system == "windows":
        if shutil.which("winget"):
            _run(["winget", "install", "--id", "Ollama.Ollama", "-e", "--source", "winget", "--silent"])
            return
        if shutil.which("powershell"):
            _run(["powershell -NoProfile -ExecutionPolicy Bypass -Command \"irm https://ollama.com/install.ps1 | iex\""], shell=True)
            return
        raise SystemExit("Install Ollama from https://ollama.com/download/windows, then rerun with context enabled.")

    raise SystemExit(f"Automatic Ollama install is not supported on {platform.system()}.")


def pull_model(model: str) -> None:
    if not shutil.which("ollama"):
        raise SystemExit("ollama command is not available after install.")
    _run(["ollama", "pull", model])


def main() -> int:
    parser = argparse.ArgumentParser(description="Install Ollama natively and pull a sentinelCam context model.")
    parser.add_argument("--install", action="store_true", help="Install Ollama if the CLI is missing.")
    parser.add_argument("--pull", action="store_true", help="Pull the selected Ollama model.")
    parser.add_argument("--profile", default=os.environ.get("DEFAULT_CONTEXT_PROFILE", "auto"))
    parser.add_argument("--model", default=os.environ.get("DEFAULT_CONTEXT_MODEL", "auto"))
    parser.add_argument("--print-model", action="store_true", help="Only print the resolved model.")
    args = parser.parse_args()

    profile, model = resolve_model(args.profile, args.model)
    if args.print_model:
        print(model)
        return 0

    print(f"Context profile: {profile} -> Ollama model: {model}", flush=True)
    if args.install:
        install_ollama_if_missing()
    if args.pull:
        pull_model(model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
