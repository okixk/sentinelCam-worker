#!/usr/bin/env python3
"""webcam.py

Smart Ultralytics YOLO webcam/stream runner with optional pose.

Adds on top of the original script:
  - Model presets (including an automatic "yolo" preset that picks CPU/GPU defaults)
  - Device "auto" mode (uses CUDA if available)
  - Hotkeys to cycle through presets at runtime (press 'm' / 'n')
  - Preset gating (e.g. "sam32" can be marked GPU-only)

Notes about "SAM32":
  This script treats "sam32" as a detection-weight preset named "sam32.pt".
  If your file is named differently, either rename it or run with --model /path/to/your.pt
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import time
import shutil
import threading
import queue
import asyncio
from dataclasses import dataclass
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import logging
import warnings

import cv2
import numpy as np
from ultralytics import YOLO

# stream_server lives in THIS worker repo.
# It exposes only the stream + JSON APIs; the actual website lives in sentinelCam-web.
from stream_server import FrameHub, ControlAPI, run_mjpeg_server
run_webrtc_server = None  # worker-only mode: no WebRTC server here


try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None


# -----------------------------
# Optional: suppress WARNING lines (grep -v "WARNING" equivalent)
# -----------------------------
class _DropWarnings(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return "WARNING" not in msg


def setup_quiet_warnings(enable: bool):
    if not enable:
        return
    warnings.filterwarnings("ignore")

    try:
        ul = logging.getLogger("ultralytics")
        ul.addFilter(_DropWarnings())
        ul.setLevel(logging.ERROR)
    except Exception:
        pass

    try:
        logging.getLogger("torch").setLevel(logging.ERROR)
    except Exception:
        pass

    logging.getLogger().setLevel(logging.ERROR)


# -----------------------------
# Helper: IoU for box matching
# -----------------------------
def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union


# -----------------------------
# Pose drawing (COCO-17 skeleton)
# -----------------------------
COCO17_SKELETON = [
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]


def draw_pose_skeleton(frame, kpts_xy, kpts_conf, conf_thr=0.35):
    """Draw keypoints + skeleton lines for one person."""
    for a, b in COCO17_SKELETON:
        if kpts_conf[a] >= conf_thr and kpts_conf[b] >= conf_thr:
            ax, ay = kpts_xy[a]
            bx, by = kpts_xy[b]
            cv2.line(
                frame,
                (int(round(ax)), int(round(ay))),
                (int(round(bx)), int(round(by))),
                (255, 0, 0), 2, cv2.LINE_AA
            )

    for i, (x, y) in enumerate(kpts_xy):
        if kpts_conf[i] >= conf_thr:
            cv2.circle(frame, (int(round(x)), int(round(y))), 3, (255, 0, 0), -1)


# -----------------------------
# Sitting heuristic from pose keypoints (COCO-17 order)
# YOLOv8-pose uses 17 keypoints:
# 11 L_hip, 12 R_hip, 13 L_knee, 14 R_knee
# -----------------------------
def is_sitting_from_kpts(kpts_xy, kpts_conf, box_h, conf_thr=0.35):
    idxs = [11, 12, 13, 14]
    if any(kpts_conf[i] < conf_thr for i in idxs):
        return None  # unknown

    l_hip_y = kpts_xy[11][1]
    r_hip_y = kpts_xy[12][1]
    l_knee_y = kpts_xy[13][1]
    r_knee_y = kpts_xy[14][1]

    hip_y = 0.5 * (l_hip_y + r_hip_y)
    knee_y = 0.5 * (l_knee_y + r_knee_y)

    dy = abs(hip_y - knee_y) / max(box_h, 1.0)
    return dy < 0.18


# -----------------------------
# Face center from pose keypoints
# COCO-17: 0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear
# We'll use nose/eyes primarily.
# -----------------------------
def face_center_from_kpts(kpts_xy, kpts_conf, conf_thr=0.35):
    face_idxs = [0, 1, 2]  # nose + eyes
    pts = []
    for i in face_idxs:
        if kpts_conf[i] >= conf_thr:
            pts.append(kpts_xy[i])
    if not pts:
        return None
    pts = np.array(pts, dtype=np.float32)
    fx, fy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
    return fx, fy


def draw_label(frame, x, y, text):
    """Draw readable text with outline."""
    cv2.putText(
        frame, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA
    )
    cv2.putText(
        frame, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
    )


def _open_capture(source: str, width: int, height: int):
    """OpenCV VideoCapture opener.

    - If source is a digit string -> webcam index
    - Otherwise -> URL/file/rtsp/http or /dev/videoX path
    """
    if isinstance(source, str) and source.isdigit():
        cam_index = int(source)
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(cam_index)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        return cap

    return cv2.VideoCapture(source)


# -----------------------------
# Presets / device selection
# -----------------------------
def _cuda_available() -> bool:
    try:
        return bool(torch is not None and torch.cuda.is_available())
    except Exception:
        return False


def resolve_device(device_arg: str) -> str:
    """Return an Ultralytics-compatible device value.

    - "auto" -> "0" if CUDA available else "cpu"
    - "cuda" / "cuda:0" -> "0"
    - otherwise passthrough ("cpu", "0", "1", ...)
    """
    d = (device_arg or "").strip().lower()
    if d in ("auto", ""):
        return "0" if _cuda_available() else "cpu"
    if d.startswith("cuda"):
        return "0"
    return device_arg


def _runtime_override_dir(env_name: str) -> Optional[str]:
    v = (os.environ.get(env_name) or "").strip()
    return v or None



def _configure_ultralytics_runtime_dirs() -> None:
    """Best-effort pin Ultralytics runtime dirs into this project's .runtime tree.

    Launchers provide absolute overrides via SC_* env vars. We keep YOLO_CONFIG_DIR
    for the settings file itself, but explicitly redirect weights/runs/datasets so
    downloads and generated artifacts do not spill into the repo root or user cache.
    """
    updates = {}
    weights_dir = _runtime_override_dir("SC_WEIGHTS_DIR")
    runs_dir = _runtime_override_dir("SC_RUNS_DIR")
    datasets_dir = _runtime_override_dir("SC_DATASETS_DIR")

    if weights_dir:
        updates["weights_dir"] = weights_dir
    if runs_dir:
        updates["runs_dir"] = runs_dir
    if datasets_dir:
        updates["datasets_dir"] = datasets_dir

    if not updates:
        return

    for path in updates.values():
        try:
            os.makedirs(path, exist_ok=True)
        except Exception:
            pass

    try:
        from ultralytics import settings  # type: ignore
        if hasattr(settings, "update"):
            settings.update(updates)
    except Exception:
        pass



def _ultra_weights_dir() -> Optional[str]:
    """Best-effort get Ultralytics weights_dir.

    Prefer the launcher's explicit .runtime override, then Ultralytics settings,
    then a YOLO_CONFIG_DIR fallback.
    """
    override = _runtime_override_dir("SC_WEIGHTS_DIR")
    if override:
        return override

    try:
        from ultralytics import settings  # type: ignore
        wd = settings.get("weights_dir") if hasattr(settings, "get") else None
        if isinstance(wd, str) and wd:
            return wd
    except Exception:
        pass

    ycd = os.environ.get("YOLO_CONFIG_DIR")
    if ycd:
        return os.path.join(ycd, "weights")

    return None


def _legacy_ultra_weight_dirs() -> List[str]:
    """Common Ultralytics weight locations outside our project.

    Why this exists:
      - You may already have custom weights (e.g. my-custom.pt) in a previous
        Ultralytics cache (~/.cache/ultralytics, ~/.config/Ultralytics, etc.).
      - run.sh/run.bat intentionally redirect Ultralytics settings into .runtime,
        so we won't see those older caches unless we explicitly look for them.
    """
    home = os.path.expanduser("~")
    bases = [
        os.path.join(home, ".cache", "ultralytics"),
        os.path.join(home, ".cache", "Ultralytics"),
        os.path.join(home, ".config", "Ultralytics"),
        os.path.join(home, ".config", "ultralytics"),
        os.path.join(home, ".local", "share", "Ultralytics"),
        os.path.join(home, ".local", "share", "ultralytics"),
    ]

    # Also honor an explicit (older) YOLO_CONFIG_DIR if the user set it.
    ycd = os.environ.get("YOLO_CONFIG_DIR")
    if ycd:
        bases.append(ycd)

    # Add both the base and common "weights" subfolder.
    out: List[str] = []
    for b in bases:
        if not b:
            continue
        out.append(b)
        out.append(os.path.join(b, "weights"))
    return out


def _is_path_like(s: str) -> bool:
    return bool(os.path.isabs(s) or os.sep in s or (os.path.altsep and os.path.altsep in s))


def _promote_weight_to_runtime(ref: Optional[str]) -> Optional[str]:
    """If a bare-name weight was downloaded into cwd/script dir, move it into weights_dir."""
    if not ref:
        return ref

    s = str(ref).strip()
    if not s or _is_path_like(s):
        return s

    wd = _ultra_weights_dir()
    if not wd:
        return s

    try:
        os.makedirs(wd, exist_ok=True)
    except Exception:
        return s

    dst = os.path.join(wd, os.path.basename(s))
    if os.path.exists(dst):
        # optional cleanup of duplicate in cwd
        src_cwd = os.path.join(os.getcwd(), s)
        try:
            if os.path.exists(src_cwd) and os.path.abspath(src_cwd) != os.path.abspath(dst):
                os.remove(src_cwd)
        except Exception:
            pass
        return dst

    for src in (
        os.path.join(os.getcwd(), s),
        os.path.join(os.path.dirname(__file__), s),
    ):
        try:
            if os.path.exists(src) and os.path.abspath(src) != os.path.abspath(dst):
                shutil.move(src, dst)
                return dst
        except Exception:
            pass

    return s


def resolve_weights(maybe_name_or_path: str) -> str:
    """Resolve a weights reference.

    If it looks like a path, return it.
    If it's a bare filename, prefer a file found in:
      - Ultralytics weights_dir (if configured)
      - cwd
      - script directory
      - legacy Ultralytics dirs
    Otherwise return the original string (Ultralytics may auto-download).
    """
    s = (maybe_name_or_path or "").strip()
    if not s:
        return s

    # already a (likely) path
    if _is_path_like(s):
        return s

    candidates: List[str] = []
    for base in (_ultra_weights_dir(), os.getcwd(), os.path.dirname(__file__), *_legacy_ultra_weight_dirs()):
        if base:
            candidates.append(os.path.join(base, s))

    for p in candidates:
        if os.path.exists(p):
            # If we found it in a legacy location, optionally copy into the
            # configured Ultralytics weights_dir so the next run is fast/clean.
            wd = _ultra_weights_dir()
            try:
                if wd:
                    os.makedirs(wd, exist_ok=True)
                    dst = os.path.join(wd, os.path.basename(p))
                    if os.path.abspath(p) != os.path.abspath(dst) and not os.path.exists(dst):
                        shutil.copy2(p, dst)
            except Exception:
                pass
            return p

    return s


# Ultralytics can auto-download *public* pretrained weights when you pass the
# filename (e.g. "yolov8n.pt", "yolo26x.pt", "yolo26x-pose.pt").
#
# The original version of this script only whitelisted YOLOv8 names, and treated
# YOLO26 as "custom", which broke auto-download.
_ULTRA_AUTO_DL_RE = re.compile(
    r"^(?:yolov8|yolo(?:11|12|26))[nslmx](?:-(?:pose|seg|cls|obb))?\.pt$",
    re.IGNORECASE,
)


def _is_ultralytics_auto_downloadable_weight(name: str) -> bool:
    """Return True if Ultralytics is expected to auto-download `name`.

    This intentionally errs on the side of allowing more names: if the weight
    doesn't exist upstream, Ultralytics will error, but we won't block it
    prematurely.
    """
    n = (name or "").strip()
    if not n:
        return False
    if _ULTRA_AUTO_DL_RE.match(n):
        return True
    # Some official releases use extra suffixes (e.g. experimental tags).
    # If it starts with a known public family name and ends with .pt, let
    # Ultralytics try.
    if n.lower().startswith(("yolov8", "yolo11", "yolo12", "yolo26")) and n.lower().endswith(".pt"):
        return True
    return False


def _ensure_weights_available(ref: str, what: str) -> None:
    """Fail fast for custom weights that can't be auto-downloaded.

    Ultralytics can auto-download many public weights when passed by name
    (e.g. YOLOv8/YOLO26 families). For truly unknown names (custom weights),
    require the file to exist (in cwd, script dir, or Ultralytics weights_dir).
    """
    if not ref:
        return

    # If this is a path, it must exist.
    is_path = _is_path_like(ref)
    if is_path:
        if not os.path.exists(ref):
            raise SystemExit(f"{what} weights not found: {ref}")
        return

    # It's a bare name.
    if _is_ultralytics_auto_downloadable_weight(ref):
        return  # Ultralytics will auto-download into weights_dir if needed

    # Unknown bare name must exist in current working dir.
    if not os.path.exists(ref):
        wd = _ultra_weights_dir()
        legacy_hint = "If you already had it from an older setup, look in ~/.cache/ultralytics or ~/.config/Ultralytics/weights and copy it over."
        hint = (
            f"Place it in {wd} (preferred) or pass --model/--pose-model with a full path. {legacy_hint}"
            if wd
            else f"Pass a full path via --model/--pose-model. {legacy_hint}"
        )
        raise SystemExit(f"{what} weights '{ref}' not found. {hint}")


@dataclass(frozen=True)
class Preset:
    name: str
    det: str
    pose: Optional[str]
    description: str
    requires_cuda: bool = False


def build_presets() -> Dict[str, Preset]:
    """Define your preset catalog here."""
    return {
        # The main one you asked for:
        # - CPU => YOLOv8n (+ yolov8n-pose)
        # - GPU => YOLO26x (+ yolo26x-pose)
        # (implemented as "yolo" selection logic in pick_preset())
        "yolov8n": Preset(
            name="yolov8n",
            det="yolov8n.pt",
            pose="yolov8n-pose.pt",
            description="Ultralytics YOLOv8 nano (fastest CPU default)",
        ),
        "yolov8s": Preset(
            name="yolov8s",
            det="yolov8s.pt",
            pose="yolov8s-pose.pt",
            description="Ultralytics YOLOv8 small",
        ),
        "yolov8m": Preset(
            name="yolov8m",
            det="yolov8m.pt",
            pose="yolov8m-pose.pt",
            description="Ultralytics YOLOv8 medium",
        ),
        "yolov8l": Preset(
            name="yolov8l",
            det="yolov8l.pt",
            pose="yolov8l-pose.pt",
            description="Ultralytics YOLOv8 large",
        ),
        "yolov8x": Preset(
            name="yolov8x",
            det="yolov8x.pt",
            pose="yolov8x-pose.pt",
            description="Ultralytics YOLOv8 extra-large (heaviest)",
        ),
        "yolo26x": Preset(
            name="yolo26x",
            det="yolo26x.pt",
            pose="yolo26x-pose.pt",
            description="Ultralytics YOLO26x pretrained weights (GPU recommended)",
        ),
        "sam32": Preset(
            name="sam32",
            det="sam32.pt",
            pose=None,
            description="SAM32 (treated as detection weights here) - GPU only",
            requires_cuda=True,
        ),
    }


def pick_preset(preset_name: str, device_resolved: str, presets: Dict[str, Preset]) -> Tuple[str, Preset]:
    """Pick an actual preset object.

    Special handling:
      preset_name == "yolo" selects CPU/GPU defaults.
    """
    pn = (preset_name or "").strip().lower()
    is_cpu = (device_resolved == "cpu")

    if pn in ("yolo", "auto", "default", ""):
        # your requested default behavior
        chosen = "yolov8n" if is_cpu else "yolo26x"
        return chosen, presets[chosen]

    if pn not in presets:
        valid = ", ".join(["yolo"] + sorted(presets.keys()))
        raise SystemExit(f"Unknown --preset '{preset_name}'. Valid: {valid}")
    return pn, presets[pn]


def print_presets(presets: Dict[str, Preset]):
    lines = []
    lines.append("Presets (use --preset <name>):")
    lines.append("  yolo    -> auto CPU/GPU default (CPU=yolov8n, GPU=yolo26x)")
    for k in sorted(presets.keys()):
        p = presets[k]
        flag = " [GPU-only]" if p.requires_cuda else ""
        pose = p.pose if p.pose else "(none)"
        lines.append(f"  {p.name:<7} det={p.det}  pose={pose}{flag}  - {p.description}")
    print("\n".join(lines))


def print_vcam_notes():
    print(
        "\n".join(
            [
                "Virtual camera testing tips:",
                "  - Cross-platform open-source option: OBS Studio (Virtual Camera).",
                "  - Linux headless option: v4l2loopback + ffmpeg.",
                "\nThis repo's run.sh supports a one-command vcam test source:",
                "  ./run.sh --vcam",
                "  ./run.sh --vcam --vcam-input demo.mp4",
            ]
        )
    )


def _parse_cycle_list(s: str) -> List[str]:
    out: List[str] = []
    for x in (s or "").split(","):
        x = x.strip()
        if x:
            out.append(x)
    return out


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Hotkeys: q=quit, m=next preset, n=prev preset, p=toggle pose\n"
            "Tip: Use --list-presets to see available pretrained choices.\n"
        ),
    )

    # Source
    ap.add_argument(
        "--source",
        type=str,
        default=None,
        help='Video source: webcam index (e.g. "0") OR URL/file (e.g. http://.../stream.mjpg or /dev/video42)',
    )
    ap.add_argument("--cam", type=int, default=0, help="Webcam index (used when --source not set)")

    # Capture resolution (display + base processing)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)

    # Performance / inference controls
    ap.add_argument("--max-fps", type=float, default=10.0)
    ap.add_argument("--imgsz", type=int, default=832, help="Inference image size (bigger -> better small objects)")
    ap.add_argument(
        "--infer-upscale",
        type=float,
        default=1.0,
        help="Optional: Upscale frame only for inference (e.g. 1.25). Boxes are rescaled back.",
    )

    # Detection thresholds
    ap.add_argument("--conf", type=float, default=0.20, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.60, help="NMS IoU threshold")

    # Model / device
    ap.add_argument(
        "--preset",
        type=str,
        default="yolo",
        help="Model preset name. 'yolo' = auto CPU/GPU default (CPU=yolov8n, GPU=yolo26x).",
    )
    ap.add_argument("--list-presets", action="store_true", help="Print preset list and exit")
    ap.add_argument("--vcam-notes", action="store_true", help="Print virtual camera tips and exit")

    ap.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override: detection weights path/name (bypasses preset det weights)",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Ultralytics device: 'auto', 'cpu' or GPU index like '0'",
    )

    # Pose
    ap.add_argument(
        "--use-pose",
        action="store_true",
        help="Enable pose model (sitting + face point + skeleton/keypoints; slower)",
    )
    ap.add_argument(
        "--no-pose",
        action="store_true",
        help="Force-disable pose even if --use-pose is set by a launcher script",
    )
    ap.add_argument(
        "--pose-model",
        type=str,
        default=None,
        help="Override: pose weights path/name (bypasses preset pose weights)",
    )
    ap.add_argument("--pose-every", type=int, default=3, help="Run pose model every N frames (if --use-pose)")
    ap.add_argument("--pose-kpt-conf", type=float, default=0.35, help="Keypoint confidence threshold")
    ap.add_argument(
        "--no-draw-pose",
        action="store_true",
        help="Disable drawing pose skeleton/keypoints (pose still used for sitting/face)",
    )

    # Tracking
    ap.add_argument("--tracker", type=str, default="bytetrack.yaml", help="Ultralytics tracker config")

    # Logging noise control
    ap.add_argument(
        "--quiet-warnings",
        action="store_true",
        help='Suppress Ultralytics "WARNING" lines (similar to: | grep -v "WARNING")',
    )

    # Runtime model switching
    ap.add_argument(
        "--cycle",
        type=str,
        default="yolov8n,yolov8s,yolov8m,yolov8l,yolov8x,yolo26x,sam32",
        help="Comma-separated preset names for runtime cycling with hotkeys m/n",
    )

    # Web streaming / server (default) + optional local window preview
    ap.add_argument(
        "--help-web",
        action="store_true",
        help="Show a short help for web-related options and exit (equivalent to run.bat/run.sh --help-web).",
    )

    # Stream/API server is the default. Use --no-web for window-only mode.
    ap.set_defaults(web=True, window=False)
    ap.add_argument(
        "--web",
        dest="web",
        action="store_true",
        help="Enable the built-in stream/API server (default).",
    )
    ap.add_argument(
        "--no-web",
        dest="web",
        action="store_false",
        help="Disable the stream/API server (window-only).",
    )
    ap.add_argument(
        "--window",
        dest="window",
        action="store_true",
        help="Also show a local OpenCV preview window + hotkeys (debug/testing).",
    )
    ap.add_argument(
        "--no-window",
        dest="window",
        action="store_false",
        help="Disable local OpenCV preview window (default).",
    )

    ap.add_argument("--host", type=str, default="0.0.0.0", help="Web server bind host (when --web)")
    ap.add_argument("--port", type=int, default=8080, help="Web server port (when --web)")
    ap.add_argument(
        "--stream",
        type=str,
        default="auto",
        choices=["auto", "mjpeg", "webrtc"],
        help="Streaming mode for --web. auto=WebRTC if available else MJPEG.",
    )
    ap.add_argument(
        "--jpeg-quality",
        type=int,
        default=80,
        help="MJPEG JPEG quality (10-95). Lower => less bandwidth, slightly more artifacts.",
    )
    ap.add_argument(
        "--webrtc-codec",
        type=str,
        default="auto",
        choices=["auto", "h264", "vp8", "vp9", "av1"],
        help="Best-effort codec preference for WebRTC (when --stream webrtc).",
    )


    ap.add_argument(
        "--advertise-ip",
        type=str,
        default="",
        help="For WebRTC: advertise / prefer this local IP address in ICE candidates (useful on LAN with multiple adapters).",
    )
    ap.add_argument(
        "--rtc-min-port",
        type=int,
        default=50000,
        help="For WebRTC: try to bind ICE/UDP sockets within this local port range (min). Default 50000.",
    )
    ap.add_argument(
        "--rtc-max-port",
        type=int,
        default=60000,
        help="For WebRTC: try to bind ICE/UDP sockets within this local port range (max). Default 60000.",
    )

    args = ap.parse_args()

    if args.help_web:
        print("sentinelCam Web-Startoptionen")
        print("  --web/--no-web --stream webrtc|mjpeg|auto --host HOST --port PORT")
        print("  Optional debug: --window (show OpenCV preview + hotkeys)")
        print("  WebRTC: --webrtc-codec auto|h264|vp8|vp9|av1 --advertise-ip IP --rtc-min-port 50000 --rtc-max-port 60000")
        print("  MJPEG:  --jpeg-quality 10-95")
        print("  Capture: --width W --height H --source N|URL")
        return

    # Early dependency check: server/web mode needs the separate sentinelCam-web repo (webstream module)
    if bool(getattr(args, 'web', False)) and FrameHub is None:
        raise SystemExit(
            "webstream module not available. Install the 'sentinelCam-web' repo/package (provides webstream.py)\n"
            "Example: python -m pip install git+https://github.com/okixk/sentinelCam-web.git\n"
            "Or run with --no-web to use window-only mode."
        )

    presets = build_presets()
    if args.list_presets:
        print_presets(presets)
        return
    if args.vcam_notes:
        print_vcam_notes()
        return

    setup_quiet_warnings(args.quiet_warnings)

    _configure_ultralytics_runtime_dirs()

    device = resolve_device(args.device)
    if device == "cpu":
        print("Device: cpu")
    else:
        print(f"Device: {device} (CUDA available={_cuda_available()})")

    # Choose preset (unless user forces --model)
    active_preset_name, active_preset = pick_preset(args.preset, device, presets)

    if active_preset.requires_cuda and device == "cpu":
        raise SystemExit(
            f"Preset '{active_preset.name}' is GPU-only, but device resolved to CPU. "
            f"Run with --device 0 (or ensure CUDA works) or pick another preset."
        )

    det_weights = resolve_weights(args.model) if args.model else resolve_weights(active_preset.det)
    pose_weights = None
    if args.use_pose:
        if args.pose_model:
            pose_weights = resolve_weights(args.pose_model)
        else:
            pose_weights = resolve_weights(active_preset.pose) if active_preset.pose else None
            if pose_weights is None:
                # fallback to YOLOv8n pose if a preset doesn't define one
                pose_weights = resolve_weights("yolov8n-pose.pt")

    # Build the cycle list
    cycle_list = _parse_cycle_list(args.cycle)
    if not cycle_list:
        cycle_list = [active_preset_name]

    # Filter cycle list based on device constraints
    filtered_cycle: List[str] = []
    for name in cycle_list:
        n = name.strip().lower()
        if n in ("yolo", "auto", "default", ""):
            # expand to concrete
            n, _ = pick_preset("yolo", device, presets)
        if n not in presets:
            continue
        if presets[n].requires_cuda and device == "cpu":
            continue
        filtered_cycle.append(n)
    if not filtered_cycle:
        filtered_cycle = [active_preset_name]

    # Pose is enabled by default when launchers pass --use-pose, but you can hard-disable via --no-pose.
    pose_enabled = bool(args.use_pose) and (not args.no_pose)

    # Runtime toggles (web UI)
    overlay_enabled = True
    inference_enabled = True
    _saved_pose_enabled = pose_enabled
    _saved_overlay_enabled = overlay_enabled
    cmd_seq_counter = 0

    # Model loaders (switchable)
    def load_models(preset_name: str) -> Tuple[str, YOLO, Optional[YOLO], str, Optional[str]]:
        """Load det+pose weights for a preset, respecting overrides."""
        pn, p = pick_preset(preset_name, device, presets)
        if p.requires_cuda and device == "cpu":
            raise RuntimeError(f"Preset '{pn}' requires CUDA")

        dw = resolve_weights(args.model) if args.model else resolve_weights(p.det)
        _ensure_weights_available(dw, "Detection")

        pw = None
        pm = None
        if pose_enabled:
            if args.pose_model:
                pw = resolve_weights(args.pose_model)
            else:
                pw = resolve_weights(p.pose) if p.pose else resolve_weights("yolov8n-pose.pt")
            _ensure_weights_available(pw, "Pose")
            pm = YOLO(pw)
            pw = _promote_weight_to_runtime(pw)

        dm = YOLO(dw)
        dw = _promote_weight_to_runtime(dw)

        return pn, dm, pm, dw, pw

    # Load initial
    active_preset_name, det_model, pose_model, det_weights, pose_weights = load_models(active_preset_name)
    names = det_model.names

    src = args.source if args.source is not None else str(args.cam)
    cap = _open_capture(src, args.width, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {src}")

    # Reduce capture buffering where supported (helps latency on RTSP/USB cams)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    # max_fps <= 0 => uncapped
    frame_interval = 0.0 if args.max_fps <= 0 else (1.0 / max(args.max_fps, 1e-6))

    # Track history for speed estimation: id -> deque[(t, cx, cy)]
    tracks = defaultdict(lambda: deque(maxlen=30))
    # Cache sitting / face / keypoints per track id
    sit_status: Dict[int, Optional[bool]] = {}
    face_point: Dict[int, Optional[Tuple[float, float]]] = {}
    pose_cache: Dict[int, Tuple[np.ndarray, np.ndarray, int]] = {}

    # Speed thresholds (pixels/sec)
    WALK_PPS = 60.0
    RUN_PPS = 160.0
    STILL_PPS = 25.0

    # Without pose: sitting heuristic via aspect ratio + stillness (very rough)
    SIT_AR_MAX = 1.45
    SIT_H_FRAC_MAX = 0.55

    frame_count = 0
    last_fps_time = time.time()
    fps_smooth = 0.0

    # -----------------------------
    # Output mode: Web server (default) + optional preview window
    # -----------------------------
    window_name = "Office Object Detection + Pose (q=quit, m/n=models, p=pose)"
    web_enabled = bool(getattr(args, 'web', False))
    window_enabled = bool(getattr(args, 'window', False))
    # Convenience: if the user disables the web server and did not request a window,
    # default to the traditional window-only mode.
    if (not web_enabled) and (not window_enabled):
        window_enabled = True

    hub: Optional[object] = None
    stop_event = threading.Event()
    shutdown_done = threading.Event()

    # Web controls (only meaningful in --web mode, but safe elsewhere)
    cmd_q: "queue.SimpleQueue[object]" = queue.SimpleQueue()
    state_lock = threading.Lock()
    web_state: Dict[str, object] = {
        "preset": active_preset_name,
        "device": device,
        "pose_enabled": pose_enabled,
        "overlay_enabled": overlay_enabled,
        "inference_enabled": inference_enabled,
        "fps": 0.0,
        "det": os.path.basename(str(det_weights)) if det_weights else None,
        "pose": os.path.basename(str(pose_weights)) if pose_weights else None,
        "cmd_seq_applied": 0,
        "cmd_last": None,
        "ts": None,
    }

    def _update_state(**kw):
        try:
            with state_lock:
                web_state.update(kw)
        except Exception:
            pass

    def _get_state() -> Dict[str, object]:
        with state_lock:
            return dict(web_state)

    def _send_cmd(payload) -> None:
        """Accept commands from web server.

        Payload may be a string (legacy) or a dict: {"cmd": "...", "seq": N}.
        seq is used by the web UI to confirm that a command was actually applied.
        """
        nonlocal cmd_seq_counter
        cmd = ""
        seq = 0
        try:
            if isinstance(payload, dict):
                cmd = str(payload.get("cmd", "")).strip().lower()
                seq = int(payload.get("seq", 0) or 0)
            else:
                cmd = str(payload).strip().lower()
        except Exception:
            cmd = str(payload).strip().lower()
            seq = 0

        if not cmd:
            return

        if seq <= 0:
            cmd_seq_counter += 1
            seq = cmd_seq_counter

        if cmd in ("stop", "quit", "exit", "q"):
            stop_event.set()

        cmd_q.put((seq, cmd))



    if web_enabled:
        if FrameHub is None:
            raise SystemExit(
                "webstream module not available. Install the 'sentinelCam-web' repo/package (provides webstream.py)\n"
                "Example: python -m pip install git+https://github.com/okixk/sentinelCam-web.git\n"
                "Or run with --no-web to use window-only mode."
            )
        hub = FrameHub(jpeg_quality=args.jpeg_quality)

    if window_enabled:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, args.width, args.height)

    cycle_idx = 0
    if active_preset_name in filtered_cycle:
        cycle_idx = filtered_cycle.index(active_preset_name)

    def _switch_to(idx: int):
        nonlocal cycle_idx, active_preset_name, det_model, pose_model, names
        nonlocal det_weights, pose_weights
        cycle_idx = idx % len(filtered_cycle)
        target = filtered_cycle[cycle_idx]
        if target == active_preset_name:
            return

        # Reload models
        active_preset_name, det_model, pose_model, det_weights, pose_weights = load_models(target)
        names = det_model.names

        # Reset tracking state so IDs don't carry over confusingly
        tracks.clear()
        sit_status.clear()
        face_point.clear()
        pose_cache.clear()

        _update_state(
            preset=active_preset_name,
            det=os.path.basename(str(det_weights)) if det_weights else None,
            pose=os.path.basename(str(pose_weights)) if pose_weights else None,
            pose_enabled=pose_enabled,
            overlay_enabled=overlay_enabled,
            inference_enabled=inference_enabled,
        )

        print(f"Switched preset -> {active_preset_name} (det={det_weights}, pose={pose_weights})")


    def _set_pose(enable: bool) -> None:
        """Enable/disable pose inference (and overlay) at runtime."""
        nonlocal pose_enabled, pose_model, pose_weights
        want = bool(enable)
        if want == pose_enabled:
            return

        pose_enabled = want

        # If pose toggled on and pose_model is missing, try to load default pose for current preset
        if pose_enabled and pose_model is None:
            try:
                _, p = pick_preset(active_preset_name, device, presets)
                pw = resolve_weights(p.pose) if p.pose else resolve_weights("yolov8n-pose.pt")
                _ensure_weights_available(pw, "Pose")
                pose_model = YOLO(pw)
                pose_weights = _promote_weight_to_runtime(pw)
            except Exception as e:
                print(f"Could not enable pose: {e}")
                pose_enabled = False

        _update_state(
            pose_enabled=pose_enabled,
            pose=os.path.basename(str(pose_weights)) if pose_weights else None,
        )
        print(f"Pose -> {'on' if pose_enabled else 'off'}")


    def _toggle_pose():
        _set_pose(not pose_enabled)



    def processing_loop():
        nonlocal frame_count, last_fps_time, fps_smooth
        nonlocal cycle_idx, active_preset_name, det_model, pose_model, names
        nonlocal det_weights, pose_weights, pose_enabled
        nonlocal overlay_enabled, inference_enabled, _saved_pose_enabled, _saved_overlay_enabled

        try:
            while not stop_event.is_set():
                # Apply pending commands from the web UI (non-blocking)
                while True:
                    try:
                        item = cmd_q.get_nowait()
                    except queue.Empty:
                        break
                    except Exception:
                        break

                    seq = 0
                    cmd = ""
                    try:
                        if isinstance(item, (tuple, list)) and len(item) >= 2:
                            seq = int(item[0] or 0)
                            cmd = str(item[1]).strip().lower()
                        elif isinstance(item, dict):
                            cmd = str(item.get("cmd", "")).strip().lower()
                            seq = int(item.get("seq", 0) or 0)
                        else:
                            cmd = str(item).strip().lower()
                            seq = 0
                    except Exception:
                        cmd = str(item).strip().lower()
                        seq = 0

                    if not cmd:
                        continue

                    applied = False

                    if cmd in ("stop", "quit", "exit", "q"):
                        stop_event.set()
                        applied = True

                    elif cmd in ("next", "m") and len(filtered_cycle) > 1:
                        _switch_to(cycle_idx + 1)
                        applied = True

                    elif cmd in ("prev", "previous", "n") and len(filtered_cycle) > 1:
                        _switch_to(cycle_idx - 1)
                        applied = True

                    elif cmd in ("toggle_pose", "pose", "p"):
                        # Pose nur sinnvoll, wenn Inference an ist
                        if inference_enabled:
                            _toggle_pose()
                        applied = True

                    elif cmd in ("toggle_overlay", "overlay", "o"):
                        # Overlay nur sinnvoll, wenn Inference an ist
                        if inference_enabled:
                            overlay_enabled = not overlay_enabled
                            _saved_overlay_enabled = overlay_enabled
                        else:
                            overlay_enabled = False
                        _update_state(overlay_enabled=overlay_enabled)
                        print(f"Overlay -> {'on' if overlay_enabled else 'off'}")
                        applied = True

                    elif cmd in ("toggle_inference", "inference", "model", "i"):
                        if inference_enabled:
                            # Stream-only: keine Inference, keine Overlays
                            _saved_pose_enabled = pose_enabled
                            _saved_overlay_enabled = overlay_enabled
                            inference_enabled = False
                            overlay_enabled = False
                            if pose_enabled:
                                _set_pose(False)
                            tracks.clear()
                            sit_status.clear()
                            face_point.clear()
                            pose_cache.clear()
                            _update_state(inference_enabled=False, overlay_enabled=False, pose_enabled=pose_enabled)
                            print("Inference -> off (stream-only)")
                        else:
                            inference_enabled = True
                            overlay_enabled = bool(_saved_overlay_enabled)
                            _update_state(inference_enabled=True, overlay_enabled=overlay_enabled)
                            print("Inference -> on")
                            # Pose ggf. wieder herstellen (wenn vorher an)
                            if _saved_pose_enabled and not pose_enabled:
                                _set_pose(True)
                        applied = True

                    if applied:
                        _update_state(
                            cmd_seq_applied=int(seq or 0),
                            cmd_last=cmd,
                            overlay_enabled=overlay_enabled,
                            inference_enabled=inference_enabled,
                            pose_enabled=pose_enabled,
                        )
                        if stop_event.is_set():
                            break

                if stop_event.is_set():
                    break

                loop_start = time.time()
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                # Stream-only mode: no model inference, no overlay (raw frames only)
                if not inference_enabled:
                    now = time.time()
                    dt = now - last_fps_time
                    last_fps_time = now
                    inst_fps = 1.0 / max(dt, 1e-6)
                    fps_smooth = 0.9 * fps_smooth + 0.1 * inst_fps if fps_smooth > 0 else inst_fps

                    _update_state(
                        preset=active_preset_name,
                        pose_enabled=pose_enabled,
                        overlay_enabled=overlay_enabled,
                        inference_enabled=inference_enabled,
                        fps=float(fps_smooth),
                        det=os.path.basename(str(det_weights)) if det_weights else None,
                        pose=os.path.basename(str(pose_weights)) if pose_weights else None,
                        ts=time.time(),
                    )

                    if hub is not None:
                        hub.update(frame)
                    if window_enabled:
                        cv2.imshow(window_name, frame)

                    frame_count += 1

                    elapsed = time.time() - loop_start
                    if frame_interval > 0 and elapsed < frame_interval:
                        time.sleep(frame_interval - elapsed)

                    if window_enabled:
                        k = cv2.waitKey(1) & 0xFF
                        if k == ord("q"):
                            break
                        if k == ord("i"):
                            # toggle inference back on
                            inference_enabled = True
                            overlay_enabled = bool(_saved_overlay_enabled)
                            if _saved_pose_enabled and not pose_enabled:
                                _set_pose(True)
                        if k == ord("o"):
                            # overlay has no effect when inference is off
                            pass
                    continue

                h_frame, _w_frame = frame.shape[:2]

                # Optional: upscale only for inference
                infer_frame = frame
                scale = float(args.infer_upscale)
                if scale != 1.0:
                    infer_frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

                # --- Detection + Tracking ---
                results = det_model.track(
                    infer_frame,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    device=device,
                    persist=True,
                    tracker=args.tracker,
                    verbose=False,
                )
                r = results[0]

                boxes = []
                clss = []
                confs = []
                ids = []

                if r.boxes is not None and len(r.boxes) > 0:
                    xyxy = r.boxes.xyxy.cpu().numpy()
                    cl = r.boxes.cls.cpu().numpy().astype(int)
                    cf = r.boxes.conf.cpu().numpy()
                    if r.boxes.id is not None:
                        tid = r.boxes.id.cpu().numpy().astype(int)
                    else:
                        tid = np.arange(len(xyxy), dtype=int)

                    if scale != 1.0:
                        xyxy = xyxy / scale

                    boxes = xyxy
                    clss = cl
                    confs = cf
                    ids = tid

                # --- Pose pass (persons) ---
                pose_boxes = []
                pose_kpts_xy = []
                pose_kpts_conf = []

                run_pose_now = bool(
                    pose_enabled and pose_model is not None and (frame_count % max(1, args.pose_every) == 0)
                )

                if run_pose_now:
                    pres = pose_model.predict(
                        infer_frame,
                        imgsz=args.imgsz,
                        conf=max(args.conf, 0.15),
                        device=device,
                        verbose=False,
                    )[0]

                    if pres.boxes is not None and len(pres.boxes) > 0 and pres.keypoints is not None:
                        pose_boxes = pres.boxes.xyxy.cpu().numpy()
                        kpts = pres.keypoints.xy.cpu().numpy()      # [N,17,2]
                        kconfs = pres.keypoints.conf.cpu().numpy()  # [N,17]

                        if scale != 1.0:
                            pose_boxes = pose_boxes / scale
                            kpts = kpts / scale

                        pose_kpts_xy = kpts
                        pose_kpts_conf = kconfs

                draw_pose = bool(pose_enabled and (not args.no_draw_pose))

                # --- Draw & infer actions ---
                for box, cls_id, conf, tid in zip(boxes, clss, confs, ids):
                    x1, y1, x2, y2 = box
                    x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])

                    # model.names is commonly a dict, but guard anyway
                    if isinstance(names, dict):
                        label = names.get(int(cls_id), str(cls_id))
                    else:
                        label = names[int(cls_id)] if int(cls_id) < len(names) else str(cls_id)
                    is_person = (label == "person")

                    cx = int(round(0.5 * (x1 + x2)))
                    cy = int(round(0.5 * (y1 + y2)))

                    if is_person:
                        now = time.time()
                        tracks[int(tid)].append((now, float(cx), float(cy)))

                        speed_pps = 0.0
                        hist = tracks[int(tid)]
                        if len(hist) >= 2:
                            t0, x0, y0 = hist[0]
                            t1, x1h, y1h = hist[-1]
                            dt = max(1e-6, (t1 - t0))
                            dist = math.hypot(x1h - x0, y1h - y0)
                            speed_pps = dist / dt

                        sitting = None
                        face_xy = None

                        # Match pose result to this tracked person by IoU
                        if pose_enabled and len(pose_boxes) > 0:
                            best_i = 0.0
                            best_j = -1
                            for j, pbox in enumerate(pose_boxes):
                                v = iou_xyxy(box, pbox)
                                if v > best_i:
                                    best_i = v
                                    best_j = j

                            if best_j >= 0 and best_i >= 0.30:
                                box_h = max(1.0, (y2 - y1))

                                s = is_sitting_from_kpts(
                                    pose_kpts_xy[best_j],
                                    pose_kpts_conf[best_j],
                                    box_h=box_h,
                                    conf_thr=args.pose_kpt_conf,
                                )
                                sitting = s
                                sit_status[int(tid)] = s

                                fp = face_center_from_kpts(
                                    pose_kpts_xy[best_j],
                                    pose_kpts_conf[best_j],
                                    conf_thr=args.pose_kpt_conf,
                                )
                                face_xy = fp
                                face_point[int(tid)] = fp

                                # cache pose for drawing between pose frames
                                pose_cache[int(tid)] = (
                                    pose_kpts_xy[best_j],
                                    pose_kpts_conf[best_j],
                                    frame_count,
                                )
                            else:
                                sitting = sit_status.get(int(tid), None)
                                face_xy = face_point.get(int(tid), None)
                        else:
                            face_xy = face_point.get(int(tid), None)

                        # Fallback sitting
                        if sitting is None:
                            bw = max(1.0, (x2 - x1))
                            bh = max(1.0, (y2 - y1))
                            ar = bh / bw
                            sitting = (speed_pps < STILL_PPS) and (ar < SIT_AR_MAX) and ((bh / h_frame) < SIT_H_FRAC_MAX)

                        if speed_pps > RUN_PPS:
                            action = "running"
                        elif speed_pps > WALK_PPS:
                            action = "walking"
                        else:
                            action = "sitting" if sitting else "standing"

                        # Draw pose skeleton if we have a recent cache for this track
                        if overlay_enabled and draw_pose and int(tid) in pose_cache:
                            kxy, kcf, last_fidx = pose_cache[int(tid)]
                            if frame_count - last_fidx <= max(2, args.pose_every * 2):
                                draw_pose_skeleton(frame, kxy, kcf, conf_thr=args.pose_kpt_conf)

                        if face_xy is not None:
                            fx, fy = face_xy
                            fxi, fyi = int(round(fx)), int(round(fy))
                            text = f"person#{int(tid)} {action} C({cx},{cy}) F({fxi},{fyi}) {conf:.2f}"
                            if overlay_enabled:
                                cv2.circle(frame, (fxi, fyi), 3, (0, 255, 0), -1)
                        else:
                            text = f"person#{int(tid)} {action} C({cx},{cy}) {conf:.2f}"
                    else:
                        text = f"{label} C({cx},{cy}) {conf:.2f}"

                    if overlay_enabled:
                        cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
                        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
                        draw_label(frame, x1i, max(15, y1i - 7), text)

                # FPS display (smoothed)
                now = time.time()
                dt = now - last_fps_time
                last_fps_time = now
                inst_fps = 1.0 / max(dt, 1e-6)
                fps_smooth = 0.9 * fps_smooth + 0.1 * inst_fps if fps_smooth > 0 else inst_fps
                if overlay_enabled:
                    draw_label(frame, 10, 25, f"FPS ~ {fps_smooth:.1f} (cap {args.max_fps})")

                # Show active config
                if overlay_enabled:
                    draw_label(
                        frame,
                        10,
                        45,
                        f"preset={active_preset_name}  det={os.path.basename(str(det_weights))}  device={device}  pose={'on' if pose_enabled else 'off'}",
                    )
                    draw_label(
                        frame,
                        10,
                        65,
                        "Keys: q quit | m next model | n prev model | p pose | o overlay | i model",
                    )

                _update_state(
                    preset=active_preset_name,
                    pose_enabled=pose_enabled,
                    overlay_enabled=overlay_enabled,
                    inference_enabled=inference_enabled,
                    fps=float(fps_smooth),
                    det=os.path.basename(str(det_weights)) if det_weights else None,
                    pose=os.path.basename(str(pose_weights)) if pose_weights else None,
                    ts=time.time(),
                )

                if hub is not None:
                    hub.update(frame)
                if window_enabled:
                    cv2.imshow(window_name, frame)

                frame_count += 1

                elapsed = time.time() - loop_start
                if frame_interval > 0 and elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)

                if window_enabled:
                    k = cv2.waitKey(1) & 0xFF
                    if k == ord("q"):
                        break
                    if k == ord("m") and len(filtered_cycle) > 1:
                        _switch_to(cycle_idx + 1)
                    if k == ord("n") and len(filtered_cycle) > 1:
                        _switch_to(cycle_idx - 1)
                    if k == ord("p"):
                        _toggle_pose()
                    if k == ord("o"):
                        if inference_enabled:
                            overlay_enabled = not overlay_enabled
                            _saved_overlay_enabled = overlay_enabled
                            _update_state(overlay_enabled=overlay_enabled)
                    if k == ord("i"):
                        # toggle inference
                        if inference_enabled:
                            _saved_pose_enabled = pose_enabled
                            _saved_overlay_enabled = overlay_enabled
                            inference_enabled = False
                            overlay_enabled = False
                            if pose_enabled:
                                _set_pose(False)
                            tracks.clear(); sit_status.clear(); face_point.clear(); pose_cache.clear()
                            _update_state(inference_enabled=False, overlay_enabled=False, pose_enabled=pose_enabled)
                        else:
                            inference_enabled = True
                            overlay_enabled = bool(_saved_overlay_enabled)
                            _update_state(inference_enabled=True, overlay_enabled=overlay_enabled)
                            if _saved_pose_enabled and not pose_enabled:
                                _set_pose(True)
        finally:
            stop_event.set()


    if web_enabled:
        # If we are headless (no window), keep the original design:
        #   - producer thread generates frames
        #   - server runs in main thread
        #
        # If a preview window is requested, run OpenCV in the main thread
        # (macOS requirement) and run the server in a background thread.
        assert hub is not None

        def _start_server_blocking():
            display_host = args.host
            if display_host in ("0.0.0.0", "::", "", None):
                display_host = "localhost"
            mode = (args.stream or "auto").lower().strip()
            try:
                if mode == "webrtc":
                    raise SystemExit("--stream webrtc is not supported in worker-only mode. Use MJPEG and let the web repo consume /stream.mjpg.")
                print(f"Stream (MJPEG): http://{display_host}:{args.port}/stream.mjpg")
                run_mjpeg_server(
                    hub,
                    host=args.host,
                    port=args.port,
                    control=ControlAPI(get_state=_get_state, command=_send_cmd),
                    stop_event=stop_event,
                )
            except KeyboardInterrupt:
                pass
            finally:
                stop_event.set()
                shutdown_done.set()

        if not window_enabled:
            # Headless server default
            t = threading.Thread(target=processing_loop, name="sentinelcam-processing", daemon=True)
            t.start()

            # On some Windows camera drivers, cap.read() can block for a long time.
            # Releasing the capture when stop_event is set helps unblock reads and
            # allows graceful shutdown without needing the watchdog.
            def _release_cap_on_stop():
                try:
                    stop_event.wait()
                    try:
                        cap.release()
                    except Exception:
                        pass
                except Exception:
                    pass

            threading.Thread(target=_release_cap_on_stop, name="sentinelcam-cap-release", daemon=True).start()

            # --- Robust local shutdown helpers (Windows-friendly) ---
            def _console_key_listener():
                # Best-effort: 'q' to stop from terminal (no window).
                try:
                    if sys.platform.startswith("win"):
                        import msvcrt  # type: ignore

                        while not stop_event.is_set():
                            try:
                                if msvcrt.kbhit():
                                    ch = msvcrt.getwch()
                                    if (ch or "").lower() == "q":
                                        stop_event.set()
                                        break
                                time.sleep(0.05)
                            except Exception:
                                time.sleep(0.1)
                    else:
                        # On POSIX we keep it simple: require Enter.
                        while not stop_event.is_set():
                            try:
                                line = sys.stdin.readline()
                            except Exception:
                                break
                            if not line:
                                break
                            if line.strip().lower() in ("q", "quit", "exit"):
                                stop_event.set()
                                break
                except Exception:
                    return

            def _force_exit_guard():
                stop_event.wait()
                if shutdown_done.wait(timeout=10.0):
                    return
                try:
                    print("\n[watchdog] Shutdown appears stuck -> forcing exit")
                except Exception:
                    pass
                try:
                    os._exit(0)
                except Exception:
                    pass

            threading.Thread(target=_console_key_listener, name="sentinelcam-console-keys", daemon=True).start()
            threading.Thread(target=_force_exit_guard, name="sentinelcam-exit-guard", daemon=True).start()

            _start_server_blocking()
            try:
                t.join(timeout=2.0)
            except Exception:
                pass
            shutdown_done.set()
        else:
            # Server + preview window: run server in background, OpenCV in main thread.
            server_t = threading.Thread(target=_start_server_blocking, name="sentinelcam-webserver", daemon=True)
            server_t.start()
            try:
                processing_loop()
            finally:
                stop_event.set()
                try:
                    server_t.join(timeout=2.0)
                except Exception:
                    pass
                shutdown_done.set()
    else:
        # Window-only
        processing_loop()


    # mark shutdown complete for watchdog
    shutdown_done.set()

    cap.release()
    if window_enabled:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
