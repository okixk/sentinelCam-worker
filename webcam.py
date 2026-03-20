#!/usr/bin/env python3
"""webcam.py

Smart Ultralytics YOLO webcam/stream runner with optional pose.

Adds on top of the original script:
  - Model presets (including an automatic "yolo" preset that picks CPU/GPU defaults)
  - Device "auto" mode (prefers CUDA, then Apple Metal/MPS, else CPU)
  - Hotkeys to cycle through presets at runtime (press 'm' / 'n')
  - Preset gating (e.g. "sam32" can be marked GPU-only)
"""

import argparse
import math
import os
import platform
import re
import sys
import time
import shutil
import threading
import queue
from dataclasses import dataclass
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import logging
import warnings
import traceback

import cv2
import numpy as np
from ultralytics import YOLO

# stream_server lives in THIS worker repo.
# It exposes only the stream + JSON APIs; the actual website lives in sentinelCam-web.
from stream_server import FrameHub, run_mjpeg_server
from security import ControlAPI, SecurityConfig
webrtc_import_error = None
try:
    from webrtc_server import run_webrtc_server
except Exception as exc:
    run_webrtc_server = None
    webrtc_import_error = exc


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


def draw_top_right_label(frame, text, y=25, pad=10):
    """Draw a readable label aligned to the top-right corner."""
    (text_w, _text_h), _baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        1,
    )
    frame_w = frame.shape[1]
    x = max(pad, frame_w - text_w - pad)
    draw_label(frame, x, y, text)


class SyntheticCapture:
    """Small OpenCV-like capture source for cross-platform testing.

    It lets Docker and local setups boot without a physical camera by generating
    animated frames in-process.
    """

    def __init__(self, width: int, height: int, fps: float = 30.0):
        self.width = max(160, int(width))
        self.height = max(120, int(height))
        self.fps = max(1.0, float(fps))
        self._opened = True
        self._started = time.time()
        self._frame_index = 0

    def isOpened(self) -> bool:  # noqa: N802
        return self._opened

    def release(self) -> None:
        self._opened = False

    def set(self, prop_id, value) -> bool:  # noqa: ANN001
        try:
            if int(prop_id) == int(cv2.CAP_PROP_FRAME_WIDTH):
                self.width = max(160, int(value))
                return True
            if int(prop_id) == int(cv2.CAP_PROP_FRAME_HEIGHT):
                self.height = max(120, int(value))
                return True
        except Exception:
            return False
        return True

    def read(self):  # noqa: ANN201
        if not self._opened:
            return False, None

        frame_period = 1.0 / self.fps
        due_at = self._started + (self._frame_index * frame_period)
        remaining = due_at - time.time()
        if remaining > 0:
            time.sleep(min(remaining, frame_period))

        elapsed = max(0.0, time.time() - self._started)
        self._frame_index += 1

        x_grad = np.linspace(30, 130, self.width, dtype=np.uint8)
        y_grad = np.linspace(20, 90, self.height, dtype=np.uint8)
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:, :, 0] = x_grad
        frame[:, :, 1] = y_grad[:, None]
        frame[:, :, 2] = 45

        cx = int((math.sin(elapsed * 1.1) * 0.4 + 0.5) * max(1, self.width - 1))
        cy = int((math.cos(elapsed * 0.9) * 0.35 + 0.5) * max(1, self.height - 1))
        radius = max(18, min(self.width, self.height) // 10)
        cv2.circle(frame, (cx, cy), radius, (0, 215, 255), -1)

        box_w = max(80, self.width // 5)
        box_h = max(50, self.height // 6)
        rect_x = int((math.sin(elapsed * 0.6 + 1.0) * 0.35 + 0.5) * max(1, self.width - box_w))
        rect_y = int((math.sin(elapsed * 0.8 + 0.4) * 0.3 + 0.5) * max(1, self.height - box_h))
        cv2.rectangle(frame, (rect_x, rect_y), (rect_x + box_w, rect_y + box_h), (255, 120, 0), -1)

        cv2.putText(
            frame,
            "sentinelCam testsrc",
            (18, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"{self.width}x{self.height}  frame={self._frame_index}",
            (18, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Use WORKER_SOURCE=testsrc for Docker smoke tests",
            (18, max(92, self.height - 26)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        return True, frame


def _open_capture(source: str, width: int, height: int):
    """OpenCV VideoCapture opener.

    - If source is a digit string -> webcam index
    - testsrc/synthetic/dummy -> generated in-process frames
    - Otherwise -> URL/file/rtsp/http or /dev/videoX path
    """
    normalized_source = str(source or "").strip().lower()
    if normalized_source in ("testsrc", "synthetic", "dummy"):
        return SyntheticCapture(width, height)

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


def _mps_available() -> bool:
    try:
        if torch is None:
            return False
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None:
            return False
        return bool(mps_backend.is_available() and mps_backend.is_built())
    except Exception:
        return False


def _is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine().lower() in ("arm64", "aarch64")


def _auto_device() -> str:
    if _cuda_available():
        return "0"
    if _mps_available():
        return "mps"
    return "cpu"


def resolve_device(device_arg: str) -> str:
    """Return an Ultralytics-compatible device value.

    - "auto" -> "0" if CUDA available, else "mps" if Apple Metal is available, else "cpu"
    - "cuda" / "cuda:0" -> "0"
    - "mps" -> "mps"
    - otherwise passthrough ("cpu", "0", "1", ...)
    """
    d = (device_arg or "").strip().lower()
    if d in ("auto", ""):
        return _auto_device()
    if d.startswith("cuda"):
        return "0"
    if d == "mps":
        return "mps"
    return device_arg


def _device_status_summary(device: str) -> str:
    return (
        f"resolved={device} "
        f"(cuda_available={_cuda_available()}, "
        f"mps_available={_mps_available()}, "
        f"apple_silicon={_is_apple_silicon()})"
    )


def _system_memory_bytes() -> int:
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        phys_pages = int(os.sysconf("SC_PHYS_PAGES"))
        if page_size > 0 and phys_pages > 0:
            return page_size * phys_pages
    except Exception:
        return 0
    return 0


def _system_memory_gb() -> float:
    total = _system_memory_bytes()
    if total <= 0:
        return 0.0
    return float(total) / float(1024 ** 3)


def _cuda_total_memory_gb(device: Optional[str] = None) -> float:
    try:
        if not _cuda_available():
            return 0.0
        index = 0
        if isinstance(device, str) and device.strip().isdigit():
            index = int(device.strip())
        props = torch.cuda.get_device_properties(index)
        total = float(getattr(props, "total_memory", 0.0) or 0.0)
        if total <= 0:
            return 0.0
        return total / float(1024 ** 3)
    except Exception:
        return 0.0


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


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name, "") or "").strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _parse_allowed_origins(raw: str) -> Tuple[str, ...]:
    out: List[str] = []
    seen = set()
    for item in (raw or "").split(","):
        origin = item.strip().rstrip("/")
        if not origin:
            continue
        if origin == "*":
            raise SystemExit("WEB_ALLOWED_ORIGINS / --web-allow-origin must list explicit origins, not '*'.")
        parsed = urlparse(origin)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise SystemExit(f"Invalid web origin: {origin!r}. Use values like http://localhost:3000")
        normalized = f"{parsed.scheme}://{parsed.netloc}"
        if normalized not in seen:
            seen.add(normalized)
            out.append(normalized)
    return tuple(out)


def _has_cli_opt(argv: List[str], name: str) -> bool:
    for item in argv:
        if item == name or item.startswith(name + "="):
            return True
    return False


STREAM_QUALITY_PRESETS: Dict[str, Dict[str, int]] = {
    "low": {"width": 640, "height": 360, "jpeg_quality": 75},
    "medium": {"width": 960, "height": 540, "jpeg_quality": 82},
    "high": {"width": 1280, "height": 720, "jpeg_quality": 88},
    "ultra": {"width": 1920, "height": 1080, "jpeg_quality": 92},
}


@dataclass(frozen=True)
class RuntimeTuningProfile:
    name: str
    stream_quality: str
    imgsz: int
    webrtc_bitrate_kbps: int
    pose_every: int
    note: str


def _pick_runtime_tuning_profile(device: str, mode: str) -> RuntimeTuningProfile:
    cpu_count = int(os.cpu_count() or 0)
    mem_gb = _system_memory_gb()

    if (mode or "").strip().lower() == "max":
        if device == "cpu":
            return RuntimeTuningProfile(
                name="cpu-max",
                stream_quality="ultra",
                imgsz=1152,
                webrtc_bitrate_kbps=6500,
                pose_every=2,
                note=f"forced max mode on CPU ({cpu_count} logical cores, {mem_gb:.1f} GiB RAM)",
            )
        if device == "mps":
            return RuntimeTuningProfile(
                name="apple-max",
                stream_quality="ultra",
                imgsz=1280,
                webrtc_bitrate_kbps=9000,
                pose_every=2,
                note=f"forced max mode on Apple Silicon ({cpu_count} logical cores, {mem_gb:.1f} GiB unified memory)",
            )
        return RuntimeTuningProfile(
            name="cuda-max",
            stream_quality="ultra",
            imgsz=1280,
            webrtc_bitrate_kbps=9000,
            pose_every=2,
            note=f"forced max mode on accelerated device ({_cuda_total_memory_gb(device):.1f} GiB VRAM)",
        )

    if device == "mps":
        if mem_gb >= 32 or cpu_count >= 10:
            return RuntimeTuningProfile(
                name="apple-performance",
                stream_quality="ultra",
                imgsz=1280,
                webrtc_bitrate_kbps=9000,
                pose_every=2,
                note=f"Apple Silicon performance tier ({cpu_count} logical cores, {mem_gb:.1f} GiB unified memory)",
            )
        if mem_gb >= 16:
            return RuntimeTuningProfile(
                name="apple-balanced",
                stream_quality="ultra",
                imgsz=1152,
                webrtc_bitrate_kbps=7500,
                pose_every=2,
                note=f"Apple Silicon balanced tier ({cpu_count} logical cores, {mem_gb:.1f} GiB unified memory)",
            )
        return RuntimeTuningProfile(
            name="apple-efficient",
            stream_quality="high",
            imgsz=960,
            webrtc_bitrate_kbps=5500,
            pose_every=3,
            note=f"Apple Silicon efficiency tier ({cpu_count} logical cores, {mem_gb:.1f} GiB unified memory)",
        )

    if device != "cpu":
        vram_gb = _cuda_total_memory_gb(device)
        if vram_gb >= 16:
            return RuntimeTuningProfile(
                name="cuda-performance",
                stream_quality="ultra",
                imgsz=1280,
                webrtc_bitrate_kbps=9000,
                pose_every=2,
                note=f"CUDA high-memory tier ({vram_gb:.1f} GiB VRAM)",
            )
        if vram_gb >= 8:
            return RuntimeTuningProfile(
                name="cuda-balanced",
                stream_quality="ultra",
                imgsz=1152,
                webrtc_bitrate_kbps=7500,
                pose_every=2,
                note=f"CUDA balanced tier ({vram_gb:.1f} GiB VRAM)",
            )
        return RuntimeTuningProfile(
            name="cuda-efficient",
            stream_quality="high",
            imgsz=960,
            webrtc_bitrate_kbps=5500,
            pose_every=3,
            note=f"CUDA entry tier ({vram_gb:.1f} GiB VRAM)",
        )

    if cpu_count >= 12 and mem_gb >= 24:
        return RuntimeTuningProfile(
            name="cpu-performance",
            stream_quality="high",
            imgsz=960,
            webrtc_bitrate_kbps=5000,
            pose_every=2,
            note=f"strong CPU tier ({cpu_count} logical cores, {mem_gb:.1f} GiB RAM)",
        )
    if cpu_count >= 8 and mem_gb >= 16:
        return RuntimeTuningProfile(
            name="cpu-balanced",
            stream_quality="high",
            imgsz=896,
            webrtc_bitrate_kbps=4000,
            pose_every=3,
            note=f"balanced CPU tier ({cpu_count} logical cores, {mem_gb:.1f} GiB RAM)",
        )
    return RuntimeTuningProfile(
        name="cpu-efficient",
        stream_quality="medium",
        imgsz=832,
        webrtc_bitrate_kbps=3000,
        pose_every=4,
        note=f"efficient CPU tier ({cpu_count} logical cores, {mem_gb:.1f} GiB RAM)",
    )


def _apply_runtime_tuning(
    args: argparse.Namespace,
    argv: List[str],
    device: str,
) -> Optional[RuntimeTuningProfile]:
    mode = (getattr(args, "performance_profile", "auto") or "auto").strip().lower()
    if mode == "off":
        return None

    profile = _pick_runtime_tuning_profile(device, mode)

    if not _has_cli_opt(argv, "--stream-quality") and (getattr(args, "stream_quality", "auto") or "auto").strip().lower() in ("", "auto"):
        args.stream_quality = profile.stream_quality
    if not _has_cli_opt(argv, "--imgsz"):
        args.imgsz = int(profile.imgsz)
    if not _has_cli_opt(argv, "--pose-every"):
        args.pose_every = int(profile.pose_every)
    if int(getattr(args, "webrtc_bitrate", -1)) < 0:
        args.webrtc_bitrate = int(profile.webrtc_bitrate_kbps)

    return profile


def _resolve_stream_quality_name(raw: str) -> str:
    name = (raw or "high").strip().lower()
    if name in ("", "auto"):
        return "high"
    if name not in STREAM_QUALITY_PRESETS:
        valid = ", ".join(["auto"] + sorted(STREAM_QUALITY_PRESETS.keys()))
        raise SystemExit(f"Invalid --stream-quality {raw!r}. Valid: {valid}")
    return name


def _apply_stream_quality_profile(args: argparse.Namespace, argv: List[str]) -> str:
    profile_name = _resolve_stream_quality_name(getattr(args, "stream_quality", "high"))
    profile = STREAM_QUALITY_PRESETS[profile_name]

    width_explicit = _has_cli_opt(argv, "--width")
    height_explicit = _has_cli_opt(argv, "--height")
    base_w = int(profile["width"])
    base_h = int(profile["height"])

    if not width_explicit and not height_explicit:
        args.width = base_w
        args.height = base_h
    elif width_explicit and not height_explicit and int(args.width) > 0:
        args.height = max(1, int(round(int(args.width) * base_h / float(base_w))))
    elif height_explicit and not width_explicit and int(args.height) > 0:
        args.width = max(1, int(round(int(args.height) * base_w / float(base_h))))

    if not _has_cli_opt(argv, "--jpeg-quality"):
        args.jpeg_quality = int(profile["jpeg_quality"])

    args.stream_quality = profile_name
    return profile_name


# ---------------------------------------------------------------------------
#  Module-level constants for detection thresholds
# ---------------------------------------------------------------------------

# Speed thresholds (pixels/second) for classifying person movement
WALK_PPS = 60.0     # Speed above which a person is considered "walking"
RUN_PPS = 160.0     # Speed above which a person is considered "running"
STILL_PPS = 25.0    # Speed below which a person is considered "still"

# Fallback sitting heuristic (used when pose keypoints are unavailable)
SIT_AR_MAX = 1.45       # Max bounding-box aspect ratio (h/w) for sitting
SIT_H_FRAC_MAX = 0.55   # Max bbox height as fraction of frame height for sitting

# Minimum IoU to match a pose detection to a tracked person
POSE_IOU_THRESHOLD = 0.30

# Regex for normalizing web command names
_CMD_WHITESPACE_RE = re.compile(r"[\s\-]+")
_CMD_UNDERSCORES_RE = re.compile(r"_+")

_log = logging.getLogger("sentinelCam.worker")


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
        help='Video source: webcam index (e.g. "0"), testsrc, or URL/file (e.g. http://.../stream.mjpg or /dev/video42)',
    )
    ap.add_argument("--cam", type=int, default=0, help="Webcam index (used when --source not set)")

    # Capture resolution (display + base processing)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)

    # Performance / inference controls
    ap.add_argument("--max-fps", type=float, default=10.0)
    ap.add_argument(
        "--imgsz",
        type=int,
        default=832,
        help="Inference image size (bigger -> better small objects). Auto-tuning may raise this on stronger devices.",
    )
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
        help="Ultralytics device: 'auto', 'cpu', 'mps' or GPU index like '0'",
    )
    ap.add_argument(
        "--performance-profile",
        type=str,
        default=(os.environ.get("DEFAULT_PERFORMANCE_PROFILE", "auto") or "auto").strip().lower(),
        choices=["auto", "max", "off"],
        help="Runtime auto-tuning: auto=hardware-aware defaults, max=aggressive quality, off=leave classic defaults unchanged.",
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
    ap.add_argument(
        "--pose-every",
        type=int,
        default=3,
        help="Run pose model every N frames (if --use-pose). Auto-tuning may lower this on stronger devices.",
    )
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
        default="yolov8n,yolov8s,yolov8m,yolov8l,yolov8x,yolo26x",
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

    ap.add_argument("--host", type=str, default="127.0.0.1", help="Web server bind host (when --web)")
    ap.add_argument("--port", type=int, default=8080, help="Web server port (when --web)")
    ap.add_argument(
        "--web-auth-token",
        type=str,
        default=(os.environ.get("WEB_AUTH_TOKEN", "") or "").strip(),
        help="Optional bearer token for stream/API access. Also read from WEB_AUTH_TOKEN.",
    )
    ap.add_argument(
        "--web-allow-origin",
        type=str,
        default=(os.environ.get("WEB_ALLOWED_ORIGINS", "") or "").strip(),
        help="Comma-separated browser origins allowed via CORS. Also read from WEB_ALLOWED_ORIGINS.",
    )
    ap.add_argument(
        "--web-max-cmd-bytes",
        type=int,
        default=_env_int("WEB_MAX_CMD_BYTES", 8192),
        help="Maximum allowed size of POST /api/cmd bodies in bytes.",
    )
    ap.add_argument(
        "--stream",
        type=str,
        default=(os.environ.get("DEFAULT_STREAM_MODE", "auto") or "auto").strip().lower(),
        choices=["auto", "mjpeg", "webrtc"],
        help="Streaming mode for --web. Default is DEFAULT_STREAM_MODE or auto. auto=WebRTC with MJPEG fallback if available, else MJPEG.",
    )
    ap.add_argument(
        "--stream-quality",
        type=str,
        default=(os.environ.get("DEFAULT_STREAM_QUALITY", "auto") or "auto").strip().lower(),
        choices=["auto", "low", "medium", "high", "ultra"],
        help="Capture + MJPEG quality preset. auto picks a hardware-aware default unless you override width/height explicitly.",
    )
    ap.add_argument(
        "--jpeg-quality",
        type=int,
        default=_env_int("DEFAULT_JPEG_QUALITY", 88),
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
        "--webrtc-bitrate",
        type=int,
        default=_env_int("DEFAULT_WEBRTC_BITRATE_KBPS", -1),
        help="Target WebRTC video bitrate in kbps (-1 = hardware-aware auto, 0 = aiortc default bitrate control).",
    )
    ap.add_argument(
        "--webrtc-port-min",
        type=int,
        default=_env_int("DEFAULT_WEBRTC_PORT_MIN", 0),
        help="Minimum UDP port for WebRTC ICE candidates (0 = OS default).",
    )
    ap.add_argument(
        "--webrtc-port-max",
        type=int,
        default=_env_int("DEFAULT_WEBRTC_PORT_MAX", 0),
        help="Maximum UDP port for WebRTC ICE candidates (0 = OS default).",
    )
    ap.add_argument(
        "--webrtc-gpu",
        type=int,
        default=_env_int("DEFAULT_WEBRTC_GPU", 1),
        choices=[0, 1],
        help="Use GPU H.264 encoder if available (1=auto-detect, 0=force CPU).",
    )
    ap.add_argument(
        "--webrtc-frame-sharing",
        type=int,
        default=_env_int("DEFAULT_WEBRTC_FRAME_SHARING", 1),
        choices=[0, 1],
        help="Encode once, share to all clients (1=on, 0=off).",
    )


    raw_argv = sys.argv[1:]
    args = ap.parse_args()
    device = resolve_device(args.device)
    tuning_profile = _apply_runtime_tuning(args, raw_argv, device)
    stream_quality_name = _apply_stream_quality_profile(args, raw_argv)

    if args.help_web:
        print("sentinelCam web options")
        print("  --web/--no-web --stream webrtc|mjpeg|auto --host HOST --port PORT")
        print("  Default bind host: 127.0.0.1 (localhost only)")
        print("  Default stream mode: auto (WebRTC preferred, MJPEG fallback)")
        print("  Runtime tuning: --performance-profile auto|max|off")
        print("  Default stream quality: hardware-aware auto (typically high or ultra) unless overridden")
        print("  Security: --web-auth-token TOKEN --web-allow-origin http://host:port[,http://host2:port]")
        print("  Optional debug: --window (show OpenCV preview + hotkeys)")
        print("  Worker-side WebRTC signaling endpoint: POST /api/webrtc/offer")
        print("  MJPEG fallback endpoint: GET /stream.mjpg")
        print("  WebRTC: --webrtc-codec auto|h264|vp8|vp9|av1 --webrtc-bitrate KBPS (-1=hardware auto)")
        print("  WebRTC port range: --webrtc-port-min PORT --webrtc-port-max PORT")
        print("  WebRTC GPU: --webrtc-gpu 0|1  (1=auto-detect GPU encoder, 0=force CPU)")
        print("  Frame sharing: --webrtc-frame-sharing 0|1  (1=encode once for all clients)")
        print("  Stream quality: --stream-quality auto|low|medium|high|ultra")
        print("  MJPEG:  --jpeg-quality 10-95")
        print("  Capture: --width W --height H --source N|URL")
        return

    if args.width <= 0 or args.height <= 0:
        raise SystemExit("--width and --height must be greater than 0.")
    if int(args.imgsz) <= 0:
        raise SystemExit("--imgsz must be greater than 0.")
    if int(args.pose_every) <= 0:
        raise SystemExit("--pose-every must be greater than 0.")
    if not (10 <= int(args.jpeg_quality) <= 95):
        raise SystemExit("--jpeg-quality must be between 10 and 95.")
    if int(args.webrtc_bitrate) < -1:
        raise SystemExit("--webrtc-bitrate must be >= -1.")
    if int(args.webrtc_port_min) < 0 or int(args.webrtc_port_max) < 0:
        raise SystemExit("--webrtc-port-min and --webrtc-port-max must be >= 0.")
    if (int(args.webrtc_port_min) > 0) != (int(args.webrtc_port_max) > 0):
        raise SystemExit("--webrtc-port-min and --webrtc-port-max must both be set or both be 0.")
    if int(args.webrtc_port_min) > 0 and int(args.webrtc_port_min) > int(args.webrtc_port_max):
        raise SystemExit("--webrtc-port-min must be <= --webrtc-port-max.")

    allowed_web_origins = _parse_allowed_origins(args.web_allow_origin)
    web_auth_token = (args.web_auth_token or "").strip()
    if args.web_max_cmd_bytes <= 0:
        raise SystemExit("--web-max-cmd-bytes must be greater than 0.")

    bind_host = (args.host or "").strip().lower()
    public_bind = bind_host not in ("127.0.0.1", "localhost", "::1", "")
    if public_bind and not web_auth_token:
        _log.warning(
            "Web stream/API is bound to a non-local interface without --web-auth-token. "
            "Anyone who can reach the port can watch the stream and send commands."
        )
    if public_bind and not allowed_web_origins:
        _log.info(
            "Browser CORS is disabled by default. Use a reverse proxy for the HTML server, "
            "or explicitly allow trusted origins with --web-allow-origin."
        )
    if allowed_web_origins and not web_auth_token:
        _log.warning("Cross-origin browser access is enabled without --web-auth-token.")
    if web_auth_token and len(web_auth_token) < 16:
        if public_bind:
            raise SystemExit(
                "ERROR: --web-auth-token must be at least 16 characters when binding to a public interface. "
                "Use a longer token or bind to 127.0.0.1."
            )
        _log.warning("--web-auth-token is short. Use at least 16 random characters.")

    requested_stream_mode = (args.stream or "auto").strip().lower()
    if bool(getattr(args, "web", False)) and requested_stream_mode == "webrtc" and run_webrtc_server is None:
        raise SystemExit(
            f"WebRTC support is not available ({webrtc_import_error}). Install aiohttp, aiortc and av, or run with --stream mjpeg."
        )
    if bool(getattr(args, "web", False)) and requested_stream_mode == "auto" and run_webrtc_server is None:
        _log.info("WebRTC dependencies are not installed; --stream auto will fall back to MJPEG.")

    presets = build_presets()
    if args.list_presets:
        print_presets(presets)
        return
    if args.vcam_notes:
        print_vcam_notes()
        return

    setup_quiet_warnings(args.quiet_warnings)

    _configure_ultralytics_runtime_dirs()

    _log.info("Device: %s", _device_status_summary(device))
    if tuning_profile is not None:
        _log.info(
            "Runtime tuning: %s  stream=%s  capture=%dx%d  imgsz=%d  webrtc=%d kbps  pose_every=%d  [%s]",
            tuning_profile.name,
            stream_quality_name,
            int(args.width),
            int(args.height),
            int(args.imgsz),
            int(args.webrtc_bitrate),
            int(args.pose_every),
            tuning_profile.note,
        )

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
        cap.release()
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
        "busy": False,
        "busy_text": None,
        "last_error": None,
        "last_error_ts": None,
        "worker_alive": True,
        "ts": None,
    }

    def _update_state(**kw):
        try:
            with state_lock:
                web_state.update(kw)
        except Exception:
            logging.getLogger("sentinelCam.worker").exception("Failed to update web state")

    def _get_state() -> Dict[str, object]:
        with state_lock:
            return dict(web_state)

    def _normalize_web_cmd_name(raw_cmd: object) -> str:
        cmd = str(raw_cmd or "").strip().lower()
        if not cmd:
            return ""
        cmd = _CMD_WHITESPACE_RE.sub("_", cmd)
        cmd = _CMD_UNDERSCORES_RE.sub("_", cmd).strip("_")

        aliases = {
            "next_model": "next",
            "model_next": "next",
            "cycle_next": "next",
            "nextmodel": "next",
            "prev_model": "prev",
            "previous_model": "prev",
            "model_prev": "prev",
            "model_previous": "prev",
            "cycle_prev": "prev",
            "prevmodel": "prev",
            "previousmodel": "prev",
            "pose_toggle": "toggle_pose",
            "togglepose": "toggle_pose",
            "overlay_toggle": "toggle_overlay",
            "toggleoverlay": "toggle_overlay",
            "inference_toggle": "toggle_inference",
            "toggleinference": "toggle_inference",
            "model_toggle": "toggle_inference",
            "togglemodel": "toggle_inference",
        }
        return aliases.get(cmd, cmd)

    def _extract_web_cmd_and_seq(payload) -> Tuple[str, int]:
        cmd = ""
        seq = 0
        try:
            if isinstance(payload, dict):
                for seq_key in ("seq", "request_id", "id"):
                    if seq_key in payload:
                        try:
                            seq = int(payload.get(seq_key) or 0)
                            break
                        except Exception:
                            pass

                for cmd_key in ("cmd", "command", "action", "event", "name", "type"):
                    value = payload.get(cmd_key)
                    if value not in (None, ""):
                        cmd = str(value).strip().lower()
                        break

                if not cmd:
                    bool_fields = (
                        ("pose_enabled", "pose_on", "pose_off"),
                        ("pose", "pose_on", "pose_off"),
                        ("overlay_enabled", "overlay_on", "overlay_off"),
                        ("overlay", "overlay_on", "overlay_off"),
                        ("inference_enabled", "inference_on", "inference_off"),
                        ("inference", "inference_on", "inference_off"),
                    )
                    for field, on_cmd, off_cmd in bool_fields:
                        if field in payload and isinstance(payload.get(field), bool):
                            cmd = on_cmd if payload.get(field) else off_cmd
                            break

                if not cmd:
                    for key, value in payload.items():
                        if isinstance(value, bool) and value:
                            cmd = str(key).strip().lower()
                            break

                if cmd and isinstance(payload.get("value"), bool):
                    value = bool(payload.get("value"))
                    normalized = _normalize_web_cmd_name(cmd)
                    if normalized in ("pose", "toggle_pose"):
                        cmd = "pose_on" if value else "pose_off"
                    elif normalized in ("overlay", "toggle_overlay"):
                        cmd = "overlay_on" if value else "overlay_off"
                    elif normalized in ("inference", "toggle_inference", "model"):
                        cmd = "inference_on" if value else "inference_off"
            else:
                cmd = str(payload).strip().lower()
        except Exception:
            cmd = str(payload).strip().lower()
            seq = 0

        return _normalize_web_cmd_name(cmd), seq

    def _send_cmd(payload) -> None:
        """Accept commands from web server.

        Payload may be a string (legacy) or a dict: {"cmd": "...", "seq": N}.
        seq is used by the web UI to confirm that a command was actually applied.
        """
        nonlocal cmd_seq_counter
        cmd, seq = _extract_web_cmd_and_seq(payload)

        if not cmd:
            return

        if seq <= 0:
            cmd_seq_counter += 1
            seq = cmd_seq_counter

        if cmd in ("stop", "quit", "exit", "q"):
            stop_event.set()

        cmd_q.put((seq, cmd))



    if web_enabled:
        hub = FrameHub(jpeg_quality=args.jpeg_quality)
        raw_frame_hub = FrameHub(jpeg_quality=args.jpeg_quality)
    else:
        raw_frame_hub = None

    if window_enabled:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, args.width, args.height)

    cycle_idx = 0
    if active_preset_name in filtered_cycle:
        cycle_idx = filtered_cycle.index(active_preset_name)

    def _switch_to(idx: int):
        nonlocal cycle_idx, active_preset_name, det_model, pose_model, names
        nonlocal det_weights, pose_weights
        target_idx = idx % len(filtered_cycle)
        target = filtered_cycle[target_idx]
        if target == active_preset_name:
            return

        # Reload models first. Only switch the active state if loading succeeded.
        new_active_preset_name, new_det_model, new_pose_model, new_det_weights, new_pose_weights = load_models(target)

        cycle_idx = target_idx
        active_preset_name = new_active_preset_name
        det_model = new_det_model
        pose_model = new_pose_model
        det_weights = new_det_weights
        pose_weights = new_pose_weights
        names = det_model.names

        # Reset tracking state so IDs do not carry over across model switches.
        _clear_runtime_tracking()

        _update_state(
            preset=active_preset_name,
            det=os.path.basename(str(det_weights)) if det_weights else None,
            pose=os.path.basename(str(pose_weights)) if pose_weights else None,
            pose_enabled=pose_enabled,
            overlay_enabled=overlay_enabled,
            inference_enabled=inference_enabled,
            last_error=None,
            last_error_ts=None,
            worker_alive=True,
        )

        _log.info("Switched preset -> %s (det=%s, pose=%s)", active_preset_name, det_weights, pose_weights)


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
            except (Exception, SystemExit) as e:
                _log.error("Could not enable pose: %s: %s", type(e).__name__, e)
                pose_enabled = False

        _update_state(
            pose_enabled=pose_enabled,
            pose=os.path.basename(str(pose_weights)) if pose_weights else None,
        )
        _log.info("Pose -> %s", "on" if pose_enabled else "off")


    def _toggle_pose():
        _set_pose(not pose_enabled)


    def _clear_runtime_tracking() -> None:
        tracks.clear()
        sit_status.clear()
        face_point.clear()
        pose_cache.clear()


    def _set_overlay(enable: bool) -> None:
        nonlocal overlay_enabled, _saved_overlay_enabled
        _saved_overlay_enabled = bool(enable)
        overlay_enabled = bool(enable) if inference_enabled else False
        _update_state(overlay_enabled=overlay_enabled)
        _log.info("Overlay -> %s", "on" if overlay_enabled else "off")


    def _set_inference(enable: bool) -> None:
        nonlocal pose_enabled, overlay_enabled, inference_enabled
        nonlocal _saved_pose_enabled, _saved_overlay_enabled
        want = bool(enable)
        if want == inference_enabled:
            return

        if want:
            inference_enabled = True
            overlay_enabled = bool(_saved_overlay_enabled)
            _update_state(inference_enabled=True, overlay_enabled=overlay_enabled)
            _log.info("Inference -> on")
            if _saved_pose_enabled and not pose_enabled:
                _set_pose(True)
            return

        _saved_pose_enabled = pose_enabled
        _saved_overlay_enabled = overlay_enabled
        inference_enabled = False
        overlay_enabled = False
        if pose_enabled:
            _set_pose(False)
        _clear_runtime_tracking()
        _update_state(inference_enabled=False, overlay_enabled=False, pose_enabled=pose_enabled)
        _log.info("Inference -> off (stream-only)")



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
                            cmd = _normalize_web_cmd_name(item[1])
                        elif isinstance(item, dict):
                            cmd, seq = _extract_web_cmd_and_seq(item)
                        else:
                            cmd = _normalize_web_cmd_name(item)
                            seq = 0
                    except Exception:
                        cmd = _normalize_web_cmd_name(item)
                        seq = 0

                    if not cmd:
                        continue

                    applied = False

                    if cmd in ("stop", "quit", "exit", "q"):
                        stop_event.set()
                        applied = True

                    elif cmd in ("next", "m") and len(filtered_cycle) > 1:
                        try:
                            _update_state(busy=True, busy_text="Switching to next model...", last_error=None, worker_alive=True)
                            _switch_to(cycle_idx + 1)
                            applied = True
                        except (Exception, SystemExit) as e:
                            _update_state(
                                busy=False,
                                busy_text=None,
                                last_error=f"Model switch failed ({type(e).__name__}): {e}",
                                last_error_ts=time.time(),
                                worker_alive=True,
                            )
                            _log.error("Model switch failed (next): %s", e)

                    elif cmd in ("prev", "previous", "n") and len(filtered_cycle) > 1:
                        try:
                            _update_state(busy=True, busy_text="Switching to previous model...", last_error=None, worker_alive=True)
                            _switch_to(cycle_idx - 1)
                            applied = True
                        except (Exception, SystemExit) as e:
                            _update_state(
                                busy=False,
                                busy_text=None,
                                last_error=f"Model switch failed ({type(e).__name__}): {e}",
                                last_error_ts=time.time(),
                                worker_alive=True,
                            )
                            _log.error("Model switch failed (prev): %s", e)

                    elif cmd in ("toggle_pose", "pose", "p"):
                        # Pose changes only make sense while inference is enabled.
                        if inference_enabled:
                            _toggle_pose()
                        else:
                            _saved_pose_enabled = not pose_enabled
                        applied = True

                    elif cmd == "pose_on":
                        if inference_enabled:
                            _set_pose(True)
                        else:
                            _saved_pose_enabled = True
                        applied = True

                    elif cmd == "pose_off":
                        _saved_pose_enabled = False
                        if pose_enabled:
                            _set_pose(False)
                        applied = True

                    elif cmd in ("toggle_overlay", "overlay", "o"):
                        if inference_enabled:
                            _set_overlay(not overlay_enabled)
                        else:
                            _set_overlay(False)
                        applied = True

                    elif cmd == "overlay_on":
                        _set_overlay(True)
                        applied = True

                    elif cmd == "overlay_off":
                        _set_overlay(False)
                        applied = True

                    elif cmd in ("toggle_inference", "inference", "model", "i"):
                        _set_inference(not inference_enabled)
                        applied = True

                    elif cmd == "inference_on":
                        _set_inference(True)
                        applied = True

                    elif cmd == "inference_off":
                        _set_inference(False)
                        applied = True

                    if applied:
                        _update_state(
                            cmd_seq_applied=int(seq or 0),
                            cmd_last=cmd,
                            overlay_enabled=overlay_enabled,
                            inference_enabled=inference_enabled,
                            pose_enabled=pose_enabled,
                            busy=False,
                            busy_text=None,
                            last_error=None,
                            worker_alive=True,
                        )
                        if stop_event.is_set():
                            break

                if stop_event.is_set():
                    break

                loop_start = time.time()
                ret, frame = cap.read()
                if not ret or frame is None:
                    _update_state(
                        busy=False,
                        busy_text=None,
                        last_error="Camera read failed; retrying...",
                        last_error_ts=time.time(),
                        worker_alive=True,
                    )
                    time.sleep(0.25)
                    continue

                # Burn in a capture timestamp immediately after grabbing the frame,
                # so it becomes part of the raw source before any YOLO processing.
                capture_stamp = time.strftime("%Y-%m-%d %H:%M:%S")
                draw_top_right_label(frame, capture_stamp, y=25, pad=10)

                # Save raw frame (with timestamp, without YOLO annotations) for /frame-raw.jpg
                if raw_frame_hub is not None:
                    raw_frame_hub.update(frame)

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
                        worker_alive=True,
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

                            if best_j >= 0 and best_i >= POSE_IOU_THRESHOLD:
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

                # Prune stale track state every 30 frames to prevent unbounded growth
                if frame_count % 30 == 0 and len(ids) > 0:
                    active_ids = set(int(t) for t in ids)
                    for d in (tracks, sit_status, face_point, pose_cache):
                        stale = [k for k in d if k not in active_ids]
                        for k in stale:
                            del d[k]

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
                            _set_overlay(not overlay_enabled)
                    if k == ord("i"):
                        _set_inference(not inference_enabled)
        except (Exception, SystemExit) as e:
            _update_state(
                busy=False,
                busy_text=None,
                last_error=f"Worker crashed ({type(e).__name__}): {e}",
                last_error_ts=time.time(),
                worker_alive=False,
            )
            _log.exception("Worker crashed: %s", e)
        finally:
            _update_state(busy=False, busy_text=None, worker_alive=False)


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
            elif display_host in ("127.0.0.1", "::1"):
                display_host = "localhost"
            requested_mode = (args.stream or "auto").lower().strip()
            mode = requested_mode
            if mode == "auto":
                mode = "webrtc" if run_webrtc_server is not None else "mjpeg"
            security = SecurityConfig(
                auth_token=web_auth_token,
                allowed_origins=allowed_web_origins,
                max_cmd_bytes=int(args.web_max_cmd_bytes),
            )
            try:
                _log.info(
                    "Stream quality: %s  capture=%dx%d  jpeg=%d",
                    stream_quality_name, int(args.width), int(args.height), int(args.jpeg_quality),
                )
                if mode == "webrtc":
                    if run_webrtc_server is None:
                        raise SystemExit(
                            f"WebRTC support is not available ({webrtc_import_error}). Install aiohttp, aiortc and av."
                        )
                    _log.info("Stream (WebRTC signaling): http://%s:%s/api/webrtc/offer", display_host, args.port)
                    _log.info("Stream (MJPEG fallback): http://%s:%s/stream.mjpg", display_host, args.port)
                    _log.info("WebRTC test page: http://%s:%s/webrtc-test", display_host, args.port)
                    _log.info("State API: http://%s:%s/api/state", display_host, args.port)
                    if int(args.webrtc_bitrate) > 0:
                        _log.info("WebRTC target bitrate: %d kbps", int(args.webrtc_bitrate))
                    elif int(args.webrtc_bitrate) < 0:
                        _log.info("WebRTC target bitrate: hardware-aware auto")
                    else:
                        _log.info("WebRTC target bitrate: aiortc default")
                    if int(args.webrtc_port_min) > 0:
                        _log.info("WebRTC ICE port range: %d-%d", int(args.webrtc_port_min), int(args.webrtc_port_max))
                    run_webrtc_server(
                        hub,
                        host=args.host,
                        port=args.port,
                        control=ControlAPI(get_state=_get_state, command=_send_cmd),
                        stop_event=stop_event,
                        security=security,
                        codec_preference=args.webrtc_codec,
                        target_bitrate_kbps=int(args.webrtc_bitrate),
                        ice_port_min=int(args.webrtc_port_min),
                        ice_port_max=int(args.webrtc_port_max),
                        use_gpu=bool(int(args.webrtc_gpu)),
                        use_frame_sharing=bool(int(args.webrtc_frame_sharing)),
                        raw_hub=raw_frame_hub,
                    )
                else:
                    if requested_mode == "auto" and run_webrtc_server is None:
                        _log.info("Stream mode auto -> MJPEG (WebRTC dependencies not available)")
                    _log.info("Stream (MJPEG): http://%s:%s/stream.mjpg", display_host, args.port)
                    run_mjpeg_server(
                        hub,
                        host=args.host,
                        port=args.port,
                        control=ControlAPI(get_state=_get_state, command=_send_cmd),
                        stop_event=stop_event,
                        security=security,
                        raw_hub=raw_frame_hub,
                    )
            except KeyboardInterrupt:
                pass
            finally:
                stop_event.set()
                shutdown_done.set()

        if not window_enabled:
            # Headless server default
            def _processing_runner():
                while not stop_event.is_set():
                    _update_state(worker_alive=True)
                    processing_loop()
                    if stop_event.is_set():
                        break
                    # processing loop exited unexpectedly; keep the HTTP API alive and retry
                    _update_state(worker_alive=False)
                    time.sleep(1.0)

            t = threading.Thread(target=_processing_runner, name="sentinelcam-processing", daemon=True)
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
                    _log.warning("Shutdown appears stuck -> forcing exit")
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
