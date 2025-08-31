from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
"""
Video utilities. We import decord lazily inside functions to avoid potential
CUDA/driver symbol clashes when used alongside large GPU models.
"""


@dataclass
class VideoMeta:
    path: Path
    width: int
    height: int
    fps: float
    frames: int
    duration_sec: float


def list_mp4_files(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*.mp4")])


def inspect_video(path: Path) -> VideoMeta:
    from decord import VideoReader, cpu  # lazy import
    vr = VideoReader(str(path), ctx=cpu(0))
    fps = float(vr.get_avg_fps()) or 0.0
    n_frames = int(len(vr))
    duration = n_frames / fps if fps > 0 else 0.0
    frame0 = vr[0].asnumpy()
    h, w = int(frame0.shape[0]), int(frame0.shape[1])
    return VideoMeta(path=Path(path), width=w, height=h, fps=fps, frames=n_frames, duration_sec=duration)


def _uniform_indices(n_total: int, n_want: int) -> List[int]:
    if n_total <= 0 or n_want <= 0:
        return []
    if n_want >= n_total:
        return list(range(n_total))
    gap = n_total / float(n_want)
    # center sampling within each segment
    return [min(n_total - 1, int(i * gap + gap / 2)) for i in range(n_want)]


def sample_frames_uniform(
    path: Path,
    choose_fps: int = 5,
    max_frames: int = 64,
) -> Tuple[List[Image.Image], List[float]]:
    """Sample frames uniformly from the video.

    Returns tuple: (frames_as_PIL, timestamps_in_seconds)
    """
    from decord import VideoReader, cpu  # lazy import
    vr = VideoReader(str(path), ctx=cpu(0))
    native_fps = float(vr.get_avg_fps()) or 0.0
    n_total = int(len(vr))
    if native_fps <= 0 or n_total <= 0:
        return [], []

    # approximate stride based on desired FPS, then cap to max_frames uniformly
    stride = max(1, round(native_fps / max(1, choose_fps)))
    idx = list(range(0, n_total, stride))
    # If max_frames <= 0, treat as unlimited (no further cap)
    if max_frames > 0 and len(idx) > max_frames:
        idx = _uniform_indices(len(idx), max_frames)
        # map back to original indices spacing
        idx = [int(i * stride) for i in idx]

    batch = vr.get_batch(idx).asnumpy()
    frames = [Image.fromarray(v.astype("uint8")).convert("RGB") for v in batch]
    timestamps = [i / native_fps for i in idx]
    return frames, timestamps


def sample_frames_with_temporal_ids(
    path: Path,
    choose_fps: int = 5,
    max_frames_after_packing: int = 180,
    max_packing: int = 3,
    force_packing: Optional[int] = None,
) -> Tuple[List[Image.Image], List[List[int]]]:
    """Sample frames and compute temporal_ids groups for 3D-Resampler.

    Returns tuple: (frames_as_PIL, temporal_ids_groups)
    temporal_ids is a list of lists (packed groups), each int ID representing a time bin.
    """
    from scipy.spatial import cKDTree  # lazy import

    TIME_SCALE = 0.1  # seconds

    from decord import VideoReader, cpu  # lazy import
    vr = VideoReader(str(path), ctx=cpu(0))
    fps = float(vr.get_avg_fps()) or 0.0
    n_total = int(len(vr))
    if fps <= 0 or n_total <= 0:
        return [], []
    duration = n_total / fps

    # Interpret non-positive cap as "unlimited" for this heuristic
    if max_frames_after_packing <= 0:
        max_frames_after_packing = 10**9

    # decide packing and number of frames to choose
    if choose_fps * int(duration) <= max_frames_after_packing:
        packing = 1
        choose_frames = round(min(choose_fps, round(fps)) * min(max_frames_after_packing, duration))
    else:
        packing = math.ceil(duration * choose_fps / max_frames_after_packing)
        if packing <= max_packing:
            choose_frames = round(duration * choose_fps)
        else:
            choose_frames = round(max_frames_after_packing * max_packing)
            packing = max_packing

    if force_packing is not None:
        packing = min(max(1, int(force_packing)), max_packing)

    # uniform indices across the entire video
    idx = _uniform_indices(n_total, max(1, choose_frames))
    batch = vr.get_batch(idx).asnumpy()
    frames = [Image.fromarray(v.astype("uint8")).convert("RGB") for v in batch]

    frame_ts = np.asarray(idx, dtype=np.float32) / float(fps)
    scale = np.arange(0, duration, TIME_SCALE, dtype=np.float32)
    tree = cKDTree(scale[:, None])
    _, indices = tree.query(frame_ts[:, None])
    ts_ids = (scale[indices] / TIME_SCALE).astype(np.int32)

    # group into packing chunks
    temporal_ids: List[List[int]] = []
    for i in range(0, len(ts_ids), packing):
        temporal_ids.append(list(ts_ids[i : i + packing]))

    return frames, temporal_ids
