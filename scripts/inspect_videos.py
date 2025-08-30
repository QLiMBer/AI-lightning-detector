#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

from decord import VideoReader, cpu


def human_time(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(int(m), 60)
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def inspect_video(path: Path) -> dict:
    vr = VideoReader(str(path), ctx=cpu(0))
    fps = float(vr.get_avg_fps())
    n_frames = int(len(vr))
    duration = n_frames / fps if fps > 0 else 0.0
    # decode first frame to infer resolution
    frame0 = vr[0].asnumpy()
    h, w = frame0.shape[0], frame0.shape[1]
    return {
        "path": str(path),
        "width": w,
        "height": h,
        "fps": fps,
        "frames": n_frames,
        "duration_sec": duration,
    }


def main():
    ap = argparse.ArgumentParser(description="Inspect mp4 videos in a folder using decord")
    ap.add_argument("folder", nargs="?", default="videos", help="Input folder (default: videos)")
    args = ap.parse_args()

    root = Path(args.folder)
    if not root.exists():
        print(f"Folder not found: {root}")
        return 1

    files = [p for p in root.rglob("*.mp4")]
    if not files:
        print(f"No .mp4 files found in {root}")
        return 0

    print(f"Found {len(files)} mp4 files under {root}\n")
    for p in sorted(files):
        try:
            meta = inspect_video(p)
            print(f"- {meta['path']}")
            print(f"  resolution: {meta['width']}x{meta['height']} | fps: {meta['fps']:.3f} | frames: {meta['frames']} | duration: {human_time(meta['duration_sec'])}")
        except Exception as e:
            print(f"- {p}")
            print(f"  ERROR: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

