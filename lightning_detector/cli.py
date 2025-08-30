from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from .json_utils import parse_detection_json, to_pretty_json
from .model import MiniCPMWrapper, ModelConfig
from .video import (
    inspect_video,
    list_mp4_files,
    sample_frames_uniform,
    sample_frames_with_temporal_ids,
)


DEFAULT_INPUT = "videos"
DEFAULT_OUTPUT = "reports"


def build_prompt_json(spec: Dict) -> str:
    schema = {
        "video": spec.get("video", ""),
        "fps_sampled": spec.get("fps_sampled", 5),
        "method": spec.get("method", "uniform"),
        "model": "openbmb/MiniCPM-V-4_5",
        "detections": [
            {"start_sec": 12.4, "end_sec": 12.9, "confidence": 0.82, "notes": "brief rationale"}
        ],
        "non_lightning_events": [
            {"start_sec": 45.0, "end_sec": 45.4, "confidence": 0.51, "label": "camera_flash"}
        ],
    }

    instructions = (
        "You analyze video frames and identify lightning flashes.\n"
        "A lightning flash is a natural atmospheric electrical discharge.\n"
        "Exclude camera flashes, fireworks, headlights, reflections, exposure changes.\n\n"
        "Return ONLY compact JSON matching the schema below. Do not include extra text.\n"
    )
    return (
        f"{instructions}\nJSON schema example to follow (use it as a template, adjust values):\n\n"
        + json.dumps(schema, ensure_ascii=False)
    )


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_pretty_json(data) + "\n", encoding="utf-8")


def update_index(report_dir: Path) -> None:
    items: List[str] = []
    for p in sorted(report_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            dets = data.get("detections", [])
            n = len(dets) if isinstance(dets, list) else 0
            items.append(f"{p.stem}.json\t{n} detections")
        except Exception:
            items.append(f"{p.stem}.json\t(unreadable)")
    write_text(report_dir / "index.txt", "\n".join(items) + ("\n" if items else ""))


def cmd_scan(args: argparse.Namespace) -> int:
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list_mp4_files(input_dir)
    if not files:
        print(f"No .mp4 files found in {input_dir}")
        return 0

    # Preload model once to fail fast and trigger weight download
    try:
        model = MiniCPMWrapper(ModelConfig(attn_impl=args.attn, torch_dtype=args.dtype))
    except Exception as e:
        if "flash_attn_required" in str(e):
            print(
                "Model requested flash-attn. Install 'flash-attn' or try --attn sdpa.\n"
                "If sdpa still fails, flash-attn wheels may be required for your CUDA."
            )
        # still proceed; per-file handling below will write error reports
        model = None

    for vid in files:
        meta = inspect_video(vid)
        method = "uniform" if args.packing == 0 else "temporal_packing"

        if args.packing == 0:
            frames, timestamps = sample_frames_uniform(
                vid, choose_fps=args.fps, max_frames=args.max_frames
            )
            temporal_ids = None
        else:
            frames, temporal_ids = sample_frames_with_temporal_ids(
                vid,
                choose_fps=args.fps,
                max_frames_after_packing=max(args.max_frames, 1),
                max_packing=max(1, min(args.packing, 6)),
                force_packing=args.packing,
            )
            timestamps = []  # not used in packed mode

        prompt = build_prompt_json(
            {
                "video": str(vid),
                "fps_sampled": args.fps,
                "method": method,
            }
        )

        raw_text: str
        try:
            if model is None:
                # last-chance lazy init if preload failed
                model = MiniCPMWrapper(ModelConfig(attn_impl=args.attn, torch_dtype=args.dtype))
            raw_text = model.chat(
                frames=frames,
                prompt=prompt,
                max_slice_nums=args.max_slice_nums,
                enable_thinking=bool(args.thinking),
                temporal_ids=temporal_ids,
            )
        except Exception as e:
            # Emit a clearer console note on first failure
            if "flash_attn_required" in str(e):
                print(
                    "Model requested flash-attn. Install 'flash-attn' or try --attn sdpa.\n"
                    "If sdpa still fails, flash-attn wheels may be required for your CUDA."
                )
            raw_text = f"ERROR: inference_failed: {e}"

        data = parse_detection_json(raw_text)

        # attach metadata
        data.setdefault("video", str(vid))
        data.setdefault("fps_sampled", int(args.fps))
        data.setdefault("method", method)
        data.setdefault("model", "openbmb/MiniCPM-V-4_5")

        base = vid.stem
        json_path = output_dir / f"{base}.json"
        txt_path = output_dir / f"{base}.txt"
        write_json(json_path, data)
        write_text(txt_path, raw_text.strip() + "\n")

        print(f"Wrote {json_path} and {txt_path}")

    update_index(output_dir)
    print(f"Updated index: {output_dir / 'index.txt'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lightning-detector", description="Detect lightning in videos using MiniCPM-V 4.5")
    sub = p.add_subparsers(dest="cmd", required=True)

    scan = sub.add_parser("scan", help="Scan input directory of .mp4 files and write reports")
    scan.add_argument("--input", default=DEFAULT_INPUT, help=f"Input directory (default: {DEFAULT_INPUT})")
    scan.add_argument("--output", default=DEFAULT_OUTPUT, help=f"Output directory (default: {DEFAULT_OUTPUT})")
    scan.add_argument("--fps", type=int, default=5, help="Target sampling FPS (default: 5)")
    scan.add_argument(
        "--packing",
        type=int,
        default=0,
        help="Temporal packing level 0(off) or 1–6 (default: 0)",
    )
    scan.add_argument("--max-frames", type=int, default=64, help="Max frames after sampling (default: 64)")
    scan.add_argument("--max-slice-nums", type=int, default=1, help="Split hi-res frames to avoid OOM (default: 1)")
    scan.add_argument("--attn", choices=["sdpa", "flash_attention_2", "eager"], default="sdpa", help="Attention backend (default: sdpa)")
    scan.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16", help="Torch dtype (default: bfloat16)")
    scan.add_argument("--thinking", action="store_true", help="Enable deep thinking mode")
    scan.set_defaults(func=cmd_scan)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
