#!/usr/bin/env python3
from __future__ import annotations

import argparse
from PIL import Image

from lightning_detector.model import MiniCPMWrapper, ModelConfig


def main() -> int:
    ap = argparse.ArgumentParser(description="Minimal MiniCPM-V chat smoke test")
    ap.add_argument("--attn", choices=["sdpa", "flash_attention_2", "eager"], default="sdpa")
    ap.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="float16")
    args = ap.parse_args()

    # Make a simple 256x256 gray image
    img = Image.new("RGB", (256, 256), color=(128, 128, 128))

    print(f"Loading model attn={args.attn} dtype={args.dtype}…")
    model = MiniCPMWrapper(ModelConfig(attn_impl=args.attn, torch_dtype=args.dtype))

    msgs = [{"role": "user", "content": [img, "What is in this image?"]}]
    print("Running chat…")
    out = model.model.chat(msgs=msgs, tokenizer=model.tokenizer, use_image_id=False)
    print("Response (first 300 chars):")
    print(str(out)[:300])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

