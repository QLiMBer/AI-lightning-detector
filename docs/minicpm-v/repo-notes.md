Upstream Repo Notes — MiniCPM‑V 4.5

Sources parsed

- Main README (MiniCPM‑V repo): sections on MiniCPM‑V 4.5, Inference, Multi‑turn Conversation, and “Chat with Video” example showing the 3D‑Resampler usage with `temporal_ids`, and code to extract frames with `decord`.
- Cookbook README and the recipe: `inference/minicpm-v4_5_video_understanding.md` (simple frame sampling + `model.chat`), which again uses `decord` to turn the `.mp4` into frames before passing them to the model.

Key API expectations

- The chat interface consumes a message array where `content` is a list containing PIL images followed by the text query. The model does not ingest raw `.mp4` bytes directly through this API.
- For long/high‑FPS videos, upstream provides the 3D‑Resampler path, which requires parallel `temporal_ids` groupings to inform packing across consecutive frames.

Environment details from upstream

- Upstream originally cited `transformers==4.44.2`, but MiniCPM‑V remote code now imports `Qwen3Config`; use `transformers>=4.47,<5` in this repo.
- Attention impl: `sdpa` by default; `flash_attention_2` optional if installed.
- Video IO: `decord` used in official snippets. PIL (`Pillow`) for images; `SciPy`’s `cKDTree` used to build `temporal_ids` in the 3D‑Resampler example.
- Model memory: ~18 GB for full MiniCPM‑V 4.5 on GPU (quantized variants exist: int4, AWQ, GGUF).

Implications for our app

- We must decode video → frames prior to `model.chat`. `decord` is the recommended lightweight path in upstream code. We’ll follow that for consistency and performance.
- For long videos, we’ll support temporal packing with `temporal_ids` to keep coverage high while controlling VRAM/time.

Repo‑specific notes

- CLI supports `--no-preload-model` to skip upfront load and initialize lazily per video. This helps surface download/progress and avoid cache lock stalls.
- Our model loader uses `dtype=…` (new API) instead of deprecated `torch_dtype`.
