MiniCPM‑V 4.5 — Overview

Note: Developer reference for the underlying model. Typical users can skip this section; use the project README (Quick Start) and User Guide.

- Model: `openbmb/MiniCPM-V-4_5` (8B params; Qwen3‑8B + SigLIP2‑400M)
- Modalities: images, multi‑image, and video (high‑FPS/long video)
- Video: 3D‑Resampler can pack up to 6 consecutive frames into 64 tokens, enabling up to ~96× token compression for video.
- Performance (high level): strong visual QA, OCR, document parsing; hybrid fast/deep thinking modes.
- Typical GPU memory: ~18 GB for the full model (fits on RTX 3090 24 GB). Quantized options exist (int4, AWQ, GGUF) if needed.
- Supported runtimes: Transformers (Python), vLLM/SGLang for serving, GGUF for llama.cpp/Ollama; local Gradio demo available.

Implications for lightning detection in videos

- Pros: Built‑in video understanding; can reason about temporal patterns; JSON‑style constrained outputs achievable via prompting.
- Cons: It’s a general MLLM, not a specialized detector; recall/precision will depend on prompting and frame sampling. We may need simple pre/post‑processing heuristics to boost recall and structure outputs.
