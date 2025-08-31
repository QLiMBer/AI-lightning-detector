Deployment Options — MiniCPM‑V 4.5

Note: Developer reference for alternative runtimes/serving. Typical users can skip this section; use the project README (Quick Start) and User Guide.

MVP choice (for this project)

- In‑process Transformers (Python) on RTX 3090.
  - Pros: simplest to build/debug; no server boundary; direct access to local videos; best for a CLI tool.
  - Cons: single‑process throughput; scale‑out needs code changes.
  - Status: This is the path assumed in `docs/plan.md` and `docs/minicpm-v/setup.md`.

Alternatives (future)

- vLLM Serving (Cookbook: `deployment/vllm/minicpm-v4_5_vllm.md`)
  - Pros: high‑throughput, tensor/kv‑cache optimizations, easy concurrent clients.
  - Cons: adds a model server to manage; we still must decode videos client‑side.
- SGLang Serving (Cookbook: `deployment/sglang/MiniCPM-v4_5_sglang.md`)
  - Similar trade‑offs to vLLM with different performance characteristics.
- Ollama / llama.cpp with GGUF (Cookbook: `deployment/ollama/minicpm-v4_5_ollama.md`, `deployment/llama.cpp/minicpm-v4_5_llamacpp.md`)
  - Pros: simpler packaging on some systems; CPU/edge‑friendly.
  - Cons: Multimodal video support and APIs vary; speed/feature gaps vs full Transformers; may require specific forks.

Quantization options

- int4 / AWQ (Hugging Face: `openbmb/MiniCPM-V-4_5-int4`, `openbmb/MiniCPM-V-4_5-AWQ`)
  - Pros: reduces VRAM to ~9 GB; can increase batch/packing headroom.
  - Cons: potential quality/runtime differences; verify outputs on your data.
- GGUF (Hugging Face: `openbmb/MiniCPM-V-4_5-gguf`)
  - For llama.cpp/Ollama path.

Recommended now

- Keep Transformers local for the first version. If/when you want to process many videos concurrently or expose an API, consider vLLM.
