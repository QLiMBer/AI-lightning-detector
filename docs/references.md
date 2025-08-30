References

- MiniCPM‑V main repo: https://github.com/OpenBMB/MiniCPM-V
- Hugging Face model: https://huggingface.co/openbmb/MiniCPM-V-4_5
- README sections used:
  - Model Zoo + memory notes
  - Multi‑turn conversation (Transformers usage)
  - Chat with Video (temporal packing via 3D‑Resampler)
- Cookbook (recipes): https://github.com/OpenSQZ/MiniCPM-V-Cookbook
  - Video QA recipe: `inference/minicpm-v4_5_video_understanding.md`
  - Single/multi‑image recipes for context
  - Serving recipes: `deployment/vllm/minicpm-v4_5_vllm.md`, `deployment/sglang/MiniCPM-v4_5_sglang.md`
  - Edge/CPU recipes: `deployment/llama.cpp/minicpm-v4_5_llamacpp.md`, `deployment/ollama/minicpm-v4_5_ollama.md`
  - Quantization: `quantization/gguf/minicpm-v4_5_gguf_quantize.md`, `quantization/awq/minicpm-v4_awq_quantize.md`, `quantization/bnb/minicpm-v4_5_bnb_quantize.md`
- Optional serving:
  - vLLM: high‑throughput GPU serving
  - SGLang: high‑throughput GPU serving
  - GGUF/Ollama/llama.cpp: CPU/edge deployment

Notes captured from upstream

- Transformers 4.44.2 recommended for compatibility.
- Use `attn_implementation='sdpa'` unless FlashAttention2 is available.
- For long/high‑FPS videos, provide `temporal_ids` groups to enable 3D‑Resampler.

Note on earlier 404s

- Some direct doc paths in the main repo are not present or move over time, so attempts can yield a short "404: Not Found" file. The authoritative sources we relied on are the main README (sections cited) and the Cookbook recipes above.
