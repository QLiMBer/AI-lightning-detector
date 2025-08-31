from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ModelConfig:
    model_id: str = "openbmb/MiniCPM-V-4_5"
    torch_dtype: str = "bfloat16"
    attn_impl: str = "sdpa"  # or "flash_attention_2" if flash-attn is installed


class MiniCPMWrapper:
    def __init__(self, cfg: Optional[ModelConfig] = None) -> None:
        # Encourage model code to avoid flash-attn if not present
        os.environ.setdefault("USE_FLASH_ATTN", "0")
        os.environ.setdefault("USE_FLASH_ATTENTION", "0")
        os.environ.setdefault("FLASH_ATTENTION", "0")

        import torch  # noqa: F401  # ensure torch is available
        from transformers import AutoModel, AutoTokenizer

        self.cfg = cfg or ModelConfig()
        torch_mod = __import__("torch")
        try:
            print(
                f"[model] torch.cuda.is_available()={torch_mod.cuda.is_available()} | CUDA={getattr(torch_mod.version, 'cuda', 'n/a')} | devices={torch_mod.cuda.device_count()} | CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','<unset>')}",
                flush=True,
            )
            print("[model] Loading model…", flush=True)
            # Choose dtype with BF16 fallback if unsupported
            want_dtype = getattr(torch_mod, self.cfg.torch_dtype)
            if self.cfg.torch_dtype == "bfloat16":
                is_bf16_ok = getattr(torch_mod.cuda, "is_bf16_supported", lambda: False)()
                if not is_bf16_ok:
                    print("[model] BF16 not supported on this device; using float16.", flush=True)
                    want_dtype = torch_mod.float16

            placed = False
            offload_dir = os.path.abspath(os.path.join(os.getcwd(), ".offload"))
            try:
                # Prefer letting HF/Accelerate shard and place weights automatically
                self.model = AutoModel.from_pretrained(
                    self.cfg.model_id,
                    trust_remote_code=True,
                    attn_implementation=self.cfg.attn_impl,
                    dtype=want_dtype,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    offload_folder=offload_dir,
                ).eval()
                placed = True
                print("[model] Loaded with device_map='auto' (with possible offload)", flush=True)
            except Exception as emap:
                print(f"[model] device_map path failed: {emap}. Falling back to CPU load.", flush=True)
                self.model = AutoModel.from_pretrained(
                    self.cfg.model_id,
                    trust_remote_code=True,
                    attn_implementation=self.cfg.attn_impl,
                    dtype=want_dtype,
                ).eval()
            if not placed:
                print("[model] Moving model to CUDA…", flush=True)
                self.model = self.model.to(device="cuda", dtype=want_dtype)
                print("[model] Model is on CUDA.", flush=True)
        except Exception as e:
            msg = str(e)
            if "flash_attn" in msg or "flash-attn" in msg:
                # Re-raise with clearer guidance for the CLI layer
                raise RuntimeError(
                    "flash_attn_required: The model requested flash-attn. "
                    "Either install it (pip install flash-attn --no-build-isolation) "
                    "or rerun with attention impl 'sdpa' if supported."
                ) from e
            raise
        print("[model] Loading tokenizer…", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_id, trust_remote_code=True)
        print("[model] Tokenizer ready.", flush=True)

    def chat(
        self,
        frames: List[Any],
        prompt: str,
        max_slice_nums: int = 1,
        enable_thinking: bool = False,
        temporal_ids: Optional[List[List[int]]] = None,
    ) -> str:
        msgs = [{"role": "user", "content": frames + [prompt]}]
        res = self.model.chat(
            msgs=msgs,
            tokenizer=self.tokenizer,
            use_image_id=False,
            max_slice_nums=max_slice_nums,
            enable_thinking=enable_thinking,
            temporal_ids=temporal_ids,
        )
        # upstream returns string
        return str(res)
