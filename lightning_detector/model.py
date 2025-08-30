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
            self.model = AutoModel.from_pretrained(
                self.cfg.model_id,
                trust_remote_code=True,
                attn_implementation=self.cfg.attn_impl,
                torch_dtype=getattr(torch_mod, self.cfg.torch_dtype),
            ).eval().cuda()
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
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_id, trust_remote_code=True)

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
