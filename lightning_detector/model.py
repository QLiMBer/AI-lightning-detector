from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ModelConfig:
    model_id: str = "openbmb/MiniCPM-V-4_5"
    torch_dtype: str = "bfloat16"
    attn_impl: str = "sdpa"


class MiniCPMWrapper:
    def __init__(self, cfg: Optional[ModelConfig] = None) -> None:
        import torch  # noqa: F401  # ensure torch is available
        from transformers import AutoModel, AutoTokenizer

        self.cfg = cfg or ModelConfig()
        self.model = AutoModel.from_pretrained(
            self.cfg.model_id,
            trust_remote_code=True,
            attn_implementation=self.cfg.attn_impl,
            torch_dtype=getattr(__import__("torch"), self.cfg.torch_dtype),
        ).eval().cuda()
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

