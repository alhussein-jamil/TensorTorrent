"""Hugging Face transformer beyond-VRAM workload (Qwen3-8B by default)."""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

DEFAULT_MODEL_ID = "Qwen/Qwen3-8B"
DEFAULT_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
DEFAULT_SEQ_LEN = 16
DEFAULT_BATCH = 1


@dataclass
class TransformerSpec:
    model_id: str
    revision: str
    dtype: str
    seq_len: int
    batch_size: int
    param_count: int
    param_bytes: int
    input_shapes: dict[str, list[int]]
    notes: str = ""


class CausalLMLogits(nn.Module):
    """Exportable single-step forward: logits only, no KV cache / generate loop."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        return out.logits


def _pad_batch(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    seq_len: int,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    from torch.nn.functional import pad

    cur = int(input_ids.shape[1])
    if cur > seq_len:
        return input_ids[:, :seq_len], attention_mask[:, :seq_len]
    if cur < seq_len:
        n = seq_len - cur
        input_ids = pad(input_ids, (0, n), value=pad_id)
        attention_mask = pad(attention_mask, (0, n), value=0)
    return input_ids, attention_mask


def load_causal_lm(
    *,
    model_id: str = DEFAULT_MODEL_ID,
    revision: str = DEFAULT_REVISION,
    dtype: torch.dtype = torch.bfloat16,
    seq_len: int = DEFAULT_SEQ_LEN,
    batch_size: int = DEFAULT_BATCH,
    prompt: str = "Hello TensorTorrent benchmark",
) -> tuple[CausalLMLogits, tuple[torch.Tensor, torch.Tensor], TransformerSpec, dict[str, Any]]:
    """Load HF causal LM on CPU and build fixed-shape example inputs."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=dtype,
        device_map="cpu",
        trust_remote_code=True,
    ).eval()
    load_s = time.perf_counter() - t0
    wrap = CausalLMLogits(model)
    encoded = tok([prompt] * batch_size, return_tensors="pt", padding=True)
    pad_id = int(tok.pad_token_id if tok.pad_token_id is not None else 0)
    input_ids, attention_mask = _pad_batch(
        encoded["input_ids"], encoded["attention_mask"], seq_len=seq_len, pad_id=pad_id
    )
    nparams = sum(int(p.numel()) for p in model.parameters())
    nbytes = sum(int(p.numel() * p.element_size()) for p in model.parameters())
    spec = TransformerSpec(
        model_id=model_id,
        revision=revision,
        dtype=str(dtype).replace("torch.", ""),
        seq_len=seq_len,
        batch_size=batch_size,
        param_count=nparams,
        param_bytes=nbytes,
        input_shapes={
            "input_ids": list(input_ids.shape),
            "attention_mask": list(attention_mask.shape),
        },
        notes="single forward (logits); use_cache=False; no generate()",
    )
    meta = {
        "load_s": load_s,
        "transformers_version": __import__("transformers").__version__,
        "tokenizer": type(tok).__name__,
        "pad_token_id": pad_id,
        "prompt": prompt,
    }
    return wrap, (input_ids, attention_mask), spec, meta


def release_model(wrap: CausalLMLogits | None) -> None:
    if wrap is None:
        return
    wrap.model = None  # type: ignore[assignment]
    del wrap
    gc.collect()
