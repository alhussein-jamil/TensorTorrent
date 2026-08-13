"""Static-shape greedy decode helpers (compile-friendly padded buffers)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn


class TinyCausalLM(nn.Module):
    """Small integer-in / logits-out stand-in for generate-suite smoke."""

    def __init__(self, vocab: int = 64, width: int = 32, depth: int = 2, max_len: int = 32) -> None:
        super().__init__()
        self.vocab = vocab
        self.max_len = max_len
        self.embed = nn.Embedding(vocab, width)
        self.pos = nn.Embedding(max_len, width)
        self.layers = nn.ModuleList(
            nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width)) for _ in range(depth)
        )
        self.lm_head = nn.Linear(width, vocab)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        _batch, seq = input_ids.shape
        pos = torch.arange(seq, device=input_ids.device)
        keep = attention_mask.unsqueeze(-1).to(dtype=self.lm_head.weight.dtype)
        hidden = (self.embed(input_ids) + self.pos(pos)) * keep
        for layer in self.layers:
            hidden = (hidden + layer(hidden)) * keep
        return self.lm_head(hidden)


def greedy_padded_decode(
    forward: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    new_tokens: int,
) -> torch.Tensor:
    """Greedy decode with static ``prompt + new_tokens`` buffers.

    TensorTorrent compiles static shapes, so the forward always sees ``max_len``.
    Mask bits grow one position per step. This is not Hugging Face KV-cache
    ``generate()``; it is the compile-legal analog.
    """
    if new_tokens < 0:
        raise ValueError("new_tokens must be >= 0")
    prompt = int(input_ids.shape[1])
    max_len = prompt + int(new_tokens)
    batch = int(input_ids.shape[0])
    ids = input_ids.new_zeros(batch, max_len)
    mask = attention_mask.new_zeros(batch, max_len)
    ids[:, :prompt] = input_ids
    mask[:, :prompt] = attention_mask
    for step in range(int(new_tokens)):
        logits = forward(ids, mask)
        read = prompt + step - 1
        write = prompt + step
        ids[:, write] = logits[:, read].argmax(dim=-1)
        mask[:, write] = 1
    return ids


def past_key_values_prefill(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, Any]:
    """One HF-style prefill with ``use_cache=True`` (logits, past)."""
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    return out.logits, out.past_key_values
