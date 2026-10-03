"""Autoregressive text generation: greedy / temperature / top-k / top-p."""

import torch
import torch.nn.functional as F


class Generator:
    def __init__(self, model, tokenizer, device, block_size: int):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.block_size = block_size
        self.model.eval()

    @torch.no_grad()
    def _next_token(self, idx: torch.Tensor,
                    temperature: float, top_k: int | None, top_p: float | None) -> torch.Tensor:
        """Return the next token ID for each sequence in the batch."""
        idx_cond = idx[:, -self.block_size:]  # sliding-window context

        logits, _ = self.model(idx_cond)      # (B, T, vocab)
        logits = logits[:, -1, :]             # last position: (B, vocab)

        # greedy decoding
        if temperature is None or temperature == 0.0:
            return torch.argmax(logits, dim=-1, keepdim=True)

        logits = logits / temperature

        # top-k: keep only the k highest logits
        if top_k is not None:
            k = min(top_k, logits.size(-1))
            top_vals, _ = torch.topk(logits, k)
            logits = logits.masked_fill(logits < top_vals[:, -1:], float("-inf"))

        # top-p (nucleus): smallest token set with cumulative prob >= p
        if top_p is not None:
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            removed = cumsum - sorted_probs > top_p
            sorted_probs[removed] = 0.0
            probs = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)
            probs = probs / probs.sum(dim=-1, keepdim=True)
        else:
            probs = F.softmax(logits, dim=-1)

        return torch.multinomial(probs, num_samples=1)  # (B, 1)

    @torch.no_grad()
    def generate(self, prompt_ids: list[int], max_new_tokens: int = 100,
                 temperature: float = 1.0, top_k: int | None = None,
                 top_p: float | None = None, eos_token_id: int | None = None) -> list[int]:
        """Autoregressively generate: prompt -> model -> sample -> append -> repeat."""
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        for _ in range(max_new_tokens):
            next_id = self._next_token(idx, temperature, top_k, top_p)
            idx = torch.cat((idx, next_id), dim=1)
            if eos_token_id is not None and next_id.item() == eos_token_id:
                break

        return idx[0].tolist()
