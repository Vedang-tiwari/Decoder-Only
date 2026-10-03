"""Data pipeline: raw text -> tokens -> (input, target) training pairs."""

import os
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------
# Simple tokenizer (no external dependencies, fully "from scratch")
# ---------------------------------------------------------------
class CharTokenizer:
    """Character-level tokenizer. Every unique character becomes a token."""

    def __init__(self):
        self.stoi = {}   # char -> id
        self.itos = {}   # id -> char

    def fit(self, text: str) -> "CharTokenizer":
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        return self

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)


class WordTokenizer:
    """Whitespace word-level tokenizer."""

    def __init__(self):
        self.stoi = {}
        self.itos = {}

    def fit(self, text: str) -> "WordTokenizer":
        vocab = sorted(list(set(text.split())))
        self.stoi = {w: i for i, w in enumerate(vocab)}
        self.itos = {i: w for i, w in enumerate(vocab)}
        return self

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[w] for w in text.split()]

    def decode(self, ids) -> str:
        return " ".join(self.itos[int(i)] for i in ids)


TOKENIZERS = {
    "char": CharTokenizer,
    "word": WordTokenizer,
}


def build_or_load_tokens(cfg) -> tuple[torch.Tensor, object]:
    """Load raw text, tokenize it, and cache the token IDs on disk.

    Returns:
        tokens: 1-D LongTensor of token IDs
        tokenizer: fitted tokenizer instance (also cached to disk)
    """
    data_cfg = cfg["data"]

    if data_cfg["tokenized_path"] and os.path.exists(data_cfg["tokenized_path"]):
        print(f"[dataset] loading cached tokens from {data_cfg['tokenized_path']}")
        bundle = torch.load(data_cfg["tokenized_path"])
        return bundle["tokens"], bundle["tokenizer"]

    print(f"[dataset] reading raw text from {data_cfg['raw_text_path']}")
    with open(data_cfg["raw_text_path"], "r", encoding="utf-8") as f:
        text = f.read()

    if data_cfg.get("max_tokens"):
        text = text[: data_cfg["max_tokens"]]

    tokenizer = TOKENIZERS[data_cfg["tokenizer_type"]]().fit(text)
    ids = tokenizer.encode(text)
    tokens = torch.tensor(ids, dtype=torch.long)

    os.makedirs(os.path.dirname(data_cfg["tokenized_path"]), exist_ok=True)
    torch.save({"tokens": tokens, "tokenizer": tokenizer}, data_cfg["tokenized_path"])
    print(f"[dataset] tokenized {len(tokens):,} tokens | vocab size = {tokenizer.vocab_size:,}")

    return tokens, tokenizer


# ---------------------------------------------------------------
# Dataset: shifted input / target pairs
# ---------------------------------------------------------------
class GPTDataset(Dataset):
    """Fixed-length sliding-window dataset for next-token prediction.

    For tokens  = [A, B, C, D, E, ...] and block_size = 3:
        sample 0: x = [A, B, C],  y = [B, C, D]
        sample 1: x = [B, C, D],  y = [C, D, E]
        ...
    """

    def __init__(self, tokens: torch.Tensor, block_size: int):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self) -> int:
        return max(len(self.tokens) - self.block_size - 1, 0)

    def __getitem__(self, idx: int):
        chunk = self.tokens[idx : idx + self.block_size + 1]
        x = chunk[:-1].clone()
        y = chunk[1:].clone()
        return x, y


def get_train_val_split(tokens: torch.Tensor, val_frac: float = 0.1):
    """Split a flat token stream into train / validation token streams."""
    n_val = int(len(tokens) * val_frac)
    return tokens[:-n_val], tokens[-n_val:]
