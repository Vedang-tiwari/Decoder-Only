"""Training entry point: wires config -> data -> model -> trainer.

Usage:
    python train_entrypoint.py --config configs/train_config.yaml
"""

import argparse
import random

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.dataset import GPTDataset, build_or_load_tokens, get_train_val_split
from src.model import GPTLanguageModel
from src.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["training"].get("seed", 1337)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # --- data ---
    tokens, tokenizer = build_or_load_tokens(cfg)
    train_tokens, val_tokens = get_train_val_split(tokens, val_frac=0.1)

    block_size = cfg["model"]["block_size"]
    train_ds = GPTDataset(train_tokens, block_size)
    val_ds = GPTDataset(val_tokens, block_size)

    batch_size = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, drop_last=True)

    # --- model ---
    vocab_size = cfg["model"].get("vocab_size") or tokenizer.vocab_size
    cfg["model"]["vocab_size"] = vocab_size
    model = GPTLanguageModel(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=cfg["model"]["n_layer"],
        n_head=cfg["model"]["n_head"],
        n_embd=cfg["model"]["n_embd"],
        dropout=cfg["model"]["dropout"],
    )
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[train] model params: {n_params:.2f}M | vocab {vocab_size} | block {block_size}")

    # --- train ---
    trainer = Trainer(model, train_loader, val_loader, cfg)
    trainer.train()


if __name__ == "__main__":
    main()
