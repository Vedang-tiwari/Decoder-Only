"""Inference entry point: load checkpoint -> prompt -> generate -> decode.

Usage:
    python generate_entrypoint.py --config configs/train_config.yaml \
        --checkpoint checkpoints/final.pt --prompt "The future of AI is"
"""

import argparse

import torch
import yaml

from src.dataset import build_or_load_tokens
from src.generator import Generator
from src.model import GPTLanguageModel
from src.prompt_template import format_prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/final.pt")
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["training"]["device"])

    _, tokenizer = build_or_load_tokens(cfg)

    model = GPTLanguageModel(
        vocab_size=tokenizer.vocab_size,
        block_size=cfg["model"]["block_size"],
        n_layer=cfg["model"]["n_layer"],
        n_head=cfg["model"]["n_head"],
        n_embd=cfg["model"]["n_embd"],
        dropout=0.0,
    )
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    g = cfg["generation"]
    prompt = format_prompt(args.prompt)
    prompt_ids = tokenizer.encode(prompt)

    gen = Generator(model, tokenizer, device, cfg["model"]["block_size"])
    out_ids = gen.generate(
        prompt_ids,
        max_new_tokens=g["max_new_tokens"],
        temperature=g["temperature"],
        top_k=g["top_k"],
        top_p=g["top_p"],
        eos_token_id=g["eos_token_id"],
    )

    print(tokenizer.decode(out_ids))


if __name__ == "__main__":
    main()