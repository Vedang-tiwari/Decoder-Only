"""Training orchestration: loop, loss, optimizer, scheduling, checkpoints."""

import math
import os
import time

import torch
import torch.nn as nn


class Trainer:
    def __init__(self, model, train_loader, val_loader, cfg):
        self.model = model
        self.cfg = cfg
        self.device = torch.device(cfg["training"]["device"])
        self.model.to(self.device)

        t = cfg["training"]
        self.grad_clip = t["grad_clip"]
        self.eval_interval = t["eval_interval"]
        self.eval_iters = t["eval_iters"]
        self.checkpoint_interval = t["checkpoint_interval"]
        self.max_epochs = t["max_epochs"]

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=t["learning_rate"],
            weight_decay=t["weight_decay"],
        )

        total_steps = self._estimate_total_steps(train_loader)
        warmup = min(t["warmup_steps"], total_steps // 10)

        if t["lr_scheduler"] == "cosine":
            def lr_lambda(step):
                if step < warmup:
                    return step / max(1, warmup)
                progress = (step - warmup) / max(1, total_steps - warmup)
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        else:
            self.scheduler = None

        self.step = 0

    def _estimate_total_steps(self, train_loader) -> int:
        return len(train_loader) * self.max_epochs

    @torch.no_grad()
    def evaluate(self) -> tuple[float, float]:
        """Average loss & perplexity over a few validation batches."""
        self.model.eval()
        losses = []
        for i, (x, y) in enumerate(self.val_loader):
            if i >= self.eval_iters:
                break
            x, y = x.to(self.device), y.to(self.device)
            _, loss = self.model(x, targets=y)
            losses.append(loss.item())
        self.model.train()

        avg = sum(losses) / max(len(losses), 1)
        return avg, math.exp(avg)   # (loss, perplexity)

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "step": self.step,
            "config": self.cfg,
        }, path)
        print(f"[trainer] checkpoint saved -> {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if self.scheduler and ckpt.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.step = ckpt.get("step", 0)
        print(f"[trainer] resumed from {path} (step {self.step})")

    def train(self):
        if self.cfg["training"].get("resume_from"):
            self.load_checkpoint(self.cfg["training"]["resume_from"])

        self.model.train()
        best_val = float("inf")
        t0 = time.time()

        for epoch in range(self.max_epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)

                _, loss = self.model(x, targets=y)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()

                if self.grad_clip:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.step += 1

                if self.step % 50 == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    print(f"step {self.step:6d} | epoch {epoch + 1}/{self.max_epochs} "
                          f"| loss {loss.item():.4f} | lr {lr:.2e} "
                          f"| elapsed {time.time() - t0:.0f}s")

                if self.step % self.eval_interval == 0 or self.step == 1:
                    val_loss, val_ppl = self.evaluate()
                    print(f"  >> validation: loss {val_loss:.4f} | perplexity {val_ppl:.2f}")
                    if val_loss < best_val:
                        best_val = val_loss
                        self.save_checkpoint("checkpoints/best.pt")

                if self.step % self.checkpoint_interval == 0:
                    self.save_checkpoint(f"checkpoints/step_{self.step}.pt")

        final_val, final_ppl = self.evaluate()
        print(f"[trainer] finished. val loss {final_val:.4f} | perplexity {final_ppl:.2f}")
        self.save_checkpoint("checkpoints/final.pt")
