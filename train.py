"""
Pre-train decoder-only LM on WikiText-2 (next-token prediction).
AdamW + warmup + cosine decay; periodic train/val loss, checkpoints, learning curve.
"""

from __future__ import annotations

import os
from collections import deque
from typing import List

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tokenizers import Tokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as cfg
from checkpoint import save_checkpoint
from dataset import LMDataset
from transformer.PreTrainingModel import PreTrainingModel


def encode_wikitext(tokenizer: Tokenizer, texts: List[str]) -> List[int]:
    ids: List[int] = []
    for t in texts:
        t = t.strip()
        if not t:
            continue
        ids.extend(tokenizer.encode(t).ids)
    return ids


@torch.no_grad()
def mean_cross_entropy(
    model: PreTrainingModel,
    loader: DataLoader,
    device: torch.device,
    vocab_size: int,
) -> float:
    model.eval()
    total, tokens = 0.0, 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            y.view(-1),
            reduction="sum",
        )
        total += loss.item()
        tokens += y.numel()
    model.train()
    return total / max(tokens, 1)


def build_scheduler(
    optimizer: AdamW,
    warmup_steps: int,
    total_steps: int,
) -> SequentialLR:
    """Linear warmup then cosine decay to 0 (per optimizer step)."""
    warmup_steps = max(1, warmup_steps)
    cosine_steps = max(1, total_steps - warmup_steps)
    warmup = LinearLR(
        optimizer,
        start_factor=1e-8,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=0.0)
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])


def plot_learning_curve(
    train_steps: List[int],
    train_losses: List[float],
    val_steps: List[int],
    val_losses: List[float],
    path: str,
) -> None:
    plt.figure(figsize=(8, 5))
    if train_steps:
        plt.plot(train_steps, train_losses, label="train", color="C0", linewidth=1.2)
    if val_steps:
        plt.plot(val_steps, val_losses, label="validation", color="C1", linewidth=1.2)
    plt.xlabel("optimizer step")
    plt.ylabel("cross-entropy loss")
    plt.title("Training / validation loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    tok_path = os.path.join(os.path.dirname(__file__), cfg.tokenizer_path)
    tokenizer = Tokenizer.from_file(tok_path)
    vocab_size = tokenizer.get_vocab_size()

    print("Loading WikiText-2 …")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", trust_remote_code=True)
    train_ids = encode_wikitext(tokenizer, ds["train"]["text"])
    val_ids = encode_wikitext(tokenizer, ds["validation"]["text"])

    train_tensor = torch.tensor(train_ids, dtype=torch.long)
    val_tensor = torch.tensor(val_ids, dtype=torch.long)

    train_ds = LMDataset(train_tensor, cfg.max_seq_len)
    val_ds = LMDataset(val_tensor, cfg.max_seq_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=cfg.grad_accum_steps > 1,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )

    steps_per_epoch = (len(train_loader) + cfg.grad_accum_steps - 1) // cfg.grad_accum_steps
    total_steps = max(1, cfg.epochs * steps_per_epoch)

    model = PreTrainingModel(
        vocab_size=vocab_size,
        max_seq_len=cfg.max_seq_len,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = build_scheduler(optimizer, cfg.warmup_steps, total_steps)

    global_step = 0
    train_log_steps: List[int] = []
    train_log_losses: List[float] = []
    val_log_steps: List[int] = []
    val_log_losses: List[float] = []
    recent_train_losses: deque[float] = deque(maxlen=cfg.log_every)

    out_dir = os.path.join(os.path.dirname(__file__), cfg.checkpoint_dir)
    os.makedirs(out_dir, exist_ok=True)

    for epoch in range(cfg.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accum = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{cfg.epochs}")

        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            ce = F.cross_entropy(
                logits.view(-1, vocab_size),
                y.view(-1),
            )
            (ce / cfg.grad_accum_steps).backward()
            accum += 1

            if accum % cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                accum = 0

                recent_train_losses.append(ce.detach().item())

                if global_step % cfg.log_every == 0:
                    avg = sum(recent_train_losses) / max(len(recent_train_losses), 1)
                    train_log_steps.append(global_step)
                    train_log_losses.append(avg)
                    pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

                if global_step % cfg.eval_every == 0:
                    vloss = mean_cross_entropy(model, val_loader, device, vocab_size)
                    val_log_steps.append(global_step)
                    val_log_losses.append(vloss)
                    pbar.write(f"[step {global_step}] val loss = {vloss:.4f}")
                    plot_learning_curve(
                        train_log_steps,
                        train_log_losses,
                        val_log_steps,
                        val_log_losses,
                        os.path.join(os.path.dirname(__file__), cfg.learning_curve_path),
                    )

                if global_step % cfg.checkpoint_every == 0:
                    ck_path = os.path.join(out_dir, f"checkpoint_step_{global_step}.pt")
                    save_checkpoint(
                        ck_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        global_step=global_step,
                    )
                    latest = os.path.join(out_dir, "checkpoint_latest.pt")
                    save_checkpoint(
                        latest,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        global_step=global_step,
                    )

        # end of epoch: flush running average if any
        if accum > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

    final_path = os.path.join(os.path.dirname(__file__), cfg.final_model_path)
    torch.save(model.state_dict(), final_path)
    print(f"Saved final weights to {final_path}")

    curve_path = os.path.join(os.path.dirname(__file__), cfg.learning_curve_path)
    plot_learning_curve(train_log_steps, train_log_losses, val_log_steps, val_log_losses, curve_path)
    print(f"Saved learning curve to {curve_path}")


if __name__ == "__main__":
    main()
