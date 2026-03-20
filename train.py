"""
Pre-train decoder-only LM on WikiText-2 (next-token prediction).
AdamW + warmup + cosine decay; periodic train/val loss, checkpoints, learning curve.
"""

from __future__ import annotations

import os
from typing import List

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tokenizers import Tokenizer
from torch.optim import AdamW
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
    max_batches: int | None = None,
) -> float:
    model.eval()
    total, tokens = 0.0, 0
    for batch_idx, (x, y) in enumerate(loader):
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
        if max_batches is not None and batch_idx + 1 >= max_batches:
            break
    model.train()
    return total / max(tokens, 1)

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

    steps_per_epoch_raw = (len(train_loader) + cfg.grad_accum_steps - 1) // cfg.grad_accum_steps
    steps_per_epoch = steps_per_epoch_raw
    if cfg.max_steps_per_epoch is not None:
        steps_per_epoch = min(steps_per_epoch_raw, cfg.max_steps_per_epoch)

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

    global_step = 0
    train_log_steps: List[int] = []
    train_log_losses: List[float] = []
    val_log_steps: List[int] = []
    val_log_losses: List[float] = []

    out_dir = os.path.join(os.path.dirname(__file__), cfg.checkpoint_dir)
    os.makedirs(out_dir, exist_ok=True)

    for epoch in range(cfg.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accum = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{cfg.epochs}")
        step_in_epoch = 0

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
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                accum = 0
                ce_value = ce.detach().item()

                if global_step % cfg.log_every == 0:
                    train_log_steps.append(global_step)
                    train_log_losses.append(ce_value)
                    pbar.set_postfix(loss=f"{ce_value:.4f}")

                if global_step % cfg.eval_every == 0:
                    # Only compute on a subset to avoid long "stalls".
                    vloss = mean_cross_entropy(
                        model,
                        val_loader,
                        device,
                        vocab_size,
                        max_batches=cfg.val_max_batches,
                    )
                    val_log_steps.append(global_step)
                    val_log_losses.append(vloss)
                    pbar.write(f"[step {global_step}] val loss = {vloss:.4f}")
                    # Plotting every time is slow; plot once after training.

                if global_step % cfg.checkpoint_every == 0:
                    ck_path = os.path.join(out_dir, "checkpoint.pt")
                    save_checkpoint(
                        ck_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=None,
                        epoch=epoch,
                        global_step=global_step,
                    )

                step_in_epoch += 1
                if step_in_epoch >= steps_per_epoch:
                    break

        # end of epoch: flush running average if any
        if accum > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
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
