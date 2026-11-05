"""
NEXA-SLM: Master Training Pipeline
Supports:
- Multi-GPU & single-GPU / CPU training via Accelerate
- Research POC (300-400M), Base (148M), Dev (41M), and Unit-Test (3.2M) configurations
- Mamba-3 trapezoidal selective scan & SW-GQA with Scratchpad
- MoE with Top-2 routing, shared expert, router z-loss, and load-balancing aux loss
- Cosine LR scheduling with verified min_lr floor
- Safe single-file checkpoint saving and resumption
"""

import os
import sys
import math
import argparse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import shutil
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator

# Import modular components from nexa package
from nexa.config import (
    NexaConfig,
    Config,
    get_unit_test_config,
    get_dev_config,
    get_base_config,
    get_poc_config,
)
from nexa.model import NexaSLM, PerfectSLM, SuperBlock, count_parameters
from nexa.mamba3 import Mamba3Layer, mamba3_trapezoidal_scan, mamba1_standard_scan
from nexa.attention import SWGQA, apply_rotary_emb
from nexa.moe import MoELayer, SwiGLU
from nexa.scratchpad import VirtualScratchpad
from nexa.data import MemmapDataset, validate_dataset, create_synthetic_dataset

# Fallback export of selective_scan_fn for legacy imports
selective_scan_fn = mamba3_trapezoidal_scan


def cycle(loader):
    """Infinite iterator to prevent StopIteration crashes across epochs in DDP."""
    while True:
        for batch in loader:
            yield batch


def get_cosine_schedule_with_min_lr(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    max_lr: float,
    min_lr: float,
):
    """
    Cosine learning rate schedule with linear warmup and min_lr floor.
    Ensures that learning rate decays to min_lr instead of 0.0.
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        # Scale between min_lr / max_lr and 1.0
        min_ratio = min_lr / max_lr
        return min_ratio + (1.0 - min_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_single_file_checkpoint(
    filepath: str,
    global_step: int,
    tokens_seen: int,
    best_val_loss: float,
    accelerator: Accelerator,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: NexaConfig,
):
    """Saves model weights, optimizer, scheduler, and training states safely into a single .pt file."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    unwrapped_model = accelerator.unwrap_model(model)

    checkpoint = {
        "global_step": global_step,
        "tokens_seen": tokens_seen,
        "best_val_loss": best_val_loss,
        "model_state_dict": unwrapped_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": vars(config),
        "torch_rng_state": torch.get_rng_state(),
        "np_rng_state": np.random.get_state(),
    }
    if torch.cuda.is_available():
        checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state_all()

    accelerator.save(checkpoint, filepath)


def parse_args():
    parser = argparse.ArgumentParser(description="Train NEXA-SLM")
    parser.add_argument(
        "--config",
        type=str,
        default="base",
        choices=["poc", "base", "dev", "test"],
        help="Configuration preset (poc: ~397M, base: ~149M, dev: ~42M, test: ~3M)",
    )
    parser.add_argument("--train_bin", type=str, default=None, help="Path to training uint16 .bin file")
    parser.add_argument("--val_bin", type=str, default=None, help="Path to validation uint16 .bin file")
    parser.add_argument("--batch_size", type=int, default=None, help="Micro-batch size per device")
    parser.add_argument("--grad_accum", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--max_lr", type=float, default=None, help="Maximum learning rate")
    parser.add_argument("--min_lr", type=float, default=None, help="Minimum learning rate")
    parser.add_argument("--steps", type=int, default=None, help="Total global training steps")
    parser.add_argument("--eval_every", type=int, default=None, help="Evaluate every N global steps")
    parser.add_argument("--save_latest_every", type=int, default=None, help="Save latest checkpoint every N global steps")
    parser.add_argument("--eval_iters", type=int, default=None, help="Max validation batches to evaluate")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Checkpoint output directory")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint file to resume from")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--create_sample_data_if_missing", action="store_true", help="Auto-generate sample data if missing")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Select base configuration preset
    if args.config == "poc":
        config = get_poc_config()
    elif args.config == "dev":
        config = get_dev_config()
    elif args.config == "test":
        config = get_unit_test_config()
    else:
        config = get_base_config()

    # Override with CLI arguments if provided
    if args.train_bin is not None:
        config.train_bin = args.train_bin
    if args.val_bin is not None:
        config.val_bin = args.val_bin
    if args.batch_size is not None:
        config.batch_size_per_gpu = args.batch_size
    if args.grad_accum is not None:
        config.gradient_accumulation_steps = args.grad_accum
    if args.max_lr is not None:
        config.max_lr = args.max_lr
    if args.min_lr is not None:
        config.min_lr = args.min_lr
    if args.steps is not None:
        config.total_global_steps = args.steps
    if args.eval_every is not None:
        config.eval_every = args.eval_every
    if args.eval_iters is not None:
        config.eval_iters = args.eval_iters
    if args.checkpoint_dir is not None:
        config.checkpoint_dir = args.checkpoint_dir
    if args.resume is not None:
        config.resume_from_checkpoint = args.resume

    # Hardware & precision adaptability
    if args.mixed_precision is not None:
        mixed_precision = args.mixed_precision
    else:
        mixed_precision = "fp16" if torch.cuda.is_available() else "no"

    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )

    accelerator.print("=" * 75)
    accelerator.print(f"  NEXA-SLM Master Training Pipeline (Preset: {args.config.upper()})")
    accelerator.print(f"  Device: {accelerator.device} | Processes: {accelerator.num_processes} | Mixed Precision: {mixed_precision}")
    accelerator.print("=" * 75)

    # 2. Check and prepare datasets
    if not os.path.exists(config.train_bin) or not os.path.exists(config.val_bin):
        if args.create_sample_data_if_missing or not os.path.exists(config.train_bin):
            accelerator.print(f"Dataset files not found at '{config.train_bin}'. Generating sample dataset...")
            os.makedirs(os.path.dirname(os.path.abspath(config.train_bin)), exist_ok=True)
            create_synthetic_dataset(config.train_bin, num_tokens=100000, vocab_size=config.vocab_size)
            create_synthetic_dataset(config.val_bin, num_tokens=20000, vocab_size=config.vocab_size, seed=123)
            accelerator.print("Sample datasets generated successfully.")

    train_val = validate_dataset(config.train_bin, config.vocab_size)
    if not train_val["valid"]:
        raise ValueError(f"Training dataset validation failed: {train_val.get('error')}")

    val_val = validate_dataset(config.val_bin, config.vocab_size)
    if not val_val["valid"]:
        raise ValueError(f"Validation dataset validation failed: {val_val.get('error')}")

    accelerator.print(f"Train Dataset: {config.train_bin} ({train_val['num_tokens']:,} tokens, {train_val['file_size_mb']} MB)")
    accelerator.print(f"Val Dataset:   {config.val_bin} ({val_val['num_tokens']:,} tokens, {val_val['file_size_mb']} MB)")

    # 3. Model construction
    model = PerfectSLM(config)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        accelerator.print("Activation Checkpointing: ENABLED")

    param_info = count_parameters(model, config)
    accelerator.print(f"Total Trainable Parameters: {param_info['total_trainable'] / 1e6:,.2f} M")
    accelerator.print(f"Active Parameters / Token:  {param_info['active_params_per_token'] / 1e6:,.2f} M")
    accelerator.print(f"Checkpoint Sizes: {param_info['checkpoint_sizes_mb']}")

    # 4. DataLoaders
    train_dataset = MemmapDataset(config.train_bin, config.max_seq_len, config.vocab_size)
    val_dataset = MemmapDataset(config.val_bin, config.max_seq_len, config.vocab_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size_per_gpu,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size_per_gpu,
        shuffle=False,
    )

    # 5. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.max_lr,
        weight_decay=config.weight_decay,
    )
    warmup_steps = int(config.warmup_steps_pct * config.total_global_steps)
    scheduler = get_cosine_schedule_with_min_lr(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=config.total_global_steps,
        max_lr=config.max_lr,
        min_lr=config.min_lr,
    )

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    # 6. Checkpoint resumption if requested
    global_step = 0
    tokens_seen = 0
    best_val_loss = float("inf")

    if config.resume_from_checkpoint:
        ckpt_path = os.path.join(config.checkpoint_dir, config.resume_from_checkpoint)
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.load_state_dict(checkpoint["model_state_dict"])
            if config.tie_embeddings:
                unwrapped.lm_head.weight = unwrapped.embedding.weight

            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            global_step = checkpoint.get("global_step", 0)
            tokens_seen = checkpoint.get("tokens_seen", 0)
            best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            accelerator.print(f"Resumed from {ckpt_path} at Global Step {global_step}")
        else:
            accelerator.print(f"Warning: Checkpoint {ckpt_path} not found. Starting from scratch.")

    # 7. Master Training Loop
    model.train()
    border = "-" * 115
    accelerator.print(f"\n{border}")
    accelerator.print(
        f"| {'Type':^12} | {'Global Step':^11} | {'CE Loss':^10} | {'Z-Loss':^10} | "
        f"{'Aux Loss':^10} | {'Total Loss':^11} | {'Tokens Seen':^15} |"
    )
    accelerator.print(border)

    train_iter = cycle(train_loader)

    while global_step < config.total_global_steps:
        accum_ce, accum_z, accum_aux, accum_total = 0.0, 0.0, 0.0, 0.0

        for micro_step in range(config.gradient_accumulation_steps):
            x, y = next(train_iter)

            with accelerator.accumulate(model):
                ce_loss, z_loss, aux_loss = model(x, y)
                total_loss = (
                    ce_loss
                    + (config.router_z_loss_coef * z_loss)
                    + (config.router_aux_loss_coef * aux_loss)
                )

                accelerator.backward(total_loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            accum_ce += ce_loss.item()
            accum_z += z_loss.item()
            accum_aux += aux_loss.item()
            accum_total += total_loss.item()
            tokens_seen += x.numel() * accelerator.num_processes

        global_step += 1
        n_accum = config.gradient_accumulation_steps
        avg_ce = accum_ce / n_accum
        avg_z = accum_z / n_accum
        avg_aux = accum_aux / n_accum
        avg_tot = accum_total / n_accum

        current_lr = scheduler.get_last_lr()[0]
        accelerator.print(
            f"| {'[ STEP ]':<12} | {global_step:11,d} | {avg_ce:10.4f} | {avg_z:10.4f} | "
            f"{avg_aux:10.4f} | {avg_tot:11.4f} | {tokens_seen:15,d} | LR: {current_lr:.2e}"
        )

        # Evaluation
        if global_step % config.eval_every == 0:
            model.eval()
            val_loss = 0.0
            batches_evaluated = 0

            with torch.no_grad():
                for i, (vx, vy) in enumerate(val_loader):
                    if i >= config.eval_iters:
                        break
                    v_ce, _, _ = model(vx, vy)
                    val_loss += accelerator.gather(v_ce).mean().item()
                    batches_evaluated += 1

            if batches_evaluated > 0:
                val_loss /= batches_evaluated
            val_ppl = math.exp(min(20.0, val_loss))

            accelerator.print(
                f">> [EVAL] Global Step {global_step} | Val Loss: {val_loss:.4f} | "
                f"Val PPL: {val_ppl:.2f} | Best Val Loss: {best_val_loss:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(config.checkpoint_dir, "best_model.pt")
                save_single_file_checkpoint(
                    best_path,
                    global_step,
                    tokens_seen,
                    best_val_loss,
                    accelerator,
                    model,
                    optimizer,
                    scheduler,
                    config,
                )
                accelerator.print(f"** Saved New Best Checkpoint -> {best_path}")

            accelerator.print(border)
            model.train()

        # Save Latest Checkpoint
        if global_step % config.save_latest_every == 0:
            latest_path = os.path.join(config.checkpoint_dir, "latest_model.pt")
            save_single_file_checkpoint(
                latest_path,
                global_step,
                tokens_seen,
                best_val_loss,
                accelerator,
                model,
                optimizer,
                scheduler,
                config,
            )
            accelerator.print(f"-- Saved Latest Checkpoint -> {latest_path} at Step {global_step}")

    accelerator.print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
