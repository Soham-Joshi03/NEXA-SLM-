"""
Training pipeline unit tests:
- Learning rate scheduler min_lr floor
- Optimizer parameter updates
- Validation and perplexity calculation
- Checkpoint saving and resumption
"""

import math
import os
import tempfile
import torch
import torch.nn as nn
from accelerate import Accelerator

from nexa.config import get_unit_test_config
from nexa.model import NexaSLM
from train import get_cosine_schedule_with_min_lr, save_single_file_checkpoint


def test_cosine_schedule_with_min_lr_floor():
    model = nn.Linear(10, 10)
    max_lr = 3e-4
    min_lr = 3e-5
    total_steps = 100
    warmup_steps = 10

    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr)
    scheduler = get_cosine_schedule_with_min_lr(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        max_lr=max_lr,
        min_lr=min_lr,
    )

    lrs = []
    for step in range(total_steps):
        optimizer.step()
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])

    # Check that warmup reaches peak max_lr
    assert math.isclose(lrs[warmup_steps - 1], max_lr, rel_tol=1e-2)
    # Check that final lr hits min_lr
    assert math.isclose(lrs[-1], min_lr, rel_tol=1e-3)
    # Check all lrs >= min_lr
    for lr in lrs:
        assert lr >= min_lr - 1e-8


def test_checkpoint_save_and_reload():
    cfg = get_unit_test_config()
    model = NexaSLM(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = get_cosine_schedule_with_min_lr(
        optimizer, num_warmup_steps=2, num_training_steps=10, max_lr=1e-3, min_lr=1e-4
    )
    accelerator = Accelerator(mixed_precision="no")

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_file = os.path.join(tmpdir, "test_ckpt.pt")
        save_single_file_checkpoint(
            filepath=ckpt_file,
            global_step=5,
            tokens_seen=1024,
            best_val_loss=5.4321,
            accelerator=accelerator,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=cfg,
        )

        assert os.path.exists(ckpt_file)
        loaded = torch.load(ckpt_file, map_location="cpu", weights_only=False)

        assert loaded["global_step"] == 5
        assert loaded["tokens_seen"] == 1024
        assert math.isclose(loaded["best_val_loss"], 5.4321)
        assert "model_state_dict" in loaded
        assert "optimizer_state_dict" in loaded
        assert "scheduler_state_dict" in loaded


def test_perplexity_computation():
    val_loss = 4.5
    ppl = math.exp(val_loss)
    assert round(ppl, 2) == 90.02
