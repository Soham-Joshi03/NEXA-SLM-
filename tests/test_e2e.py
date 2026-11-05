"""
End-to-End Workflow Integration Test:
Validates the complete pipeline:
Tokenizer -> Synthetic Dataset -> Model -> Training -> Checkpoint -> Reload -> Cached Inference.
"""

import math
import os
import tempfile
import torch
from accelerate import Accelerator
from transformers import PreTrainedTokenizerFast

from nexa.config import get_unit_test_config
from nexa.model import NexaSLM, count_parameters
from nexa.data import create_synthetic_dataset, MemmapDataset
from nexa.inference import generate
from train import get_cosine_schedule_with_min_lr, save_single_file_checkpoint


def test_full_end_to_end_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Tokenizer validation
        tok_file = "jamba_bbpe_tokenizer.json"
        assert os.path.exists(tok_file)
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tok_file)
        assert len(tokenizer) == 32768

        # 2. Dataset creation
        train_bin = os.path.join(tmpdir, "train.bin")
        val_bin = os.path.join(tmpdir, "val.bin")
        create_synthetic_dataset(train_bin, num_tokens=2000, vocab_size=32768, seed=42)
        create_synthetic_dataset(val_bin, num_tokens=500, vocab_size=32768, seed=123)

        cfg = get_unit_test_config()
        cfg.train_bin = train_bin
        cfg.val_bin = val_bin
        cfg.checkpoint_dir = tmpdir

        # 3. Model construction and parameter verification
        model = NexaSLM(cfg)
        param_info = count_parameters(model, cfg)
        assert param_info["total_trainable"] > 0
        assert param_info["active_params_per_token"] <= param_info["total_trainable"]

        # 4. Short Training Step
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.max_lr)
        scheduler = get_cosine_schedule_with_min_lr(
            optimizer, num_warmup_steps=1, num_training_steps=4, max_lr=cfg.max_lr, min_lr=cfg.min_lr
        )
        accelerator = Accelerator(mixed_precision="no")

        dataset = MemmapDataset(train_bin, cfg.max_seq_len, cfg.vocab_size)
        loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size_per_gpu, shuffle=True)

        model.train()
        batch_x, batch_y = next(iter(loader))
        ce, z, aux = model(batch_x, batch_y)
        loss = ce + 0.001 * z + 0.01 * aux
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # 5. Checkpoint saving
        ckpt_path = os.path.join(tmpdir, "best_model.pt")
        save_single_file_checkpoint(
            ckpt_path,
            global_step=1,
            tokens_seen=batch_x.numel(),
            best_val_loss=ce.item(),
            accelerator=accelerator,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=cfg,
        )
        assert os.path.exists(ckpt_path)

        # 6. Checkpoint Reload
        loaded_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        new_model = NexaSLM(cfg)
        new_model.load_state_dict(loaded_ckpt["model_state_dict"])
        new_model.eval()

        # 7. Cached Autoregressive Inference
        prompt_text = "NEXA-SLM is"
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
        output_tokens = generate(new_model, prompt_tokens, max_new_tokens=8, temperature=0.7)

        assert len(output_tokens) == len(prompt_tokens) + 8
        decoded_text = tokenizer.decode(output_tokens)
        assert len(decoded_text) > 0
        print("End-to-End Test completed with decoded output:", repr(decoded_text))

        # Close memmap file handles to allow temp directory cleanup on Windows
        dataset.close()
        del dataset
        del loader
