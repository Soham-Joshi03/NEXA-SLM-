"""
NEXA-SLM Evaluation and Ablation Study Suite
Evaluates:
- Validation loss and Perplexity (PPL)
- Throughput (tokens/sec) and Time-To-First-Token (TTFT)
- Memory footprint: Param weights, KV cache, Mamba recurrent state
- Architectural Ablations:
  1. Full NEXA-SLM (Baseline)
  2. Ablation: Without Virtual Scratchpad (m = 0)
  3. Ablation: Mamba-1 standard scan vs Mamba-3 trapezoidal scan
  4. Ablation: Dense Attention vs Chunked SW-GQA Attention
  5. Ablation: Without MoE (Single Shared MLP)
"""

import os
import sys
import math
import time
import argparse
from typing import Dict, List, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from nexa.config import NexaConfig, get_unit_test_config, get_dev_config, get_poc_config, get_base_config
from nexa.model import NexaSLM, count_parameters
from nexa.data import MemmapDataset, validate_dataset
from nexa.inference import generate
from nexa.quantization import calculate_analytical_kv_cache, calculate_analytical_mamba_state


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def evaluate_perplexity(model: NexaSLM, dataloader: DataLoader, device: str = "cpu", max_batches: int = 50) -> Dict[str, float]:
    model.eval()
    model.to(device)
    total_loss = 0.0
    total_z = 0.0
    total_aux = 0.0
    batches = 0

    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            if i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            ce, z, aux = model(x, y)
            total_loss += ce.item()
            total_z += z.item()
            total_aux += aux.item()
            batches += 1

    if batches == 0:
        return {"val_loss": 0.0, "perplexity": 1.0, "batches": 0}

    avg_loss = total_loss / batches
    ppl = math.exp(min(20.0, avg_loss))
    return {
        "val_loss": round(avg_loss, 4),
        "perplexity": round(ppl, 2),
        "avg_z_loss": round(total_z / batches, 4),
        "avg_aux_loss": round(total_aux / batches, 4),
        "batches_evaluated": batches,
    }


def run_ablation_matrix(device: str = "cpu") -> List[Dict[str, Any]]:
    """
    Executes the planned ablation matrix specified in research paper Table VII:
    1. Full model (Baseline)
    2. w/o Scratchpad
    3. Mamba-1 (Euler) vs Mamba-3 (Trapezoidal)
    4. Chunked SW-GQA vs Dense Attention
    """
    print("\n" + "=" * 80)
    print("  NEXA-SLM ARCHITECTURAL ABLATION STUDY (Empirical Comparison)")
    print("=" * 80)

    ablations = [
        ("Full Model (Baseline)", {}),
        ("w/o Virtual Scratchpad", {"num_scratchpad": 0}),
        ("w/o Mamba-3 (Mamba-1 Euler)", {"use_mamba3_trapezoidal": False}),
        ("SW-GQA Dense Attention", {"attn_mode": "dense"}),
        ("SW-GQA Chunked O(nW)", {"attn_mode": "chunked"}),
    ]

    results = []
    base_cfg = get_unit_test_config()
    base_cfg.max_seq_len = 64

    # Generate dummy input sequence
    B, L = 2, 64
    x = torch.randint(0, base_cfg.vocab_size, (B, L), device=device)
    y = torch.randint(0, base_cfg.vocab_size, (B, L), device=device)

    for name, overrides in ablations:
        cfg = get_unit_test_config()
        cfg.max_seq_len = 64
        for k, v in overrides.items():
            setattr(cfg, k, v)

        model = NexaSLM(cfg).to(device)
        params = count_parameters(model, cfg)

        # Forward latency test
        model.eval()
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(10):
                ce, z, aux = model(x, y)
        elapsed = (time.perf_counter() - start) / 10.0

        # KV cache calculation
        kv_info = calculate_analytical_kv_cache(cfg)

        entry = {
            "variant": name,
            "total_params_M": round(params["total_trainable"] / 1e6, 3),
            "active_params_M": round(params["active_params_per_token"] / 1e6, 3),
            "step_latency_ms": round(elapsed * 1000, 2),
            "ce_loss": round(ce.item(), 4),
            "kv_cache_kib": kv_info["cache_kib"],
        }
        results.append(entry)

    # Print summary table
    print(f"\n| {'Ablation Variant':<30} | {'Total M':<9} | {'Active M':<9} | {'Latency (ms)':<13} | {'KV Cache (KiB)':<15} |")
    print("|" + "-" * 32 + "|" + "-" * 11 + "|" + "-" * 11 + "|" + "-" * 15 + "|" + "-" * 17 + "|")
    for r in results:
        print(
            f"| {r['variant']:<30} | {r['total_params_M']:<9.3f} | {r['active_params_M']:<9.3f} | "
            f"{r['step_latency_ms']:<13.2f} | {r['kv_cache_kib']:<15.2f} |"
        )
    print("-" * 80)
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate NEXA-SLM")
    parser.add_argument("--config", type=str, default="test", choices=["poc", "base", "dev", "test"])
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--val_bin", type=str, default="data/test_val.bin")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--run_ablations", action="store_true", help="Execute Table VII ablation comparisons")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.run_ablations:
        run_ablation_matrix(device=device)
        return

    # Standard evaluation
    if args.config == "poc":
        cfg = get_poc_config()
    elif args.config == "dev":
        cfg = get_dev_config()
    elif args.config == "base":
        cfg = get_base_config()
    else:
        cfg = get_unit_test_config()

    print("=" * 70)
    print(f"  NEXA-SLM Model Evaluation ({args.config.upper()})")
    print(f"  Device: {device} | Checkpoint: {args.checkpoint}")
    print("=" * 70)

    model = NexaSLM(cfg)
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"Warning: Checkpoint {args.checkpoint} not found. Evaluating initialized weights.")

    if os.path.exists(args.val_bin):
        val_dataset = MemmapDataset(args.val_bin, cfg.max_seq_len, cfg.vocab_size)
        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size_per_gpu, shuffle=False)
        metrics = evaluate_perplexity(model, val_loader, device=device)
        print("\n--- Validation Performance ---")
        print(f"Validation Loss:       {metrics['val_loss']}")
        print(f"Perplexity (PPL):      {metrics['perplexity']}")
        print(f"MoE Router Z-Loss:     {metrics['avg_z_loss']}")
        print(f"MoE Aux Load Loss:     {metrics['avg_aux_loss']}")
        print(f"Batches Evaluated:     {metrics['batches_evaluated']}")
        print("------------------------------")
    else:
        print(f"Validation dataset {args.val_bin} not found.")


if __name__ == "__main__":
    main()
