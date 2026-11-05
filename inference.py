"""
NEXA-SLM: Master Inference and Generation Pipeline
Supports:
- High-efficiency cached autoregressive decoding with O(W + m) constant-size KV cache
- Mamba-3 recurrent state caching
- Temperature, Top-K, and Top-P (nucleus) sampling
- Optional INT8 dynamic quantization for edge deployment
- Latency and throughput benchmarking
"""

import os
import sys
import time
import argparse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast

# Import components from nexa
from nexa.config import (
    NexaConfig,
    Config,
    get_unit_test_config,
    get_dev_config,
    get_base_config,
    get_poc_config,
)
from nexa.model import NexaSLM, PerfectSLM, SuperBlock
from nexa.mamba3 import Mamba3Layer
from nexa.attention import SWGQA, apply_rotary_emb
from nexa.inference import generate, inference_step, sample_next_token
from nexa.quantization import quantize_model_int8, profile_inference_speed


# Legacy alias for sampling
def sample(logits: torch.Tensor, temperature: float = 1.0, top_k: Optional[int] = None) -> int:
    return sample_next_token(logits, temperature=temperature, top_k=top_k)


# Monkey-patch step methods for legacy imports if needed
mamba_step = Mamba3Layer.step
swgqa_step = SWGQA.step


def parse_args():
    parser = argparse.ArgumentParser(description="NEXA-SLM Text Generation and Inference")
    parser.add_argument(
        "--config",
        type=str,
        default="base",
        choices=["poc", "base", "dev", "test"],
        help="Model configuration preset",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained checkpoint (.pt file)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="In the era of artificial intelligence, small language models",
        help="Prompt text for generation",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="jamba_bbpe_tokenizer.json",
        help="Path to tokenizer JSON file",
    )
    parser.add_argument("--max_tokens", type=int, default=50, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=40, help="Top-K sampling cutoff")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-P nucleus sampling threshold")
    parser.add_argument("--device", type=str, default=None, help="Execution device (cuda/cpu)")
    parser.add_argument("--quantize_int8", action="store_true", help="Apply dynamic INT8 quantization")
    parser.add_argument("--benchmark", action="store_true", help="Run speed and latency benchmark")
    return parser.parse_args()


def load_model_for_inference(
    config: NexaConfig,
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
    quantize_int8: bool = False,
) -> Tuple[NexaSLM, bool]:
    """
    Constructs model, loads checkpoint if available, and applies optional INT8 quantization.
    Returns (model, is_trained).
    """
    is_trained = False

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

        # Auto-detect Mamba discretization from x_proj weight shape
        for k, v in state_dict.items():
            if "mamba.x_proj.weight" in k:
                euler_dim = config.mamba_dt_rank + config.mamba_d_state * 2
                if v.shape[0] == euler_dim:
                    config.use_mamba3_trapezoidal = False
                break

        model = PerfectSLM(config)

        # Scratchpad alias compatibility
        if "scratchpad" in state_dict and "scratchpad_manager.scratchpad" not in state_dict:
            state_dict["scratchpad_manager.scratchpad"] = state_dict["scratchpad"]

        model.load_state_dict(state_dict, strict=False)
        is_trained = True
        step = checkpoint.get("global_step", "N/A")
        loss = checkpoint.get("best_val_loss", "N/A")
        print(f"Checkpoint loaded successfully! (Global Step: {step}, Best Val Loss: {loss})")
    else:
        model = PerfectSLM(config)
        if checkpoint_path:
            print(f"Warning: Checkpoint '{checkpoint_path}' not found.")
        print("[NOTICE: Generating with initialized (untrained) weights for architecture validation]")

    # Ensure tied weights
    if config.tie_embeddings:
        model.lm_head.weight = model.embedding.weight

    if quantize_int8:
        print("Applying INT8 weight-only quantization to linear layers...")
        model = quantize_model_int8(model)

    model.to(device)
    model.eval()
    return model, is_trained


def main():
    args = parse_args()

    # Determine device
    if args.device is not None:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Select configuration
    if args.config == "poc":
        config = get_poc_config()
    elif args.config == "dev":
        config = get_dev_config()
    elif args.config == "test":
        config = get_unit_test_config()
    else:
        config = get_base_config()

    print("=" * 70)
    print(f"  NEXA-SLM Inference Engine")
    print(f"  Preset: {args.config.upper()} | Device: {device} | INT8: {args.quantize_int8}")
    print("=" * 70)

    # Load tokenizer
    if not os.path.exists(args.tokenizer):
        # Check if parts exist to assemble
        from scripts import assemble_tokenizer
        print("Assembling tokenizer from parts...")

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer)
    print(f"Tokenizer loaded: {args.tokenizer} (Vocab: {len(tokenizer):,})")

    # Load model
    model, is_trained = load_model_for_inference(
        config=config,
        checkpoint_path=args.checkpoint,
        device=device,
        quantize_int8=args.quantize_int8,
    )

    if args.benchmark:
        print("\n--- Running Inference Speed Benchmark ---")
        perf = profile_inference_speed(
            model, prompt_length=16, num_new_tokens=args.max_tokens, device=device
        )
        print(f"Throughput:       {perf['tokens_per_sec']} tokens/sec")
        print(f"Latency / Token:  {perf['latency_per_token_ms']} ms")
        print(f"Total Time:       {perf['total_time_sec']} s ({perf['generated_tokens']} tokens)")
        print("------------------------------------------")

    # Encode prompt
    print(f"\nPrompt: '{args.prompt}'")
    prompt_tokens = tokenizer.encode(args.prompt, add_special_tokens=False)
    if len(prompt_tokens) == 0:
        prompt_tokens = [1]  # Fallback to pad/bos

    start_t = time.perf_counter()
    output_tokens = generate(
        model=model,
        prompt_tokens=prompt_tokens,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device,
    )
    gen_time = time.perf_counter() - start_t

    new_tokens_count = len(output_tokens) - len(prompt_tokens)
    tps = new_tokens_count / max(1e-6, gen_time)

    generated_text = tokenizer.decode(output_tokens)
    print("\n--- Final Generated Output ---")
    print(generated_text)
    print("------------------------------")
    print(f"Generated {new_tokens_count} tokens in {gen_time:.3f}s ({tps:.1f} tokens/sec)")
    if not is_trained:
        print("[Note: Output generated from initialized architecture weights. Train model on text corpus for coherent text.]")


if __name__ == "__main__":
    main()
