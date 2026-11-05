"""
Data loading, validation, and preparation pipeline for NEXA-SLM.
Supports memory-mapped uint16 token arrays, dataset validation,
and reproducible dataset generation for smoke testing and benchmark training.
"""

import os
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast


class MemmapDataset(Dataset):
    """
    Memory-mapped uint16 token dataset for high-throughput causal LM training.
    Produces (input_ids, target_ids) shifted by 1 position for next-token prediction.
    """

    def __init__(self, bin_path: str, seq_len: int, vocab_size: int = 32768):
        if not os.path.exists(bin_path):
            raise FileNotFoundError(
                f"Dataset binary file not found at: {bin_path}\n"
                f"To generate a local development dataset, run: python -m nexa.data --create-sample"
            )

        self.bin_path = bin_path
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        # Open file in read-only memmap mode
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.num_tokens = len(self.data)

        if self.num_tokens <= seq_len:
            raise ValueError(
                f"Dataset has {self.num_tokens} tokens, but seq_len is {seq_len}. "
                f"Dataset must have at least {seq_len + 1} tokens."
            )

    def __len__(self) -> int:
        return (self.num_tokens - 1) // self.seq_len

    def close(self):
        """Closes the underlying memory-mapped file buffer (critical for Windows)."""
        if hasattr(self, "data") and hasattr(self.data, "_mmap") and self.data._mmap is not None:
            self.data._mmap.close()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = torch.from_numpy(self.data[start:end].astype(np.int64))
        # x is input sequence of length seq_len, y is shifted targets
        return chunk[:-1], chunk[1:]


def validate_dataset(bin_path: str, vocab_size: int = 32768) -> dict:
    """
    Validates binary token dataset file integrity and token ID bounds.
    """
    if not os.path.isfile(bin_path):
        return {"valid": False, "error": f"File does not exist: {bin_path}"}

    file_size = os.path.getsize(bin_path)
    if file_size == 0 or file_size % 2 != 0:
        return {
            "valid": False,
            "error": f"File size {file_size} bytes is not a valid non-empty uint16 array.",
        }

    num_tokens = file_size // 2
    # Sample up to 100,000 tokens for bounds checking
    sample_size = min(num_tokens, 100000)
    data = np.memmap(bin_path, dtype=np.uint16, mode="r", shape=(num_tokens,))
    sample = data[:sample_size]

    min_val = int(sample.min())
    max_val = int(sample.max())

    if max_val >= vocab_size:
        return {
            "valid": False,
            "num_tokens": num_tokens,
            "error": f"Token ID {max_val} exceeds vocabulary size {vocab_size}.",
        }

    return {
        "valid": True,
        "num_tokens": num_tokens,
        "file_size_mb": round(file_size / (1024 ** 2), 2),
        "min_token_id": min_val,
        "max_token_id": max_val,
    }


def create_synthetic_dataset(
    output_bin_path: str,
    num_tokens: int = 50000,
    vocab_size: int = 32768,
    seed: int = 42,
) -> str:
    """
    Generates a synthetic binary token dataset in uint16 format for smoke testing and CI.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_bin_path)), exist_ok=True)
    rng = np.random.default_rng(seed)

    # Use a Zipf distribution to simulate realistic natural language token frequencies
    # Zipfian parameter a=1.2, clipped to vocab_size
    tokens = rng.zipf(a=1.2, size=num_tokens).astype(np.uint16)
    tokens = np.clip(tokens, 0, vocab_size - 1)

    # Insert end-of-text tokens (ID 0) periodically
    eot_indices = rng.choice(num_tokens, size=max(1, num_tokens // 256), replace=False)
    tokens[eot_indices] = 0

    with open(output_bin_path, "wb") as f:
        f.write(tokens.tobytes())

    return output_bin_path


def tokenize_text_to_bin(
    text: str,
    tokenizer_file: str,
    output_bin_path: str,
    bos_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = 0,
) -> int:
    """
    Tokenizes raw text using the ByteLevel BPE tokenizer and writes to a uint16 .bin file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_bin_path)), exist_ok=True)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)

    encoding = tokenizer.encode(text, add_special_tokens=False)
    token_list = []
    if bos_token_id is not None:
        token_list.append(bos_token_id)
    token_list.extend(encoding)
    if eos_token_id is not None:
        token_list.append(eos_token_id)

    tokens_arr = np.array(token_list, dtype=np.uint16)
    with open(output_bin_path, "wb") as f:
        f.write(tokens_arr.tobytes())

    return len(tokens_arr)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NEXA-SLM Data Preparation Utility")
    parser.add_argument("--create-sample", action="store_true", help="Generate sample datasets")
    parser.add_argument("--train-tokens", type=int, default=100000, help="Tokens in train set")
    parser.add_argument("--val-tokens", type=int, default=20000, help="Tokens in val set")
    parser.add_argument("--data-dir", type=str, default="data", help="Output directory")
    args = parser.parse_args()

    if args.create_sample:
        train_p = os.path.join(args.data_dir, "dev_train.bin")
        val_p = os.path.join(args.data_dir, "dev_val.bin")
        print(f"Creating sample train dataset: {train_p} ({args.train_tokens:,} tokens)...")
        create_synthetic_dataset(train_p, num_tokens=args.train_tokens)
        print(f"Creating sample val dataset: {val_p} ({args.val_tokens:,} tokens)...")
        create_synthetic_dataset(val_p, num_tokens=args.val_tokens, seed=123)
        print("Dataset creation complete!")
        print("Train info:", validate_dataset(train_p))
        print("Val info:", validate_dataset(val_p))
