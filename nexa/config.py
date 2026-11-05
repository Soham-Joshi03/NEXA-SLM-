"""
Configuration module for NEXA-SLM architecture, training, and inference.
"""

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NexaConfig:
    """Master configuration class for NEXA-SLM."""
    
    # Model Specs
    d_model: int = 512
    n_super_blocks: int = 2  # Total sub-layers = 5 * n_super_blocks
    vocab_size: int = 32768
    max_seq_len: int = 2048
    num_scratchpad: int = 256
    tie_embeddings: bool = True
    initializer_range: float = 0.02
    
    # Mamba-3 Specs (Trapezoidal Discretization)
    mamba_expand: int = 2
    mamba_d_state: int = 16
    mamba_conv_size: int = 4
    mamba_dt_rank: Optional[int] = None  # Default: ceil(d_model / 16)
    use_mamba3_trapezoidal: bool = True  # True: Lahoti et al. 2026 Eq 3-4; False: Mamba-1
    
    # SW-GQA Specs
    q_heads: int = 8
    kv_heads: int = 2
    head_dim: int = 64
    window_size: int = 256
    attn_mode: str = "chunked"  # "chunked" for true O(n*W) sliding window, "dense" for reference
    rope_base: float = 10000.0
    
    # MoE Specs
    num_experts: int = 8
    num_active_experts: int = 2
    moe_ffn_dim: int = 1024
    router_aux_loss_coef: float = 0.01
    router_z_loss_coef: float = 0.001
    normalize_topk_prob: bool = True
    
    # Training Specs
    batch_size_per_gpu: int = 4
    gradient_accumulation_steps: int = 64
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.01
    warmup_steps_pct: float = 0.10
    total_global_steps: int = 10000
    eval_every: int = 50
    save_latest_every: int = 20
    eval_iters: int = 100
    mixed_precision: str = "fp16"  # "fp16", "bf16", or "no"
    seed: int = 42
    
    # Dataset File Paths
    train_bin: str = "data/train.bin"
    val_bin: str = "data/val.bin"
    dataset_dir: str = "data"
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    resume_from_checkpoint: Optional[str] = None
    tokenizer_file: str = "jamba_bbpe_tokenizer.json"

    def __post_init__(self):
        if self.mamba_dt_rank is None:
            self.mamba_dt_rank = math.ceil(self.d_model / 16)
        self.mamba_d_inner = self.d_model * self.mamba_expand
        if self.q_heads % self.kv_heads != 0:
            raise ValueError(f"q_heads ({self.q_heads}) must be divisible by kv_heads ({self.kv_heads})")

    @property
    def total_sublayers(self) -> int:
        return self.n_super_blocks * 5

    @property
    def total_mamba_moe_layers(self) -> int:
        return self.n_super_blocks * 4

    @property
    def total_attention_layers(self) -> int:
        return self.n_super_blocks * 1


def get_unit_test_config() -> NexaConfig:
    """Ultra-lightweight configuration (~3.2M parameters) for fast unit and gradient testing."""
    return NexaConfig(
        d_model=64,
        n_super_blocks=1,
        vocab_size=32768,
        max_seq_len=64,
        num_scratchpad=16,
        mamba_expand=2,
        mamba_d_state=8,
        mamba_conv_size=4,
        mamba_dt_rank=4,
        q_heads=2,
        kv_heads=1,
        head_dim=32,
        window_size=16,
        num_experts=4,
        num_active_experts=2,
        moe_ffn_dim=128,
        batch_size_per_gpu=2,
        gradient_accumulation_steps=2,
        total_global_steps=10,
        eval_every=5,
        save_latest_every=5,
        eval_iters=5,
        mixed_precision="no",
        train_bin="data/test_train.bin",
        val_bin="data/test_val.bin",
    )


def get_dev_config() -> NexaConfig:
    """Modest configuration (~41.6M parameters) suitable for local development and debugging."""
    return NexaConfig(
        d_model=256,
        n_super_blocks=2,
        vocab_size=32768,
        max_seq_len=512,
        num_scratchpad=64,
        mamba_expand=2,
        mamba_d_state=16,
        mamba_conv_size=4,
        mamba_dt_rank=16,
        q_heads=4,
        kv_heads=2,
        head_dim=64,
        window_size=128,
        num_experts=8,
        num_active_experts=2,
        moe_ffn_dim=512,
        batch_size_per_gpu=4,
        gradient_accumulation_steps=16,
        total_global_steps=1000,
        eval_every=50,
        save_latest_every=50,
        eval_iters=50,
        mixed_precision="fp16",
        train_bin="data/dev_train.bin",
        val_bin="data/dev_val.bin",
    )


def get_base_config() -> NexaConfig:
    """Original repository configuration (~148.7M total, ~73.2M active)."""
    return NexaConfig(
        d_model=512,
        n_super_blocks=2,
        vocab_size=32768,
        max_seq_len=2048,
        num_scratchpad=256,
        mamba_expand=2,
        mamba_d_state=16,
        mamba_conv_size=4,
        mamba_dt_rank=32,
        q_heads=8,
        kv_heads=2,
        head_dim=64,
        window_size=256,
        num_experts=8,
        num_active_experts=2,
        moe_ffn_dim=1024,
        batch_size_per_gpu=4,
        gradient_accumulation_steps=64,
        total_global_steps=10000,
        eval_every=50,
        save_latest_every=20,
        eval_iters=100,
        mixed_precision="fp16",
    )


def get_poc_config() -> NexaConfig:
    """
    Research Proof-of-Concept configuration (~396.9M parameters total, ~170.5M active per token).
    Targets the 300M–400M parameter research specification with 4 SuperBlocks (20 sub-layers).
    """
    return NexaConfig(
        d_model=512,
        n_super_blocks=4,  # 16 Mamba-MoE layers + 4 Attention layers = 20 total layers
        vocab_size=32768,
        max_seq_len=2048,
        num_scratchpad=256,
        mamba_expand=2,
        mamba_d_state=16,
        mamba_conv_size=4,
        mamba_dt_rank=32,
        q_heads=8,
        kv_heads=2,
        head_dim=64,
        window_size=256,
        num_experts=8,
        num_active_experts=2,
        moe_ffn_dim=1536,
        batch_size_per_gpu=2,
        gradient_accumulation_steps=64,
        total_global_steps=10000,
        eval_every=50,
        save_latest_every=20,
        eval_iters=100,
        mixed_precision="fp16",
    )


# Backward compatibility with existing scripts
class Config(NexaConfig):
    pass
