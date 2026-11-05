"""
NEXA-SLM: A Scalable Hybrid Mamba-Attention MoE Small Language Model
Department of AI & Data Science, ABMSP's Anantrao Pawar College of Engineering & Research, Pune.
"""

__version__ = "0.2.0"
__author__ = "Soham Joshi, Srihari Mhaisekar, Tanmayee Ozarkar, Tanay Mehta"

from nexa.config import NexaConfig, Config
from nexa.model import NexaSLM, PerfectSLM, SuperBlock, count_parameters
from nexa.mamba3 import Mamba3Layer
from nexa.attention import SWGQA, apply_rotary_emb
from nexa.moe import MoELayer, SwiGLU

__all__ = [
    "NexaConfig",
    "Config",
    "NexaSLM",
    "PerfectSLM",
    "SuperBlock",
    "Mamba3Layer",
    "SWGQA",
    "MoELayer",
    "SwiGLU",
    "apply_rotary_emb",
    "count_parameters",
]
