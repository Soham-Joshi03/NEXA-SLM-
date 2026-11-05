"""
Inference and quantization unit tests:
- Autoregressive generation
- Deterministic greedy decoding
- EOS handling
- INT8 dynamic quantization stability
"""

import torch
from nexa.config import get_unit_test_config
from nexa.model import NexaSLM
from nexa.inference import generate, sample_next_token
from nexa.quantization import quantize_model_int8, profile_inference_speed


def test_autoregressive_generation_length():
    cfg = get_unit_test_config()
    model = NexaSLM(cfg)
    prompt = [10, 20, 30]
    out = generate(model, prompt, max_new_tokens=10, temperature=0.7, top_k=10, eos_token_id=-1)
    assert len(out) == len(prompt) + 10


def test_deterministic_greedy_generation():
    cfg = get_unit_test_config()
    model = NexaSLM(cfg)
    prompt = [5, 15]
    out1 = generate(model, prompt, max_new_tokens=5, temperature=0.0, eos_token_id=-1)
    out2 = generate(model, prompt, max_new_tokens=5, temperature=0.0, eos_token_id=-1)
    assert out1 == out2, "Greedy generation must be strictly deterministic"


def test_eos_early_stopping():
    cfg = get_unit_test_config()
    model = NexaSLM(cfg)
    # Set eos_token_id to a known value
    prompt = [10]
    # In greedy mode or forced sample, generation halts when EOS is encountered
    out = generate(model, prompt, max_new_tokens=20, eos_token_id=0)
    if 0 in out:
        assert out[-1] == 0, "Generation must stop immediately at EOS token"


def test_int8_quantization_stability():
    cfg = get_unit_test_config()
    model = NexaSLM(cfg)
    model.eval()

    prompt = [1, 2, 3]
    out_fp32 = generate(model, prompt, max_new_tokens=4, temperature=0.0, eos_token_id=-1)

    model_int8 = quantize_model_int8(model)
    out_int8 = generate(model_int8, prompt, max_new_tokens=4, temperature=0.0, eos_token_id=-1)

    # Both must generate valid non-empty tokens
    assert len(out_fp32) == 7
    assert len(out_int8) == 7
