# inference.py

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from train import Config, PerfectSLM, Mamba3Layer, SWGQA, apply_rotary_emb
from transformers import PreTrainedTokenizerFast

# ---------------------------------------------------------
# 1. ADD `step` METHODS FOR AUTOREGRESSIVE CACHING
# ---------------------------------------------------------

def mamba_step(self, x, cache, block_idx, layer_idx):
    """
    Single-token step for Mamba3Layer with state caching.
    x: (B, 1, D)
    """
    B, L, D = x.shape
    conv_key = f"mamba_{block_idx}_{layer_idx}_conv"
    ssm_key = f"mamba_{block_idx}_{layer_idx}_ssm"

    conv_state = cache.get(conv_key)
    ssm_state = cache.get(ssm_key)
    
    if conv_state is None:
        conv_state = torch.zeros(B, self.d_inner, self.conv1d.kernel_size[0] - 1,
                                 device=x.device, dtype=x.dtype)
        ssm_state = torch.zeros(B, self.d_inner, self.d_state,
                                device=x.device, dtype=x.dtype)

    # In-projection
    xz = self.in_proj(x)                         # (B, 1, 2*d_inner)
    x_in, z = torch.split(xz, self.d_inner, dim=-1)

    # 1. Update 1D Conv state & apply convolution (CRITICAL: added bias)
    conv_state = torch.cat([conv_state, x_in.transpose(1, 2)], dim=-1)
    x_conv = F.conv1d(conv_state, self.conv1d.weight, bias=self.conv1d.bias,
                      groups=self.conv1d.groups, padding=0)
    x_conv = x_conv.squeeze(-1)                  # (B, d_inner)
    x_conv = F.silu(x_conv)

    # 2. State-Space Projections
    dt_rank = self.dt_proj.in_features
    x_dbl = self.x_proj(x_conv.unsqueeze(1))     # (B, 1, dt_rank + 2*d_state)
    dt, B_proj, C_proj = torch.split(x_dbl, [dt_rank, self.d_state, self.d_state], dim=-1)
    
    dt = F.softplus(self.dt_proj(dt)).squeeze(1) # (B, d_inner)
    B_proj = B_proj.squeeze(1)                   # (B, d_state)
    C_proj = C_proj.squeeze(1)                   # (B, d_state)

    # 3. Discretization & SSM State Update
    A = -torch.exp(self.A_log)                   # (d_inner, d_state)
    dA = torch.exp(dt.unsqueeze(-1) * A)         # (B, d_inner, d_state)
    dB = dt.unsqueeze(-1) * B_proj.unsqueeze(1)  # (B, d_inner, d_state)
    
    u = x_conv.unsqueeze(-1)                     # (B, d_inner, 1)
    ssm_state = ssm_state * dA + dB * u          # (B, d_inner, d_state)

    # 4. Output Projection (FIXED EINSUM STRING)
    y = torch.einsum('bs,bds->bd', C_proj, ssm_state) + self.D * x_conv
    y = y * F.silu(z.squeeze(1))                 
    out = self.out_proj(y.unsqueeze(1))          # (B, 1, D)

    # Trim conv_state to maintain rolling window
    conv_state = conv_state[:, :, -(self.conv1d.kernel_size[0] - 1):]
    
    cache[conv_key] = conv_state
    cache[ssm_key] = ssm_state
    
    return out, cache

def swgqa_step(self, x, cache, block_idx, layer_idx):
    """
    Single-token step for SWGQA with O(1) pruned KV caching.
    x: (B, 1, D)
    """
    B, L, D = x.shape
    pos_key = f"pos_{block_idx}_{layer_idx}"
    pos = cache.get(pos_key, 0)

    # Project Q, K, V for current token
    q = self.q_proj(x).view(B, L, self.q_heads, self.head_dim).transpose(1, 2)
    k = self.k_proj(x).view(B, L, self.kv_heads, self.head_dim).transpose(1, 2)
    v = self.v_proj(x).view(B, L, self.kv_heads, self.head_dim).transpose(1, 2)

    # Apply RoPE mapped to the absolute sequence position
    q_rot, k_rot = apply_rotary_emb(q, k,
                                    self.freqs_cos[pos:pos+1],
                                    self.freqs_sin[pos:pos+1])

    k_key = f"swgqa_{block_idx}_{layer_idx}_k"
    v_key = f"swgqa_{block_idx}_{layer_idx}_v"
    
    k_cache = cache.get(k_key, [])
    v_cache = cache.get(v_key, [])
    
    k_cache.append(k_rot)
    v_cache.append(v)
    
    # MEMORY OPTIMIZATION: Prune oldest non-scratchpad token out of the window
    if len(k_cache) > self.num_scratchpad + self.window_size:
        k_cache.pop(self.num_scratchpad) # Index 256 represents the oldest sliding window token
        v_cache.pop(self.num_scratchpad)
        
    cache[k_key] = k_cache
    cache[v_key] = v_cache
    cache[pos_key] = pos + 1

    # Concatenate active KVs (O(1) complexity since size is capped at 512 tokens)
    keys = torch.cat(k_cache, dim=2)
    vals = torch.cat(v_cache, dim=2)

    # GQA Repetition
    keys = keys.repeat_interleave(self.q_heads // self.kv_heads, dim=1)
    vals = vals.repeat_interleave(self.q_heads // self.kv_heads, dim=1)

    # Calculate Attention (No mask needed since pruning handles eviction inherently)
    scores = torch.matmul(q_rot, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
    attn = F.softmax(scores, dim=-1)
    
    out = torch.matmul(attn, vals).transpose(1, 2).contiguous().view(B, 1, -1)
    return self.o_proj(out), cache

# Monkey-patch inference steps into class definitions
Mamba3Layer.step = mamba_step
SWGQA.step = swgqa_step


# ---------------------------------------------------------
# 2. INFERENCE LOGIC & SAMPLING
# ---------------------------------------------------------

def inference_step(model, hidden, cache):
    """Passes a single token (B, 1, D) through all architectural layers."""
    for block_idx, block in enumerate(model.blocks):
        for layer_idx, layer in enumerate(block.layers):
            
            if layer_idx == 2:  # SWGQA + Shared MLP
                x_norm = layer['norm1'](hidden)
                attn_out, cache = layer['attn'].step(x_norm, cache, block_idx, layer_idx)
                hidden = hidden + attn_out
                
                x_norm2 = layer['norm2'](hidden)
                mlp_out = layer['shared_mlp'](x_norm2)
                hidden = hidden + mlp_out
                
            else:               # Mamba + MoE
                x_norm = layer['norm1'](hidden)
                mamba_out, cache = layer['mamba'].step(x_norm, cache, block_idx, layer_idx)
                hidden = hidden + mamba_out
                
                x_norm2 = layer['norm2'](hidden)
                moe_out, _, _ = layer['moe'](x_norm2) 
                hidden = hidden + moe_out

    return model.norm(hidden), cache

def sample(logits, temperature=1.0, top_k=None):
    """1D Sampling logic."""
    logits = logits[0, -1, :] # Flatten from (1, 1, Vocab) to (Vocab,)
    
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1).item()
        
    probs = F.softmax(logits / temperature, dim=-1)
    
    if top_k is not None:
        values, indices = torch.topk(probs, top_k, dim=-1)
        probs = torch.zeros_like(probs).scatter_(-1, indices, values)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        
    return torch.multinomial(probs, 1).item()

@torch.no_grad()
def generate(model, prompt_tokens, max_new_tokens=50, device='cpu', temperature=0.7, top_k=40):
    model.eval()
    model.to(device)
    
    cache = {}
    
    # 1. Pre-fill the Virtual Scratchpad
    for i in range(model.config.num_scratchpad):
        token_emb = model.scratchpad[i].view(1, 1, -1).to(device)
        _, cache = inference_step(model, token_emb, cache)

    # 2. Pre-fill the Actual Prompt
    generated = prompt_tokens.copy()
    for idx, token_id in enumerate(prompt_tokens):
        tok_tensor = torch.tensor([[token_id]], device=device)
        token_emb = model.embedding(tok_tensor)
        
        hidden, cache = inference_step(model, token_emb, cache)
        
        # Capture logits on the very last prompt token to begin generation
        if idx == len(prompt_tokens) - 1:
            logits = model.lm_head(hidden)
            next_token = sample(logits, temperature, top_k)
            generated.append(next_token)

    # 3. Autoregressive Decoding Loop
    for _ in range(max_new_tokens - 1):
        tok_tensor = torch.tensor([[generated[-1]]], device=device)
        token_emb = model.embedding(tok_tensor)
        
        hidden, cache = inference_step(model, token_emb, cache)
        logits = model.lm_head(hidden)
        
        next_token = sample(logits, temperature, top_k)
        generated.append(next_token)
        
        if next_token == 0:  # Assumes 0 is EOS token ID
            break

    return generated

# ---------------------------------------------------------
# 3. EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    from transformers import AutoTokenizer # <-- Added Tokenizer
    
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = PerfectSLM(config)
    
    # Load Checkpoint
    ckpt_path = config.resume_from_checkpoint 
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    else:
        print("Warning: No checkpoint found, generating with random weights.")

    model.lm_head.weight = model.embedding.weight  # Enforce weight tying

    # --- TOKENIZER SETUP ---
    # Put the exact tokenizer name you used to build your .bin files here!
    tokenizer_name = "jamba_bbpe_tokenizer.json" 
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_name)

    # --- PROMPTS ---
    prompt_text = "There was a boy named Om who"
    
    print(f"\nPrompt: '{prompt_text}'")
    print(f"Generating on {device}...")
    
    # Convert text to tokens
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    
    # Generate 15 next tokens
    output_tokens = generate(model, prompt_tokens, max_new_tokens=200, device=device, temperature=0.3, top_k=40)
    
    # Convert tokens back to text
    generated_text = tokenizer.decode(output_tokens)
    print("\n--- Final Output ---")
    print(generated_text)
    print("--------------------")