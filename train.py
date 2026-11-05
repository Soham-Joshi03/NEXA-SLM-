import os
import math
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import RMSNorm
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup

# Ensure the official mamba-ssm library is installed
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    raise ImportError("Please install the official library: pip install mamba-ssm")

# ==========================================
# 1. STATIC CONFIGURATION
# ==========================================
class Config:
    # Model Specs
    d_model = 512
    n_super_blocks = 2  # Total 10 sub-layers
    vocab_size = 32768
    max_seq_len = 2048
    num_scratchpad = 256
    
    # Mamba-3 Specs
    mamba_expand = 2
    mamba_d_state = 16
    mamba_d_inner = d_model * mamba_expand
    mamba_dt_rank = math.ceil(d_model / 16)
    mamba_conv_size = 4
    
    # SW-GQA Specs
    q_heads = 8
    kv_heads = 2
    head_dim = 64
    window_size = 256
    
    # MoE Specs
    num_experts = 8
    num_active_experts = 2
    moe_ffn_dim = 1024
    router_aux_loss_coef = 0.01
    router_z_loss_coef = 0.001
    
    # Training Specs (Tuned for ~1M Tokens per Global Step on 2 GPUs)
    batch_size_per_gpu = 4               
    gradient_accumulation_steps = 64       
    max_lr = 3e-4                         
    min_lr = 3e-5
    weight_decay = 0.01
    warmup_steps_pct = 0.10               
    total_global_steps = 10000
    eval_every = 50                       
    save_latest_every = 20
    eval_iters = 1000                      # Cap validation to 100 batches (~3 sec)
    
    # Dataset File Paths
    train_bin = "/kaggle/input/datasets/ommahadeokar/dataandtok/train.bin"
    val_bin = "/kaggle/input/datasets/ommahadeokar/dataandtok/val.bin"
    
    # SINGLE FILE CHECKPOINT OPTIONS:
    checkpoint_dir = "checkpoints"
    # Set to "best_model.pt", "latest_model.pt", or None
    resume_from_checkpoint = None 

# ==========================================
# 2. DATASET (Memmap Loading)
# ==========================================
class MemmapDataset(Dataset):
    def __init__(self, bin_path, seq_len):
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"{bin_path} missing. Generate it before training.")
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        self.seq_len = seq_len
        
    def __len__(self):
        return (len(self.data) - 1) // self.seq_len
        
    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = torch.from_numpy((self.data[start:end]).astype(np.int64))
        return chunk[:-1], chunk[1:]

def cycle(loader):
    """Infinite iterator to prevent StopIteration crashes across epochs in DDP"""
    while True:
        for batch in loader:
            yield batch

# ==========================================
# 3. ARCHITECTURE COMPONENTS
# ==========================================
class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(in_features, hidden_features, bias=False)
        self.w3 = nn.Linear(hidden_features, in_features, bias=False)
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class MoELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_active = config.num_active_experts
        self.router = nn.Linear(config.d_model, self.num_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(config.d_model, config.moe_ffn_dim) for _ in range(self.num_experts)])
        self.shared_expert = SwiGLU(config.d_model, config.moe_ffn_dim)

    def forward(self, x):
        B, L, D = x.shape
        shared_out = self.shared_expert(x)
        
        logits = self.router(x)
        routing_weights = F.softmax(logits, dim=-1)
        top_weights, top_indices = torch.topk(routing_weights, self.num_active, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
        
        flat_x = x.view(-1, D)
        flat_indices = top_indices.view(-1, self.num_active)
        flat_weights = top_weights.view(-1, self.num_active)
        flat_out = torch.zeros_like(flat_x)
        
        for i in range(self.num_experts):
            expert_mask = (flat_indices == i)
            if expert_mask.any():
                token_idx, active_idx = torch.where(expert_mask)
                expert_outputs = self.experts[i](flat_x[token_idx])
                weights_i = flat_weights[token_idx, active_idx].unsqueeze(-1)
                flat_out[token_idx] += expert_outputs * weights_i
                
        out = flat_out.view(B, L, D) + shared_out
        
        z_loss = torch.mean(torch.logsumexp(logits, dim=-1)**2)
        expert_mask_oh = F.one_hot(top_indices, num_classes=self.num_experts).float()
        expert_fraction = expert_mask_oh.mean(dim=(0, 1)).sum(dim=0)
        routing_prob = routing_weights.mean(dim=(0, 1))
        aux_loss = torch.sum(expert_fraction * routing_prob) * self.num_experts
        
        return out, z_loss, aux_loss

class Mamba3Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_inner = config.mamba_d_inner
        self.d_state = config.mamba_d_state
        
        self.in_proj = nn.Linear(config.d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, config.mamba_conv_size, groups=self.d_inner, padding=config.mamba_conv_size - 1)
        self.x_proj = nn.Linear(self.d_inner, config.mamba_dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(config.mamba_dt_rank, self.d_inner, bias=True)
        
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, config.d_model, bias=False)

    def forward(self, x):
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        
        x_conv = self.conv1d(x_in.transpose(1, 2))[:, :, :L]
        x_conv = F.silu(x_conv)
        
        x_dbl = self.x_proj(x_conv.transpose(1, 2))
        dt, B_proj, C_proj = torch.split(x_dbl, [Config.mamba_dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        
        A = -torch.exp(self.A_log.float())
        
        y = selective_scan_fn(
            u=x_conv,
            delta=dt.transpose(1, 2).to(x_conv.dtype),
            A=A,
            B=B_proj.transpose(1, 2),
            C=C_proj.transpose(1, 2),
            D=self.D.float(),
            z=None,
            delta_bias=None,
            delta_softplus=False
        )
        
        y = y.transpose(1, 2)
        return self.out_proj(y * F.silu(z))

def apply_rotary_emb(xq, xk, freqs_cos, freqs_sin):
    xq_r, xq_i = xq.float().reshape(*xq.shape[:-1], -1, 2).unbind(-1)
    xk_r, xk_i = xk.float().reshape(*xk.shape[:-1], -1, 2).unbind(-1)
    
    freqs_cos = freqs_cos.view(1, 1, xq.shape[2], xq.shape[3] // 2)
    freqs_sin = freqs_sin.view(1, 1, xq.shape[2], xq.shape[3] // 2)
    
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos
    
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class SWGQA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.q_heads, self.kv_heads = config.q_heads, config.kv_heads
        self.head_dim, self.window_size = config.head_dim, config.window_size
        self.num_scratchpad = config.num_scratchpad
        
        self.q_proj = nn.Linear(config.d_model, self.q_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, self.kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.q_heads * self.head_dim, config.d_model, bias=False)
        
        dim, end = self.head_dim, config.max_seq_len + self.num_scratchpad
        freqs = 1.0 / (10000.0 ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end, dtype=torch.float32)
        freqs = torch.outer(t, freqs).float()
        self.register_buffer("freqs_cos", torch.cos(freqs))
        self.register_buffer("freqs_sin", torch.sin(freqs))
        
    def forward(self, x):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.kv_heads, self.head_dim).transpose(1, 2)
        
        q, k = apply_rotary_emb(q, k, self.freqs_cos[:L], self.freqs_sin[:L])
        
        k = k.repeat_interleave(self.q_heads // self.kv_heads, dim=1)
        v = v.repeat_interleave(self.q_heads // self.kv_heads, dim=1)
        
        mask_tril = torch.tril(torch.ones(L, L, dtype=torch.bool, device=x.device))
        mask_sw = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=-self.window_size + 1)
        mask_scratchpad = torch.zeros(L, L, dtype=torch.bool, device=x.device)
        mask_scratchpad[:, :self.num_scratchpad] = True
        
        valid_attend = mask_tril & (mask_sw | mask_scratchpad)
        attn_bias = torch.zeros(L, L, device=x.device).masked_fill_(~valid_attend, float('-inf'))
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores + attn_bias.unsqueeze(0).unsqueeze(0), dim=-1)
        
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, -1)
        return self.o_proj(out)

class SuperBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(5):
            if i == 2:
                self.layers.append(nn.ModuleDict({
                    'norm1': RMSNorm(config.d_model), 'attn': SWGQA(config),
                    'norm2': RMSNorm(config.d_model), 'shared_mlp': SwiGLU(config.d_model, config.moe_ffn_dim)
                }))
            else:
                self.layers.append(nn.ModuleDict({
                    'norm1': RMSNorm(config.d_model), 'mamba': Mamba3Layer(config),
                    'norm2': RMSNorm(config.d_model), 'moe': MoELayer(config)
                }))
                
    def forward(self, x):
        tot_z, tot_aux = 0.0, 0.0
        for i, layer in enumerate(self.layers):
            if i == 2:
                x = x + layer['attn'](layer['norm1'](x))
                x = x + layer['shared_mlp'](layer['norm2'](x))
            else:
                x = x + layer['mamba'](layer['norm1'](x))
                moe_out, z, aux = layer['moe'](layer['norm2'](x))
                x = x + moe_out
                tot_z += z; tot_aux += aux
        return x, tot_z, tot_aux

class PerfectSLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.scratchpad = nn.Parameter(torch.randn(config.num_scratchpad, config.d_model) * 0.02)
        
        self.blocks = nn.ModuleList([SuperBlock(config) for _ in range(config.n_super_blocks)])
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight 
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self, x, targets=None):
        B = x.size(0)
        hidden = torch.cat([self.scratchpad.unsqueeze(0).expand(B, -1, -1), self.embedding(x)], dim=1)
        
        total_z_loss, total_aux_loss = 0.0, 0.0
        for block in self.blocks:
            hidden, z, aux = block(hidden)
            total_z_loss += z; total_aux_loss += aux
            
        hidden = self.norm(hidden)[:, self.config.num_scratchpad:, :]
        logits = self.lm_head(hidden)
        
        ce_loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), targets.view(-1)) if targets is not None else None
        return ce_loss, total_z_loss, total_aux_loss

# ==========================================
# 4. SINGLE-FILE CHECKPOINT HELPER FUNCTIONS
# ==========================================
def save_single_file_checkpoint(filepath, global_step, tokens_seen, best_val_loss, accelerator, model, optimizer, scheduler):
    """Saves the model, optimizer, and training states into a single .pt file safely in DDP"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    unwrapped_model = accelerator.unwrap_model(model)
    
    checkpoint = {
        'global_step': global_step,
        'tokens_seen': tokens_seen,
        'best_val_loss': best_val_loss,
        'model_state_dict': unwrapped_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    # accelerator.save ensures only Rank 0 writes to disk
    accelerator.save(checkpoint, filepath)

# ==========================================
# 5. TRAINING LOOP
# ==========================================
def main():
    config = Config()
    
    # 1. Copy dataset to fast local NVMe disk to prevent FUSE IO crashes
    local_train = "/kaggle/working/local_train.bin"
    if not os.path.exists(local_train) and os.path.exists(config.train_bin):
        print("Copying 9.5GB dataset to local NVMe... (takes ~2 mins)")
        shutil.copy(config.train_bin, local_train)
        config.train_bin = local_train

    accelerator = Accelerator(
        mixed_precision='fp16', 
        gradient_accumulation_steps=config.gradient_accumulation_steps
    )
    
    train_loader = DataLoader(MemmapDataset(config.train_bin, config.max_seq_len), batch_size=config.batch_size_per_gpu, shuffle=True, drop_last=True)
    val_loader = DataLoader(MemmapDataset(config.val_bin, config.max_seq_len), batch_size=config.batch_size_per_gpu, shuffle=False)
    
    model = PerfectSLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.max_lr, weight_decay=config.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(config.warmup_steps_pct * config.total_global_steps), num_training_steps=config.total_global_steps)
    
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, val_loader, scheduler)
    
    # -------------------------------------------------------------
    # RESUME FROM SINGLE CHECKPOINT FILE
    # -------------------------------------------------------------
    global_step = 0
    tokens_seen = 0
    best_val_loss = float('inf')
    
    if config.resume_from_checkpoint:
        ckpt_path = os.path.join(config.checkpoint_dir, config.resume_from_checkpoint)
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            
            # Load model state dict
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.load_state_dict(checkpoint['model_state_dict'])
            unwrapped_model.lm_head.weight = unwrapped_model.embedding.weight  # Re-tie weights
            
            # Load optimizer & scheduler
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Load step state
            global_step = checkpoint['global_step']
            tokens_seen = checkpoint['tokens_seen']
            best_val_loss = checkpoint['best_val_loss']
            accelerator.print(f"✅ Successfully Resumed from single-file: {ckpt_path} (Global Step {global_step})")
        else:
            accelerator.print(f"⚠️ Checkpoint file {ckpt_path} not found. Starting from scratch.")

    # -------------------------------------------------------------
    # MASTER TRAINING LOOP
    # -------------------------------------------------------------
    model.train()
    border = "-" * 115
    accelerator.print(f"\n{border}")
    accelerator.print(f"| {'Type':^12} | {'Global Step':^11} | {'CE Loss':^10} | {'Z-Loss':^10} | {'Aux Loss':^10} | {'Total Loss':^11} | {'Tokens Seen':^15} |")
    accelerator.print(f"{border}")

    train_iter = cycle(train_loader)
    
    while global_step < config.total_global_steps:
        for micro_step in range(config.gradient_accumulation_steps):
            x, y = next(train_iter)
            
            with accelerator.accumulate(model):
                ce_loss, z_loss, aux_loss = model(x, y)
                total_loss = ce_loss + (config.router_z_loss_coef * z_loss) + (config.router_aux_loss_coef * aux_loss)
                
                accelerator.backward(total_loss)
                
                if accelerator.sync_gradients:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
            tokens_seen += x.numel() * accelerator.num_processes
            lr = scheduler.get_last_lr()[0]
            
            accelerator.print(f"| {'Micro ' + str(micro_step):<12} | {global_step + 1:11,d} | {ce_loss.item():10.4f} | {z_loss.item():10.4f} | {aux_loss.item():10.4f} | {total_loss.item():11.4f} | {tokens_seen:15,d} |")
            
        global_step += 1
        
        accelerator.print(f"| {'[ GLOBAL ]':<12} | {global_step:11,d} | {ce_loss.item():10.4f} | {z_loss.item():10.4f} | {aux_loss.item():10.4f} | {total_loss.item():11.4f} | {tokens_seen:15,d} |\n{border}")

        # Evaluation & Best Checkpoint (Single File)
        if global_step % config.eval_every == 0:
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for i, (vx, vy) in enumerate(val_loader):
                    if i >= config.eval_iters:
                        break
                    v_ce, _, _ = model(vx, vy)
                    val_loss += accelerator.gather(v_ce).mean().item()
                    
            val_loss /= config.eval_iters
            accelerator.print(f"🌟 [VALIDATION] Global Step {global_step} | Val Loss: {val_loss:.4f} | Best Loss: {best_val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(config.checkpoint_dir, "best_model.pt")
                save_single_file_checkpoint(best_path, global_step, tokens_seen, best_val_loss, accelerator, model, optimizer, scheduler)
                accelerator.print(f"🚀 New Best Single-File Checkpoint Saved to {best_path}!")
                
            accelerator.print(border)
            model.train()

        # Save Latest Checkpoint (Single File)
        if global_step % config.save_latest_every == 0:
            latest_path = os.path.join(config.checkpoint_dir, "latest_model.pt")
            save_single_file_checkpoint(latest_path, global_step, tokens_seen, best_val_loss, accelerator, model, optimizer, scheduler)
            accelerator.print(f"💾 Latest Single-File Checkpoint Saved to {latest_path} at Step {global_step}")
            accelerator.print(border)

if __name__ == "__main__":
    main()
