import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
import xformers.ops as xops
from omegaconf import OmegaConf
 
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim
        
    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None,None,:]
 
class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
 
    def timestep_embedding(self, t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
 
    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)
 
class DDiTBlock(nn.Module):
    """SEDD的Transformer块，使用xformers替代flash-attention"""
    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        # Attention部分
        self.norm1 = LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)
        
        # MLP部分
        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True)
        )
        self.dropout2 = nn.Dropout(dropout)
        
        # AdaLN模块
        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()
 
    def forward(self, x, rotary_cos_sin, c, seqlens=None):
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # AdaLN模块参数
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
        
        # Self-attention
        x_skip = x
        x = modulate(self.norm1(x), shift_msa, scale_msa)
        
        # 分别计算Q、K、V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 应用Rotary Position Embedding
        q = rearrange(q, 'b s (h d) -> b s h d', h=self.n_heads)
        k = rearrange(k, 'b s (h d) -> b s h d', h=self.n_heads)
        v = rearrange(v, 'b s (h d) -> b s h d', h=self.n_heads)
        
        cos, sin = rotary_cos_sin
        q, k, v = apply_rotary_pos_emb(q, k, v, cos[:,:,0].to(q.dtype), sin[:,:,0].to(q.dtype))
        
        # 使用xformers的attention
        x = xops.memory_efficient_attention(
            q.reshape(batch_size, seq_len, self.n_heads, self.head_dim),
            k.reshape(batch_size, seq_len, self.n_heads, self.head_dim),
            v.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        )
        
        x = rearrange(x, 'b s h d -> b s (h d)')
        x = self.dropout1(self.attn_out(x))
        x = x_skip + gate_msa * x
        
        # MLP
        x_skip = x
        x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.dropout2(self.mlp(x))
        x = x_skip + gate_mlp * x
        
        return x
 
class SEDD(nn.Module):
    """SEDD主模型类"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.vocab_embed = nn.Embedding(config.tokens, config.hidden_size)
        self.sigma_map = TimestepEmbedder(config.hidden_size)
        
        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(config.hidden_size // config.n_heads)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DDiTBlock(
                config.hidden_size, 
                config.n_heads,
                config.hidden_size,
                dropout=config.dropout
            ) for _ in range(config.n_blocks)
        ])
        
        # Output layers
        self.norm_final = LayerNorm(config.hidden_size)
        self.linear = nn.Linear(config.hidden_size, config.tokens)
        
        self.scale_by_sigma = config.get("scale_by_sigma", True)
        
    def forward(self, indices, sigma):
        x = self.vocab_embed(indices)
        c = F.silu(self.sigma_map(sigma))
        
        rotary_cos_sin = self.rotary_emb(x)
        
        # Transformer blocks forward
        for block in self.blocks:
            x = block(x, rotary_cos_sin, c)
            
        x = self.norm_final(x)
        x = self.linear(x)
        
        # Scale output by sigma if needed
        if self.scale_by_sigma and self.config.graph.type == "absorb":
            esigm1_log = torch.where(
                sigma < 0.5, 
                torch.expm1(sigma),
                sigma.exp() - 1
            ).log().to(x.dtype)[:, None, None]
            x = x - esigm1_log - math.log(x.shape[-1] - 1)
            
        return x
 
class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding实现"""
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
        
    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1,1,3,1,1)
            self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1,1,3,1,1)
            self.cos_cached[:,:,2,:,:].fill_(1.)
            self.sin_cached[:,:,2,:,:].fill_(0.)
            
        return self.cos_cached, self.sin_cached
 
def modulate(x, shift, scale):
    return x * (1 + scale) + shift
 
def apply_rotary_pos_emb(q, k, v, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin), v
 
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)