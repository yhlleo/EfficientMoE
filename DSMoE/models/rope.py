import torch
import torch.nn as nn
import torch.nn.functional as F

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.split(x.size(-1) // 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def _apply_rope_1d(
    q: torch.Tensor, 
    k: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # q,k: [B, H, S, Dh]
    # cos,sin: [S, Dh]  -> broadcast to [1,1,S,Dh]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return q, k

class DeepseekV3RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )
        self.max_seq_len_cached = None

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.outer(t, self.inv_freq.to(t.device))
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

def _apply_rope_2d(q, k, cos_y, sin_y, cos_x, sin_x):
    """
    q,k: [B, Hh, HW, D]
    cos_y/sin_y/cos_x/sin_x: [HW, D/2] each
    """
    B, Hh, HW, D = q.shape
    D_half = D // 2

    qy, qx = q.split(D_half, dim=-1)  # [B,Hh,HW,D/2]
    ky, kx = k.split(D_half, dim=-1)

    # broadcast to [1,1,HW,D/2]
    cy = cos_y[None, None, :, :]
    sy = sin_y[None, None, :, :]
    cx = cos_x[None, None, :, :]
    sx = sin_x[None, None, :, :]

    qy = (qy * cy) + (_rotate_half(qy) * sy)
    ky = (ky * cy) + (_rotate_half(ky) * sy)
    qx = (qx * cx) + (_rotate_half(qx) * sx)
    kx = (kx * cx) + (_rotate_half(kx) * sx)

    q = torch.cat([qy, qx], dim=-1)
    k = torch.cat([ky, kx], dim=-1)
    return q, k

class RotaryEmbedding2D(nn.Module):
    """
    2-D RoPE cache builder for a single head-dim D.
    Produces per-axis cos/sin of shape [HW, D/2] each (for y and x halves).
    """
    def __init__(self, dim: int, HW: int = 16, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.HW = HW
        self.base = base
        if (self.dim % 4) != 0:
            raise ValueError("2D RoPE requires head_dim % 4 == 0")

        # Per-axis half-dim:
        # total D -> split into [D/2 | D/2] for (y | x)
        # within each axis half, we rotate in pairs -> inv_freq length is (D/2)/2 = D/4
        self.axis_dim = self.dim // 2
        self.pair_dim = self.axis_dim // 2  # number of distinct frequencies per axis

        # Build per-axis frequency vector (length D/4). Device/dtype will follow on forward().
        inv = torch.arange(0, self.pair_dim).float() / self.pair_dim
        inv_freq = 1.0 / (self.base ** inv)  # [D/4]
        self.register_buffer("inv_freq_base", inv_freq, persistent=False)

    def forward(self, x):
        """
        Returns:
          cos_y, sin_y: [HW, D/2]
          cos_x, sin_x: [HW, D/2]
        """
        device = x.device
        inv_freq = self.inv_freq_base.to(device)

        ys = torch.arange(self.HW, device=inv_freq.device, dtype=inv_freq.dtype)  # [H]
        xs = torch.arange(self.HW, device=inv_freq.device, dtype=inv_freq.dtype)  # [W]
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")                      # [H,W]
        pos_y = yy.reshape(-1)                                              # [HW]
        pos_x = xx.reshape(-1)                                              # [HW]

        # phases per axis: [HW, D/4]
        freqs_y = torch.einsum("p,d->pd", pos_y, inv_freq)
        freqs_x = torch.einsum("p,d->pd", pos_x, inv_freq)

        # duplicate to fill axis_dim (so rotate_half can split pairs inside axis half)
        # -> [HW, D/2]
        emb_y = torch.cat([freqs_y, freqs_y], dim=-1)
        emb_x = torch.cat([freqs_x, freqs_x], dim=-1)

        cos_y, sin_y = emb_y.cos(), emb_y.sin()
        cos_x, sin_x = emb_x.cos(), emb_x.sin()
        return cos_y, sin_y, cos_x, sin_x  # each [HW, D/2]