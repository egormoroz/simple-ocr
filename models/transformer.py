import torch, torch.nn as nn, torch.nn.functional as F
from einops import rearrange


def sdp_attn(q, k, v, is_causal, n_embd, n_heads):
    nh, hc = n_heads, n_embd // n_heads
    q = rearrange(q, 'b t (nh hc) -> b nh t hc', nh=nh, hc=hc)
    k = rearrange(k, 'b t (nh hc) -> b nh t hc', nh=nh, hc=hc)
    v = rearrange(v, 'b t (nh hc) -> b nh t hc', nh=nh, hc=hc)

    attn = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    attn = rearrange(attn, 'b nh t hc -> b t (nh hc)')
    return attn


def patchify(x, patch_size, pad_val=None):
    H, W = x.shape[-2], x.shape[-1]
    pw, ph = patch_size
    dy, dx = H % ph, W % pw
    if pad_val is not None:  
        pad_left, pad_top = dx // 2, dy // 2
        pad_right, pad_bot = dx - pad_left, dy - pad_top
        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bot), value=pad_val)
    else:
        assert dx == 0 and dy == 0

    H, W = x.shape[-2], x.shape[-1]
    # split into patches, then flatten them
    x = rearrange(x, 'b c (hs h) (ws w) -> b (hs ws) (h w c)', hs=H//ph, ws=W//pw)
    return x


class SelfAttn(nn.Module):
    def __init__(self, cfg, is_causal):
        super().__init__()
        self.n_embd = cfg['n_embd']
        self.n_heads = cfg['n_heads']
        self.is_causal = is_causal
        self.QKV = nn.Linear(self.n_embd, 3*self.n_embd, bias=False)
        self.out_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x):
        q, k, v = torch.split(self.QKV(x), self.n_embd, dim=-1)
        attn = sdp_attn(q, k, v, self.is_causal, self.n_embd, self.n_heads)
        return self.out_proj(attn)


class CrossAttn(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_embd = cfg['n_embd']
        self.n_heads = cfg['n_heads']
        self.KV = nn.Linear(self.n_embd, 2*self.n_embd, bias=False)
        self.Q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.out_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, x_enc):
        k, v = torch.split(self.KV(x_enc), self.n_embd, dim=-1)
        q = self.Q(x)
        attn = sdp_attn(q, k, v, True, self.n_embd, self.n_heads)
        return self.out_proj(attn)


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        n_embd = cfg['n_embd']
        self.layers = nn.Sequential(
            nn.Linear(n_embd, n_embd*4, bias=False),
            nn.ReLU(),
            nn.Linear(n_embd*4, n_embd, bias=False),
        )

    def forward(self, x):
        return self.layers(x)


class EncoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln0 = nn.LayerNorm(cfg['n_embd'])
        self.sa = SelfAttn(cfg, is_causal=False)
        self.ln1 = nn.LayerNorm(cfg['n_embd'])
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.sa(self.ln0(x))
        x = x + self.mlp(self.ln1(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        n_embd = cfg['n_embd']
        self.ln0 = nn.LayerNorm(n_embd)
        self.sa = SelfAttn(cfg, is_causal=True)
        
        self.ln1 = nn.LayerNorm(n_embd)
        self.ca = CrossAttn(cfg)

        self.ln2 = nn.LayerNorm(n_embd)        
        self.mlp = MLP(cfg)

    def forward(self, x, x_enc):
        x = x + self.sa(self.ln0(x))
        x = x + self.ca(self.ln1(x), x_enc)
        x = x + self.mlp(self.ln2(x))
        return x


class PosEmbd2d(nn.Module):
    def __init__(self, shape):
        super().__init__()
        C, H, W = shape
        scale = C**-0.5
        self.embd_height = nn.Parameter(torch.randn(H, 1)*scale)
        self.embd_width = nn.Parameter(torch.randn(1, W)*scale)

    def forward(self):
        embd = self.embd_width + self.embd_height
        embd = rearrange(embd, 'h w -> (h w)')
        return embd


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        channels = cfg['ch_in']
        pw, ph = cfg['patch_size']
        n_ctx = cfg['n_ctx']
        n_embd = cfg['n_embd']
        n_blocks = cfg['n_blocks']
        
        self.patch_size = (pw, ph)
        self.pad_value = cfg.get('pad_value', None)

        self.embd = nn.Embedding(n_ctx, n_embd)
        self.project_patch = nn.Linear(pw*ph*channels, n_embd, bias=False)
        self.blocks = nn.ModuleList([
            EncoderBlock(cfg) for _ in range(n_blocks)
        ])
        self.ln_out = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = patchify(x, self.patch_size, self.pad_value)
        x = self.project_patch(x)
        
        device = next(self.parameters()).device
        # B, T, E = x.shape
        T = x.shape[1]
        x = x + self.embd(torch.arange(T, device=device))
            
        for block in self.blocks:
            x = block(x)
        return self.ln_out(x)


class TiedLinear(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.w = w

    def forward(self, x):
        return F.linear(x, self.w, bias=None)
        

class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        n_ctx, n_vocab = cfg['n_ctx'], cfg['n_vocab']
        n_blocks, n_embd = cfg['n_blocks'], cfg['n_embd']

        self.embd_tok = nn.Embedding(n_vocab, n_embd)
        self.embd_pos = nn.Embedding(n_ctx, n_embd)
        
        self.blocks = nn.ModuleList([
            DecoderBlock(cfg) for _ in range(n_blocks)
        ])
        self.ln_out = nn.LayerNorm(n_embd)
        self.head = TiedLinear(self.embd_tok.weight)

    def forward(self, x, x_enc):
        _, T = x.shape
        device = next(self.parameters()).device
        
        tok_embd = self.embd_tok(x)
        pos_embd = self.embd_pos(torch.arange(T, device=device))
        x = tok_embd + pos_embd

        for block in self.blocks:
            x = block(x, x_enc)

        x = self.ln_out(x)
        x = self.head(x)
        return x
