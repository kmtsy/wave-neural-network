
"""
Rough Model Architecture:
  • RPM grid & static-feature constants
  • Multi-scale CNN stem
  • Transformer encoder
  • GRU prediction head
  • Physics-inspired RPM positional encoding
"""
import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------------------- #
#                             CONSTANTS                                #
# -------------------------------------------------------------------- #
RPM_GRID   = np.linspace(4_000, 13_000, 50, dtype=np.float32)   # 50-point grid[1]
SEQ_LEN    = len(RPM_GRID)
STATIC_DIM = 4                                                  # hdr1,hdr2,runner,plenum

RPM_MEAN   = float(RPM_GRID.mean())
RPM_STD    = float(RPM_GRID.std())

# -------------------------------------------------------------------- #
#                    PHYSICS-INSPIRED RPM ENCODING                     #
# -------------------------------------------------------------------- #
def physics_rpm_encoding(rpm: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    Vectorised sinusoidal encoding for engine speed.

    rpm      : (B, L) tensor of *raw* RPM values
    d_model  : embedding dimension
    returns  : (B, L, d_model)
    """
    freqs = torch.tensor([1, 2, 4, 6, 8, 12, 16, 24],
                         dtype=torch.float32, device=rpm.device)
    B, L = rpm.shape
    pe   = torch.zeros(B, L, d_model, device=rpm.device)
    for i, f in enumerate(freqs):
        if i * 2 + 1 >= d_model:
            break
        angle = rpm * f * 2.0 * math.pi / 13_000.0
        pe[:, :, i * 2]     = torch.sin(angle)
        pe[:, :, i * 2 + 1] = torch.cos(angle)
    return pe                                                   # (B,L,d_model)


# -------------------------------------------------------------------- #
#                         MODEL BUILDING BLOCKS                        #
# -------------------------------------------------------------------- #
class MultiScaleDilatedStem(nn.Module):
    def __init__(self, out_ch: int = 256) -> None:
        super().__init__()
        ks_dil = [(3, 1), (5, 1), (7, 1), (3, 2), (5, 2), (3, 4)]
        n_br   = len(ks_dil)
        base   = out_ch // n_br
        extra  = out_ch - base * n_br
        chans  = [base + 1 if i < extra else base for i in range(n_br)]

        self.branches = nn.ModuleList(
            nn.Conv1d(1, c, k, padding=d * (k // 2), dilation=d)
            for (k, d), c in zip(ks_dil, chans)
        )
        self.post = nn.Sequential(
            nn.Conv1d(sum(chans), out_ch, 1),
            nn.BatchNorm1d(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [br(x) for br in self.branches]
        return F.gelu(self.post(torch.cat(feats, 1)))           # (B,out_ch,50)


class LocalSelfAttention(nn.Module):
    def __init__(self, dim: int, nhead: int, window: int) -> None:
        super().__init__()
        assert dim % nhead == 0
        self.nhead, self.dk, self.window = nhead, dim // nhead, window
        self.qkv  = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim,     bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, self.nhead, self.dk).transpose(1, 2) for t in (q, k, v)]
        attn    = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)

        idx   = torch.arange(T, device=x.device)
        mask  = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1)) > self.window
        attn  = attn.masked_fill(mask, -1e9)

        out = (attn.softmax(-1) @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class AdaptiveSparseSelfAttention(nn.Module):
    def __init__(self, dim: int, nhead: int, topk: int) -> None:
        super().__init__()
        assert dim % nhead == 0
        self.nhead, self.dk, self.topk = nhead, dim // nhead, topk
        self.qkv   = nn.Linear(dim, dim * 3, bias=False)
        self.out   = nn.Linear(dim, dim,     bias=False)
        self.alpha = nn.Parameter(torch.zeros(nhead))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, self.nhead, self.dk).transpose(1, 2) for t in (q, k, v)]
        attn    = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)

        dense  = attn.softmax(-1) @ v
        if self.topk >= T:
            sparse = dense
        else:
            idx   = attn.topk(self.topk, -1).indices
            mask  = torch.ones_like(attn, dtype=torch.bool)
            mask.scatter_(-1, idx, False)
            sparse = (attn.masked_fill(mask, -1e9).softmax(-1) @ v)

        gate = torch.sigmoid(self.alpha).view(1, self.nhead, 1, 1)
        out  = (gate * dense + (1.0 - gate) * sparse).transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        nhead: int,
        topk: int,
        local_window: Optional[int],
        mlp_ratio: float = 8.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.ln1  = nn.LayerNorm(dim)
        self.attn = (
            LocalSelfAttention(dim, nhead, local_window)
            if local_window is not None
            else AdaptiveSparseSelfAttention(dim, nhead, topk)
        )
        self.ln2  = nn.LayerNorm(dim)
        self.ffn  = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.alpha * self.attn(self.ln1(x))
        x = x + self.alpha * self.ffn(self.ln2(x))
        return x


# -------------------------------------------------------------------- #
#                           FULL TORQUE MODEL                          #
# -------------------------------------------------------------------- #
class ASTorqueModel(nn.Module):
    def __init__(
        self,
        *,
        hid: int = 256,
        nhead: int = 4,
        nlayer: int = 8,
        topk: int = 8,
        local_window: Optional[int] = 4,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.hid = hid

        self.stem        = MultiScaleDilatedStem(hid)
        self.static_proj = nn.Linear(STATIC_DIM, hid)
        self.rpm_proj    = nn.Linear(1, hid)     # learned embedding

        self.pos_emb = nn.Parameter(torch.zeros(1, SEQ_LEN, hid))
        self.dropout = nn.Dropout(dropout)
        self.fusion  = nn.Linear(hid * 3, hid)

        self.skip_proj = nn.Linear(hid, 1)
        self.alpha     = nn.Parameter(torch.tensor(0.5))

        self.layers = nn.ModuleList(
            TransformerBlock(
                dim=hid,
                nhead=nhead,
                topk=topk,
                local_window=local_window,
                dropout=dropout,
            )
            for _ in range(nlayer)
        )

        self.chain_rnn = nn.GRU(hid, hid, batch_first=True)
        self.chain_act = nn.GELU()
        self.head      = nn.Linear(hid, 1)

        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    # ---------------------------------------------------------------- #
    #                            FORWARD                                #
    # ---------------------------------------------------------------- #
    def forward(self, rpm_norm: torch.Tensor, statics: torch.Tensor) -> torch.Tensor:
        """
        rpm_norm : (B,1,50) – normalised RPM grid
        statics  : (B,4)    – hdr1,hdr2,runner,plenum
        returns  : (B,50)   – normalised torque curve
        """
        # CNN stem on raw RPM waveform
        stem = self.stem(rpm_norm).permute(0, 2, 1)                 # (B,50,hid)

        # ----------------------------------------------------------------
        # Learned + Physics-inspired RPM embedding
        # ----------------------------------------------------------------
        rpm_learn = self.rpm_proj(rpm_norm.permute(0, 2, 1))        # (B,50,hid)

        rpm_raw   = rpm_norm.squeeze(1) * RPM_STD + RPM_MEAN        # back-transform
        rpm_phys  = physics_rpm_encoding(rpm_raw, self.hid)         # (B,50,hid)[1]

        rpm_e = rpm_learn + rpm_phys                                # combined

        # Static-feature embedding
        stat_e = self.static_proj(statics).unsqueeze(1).expand(-1, SEQ_LEN, -1)

        # Fuse three sources of information
        x = self.dropout(self.fusion(torch.cat([stem, rpm_e, stat_e], -1))) + self.pos_emb

        for blk in self.layers:
            x = blk(x)

        x, _  = self.chain_rnn(x)
        x     = self.chain_act(x)
        y_main = self.head(x).squeeze(-1)
        y_skip = self.skip_proj(stem).squeeze(-1)
        return y_main + self.alpha * y_skip


__all__: Tuple[str, ...] = (
    "RPM_GRID",
    "SEQ_LEN",
    "STATIC_DIM",
    "ASTorqueModel",
    "physics_rpm_encoding",
)
