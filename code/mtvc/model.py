#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""Top-level ``Model`` (``model.core``): wire encoders + joint stack + PredFusion.

Role
----
- Own text/price tokenizers, **virtual nodes**, PredFusion.
- Shared tokenizers: SeqTF / TextTF; ``full_unified`` only swaps price → ``TokenMLP``.
- Three paper stacks (CLI ``--price_encoder``):
  * ``partial_unified`` → ``PartialUnifiedModel`` (main MTVC)
  * ``full_unified`` → same joint Encoder + price TokenMLP (news = random TextTF)
  * ``cross_attn`` → ``CrossAttentionEncoder`` (Cross-Attention ablation;
    uses ``CrossAttnPostFusion``)
- ``forward`` → movement logits for BCE in ``train.py``.
- Contrast branch (``contrast_aux.py``) reuses the same encoders + ``unitrans``.
"""
import math
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as dist
import pickle as pkl

# U(0,1) 的理论标准差，type 边多步噪声矩匹配目标
UNIFORM01_MEAN = 0.5
UNIFORM01_STD = 1.0 / math.sqrt(12.0)
MULTISTEP_UNIFORM_TOL_MEAN = 0.04
MULTISTEP_UNIFORM_TOL_STD = 0.04
MULTISTEP_UNIFORM_MIN_EDGES = 16
MULTISTEP_UNIFORM_MIN_STEPS = 1
EDGE_NOISE_CLAMP = 1e-4


def _nhead_for(d: int, prefer: int = 4) -> int:
    """Pick attention heads dividing d (prefer small for memory)."""
    d = int(d)
    h = min(int(prefer), d)
    while h > 1 and d % h != 0:
        h //= 2
    return max(1, h)


# Canonical CLI / Model names.
PRICE_ENCODER_CANONICAL = ("cross_attn", "partial_unified", "full_unified")
def normalize_price_encoder(name: str | None) -> str:
    """Return canonical ``price_encoder`` (partial_unified / cross_attn / full_unified)."""
    key = str(name or "partial_unified").strip().lower()
    if key not in PRICE_ENCODER_CANONICAL:
        raise ValueError(
            f"price_encoder must be one of {list(PRICE_ENCODER_CANONICAL)}; got {name!r}"
        )
    return key


def _mtvc_env(key: str, default: str = "") -> str:
    """Read ``MTVC_<KEY>`` (``key`` may be bare or already prefixed)."""
    k = str(key).removeprefix("MTVC_")
    v = os.environ.get(f"MTVC_{k}")
    if v is None:
        return default
    return str(v)


_MODEL_DIR = os.environ.get(
    "DUAL_TF_MODEL_DIR",
    os.path.dirname(os.path.abspath(__file__)),
)


# =============================================================================
# Cross-Attention ablation (CLI: --price_encoder cross_attn)
# Paper name: Cross-Attention. NOT partial / NOT full unified.
# This whole block (helpers → CrossAttnPostFusion → CrossAttentionEncoder) is
# Cross-Attention-only. Shared SeqTF / PredFusion / PartialUnified live below.
# Stack: price Enc + within-day news Enc
#      → causal price←news Cross Transformer (Q=price, K/V=news slots)
#      → CrossAttnPostFusion (dual-stream) → h for PredFusion
# =============================================================================

class _CrossAttnRMSNorm(nn.Module):
    """RMSNorm used inside ``CrossAttnPostFusion`` (cross_attn only)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inv = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * inv * self.weight


class _CrossAttnDualStreamAttention(nn.Module):
    """Dual-stream MHA for Cross-Attention post-fusion (price side + news side).

    Used only by ``CrossAttnPostFusion`` after the price←news Cross Transformer.
    Not part of ``PartialUnifiedModel`` / full-unified.
    """

    def __init__(self, d_s: int, d_ns: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        d_s, d_ns, nhead = int(d_s), int(d_ns), int(nhead)
        while d_s % nhead != 0 and nhead > 1:
            nhead //= 2
        self.d_s = d_s
        self.d_ns = d_ns
        self.nhead = nhead
        self.head_dim = d_s // nhead
        self.scale = self.head_dim ** -0.5
        self.qkv_s = nn.Linear(d_s, 3 * d_s, bias=False)
        self.qkv_ns = nn.Linear(d_ns, 3 * d_s, bias=False)
        self.out_s = nn.Linear(d_s, d_s, bias=False)
        self.out_ns = nn.Linear(d_s, d_ns, bias=False)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        x_s: torch.Tensor,
        x_ns: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        bsz, ls, _ = x_s.shape
        qkv_s = self.qkv_s(x_s)
        if x_ns is None or x_ns.numel() == 0 or x_ns.size(1) == 0:
            qkv = qkv_s
            lns = 0
        else:
            qkv = torch.cat([qkv_s, self.qkv_ns(x_ns)], dim=1)
            lns = int(x_ns.size(1))
        length = ls + lns
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bsz, length, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(bsz, length, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(bsz, length, self.nhead, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        try:
            from tool.numeric import stable_attn_softmax
        except ImportError:
            from tool.numeric import stable_attn_softmax
        attn = self.attn_drop(stable_attn_softmax(scores, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bsz, length, self.d_s)
        out_s = self.out_s(out[:, :ls, :])
        if lns == 0:
            return out_s, None
        return out_s, self.out_ns(out[:, ls:, :])


class _CrossAttnDualStreamFFN(nn.Module):
    """Per-stream FFN for ``CrossAttnPostFusion`` (cross_attn only)."""

    def __init__(self, d_s: int, d_ns: int, dim_ff_s: int, dim_ff_ns: int, dropout: float = 0.1):
        super().__init__()
        self.ff_s = nn.Sequential(
            nn.Linear(d_s, dim_ff_s),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff_s, d_s),
            nn.Dropout(dropout),
        )
        self.ff_ns = nn.Sequential(
            nn.Linear(d_ns, dim_ff_ns),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff_ns, d_ns),
            nn.Dropout(dropout),
        )

    def forward(
        self, x_s: torch.Tensor, x_ns: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        y_s = self.ff_s(x_s)
        if x_ns is None or x_ns.numel() == 0 or x_ns.size(1) == 0:
            return y_s, None
        return y_s, self.ff_ns(x_ns)


class CrossAttnPostFusion(nn.Module):
    """Post-cross dual-stream Fusion for the Cross-Attention ablation.

    Runs **after** the causal price←news Cross Transformer inside
    ``CrossAttentionEncoder`` (``--price_encoder cross_attn``).
    Not used by partial-unified or full-unified.

    State-dict attribute on the encoder remains ``price_news_fusion`` for ckpt compat.
    """

    def __init__(
        self,
        d_s: int,
        d_ns: int,
        nhead: int,
        dim_ff_s: int,
        dim_ff_ns: int,
        dropout: float = 0.1,
        num_layers: int = 1,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(max(1, int(num_layers))):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "norm_s1": _CrossAttnRMSNorm(d_s),
                        "norm_ns1": _CrossAttnRMSNorm(d_ns),
                        "attn": _CrossAttnDualStreamAttention(d_s, d_ns, nhead, dropout=dropout),
                        "norm_s2": _CrossAttnRMSNorm(d_s),
                        "norm_ns2": _CrossAttnRMSNorm(d_ns),
                        "ffn": _CrossAttnDualStreamFFN(
                            d_s, d_ns, dim_ff_s, dim_ff_ns, dropout=dropout
                        ),
                    }
                )
            )

    def forward(
        self, x_s: torch.Tensor, x_ns: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        for layer in self.layers:
            hs = layer["norm_s1"](x_s)
            hns = layer["norm_ns1"](x_ns) if x_ns is not None and x_ns.size(1) > 0 else None
            ds, dns = layer["attn"](hs, hns)
            x_s = x_s + ds
            if dns is not None and x_ns is not None:
                x_ns = x_ns + dns
            hs = layer["norm_s2"](x_s)
            hns = layer["norm_ns2"](x_ns) if x_ns is not None and x_ns.size(1) > 0 else None
            ds, dns = layer["ffn"](hs, hns)
            x_s = x_s + ds
            if dns is not None and x_ns is not None:
                x_ns = x_ns + dns
        return x_s, x_ns


def _make_transformer_encoder(layer: nn.Module, num_layers: int) -> nn.TransformerEncoder:
    """Shared TransformerEncoder builder (Cross-Attention Enc + SeqTF / PredFusion)."""
    try:
        return nn.TransformerEncoder(
            layer, num_layers=int(num_layers), enable_nested_tensor=False
        )
    except TypeError:
        return nn.TransformerEncoder(layer, num_layers=int(num_layers))


class CrossAttentionEncoder(nn.Module):
    """Cross-Attention ablation encoder (``--price_encoder cross_attn``).

    Paper name: Cross-Attention.
    Not ``PartialUnifiedModel`` (partial) and not full-unified.

    Pipeline
    --------
    1. Shared day pos → price Enc (K days, causal) + news Enc (within-day over M slots)
    2. Flatten news to ``K*M`` memory tokens (no day mean-pool)
    3. Cross TransformerDecoder: Q=price days, K/V=news; causal price self-attn
       + causal price←news cross (day ``i`` sees news with day ``j<=i`` only)
    4. ``CrossAttnPostFusion`` (state-dict attr ``price_news_fusion``) → h

    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_days: int = 5,
        d_news: int = 32,
        news_per_day: int = 5,
        dropout: float = 0.1,
        price_encoder_layers: int = 1,
        news_encoder_layers: int = 1,
        transformer_layers: int = 2,
        include_news: bool = True,
        include_price: bool = True,
        include_virtual: bool = True,
        num_v_window: int | None = None,
        price_news_align: str = "strict",
        news_emb_dim: int | None = None,
        use_type_emb: bool = True,
        **_ignored,
    ):
        super().__init__()
        del _ignored

        self.d_model = int(d_model)
        self.num_days = max(1, int(num_days))
        self.d_news = int(news_emb_dim) if news_emb_dim is not None else int(d_news)
        self.news_per_day = max(1, int(news_per_day))
        self.include_news = bool(include_news)
        self.include_price = bool(include_price)
        self.include_virtual = bool(include_virtual)
        self.price_news_align = str(price_news_align or "strict").lower()
        self.use_type_emb = bool(use_type_emb)
        self.num_v_window = (
            int(num_v_window)
            if num_v_window is not None
            else (self.num_days if self.include_virtual else 0)
        )
        self.price_encoder_layers = max(1, int(price_encoder_layers))
        self.news_encoder_layers = max(1, int(news_encoder_layers))
        self.transformer_layers = max(1, int(transformer_layers))

        nhead = int(nhead)
        while self.d_model % nhead != 0 and nhead > 1:
            nhead //= 2
        self.nhead = nhead
        dim_ff = self.d_model * 2
        drop = float(dropout)

        self.day_pos = nn.Embedding(self.num_days, self.d_model)
        nn.init.trunc_normal_(self.day_pos.weight, std=0.02)
        # Shared calendar-day PE: price day t and all news slots of day t use day_pos[t]
        # (same Embedding, same vector — not separate price/news PE tables).
        # 0=price, 1=news, 2=virtual/VIN
        if self.use_type_emb:
            self.type_emb = nn.Embedding(3, self.d_model)
            nn.init.trunc_normal_(self.type_emb.weight, std=0.02)
        else:
            self.type_emb = None

        self.news_up = (
            nn.Identity()
            if self.d_news == self.d_model
            else nn.Linear(self.d_news, self.d_model)
        )
        self.news_down = (
            nn.Identity()
            if self.d_news == self.d_model
            else nn.Linear(self.d_model, self.d_news)
        )

        self.state_tok = nn.Parameter(torch.zeros(1, 1, self.d_model))
        nn.init.trunc_normal_(self.state_tok, std=0.02)
        if self.include_virtual and self.num_v_window > 0:
            self.vin_window_emb = nn.Parameter(torch.zeros(self.num_v_window, self.d_model))
            nn.init.trunc_normal_(self.vin_window_emb, std=0.02)
        else:
            self.vin_window_emb = None

        def _make_transformer_encoder_stack(num_layers: int) -> nn.TransformerEncoder:
            layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=dim_ff,
                dropout=drop,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            return _make_transformer_encoder(layer, max(1, int(num_layers)))

        # price: K-day causal self-attn; news slots: within-day; news days: K-day causal
        self.price_transformer_encoder = _make_transformer_encoder_stack(
            self.price_encoder_layers
        )
        self.news_transformer_encoder = _make_transformer_encoder_stack(
            self.news_encoder_layers
        )
        self.news_day_transformer_encoder = _make_transformer_encoder_stack(
            self.news_encoder_layers
        )

        transformer_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=dim_ff,
            dropout=drop,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        try:
            self.transformer = nn.TransformerDecoder(
                transformer_layer,
                num_layers=self.transformer_layers,
                enable_nested_tensor=False,
            )
        except TypeError:
            self.transformer = nn.TransformerDecoder(
                transformer_layer, num_layers=self.transformer_layers
            )

        # ckpt key kept as ``price_news_fusion`` (CrossAttnPostFusion module)
        self.price_news_fusion = CrossAttnPostFusion(
            d_s=self.d_model,
            d_ns=self.d_news,
            nhead=self.nhead,
            dim_ff_s=dim_ff,
            dim_ff_ns=max(self.d_news * 2, 64),
            dropout=drop,
            num_layers=1,
        )
        self.out_norm = nn.LayerNorm(self.d_model)

        self.last_news_keep: torch.Tensor | None = None
        self.last_cross_keep: torch.Tensor | None = None

    def _shared_day_pos(self, k: int) -> torch.Tensor:
        """[K, D] calendar-day PE; shared by price day t and news slots of day t."""
        return self.day_pos.weight[:k]

    def _expand_day_pos_to_news_slots(
        self, day_p: torch.Tensor, m_slots: int
    ) -> torch.Tensor:
        """[K,D] → [K*M, D]: slot (t,m) gets the same vector as price day t."""
        # day_p[t] repeated M times consecutively
        return day_p.repeat_interleave(max(1, int(m_slots)), dim=0)

    def _causal_bool_mask(self, k: int, device: torch.device) -> torch.Tensor:
        """[K,K] True=block; day i cannot attend to future j>i."""
        return torch.triu(torch.ones(k, k, device=device, dtype=torch.bool), diagonal=1)

    def _causal_memory_mask(
        self, k: int, device: torch.device, news_per_day: int = 1
    ) -> torch.Tensor | None:
        """Price day i ↔ expanded news slots: block news with day j>i. Shape [K, K*M]."""
        if self.price_news_align == "none":
            return None
        m_slots = max(1, int(news_per_day))
        q_day = torch.arange(k, device=device).view(k, 1)
        key_day = torch.arange(k, device=device).repeat_interleave(m_slots).view(1, k * m_slots)
        return key_day > q_day

    @staticmethod
    def _causal_bool_mask_with_prefix(k: int, device: torch.device, n_prefix: int = 1) -> torch.Tensor:
        """Causal mask for [prefix | K days]; prefix may attend to all; True=block."""
        n_prefix = max(1, int(n_prefix))
        n = n_prefix + int(k)
        m = torch.ones(n, n, device=device, dtype=torch.bool)
        # prefix tokens: attend to all prefix + all days
        m[:n_prefix, :] = False
        for i in range(int(k)):
            # day i at index n_prefix+i attends to prefix + days 0..i
            m[n_prefix + i, : n_prefix + i + 1] = False
        return m

    def _causal_memory_mask_with_prefix(
        self,
        k: int,
        device: torch.device,
        n_prefix: int = 1,
        news_per_day: int = 1,
    ) -> torch.Tensor | None:
        """Memory mask for tgt=[prefix|days] vs expanded news [K*M]; True=block."""
        if self.price_news_align == "none":
            return None
        n_prefix = max(1, int(n_prefix))
        m_slots = max(1, int(news_per_day))
        mem_len = int(k) * m_slots
        # [n_prefix+K, K*M]
        m = torch.zeros(n_prefix + int(k), mem_len, device=device, dtype=torch.bool)
        key_day = torch.arange(k, device=device).repeat_interleave(m_slots).view(1, mem_len)
        for i in range(int(k)):
            m[n_prefix + i] = key_day > i
        return m

    def _normalize_industry_tok(
        self, industry_tok: torch.Tensor, bsz: int, d: int
    ) -> torch.Tensor:
        ind = industry_tok
        if ind.dim() == 2:
            ind = ind.unsqueeze(1)
        if ind.dim() != 3:
            raise ValueError(f"industry_tok expected [B,D] or [B,L,D], got {tuple(ind.shape)}")
        if int(ind.size(0)) != bsz or int(ind.size(-1)) != d:
            raise ValueError(f"industry_tok {tuple(ind.shape)} incompatible with B={bsz} D={d}")
        if int(ind.size(1)) != 1:
            ind = ind.mean(dim=1, keepdim=True)
        return ind

    def _run_cross_attention_transformer(
        self,
        price_h: torch.Tensor,
        news_memory: torch.Tensor,
        industry_tok: torch.Tensor | None,
        device: torch.device,
        memory_key_padding_mask: torch.Tensor | None = None,
        news_per_day: int = 1,
    ) -> torch.Tensor:
        """Causal price<-news Cross Transformer (cross_attn core).

        nn.TransformerDecoder: tgt/Q = price days; memory/K,V = news slots.
        Industry VN may prepend on last layer.
        """
        bsz, k, d = price_h.shape
        m_slots = max(1, int(news_per_day))
        causal = self._causal_bool_mask(k, device)
        mem_mask = self._causal_memory_mask(k, device, news_per_day=m_slots)
        layers = list(self.transformer.layers)
        tgt = price_h
        prepended = False
        self.last_cross_keep = None
        for i, layer in enumerate(layers):
            tgt_mask = causal
            cur_mem_mask = mem_mask
            if (
                i == len(layers) - 1
                and industry_tok is not None
                and industry_tok.numel() > 0
            ):
                ind = self._normalize_industry_tok(industry_tok, bsz, d)
                te = self.type_emb.weight if self.type_emb is not None else None
                if te is not None and int(te.size(0)) > 2:
                    ind = ind + te[2].view(1, 1, -1)
                tgt = torch.cat([ind.to(dtype=tgt.dtype, device=tgt.device), tgt], dim=1)
                tgt_mask = self._causal_bool_mask_with_prefix(k, device, 1)
                cur_mem_mask = self._causal_memory_mask_with_prefix(
                    k, device, 1, news_per_day=m_slots
                )
                prepended = True
            tgt = layer(
                tgt,
                news_memory,
                tgt_mask=tgt_mask,
                memory_mask=cur_mem_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        if prepended:
            tgt = tgt[:, 1:, :]
        return tgt

    def encode_ssl_towers(self, price_tok, news_tok=None, **_ignored):
        """MTVC aux expects this. Price tokens are already day embeddings."""
        del news_tok, _ignored
        return {
            "price_h": price_tok,
            "price_tok": price_tok,
            "news_day": None,
            "news_day_valid": None,
            "price_day_mask": None,
        }

    def forward_unified(
        self,
        price_tok: torch.Tensor | None = None,
        news_tok: torch.Tensor | None = None,
        virt_tok: torch.Tensor | None = None,
        industry_tok: torch.Tensor | None = None,
        news_day_idx: torch.Tensor | None = None,
        news_pad: torch.Tensor | None = None,
        vol_tok: torch.Tensor | None = None,  # ignored; volume path removed
    ) -> torch.Tensor:
        # Industry VN: only the last cross-Transformer layer (3rd TF block).
        del vol_tok
        if price_tok is None:
            raise ValueError("CrossAttentionEncoder requires price_tok [B,K,D]")
        if price_tok.dim() != 3:
            raise ValueError(f"price_tok expected [B,K,D], got {tuple(price_tok.shape)}")
        bsz, k, d = price_tok.shape
        if k != self.num_days or d != self.d_model:
            raise ValueError(
                f"price_tok {tuple(price_tok.shape)} != K={self.num_days} D={self.d_model}"
            )
        device = price_tok.device
        # MTVC L_pair packs one news bag [B,N,D] plus day index, not [B,K,M,D].
        if news_day_idx is not None and news_tok is not None and news_tok.dim() == 3:
            bsz_n, nmax, dn = news_tok.shape
            src = news_tok
            if dn != int(self.d_news):
                if dn == int(self.d_model):
                    src = self.news_down(news_tok)
                else:
                    raise ValueError(
                        f"mtvc news_tok D={dn} != d_news={self.d_news}"
                    )
            if news_pad is None:
                news_pad = torch.zeros(bsz_n, nmax, dtype=torch.bool, device=device)
            di = news_day_idx.long().clamp(0, k - 1)
            slots = src.new_zeros(bsz_n, k, 1, int(self.d_news))
            for i in range(nmax):
                idx = (~news_pad[:, i]).nonzero(as_tuple=False).view(-1)
                if idx.numel() > 0:
                    slots[idx, di[idx], 0, :] = src[idx, i, :]
            news_tok = slots
        day_p = self._shared_day_pos(k)
        te = self.type_emb.weight if self.type_emb is not None else None
        causal = self._causal_bool_mask(k, device)

        p = price_tok + day_p.unsqueeze(0)
        if te is not None:
            p = p + te[0].view(1, 1, -1)
        if self.include_price:
            price_h = self.price_transformer_encoder(p, mask=causal)
        else:
            price_h = price_tok.new_zeros(bsz, k, d)

        news_memory = None
        news_ns = None
        news_pad_flat = None
        m_slots = 1
        self.last_news_keep = None
        if self.include_news and news_tok is not None:
            if news_tok.dim() == 3:
                news_tok = news_tok.unsqueeze(2)
            if news_tok.dim() != 4:
                raise ValueError(f"news_tok expected [B,K,M,d], got {tuple(news_tok.shape)}")
            m_slots = int(news_tok.size(2))
            n = self.news_up(news_tok)
            n = n + day_p.view(1, k, 1, d)
            if te is not None:
                n = n + te[1].view(1, 1, 1, -1)
            pad = news_tok.abs().sum(dim=-1) == 0  # [B,K,M] True=pad
            flat = n.reshape(bsz * k, m_slots, d)
            pad_f = pad.reshape(bsz * k, m_slots)
            all_pad = pad_f.all(dim=1)
            if bool(all_pad.any()):
                pad_f = pad_f.clone()
                pad_f[all_pad, 0] = False
            flat = self.news_transformer_encoder(flat, src_key_padding_mask=pad_f)
            pad_out = pad.reshape(bsz * k, m_slots)
            flat = flat.masked_fill(pad_out.unsqueeze(-1), 0.0)

            news_memory = flat.reshape(bsz, k * m_slots, d)
            news_pad_flat = pad_out.reshape(bsz, k * m_slots)

            day_p_news = self._expand_day_pos_to_news_slots(day_p, m_slots)
            price_h = price_h + day_p.unsqueeze(0)
            news_memory = news_memory + day_p_news.unsqueeze(0)
            news_memory = news_memory.masked_fill(news_pad_flat.unsqueeze(-1), 0.0)

            news_ns = self.news_down(news_memory)
            news_ns = news_ns.masked_fill(news_pad_flat.unsqueeze(-1), 0.0)

        if news_memory is not None:
            price_h = self._run_cross_attention_transformer(
                price_h,
                news_memory,
                industry_tok,
                device,
                memory_key_padding_mask=news_pad_flat,
                news_per_day=m_slots,
            )
            if (
                self.last_cross_keep is not None
                and news_ns is not None
                and news_ns.shape[:2] == self.last_cross_keep.shape
            ):
                from tool.numeric import decay_residual_enabled as _dr_ns
                from tool.numeric import is_attn_decay_prune_mode as _id_ns

                # Residual decay: keep news slot features for fusion (attn already masked)
                if not (_id_ns(self.prune_mode) and _dr_ns()):
                    news_ns = news_ns * self.last_cross_keep.unsqueeze(-1).to(
                        dtype=news_ns.dtype
                    )
        elif industry_tok is not None and industry_tok.numel() > 0:
            ind = self._normalize_industry_tok(industry_tok, bsz, d)
            price_h = price_h + ind

        state = self.state_tok.expand(bsz, -1, -1)
        s_parts = [state]
        if virt_tok is not None and virt_tok.numel() > 0:
            vw = virt_tok.unsqueeze(1) if virt_tok.dim() == 2 else virt_tok
            if te is not None:
                vw = vw + te[2].view(1, 1, -1)
            s_parts.append(vw)
        elif self.vin_window_emb is not None:
            vw = (
                self.vin_window_emb[: min(self.num_v_window, k)]
                .unsqueeze(0)
                .expand(bsz, -1, -1)
            )
            if te is not None:
                vw = vw + te[2].view(1, 1, -1)
            s_parts.append(vw)
        ph = price_h
        if te is not None:
            ph = ph + te[0].view(1, 1, -1)
        s_parts.append(ph)
        x_s = torch.cat(s_parts, dim=1)
        # CrossAttnPostFusion (attr name price_news_fusion for ckpt)
        x_s, _ = self.price_news_fusion(x_s, news_ns)
        self.last_mtvc_gate_w_mean = None
        self.last_mtvc_case_price = price_h.mean(dim=1)
        if news_memory is not None:
            valid = (
                (~news_pad_flat).unsqueeze(-1).to(dtype=news_memory.dtype)
                if news_pad_flat is not None
                else news_memory.new_ones(news_memory.shape[:2] + (1,))
            )
            self.last_mtvc_case_news = (news_memory * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        else:
            self.last_mtvc_case_news = price_h.new_zeros(bsz, d)
        return self.out_norm(x_s[:, 0, :])

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.forward_unified(*args, **kwargs)

    def forward_with_price_aux(self, *args, **kwargs):
        h = self.forward_unified(*args, **kwargs)
        return h, h

    # Back-compat method name used by older call sites / notes.
    _run_price_news_transformer = _run_cross_attention_transformer



# =============================================================================
# 1) Shared modality tokenizers / PredFusion (used by all three stacks)
# =============================================================================

class SeqTransformerPool(nn.Module):
    """Shared short-sequence Transformer pool (price HLC tokens + news char tokens).

    Used as:
    - ``unitrans_price_tok``: H/L/C (or lookback) → day price vector
    - backbone inside ``NewsTransformerEncoder``: token emb sequence → news vector

    Optional soft keyword pruning between layers (learnable keep∈(0,1), no hard mask).
    """

    def __init__(
        self,
        in_dim: int,
        d_model: int,
        nhead: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
        max_len: int = 64,
        out_dim: int | None = None,
        dim_ff_mult: int = 2,
        **_ignored,
    ):
        super().__init__()
        del _ignored
        self.d_model = int(d_model)
        self.max_len = max(1, int(max_len))
        self.out_dim = int(out_dim) if out_dim is not None else self.d_model
        self.num_layers = max(1, int(num_layers))
        nhead = int(nhead)
        while self.d_model % nhead != 0 and nhead > 1:
            nhead //= 2
        self.nhead = max(1, nhead)
        self.in_proj = nn.Linear(int(in_dim), self.d_model) if int(in_dim) != self.d_model else nn.Identity()
        self.pos_emb = nn.Parameter(torch.zeros(1, self.max_len, self.d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        ff = max(self.d_model, self.d_model * max(1, int(dim_ff_mult)))
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=ff,
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = _make_transformer_encoder(layer, self.num_layers)
        self.out_norm = nn.LayerNorm(self.d_model)
        self.out_proj = (
            nn.Linear(self.d_model, self.out_dim) if self.out_dim != self.d_model else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        chunk_size: int = 0,
    ) -> torch.Tensor:
        """x: [N, L, Fin] -> pooled [N, out_dim]. Soft-token prune removed (paper package)."""
        if x.dim() != 3:
            raise ValueError(f"expected [N,L,F], got {tuple(x.shape)}")
        n = int(x.size(0))
        if chunk_size and chunk_size > 0 and n > chunk_size:
            outs = []
            for i in range(0, n, chunk_size):
                sl = slice(i, i + chunk_size)
                msk = None if key_padding_mask is None else key_padding_mask[sl]
                outs.append(self._forward_one(x[sl], msk))
            return torch.cat(outs, dim=0)
        return self._forward_one(x, key_padding_mask)

    def _forward_one(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        length = min(int(x.size(1)), self.max_len)
        x = x[:, :length, :]
        h = self.in_proj(x)
        h = h + self.pos_emb[:, :length, :]
        mask = None
        if key_padding_mask is not None:
            mask = key_padding_mask[:, :length].clone()
            try:
                from tool.numeric import sanitize_pad_mask
            except ImportError:
                from tool.numeric import sanitize_pad_mask
            mask = sanitize_pad_mask(mask)
        h = self.encoder(h, src_key_padding_mask=mask)
        h = self.out_norm(h)
        if mask is None:
            pooled = h.mean(dim=1)
        else:
            keep_m = (~mask).to(dtype=h.dtype).unsqueeze(-1)
            pooled = (h * keep_m).sum(dim=1) / keep_m.sum(dim=1).clamp(min=1.0)
        return self.out_proj(pooled)


class NewsTransformerEncoder(nn.Module):
    """News text encoder for ``news_word_emb=random`` (paper path).

    ``vocab ids → learnable Embedding → SeqTransformerPool → news vector``.
    Not a pretrained BERT/FinBERT; name kept clear of that confusion.
    Legacy alias: ``BertTokTransformer``.
    """

    def __init__(
        self,
        vocab_size: int,
        pad_id: int = 0,
        d_model: int = 128,
        out_dim: int = 512,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_len: int = 64,
        dim_ff_mult: int = 2,
    ):
        super().__init__()
        self.pad_id = int(pad_id)
        self.max_len = max(1, int(max_len))
        self.embed = nn.Embedding(int(vocab_size), int(d_model), padding_idx=self.pad_id)
        nn.init.normal_(self.embed.weight, std=0.02)
        if self.pad_id >= 0 and self.pad_id < int(vocab_size):
            with torch.no_grad():
                self.embed.weight[self.pad_id].zero_()
        self.encoder = SeqTransformerPool(
            in_dim=int(d_model),
            d_model=int(d_model),
            nhead=int(nhead),
            num_layers=int(num_layers),
            dropout=float(dropout),
            max_len=self.max_len,
            out_dim=int(out_dim),
            dim_ff_mult=int(dim_ff_mult),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        chunk_size: int = 128,
    ) -> torch.Tensor:
        """
        input_ids: [N, L]
        attention_mask: [N, L] 1=keep, 0=pad (optional)
        returns: [N, out_dim]
        """
        if input_ids.dim() != 2:
            input_ids = input_ids.reshape(input_ids.size(0), -1)
        length = min(int(input_ids.size(1)), self.max_len)
        input_ids = input_ids[:, :length]
        if attention_mask is not None:
            attention_mask = attention_mask[:, :length]
            pad_mask = attention_mask.eq(0)
        else:
            pad_mask = input_ids.eq(self.pad_id)
        tok_emb = self.embed(input_ids.clamp(0, self.embed.num_embeddings - 1))
        return self.encoder(tok_emb, key_padding_mask=pad_mask, chunk_size=chunk_size)


# Legacy name — not a BERT backbone.
BertTokTransformer = NewsTransformerEncoder

class PredFusionTransformer(nn.Module):
    """
    Outer task tower:
      tokens = [unified_h, p_0..p_{K-1}, (optional news...), global?, industry?]
      unified_h = day-serial pyramid final state; p_* = K-day price encodings.
      纯股价路径也保留 h 与 K 天股价两路。
      small Transformer over tokens → pool → logit.
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 2,
        num_tokens: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.num_tokens = max(1, int(num_tokens))
        nhead = int(nhead)
        while self.d_model % nhead != 0 and nhead > 1:
            nhead //= 2
        self.pos_emb = nn.Parameter(torch.zeros(1, self.num_tokens, self.d_model))
        self.slot_emb = nn.Parameter(torch.zeros(self.num_tokens, self.d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        nn.init.trunc_normal_(self.slot_emb, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=self.d_model * 2,
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = _make_transformer_encoder(layer, max(1, int(num_layers)))
        self.out_norm = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, 1)

    def forward(self, tokens: torch.Tensor, return_pooled: bool = False):
        """
        tokens: [N, T, D], T == num_tokens
        returns: [N] logits; if return_pooled, also the [N,D] vector before Linear.
        """
        if tokens.dim() != 3:
            raise ValueError(f"expected [N,T,D], got {tuple(tokens.shape)}")
        t = min(int(tokens.size(1)), self.num_tokens)
        x = tokens[:, :t, :]
        x = x + self.pos_emb[:, :t, :] + self.slot_emb[:t].unsqueeze(0)
        x = self.encoder(x)
        x = self.out_norm(x)
        pooled = x.mean(dim=1)
        logits = self.head(pooled).squeeze(-1)
        if return_pooled:
            return logits, pooled
        return logits

# =============================================================================
# 2) PartialUnifiedModel (joint Transformer + layer-wise news gates)
#    Used by partial_unified / full_unified — NOT Cross-Attention.
# =============================================================================


def _env_causal_default_on() -> bool:
    return _mtvc_env("UNIFIED_CAUSAL", "1").strip().lower() not in (
        "0",
        "false",
        "off",
        "no",
    )


def _env_same_day_on() -> bool:
    return _mtvc_env("UNIFIED_SAME_DAY", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _make_encoder(d_model: int, nhead: int, num_layers: int, dropout: float) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=d_model * 2,
        dropout=float(dropout),
        batch_first=True,
        activation="gelu",
        norm_first=True,
    )
    try:
        return nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
    except TypeError:
        return nn.TransformerEncoder(layer, num_layers=num_layers)


class PartialUnifiedModel(nn.Module):
    """Shared self-attn over concatenated price-day + variable news tokens."""

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_days: int = 5,
        num_layers: int = 6,
        dropout: float = 0.1,
        use_type_emb: bool = True,
        mode: str = "partial",
        causal: bool | None = None,
        same_day: bool | None = None,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.num_days = max(1, int(num_days))
        self.num_layers = max(1, int(num_layers))
        self.use_type_emb = bool(use_type_emb)
        self.mode = str(mode or "partial").lower()
        self.same_day = _env_same_day_on() if same_day is None else bool(same_day)
        # same-day is stricter; when on, ignore causal (j<=t would re-open past days)
        if self.same_day:
            self.causal = False
        else:
            self.causal = _env_causal_default_on() if causal is None else bool(causal)
        nhead = _nhead_for(self.d_model, int(nhead))
        self.nhead = nhead

        self.state_tok = nn.Parameter(torch.zeros(1, 1, self.d_model))
        nn.init.trunc_normal_(self.state_tok, std=0.02)
        self.day_pos = nn.Embedding(self.num_days, self.d_model)
        nn.init.trunc_normal_(self.day_pos.weight, std=0.02)
        if self.use_type_emb:
            # 0=price, 1=news
            self.type_emb = nn.Embedding(2, self.d_model)
            nn.init.trunc_normal_(self.type_emb.weight, std=0.02)
        else:
            self.type_emb = None

        self.encoder = _make_encoder(self.d_model, nhead, self.num_layers, dropout)
        self.out_norm = nn.LayerNorm(self.d_model)
        # SSL MLM: replace masked price-day tokens before joint encode
        self.ssl_mask_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        nn.init.trunc_normal_(self.ssl_mask_token, std=0.02)
        # MTVC: news weight after EVERY layer 1..L-1 (each layer has its own w)
        self.news_layer_gate = False
        self.news_gates: nn.ModuleList | None = None
        # 1-indexed joint layer used to cache price/news pools for case mining
        self.case_mine_layer: int = 6
        self.last_mtvc_gate_w_mean: float | None = None
        self.last_mtvc_case_price: torch.Tensor | None = None
        self.last_mtvc_case_news: torch.Tensor | None = None
        self.last_news_keep = None
        self.last_cross_keep = None
        self.include_news = True
        self.include_price = True

    def _day_layout(
        self,
        bsz: int,
        k: int,
        news_day_idx: torch.Tensor | None,
        nmax: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (day[B,L], is_state_q[B,L,1], is_state_k[B,1,L]). State day=-1."""
        L = 1 + k + max(0, int(nmax))
        day = torch.full((bsz, L), -1, device=device, dtype=torch.long)
        day[:, 1 : 1 + k] = torch.arange(k, device=device, dtype=torch.long).view(1, k)
        if nmax > 0 and news_day_idx is not None:
            day[:, 1 + k :] = news_day_idx.long().clamp(0, k - 1)
        is_state_q = torch.zeros(bsz, L, 1, dtype=torch.bool, device=device)
        is_state_q[:, 0, :] = True
        is_state_k = torch.zeros(bsz, 1, L, dtype=torch.bool, device=device)
        is_state_k[:, :, 0] = True
        return day, is_state_q, is_state_k

    def _causal_src_mask(
        self,
        bsz: int,
        k: int,
        news_day_idx: torch.Tensor | None,
        nmax: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Bool attn mask [B*nhead, L, L]: True=block. Day rule: key.day <= query.day."""
        day, is_state_q, is_state_k = self._day_layout(bsz, k, news_day_idx, nmax, device)
        q_day = day.unsqueeze(2)
        k_day = day.unsqueeze(1)
        allow = is_state_q | is_state_k | (k_day <= q_day)
        L = day.size(1)
        block = (~allow).unsqueeze(1).expand(bsz, self.nhead, L, L)
        return block.reshape(bsz * self.nhead, L, L).contiguous()

    def _same_day_src_mask(
        self,
        bsz: int,
        k: int,
        news_day_idx: torch.Tensor | None,
        nmax: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Bool attn mask [B*nhead, L, L]: True=block. Day rule: key.day == query.day.

        Price day t and news of day t can attend each other; cross-day blocked.
        State remains a free prefix (sees all / visible to all).
        """
        day, is_state_q, is_state_k = self._day_layout(bsz, k, news_day_idx, nmax, device)
        q_day = day.unsqueeze(2)
        k_day = day.unsqueeze(1)
        allow = is_state_q | is_state_k | (k_day == q_day)
        L = day.size(1)
        block = (~allow).unsqueeze(1).expand(bsz, self.nhead, L, L)
        return block.reshape(bsz * self.nhead, L, L).contiguous()

    def enable_news_gate(self, enabled: bool = True) -> None:
        """Create / toggle per-layer news gates (MTVC eqs 1-2)."""
        self.news_layer_gate = bool(enabled)
        if self.news_layer_gate and self.news_gates is None:
            n_gate = max(0, self.num_layers - 1)
            self.news_gates = nn.ModuleList(
                [nn.Linear(self.d_model, 1) for _ in range(n_gate)]
            )
            for g in self.news_gates:
                nn.init.zeros_(g.weight)
                nn.init.zeros_(g.bias)  # sigmoid → w≈0.5 at start
            # ensure gates live on same device as encoder
            try:
                dev = next(self.encoder.parameters()).device
                self.news_gates.to(dev)
            except StopIteration:
                pass

    def _apply_news_layer_gate(
        self,
        h: torch.Tensor,
        pad: torch.Tensor,
        k: int,
        nmax: int,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Scale news tokens by sample-level wi after layer ``layer_idx`` (0-based)."""
        if (
            not self.news_layer_gate
            or self.news_gates is None
            or nmax <= 0
            or layer_idx >= len(self.news_gates)
        ):
            return h, None
        news_h = h[:, 1 + k : 1 + k + nmax, :]
        news_pad = pad[:, 1 + k : 1 + k + nmax]
        valid = ~news_pad
        # masked mean pool over news tokens → [B, D]
        w_mask = valid.unsqueeze(-1).to(dtype=news_h.dtype)
        summed = (news_h * w_mask).sum(dim=1)
        denom = w_mask.sum(dim=1).clamp_min(1.0)
        pooled = summed / denom
        # rows with no news: keep w=1 (no effect)
        has_news = valid.any(dim=1)
        logit = self.news_gates[layer_idx](pooled)  # [B,1]
        w = torch.sigmoid(logit).squeeze(-1)  # [B]
        w = torch.where(has_news, w, torch.ones_like(w))
        news_scaled = news_h * w.view(-1, 1, 1)
        news_scaled = news_scaled.masked_fill(news_pad.unsqueeze(-1), 0.0)
        h = h.clone()
        h[:, 1 + k : 1 + k + nmax, :] = news_scaled
        return h, w

    def _run_encoder(
        self,
        x: torch.Tensor,
        pad: torch.Tensor,
        src_mask: torch.Tensor | None,
        nmax: int,
        k: int,
    ) -> torch.Tensor:
        """Joint SA; optional news-token attn_decay residual halt (enc site)."""
        pad = pad
        try:
            from tool.numeric import sanitize_hidden, sanitize_pad_mask
        except ImportError:
            from tool.numeric import sanitize_hidden, sanitize_pad_mask
        pad = sanitize_pad_mask(pad)
        self.last_mtvc_gate_w_mean = None
        self.last_mtvc_case_price = None
        self.last_mtvc_case_news = None

        # MTVC contrast news-layer gate (case mining)
        if self.news_layer_gate and self.news_gates is not None:
            layers = list(self.encoder.layers)
            h = x
            w_acc: list[torch.Tensor] = []
            for i, layer in enumerate(layers):
                h = layer(h, src_mask=src_mask, src_key_padding_mask=pad)
                h = sanitize_hidden(h)
                # cache price / news pools after the configured joint layer (1-indexed)
                _case_l = int(getattr(self, "case_mine_layer", 6) or 6)
                _case_l = max(1, min(_case_l, len(layers)))
                if i == _case_l - 1:
                    price_h = h[:, 1 : 1 + k, :]
                    self.last_mtvc_case_price = price_h.mean(dim=1)
                    if nmax > 0:
                        news_h = h[:, 1 + k : 1 + k + nmax, :]
                        news_pad = pad[:, 1 + k : 1 + k + nmax]
                        valid = (~news_pad).unsqueeze(-1).to(dtype=news_h.dtype)
                        summed = (news_h * valid).sum(dim=1)
                        denom = valid.sum(dim=1).clamp_min(1.0)
                        self.last_mtvc_case_news = summed / denom
                    else:
                        self.last_mtvc_case_news = h.new_zeros(h.size(0), h.size(-1))
                if i < len(layers) - 1 and nmax > 0:
                    h, w = self._apply_news_layer_gate(h, pad, k, nmax, i)
                    if w is not None:
                        w_acc.append(w.detach())
            if w_acc:
                self.last_mtvc_gate_w_mean = float(torch.stack(w_acc).mean().item())
            return sanitize_hidden(h)


        return sanitize_hidden(self.encoder(x, mask=src_mask, src_key_padding_mask=pad))

    def _attn_src_mask(
        self,
        bsz: int,
        k: int,
        news_day_idx: torch.Tensor | None,
        nmax: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if self.same_day:
            return self._same_day_src_mask(bsz, k, news_day_idx, nmax, device)
        if self.causal:
            return self._causal_src_mask(bsz, k, news_day_idx, nmax, device)
        return None

    def forward_unified(
        self,
        price_tok: torch.Tensor,
        news_tok: torch.Tensor | None = None,
        news_day_idx: torch.Tensor | None = None,
        news_pad: torch.Tensor | None = None,
        **_ignored,
    ) -> torch.Tensor:
        """
        price_tok: [B, K, D]
        news_tok:  [B, Nmax, D]  (padded within batch; may be Nmax=0)
        news_day_idx: [B, Nmax] long in [0,K) for valid news; ignored if pad
        news_pad: [B, Nmax] bool True=pad
        returns h: [B, D] from state token
        """
        del _ignored
        if price_tok.dim() != 3:
            raise ValueError(f"price_tok expected [B,K,D], got {tuple(price_tok.shape)}")
        bsz, k, d = price_tok.shape
        if k != self.num_days or d != self.d_model:
            raise ValueError(
                f"price_tok {tuple(price_tok.shape)} != K={self.num_days} D={self.d_model}"
            )
        device = price_tok.device
        day_p = self.day_pos.weight[:k]  # [K,D]

        # price: + day_pos + type
        p = price_tok + day_p.unsqueeze(0)
        if self.type_emb is not None:
            p = p + self.type_emb.weight[0].view(1, 1, -1)

        parts = [self.state_tok.expand(bsz, -1, -1), p]
        # pad mask: False=keep. state+price always keep
        pad_parts = [
            torch.zeros(bsz, 1, dtype=torch.bool, device=device),
            torch.zeros(bsz, k, dtype=torch.bool, device=device),
        ]

        nmax = 0
        news_day_for_mask: torch.Tensor | None = None
        if news_tok is not None and news_tok.numel() > 0 and news_tok.size(1) > 0:
            if news_tok.dim() != 3:
                raise ValueError(f"news_tok expected [B,N,D], got {tuple(news_tok.shape)}")
            nmax = int(news_tok.size(1))
            if news_day_idx is None:
                raise ValueError("news_day_idx required with news_tok")
            if news_pad is None:
                news_pad = torch.zeros(bsz, nmax, dtype=torch.bool, device=device)
            # clamp day idx for safety
            di = news_day_idx.long().clamp(0, k - 1)
            news_day_for_mask = di
            n = news_tok + day_p[di]  # gather day pos
            if self.type_emb is not None:
                n = n + self.type_emb.weight[1].view(1, 1, -1)
            # zero out pads so they don't leak before mask
            n = n.masked_fill(news_pad.unsqueeze(-1), 0.0)
            parts.append(n)
            pad_parts.append(news_pad)

        x = torch.cat(parts, dim=1)
        pad = torch.cat(pad_parts, dim=1)
        try:
            from tool.numeric import sanitize_pad_mask
        except ImportError:
            from tool.numeric import sanitize_pad_mask
        pad = sanitize_pad_mask(pad)

        src_mask = self._attn_src_mask(bsz, k, news_day_for_mask, nmax, device=device)
        x = self._run_encoder(x, pad, src_mask, nmax=nmax, k=k)
        h = self.out_norm(x[:, 0, :])
        return h

    def encode_ssl_towers(
        self,
        price_tok: torch.Tensor,
        news_tok: torch.Tensor | None = None,
        news_day_idx: torch.Tensor | None = None,
        news_pad: torch.Tensor | None = None,
        price_day_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """Joint-encode for SSL; return price-day / news-day pools (pre-PredFusion).

        Unlike CrossAttentionEncoder (cross_attn), this runs the *same* joint encoder used at
        finetune so SSL gradients warm-start ``unitrans.encoder``.

        Returns:
          price_h [B,K,D], price_tok [B,K,D] (unmasked recon target),
          news_day [B,K,D] or None, news_day_valid [B,K] or None,
          price_day_mask [B,K] or None.
        """
        if price_tok.dim() != 3:
            raise ValueError(f"price_tok expected [B,K,D], got {tuple(price_tok.shape)}")
        bsz, k, d = price_tok.shape
        if k != self.num_days or d != self.d_model:
            raise ValueError(
                f"price_tok {tuple(price_tok.shape)} != K={self.num_days} D={self.d_model}"
            )
        device = price_tok.device
        price_tgt = price_tok
        p_in = price_tok
        if price_day_mask is not None:
            if price_day_mask.shape != (bsz, k):
                raise ValueError(
                    f"price_day_mask {tuple(price_day_mask.shape)} != {(bsz, k)}"
                )
            mt = self.ssl_mask_token.to(dtype=p_in.dtype, device=device)
            p_in = torch.where(price_day_mask.unsqueeze(-1), mt.expand_as(p_in), p_in)

        day_p = self.day_pos.weight[:k]
        p = p_in + day_p.unsqueeze(0)
        if self.type_emb is not None:
            p = p + self.type_emb.weight[0].view(1, 1, -1)

        parts = [self.state_tok.expand(bsz, -1, -1), p]
        pad_parts = [
            torch.zeros(bsz, 1, dtype=torch.bool, device=device),
            torch.zeros(bsz, k, dtype=torch.bool, device=device),
        ]
        nmax = 0
        news_day_for_mask: torch.Tensor | None = None
        di: torch.Tensor | None = None
        if news_tok is not None and news_tok.numel() > 0 and news_tok.size(1) > 0:
            if news_tok.dim() != 3:
                raise ValueError(f"news_tok expected [B,N,D], got {tuple(news_tok.shape)}")
            nmax = int(news_tok.size(1))
            if news_day_idx is None:
                raise ValueError("news_day_idx required with news_tok")
            if news_pad is None:
                news_pad = torch.zeros(bsz, nmax, dtype=torch.bool, device=device)
            di = news_day_idx.long().clamp(0, k - 1)
            news_day_for_mask = di
            n = news_tok + day_p[di]
            if self.type_emb is not None:
                n = n + self.type_emb.weight[1].view(1, 1, -1)
            n = n.masked_fill(news_pad.unsqueeze(-1), 0.0)
            parts.append(n)
            pad_parts.append(news_pad)

        x = torch.cat(parts, dim=1)
        pad = torch.cat(pad_parts, dim=1)
        try:
            from tool.numeric import sanitize_pad_mask
        except ImportError:
            from tool.numeric import sanitize_pad_mask
        pad = sanitize_pad_mask(pad)

        src_mask = self._attn_src_mask(bsz, k, news_day_for_mask, nmax, device=device)
        x = self._run_encoder(x, pad, src_mask, nmax=nmax, k=k)
        x = self.out_norm(x)
        price_h = x[:, 1 : 1 + k, :]

        news_day: torch.Tensor | None = None
        news_day_valid: torch.Tensor | None = None
        if nmax > 0 and di is not None and news_pad is not None:
            news_out = x[:, 1 + k :, :]
            valid = ~news_pad
            news_day = price_tok.new_zeros(bsz, k, d)
            idx = di.unsqueeze(-1).expand(-1, -1, d)
            news_day.scatter_add_(
                1, idx, news_out.masked_fill(~valid.unsqueeze(-1), 0.0)
            )
            cnt = price_tok.new_zeros(bsz, k)
            cnt.scatter_add_(1, di, valid.to(dtype=cnt.dtype))
            news_day = news_day / cnt.clamp_min(1.0).unsqueeze(-1)
            news_day_valid = cnt > 0

        return {
            "price_h": price_h,
            "price_tok": price_tgt,
            "news_day": news_day,
            "news_day_valid": news_day_valid,
            "price_day_mask": price_day_mask,
        }

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.forward_unified(*args, **kwargs)


# =============================================================================
# full_unified: TokenMLP price tokenizer (+ Model helpers below)
# Also reused for optional vol token projection.
# =============================================================================

class TokenMLP(nn.Module):
    """OneTrans-style tokenizer: concat raw features → MLP → one token.

    Primary use: ``full_unified`` price day tokens (HLC → MLP).
        """

    def __init__(self, in_dim: int, d_model: int, hidden: int | None = None):
        super().__init__()
        self.in_dim = max(1, int(in_dim))
        hid = int(hidden) if hidden is not None else max(int(d_model), self.in_dim)
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, hid),
            nn.GELU(),
            nn.Linear(hid, int(d_model)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.reshape(x.size(0), -1)
        elif x.dim() != 2:
            raise ValueError(f"TokenMLP expected [N,F] or [N,T,F], got {tuple(x.shape)}")
        n, f = x.shape
        if f < self.in_dim:
            x = torch.cat([x, x.new_zeros((n, self.in_dim - f))], dim=-1)
        elif f > self.in_dim:
            x = x[:, : self.in_dim]
        return self.net(x)


# =============================================================================
# 3) Top-level Model (virtual nodes + wire §1/§2 + forward → logits)
#    Contrast: contrast_aux.py reuses unitrans_price_tok / news TF / unitrans
# =============================================================================

class Model(nn.Module):
    """
    价格序列 + batch['news']（vocab + random word emb）经 encoder 编码后 PredFusion 预测。

    Paper path: news_encode_mode=vocab_* + NewsTransformerEncoder (random Embedding).
    """

    def __init__(
        self,
        price_emb_dim: int = 128,
        news_emb_dim: int = 128,
        hidden_dim: int = 128,
        news_encode_mode: str = "vocab_padding",
        news_top_k: int = 5,
        grid_num_days: int = 20,
        grid_num_stocks: int = 50,
        industry_window_count: int = 20,
        num_industry: int = 27,
        num_rule_trend: int = 8,
        rule_trend_feat_dim: int = 64,
        vocab_path: str | None = None,
        price_input_dim: int = 3,
        no_price_token: bool = False,
        industry_no_mean: bool = False,
        num_score_windows: int | None = None,
        news_ablation_mode: str = "full",
        pred_fusion_ablation_mode: str | None = None,
        main_mlp_ablation_mode: str | None = None,  # legacy alias
        vin_ablation_mode: str = "no_window",
        # Where industry VN joins (mainly Cross-Attention / cross_attn):
        # pred_fusion | cross (last Cross Transformer layer) | fusion (CrossAttnPostFusion).
        industry_fuse_site: str = "pred_fusion",
        news_word_emb: str = "random",
        news_text_tf_layers: int = 1,
        news_process_mode: str = "default",
        dual_own_prop_news: bool = False,
        own_news_only: bool = False,
        prop_same_trend_only: bool = False,
        companies: list | None = None,
        price_encoder: str = "partial_unified",
        # PredFusion tokens: [h, global?, score?, industry?].
        # Paper default / new runs: score off (tokens=3 with global+industry).
        # Keep True for loading older cross_attn / full_unified ckpts that ship score.
        use_pred_fusion_global_score: bool = False,
        # Virtual-node connect ablation (paper default keeps market=Hadamard, industry=concat):
        #   default   : mkt = Proj(mean)⊙g_day ; ind = Proj([mean‖v_k])
        #   mkt_as_ind : mkt = Proj([mean‖g_day]) (g_day still day-phase → keep periodicity)
        #   ind_as_mkt : ind = mean⊙v_k (industry prototype as multiplicative modulator)
        vin_connect_mode: str = "default",
    ):
        super().__init__()
        self.use_pred_fusion_global_score = bool(use_pred_fusion_global_score)
        _vcm = str(vin_connect_mode or "default").lower().strip()
        if _vcm not in ("default", "mkt_as_ind", "ind_as_mkt"):
            raise ValueError(
                f"vin_connect_mode must be default|mkt_as_ind|ind_as_mkt, got {vin_connect_mode!r}"
            )
        self.vin_connect_mode = _vcm
        self.industry_window_count = int(industry_window_count)
        self.num_industry = int(num_industry)
        self.num_rule_trend = max(1, int(num_rule_trend))
        self.rule_trend_feat_dim = max(8, int(rule_trend_feat_dim))
        # 价格 tokenizer 固定 H/L/C（图里 weekday 不进 tokenizer）。
        self.price_encoder = normalize_price_encoder(price_encoder)
        self.unitrans_num_days = 5
        # Paper modality ablations (orthogonal):
        #   --no_price_token → price day tokens = 0 (Enc+cross kept)
        #   --news_ablation_mode no_news → drop news tower / news PredFusion token
        self.price_token_fill = "zero" if bool(no_price_token) else "real"
        self.include_price_token = self.price_token_fill == "real"
        self.include_news_token = True  # news off only via news_ablation_mode
        self.news_token_fill = "real"
        # Industry PredFusion token: False=proj([mean‖v_k]); True=proj(v_k) only.
        self.industry_no_mean = bool(industry_no_mean)
        self.price_input_dim = 3
        self.dual_own_prop_news = bool(dual_own_prop_news)
        self.own_news_only = bool(own_news_only)
        self.prop_same_trend_only = bool(prop_same_trend_only)
        self.companies = list(companies or [])
        self._company2id = {c: i for i, c in enumerate(self.companies)}
        if num_score_windows is not None:
            self.num_score_windows = max(1, int(num_score_windows))
        else:
            self.num_score_windows = 5
        # ----- Virtual nodes (checkpoint keys; do not rename) -----
        # virtual_industry_node: learnable per-industry embedding [1, num_industry, price_emb_dim]
        self.virtual_industry_node = nn.Parameter(
            torch.randn(1, self.num_industry, price_emb_dim) * 0.1
        )
        self.news_encode_mode = news_encode_mode
        self.news_transformer_encoder = None
        self.bert_tok_pad_id = 0
        if not str(news_encode_mode).startswith("vocab"):
            raise ValueError(
                f"news_encode_mode={news_encode_mode!r} unsupported; paper path uses vocab_* only"
            )
        _ablation = str(news_ablation_mode or "full").lower()
        if _ablation not in ("full", "no_news"):
            raise ValueError(
                f"news_ablation_mode must be full|no_news, got {news_ablation_mode!r}"
            )
        self.news_ablation_mode = _ablation
        # no_news：去掉新闻支路；市场全局 / 行业虚拟结点仍默认保留
        # （仅当 --pred_fusion_ablation_mode no_global / no_industry / … 时才去掉）
        self.include_news_in_main_mlp = self.news_ablation_mode != "no_news"
        self.encoder_include_news = bool(self.include_news_in_main_mlp)
        # --no_price_token still runs price Enc + cross (only kills real HLC encode).
        self.encoder_include_price = True
        if self.news_ablation_mode == "no_news" or not self.include_price_token:
            print(
                f"[MODALITY] news_ablation={self.news_ablation_mode} "
                f"no_price_token={int(not self.include_price_token)} "
                f"encoder(price={self.encoder_include_price}, news={self.encoder_include_news}) "
                f"PredFusion(news_tok={self.include_news_in_main_mlp})",
                flush=True,
            )
        # news_global（横截面新闻均值 token）：按设计关闭，PredFusion 不用
        self.include_news_global_in_main_mlp = False
        self.use_aux_branch = False
        self.include_aux_in_main_mlp = False
        _mlp_ab = str(
            pred_fusion_ablation_mode
            if pred_fusion_ablation_mode is not None
            else (main_mlp_ablation_mode if main_mlp_ablation_mode is not None else "full")
        ).lower()
        if _mlp_ab not in ("full", "no_global", "no_industry", "no_global_industry"):
            _mlp_ab = "full"
        self.pred_fusion_ablation_mode = _mlp_ab
        self.main_mlp_ablation_mode = _mlp_ab  # legacy alias
        # Paper path drops dual_tf virtual day-window nodes (always no_window).
        _vin_ab = str(vin_ablation_mode or "no_window").lower().strip()
        if _vin_ab not in ("full", "no_window"):
            raise ValueError(
                f"vin_ablation_mode must be full|no_window, got {vin_ablation_mode!r}"
            )
        self.vin_ablation_mode = _vin_ab
        # Paper path: learnable Embedding inside NewsTransformerEncoder only.
        self.news_word_emb = "random"
        if str(news_word_emb or "random").lower().strip() != "random":
            print(
                f"[NEWS] WARN: news_word_emb={news_word_emb!r} ignored; random Embedding only",
                flush=True,
            )
        self.include_global_in_main_mlp = _mlp_ab not in ("no_global", "no_global_industry")
        self.include_industry_in_main_mlp = _mlp_ab not in ("no_industry", "no_global_industry")
        _ifs = str(industry_fuse_site or "pred_fusion").lower().strip()
        # pred_fusion: industry in final PredFusion with global (default / paper).
        # cross: industry only in last Cross-Attention Transformer layer.
        # fusion: industry in CrossAttnPostFusion (cross_attn only).
        if _ifs not in ("cross", "fusion", "pred_fusion"):
            raise ValueError(
                f"industry_fuse_site must be pred_fusion|cross|fusion, "
                f"got {industry_fuse_site!r}"
            )
        _paper_enc = self.price_encoder in PRICE_ENCODER_CANONICAL
        self.industry_fuse_site = _ifs
        self.fuse_industry_with_price_news = bool(
            _paper_enc and _ifs == "cross" and self.include_industry_in_main_mlp
        )
        self.fuse_industry_in_fusion = bool(
            _paper_enc and _ifs == "fusion" and self.include_industry_in_main_mlp
        )
        self.include_industry_in_pred_fusion = bool(
            self.include_industry_in_main_mlp and _ifs == "pred_fusion"
        )
        self.price_emb_dim = int(price_emb_dim)
        self.news_emb_dim = max(8, int(news_emb_dim))
        self.global_feat_dim = 128
        # 主 MLP 中 news 分支维度 = price + global + industry（不含 score 标量）
        self.news_main_feat_dim = self.price_emb_dim
        if self.include_global_in_main_mlp:
            self.news_main_feat_dim += self.price_emb_dim
        if self.include_industry_in_main_mlp:
            self.news_main_feat_dim += self.price_emb_dim
        # 双路：自有新闻 + 波及行业/市场，主 MLP 新闻槽为 2×news_main_feat_dim
        self.news_mlp_slot_dim = int(self.news_main_feat_dim) * (2 if self.dual_own_prop_news else 1)
        _proc = str(news_process_mode or "default").lower()
        if _proc != "default":
            raise ValueError(
                f"news_process_mode={news_process_mode!r} removed; paper path uses default only"
            )
        self.news_process_mode = "default"
        # News aggregation: mean over price nodes (freq-quantile / sum-gate path removed).
        self.news_window_agg_mode = "false"
        self.news_top_k = int(news_top_k)
        self.grid_num_days = max(1, int(grid_num_days))
        self.grid_num_stocks = int(grid_num_stocks)

        self.reduce_dim = nn.Linear(self.global_feat_dim, 1)

        # vocab_token CNN encoder（需与 Dataset 使用相同词表，否则 vocab_input_ids 索引越界）
        # Prefer package dict/ relative to DUAL_TF_MODEL_DIR (MTVC root) or this file.
        _pkg_root = os.environ.get("DUAL_TF_MODEL_DIR") or os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
        )
        _candidates = (
            [vocab_path] if vocab_path else []
        ) + [
            os.path.join(_pkg_root, "dict", "dict_csmd.pkl"),
            os.path.join(_pkg_root, "dict", "dict_massive.pkl"),
            os.path.join(_MODEL_DIR, "dict", "dict_csmd.pkl"),
            os.path.join(_MODEL_DIR, "dict", "dict_massive.pkl"),
        ]
        _loaded_path = None
        for _p in _candidates:
            if _p and os.path.isfile(_p):
                self.vocab_dict = pkl.load(open(_p, "rb"))
                _loaded_path = _p
                break
        if _loaded_path is None:
            raise FileNotFoundError(f"vocab not found, tried: {_candidates}")
        self.vocab_path = str(_loaded_path)
        self.vocab_pad_id = int(self.vocab_dict.get("<pad>", 0))
        # 兼容稀疏/异常词表：num_embeddings 必须 > padding_idx，且覆盖最大 token id
        _vocab_vals = [int(v) for v in self.vocab_dict.values() if isinstance(v, (int, np.integer))]
        _max_vocab_id = max(_vocab_vals) if _vocab_vals else 0
        self.vocab_size = max(len(self.vocab_dict), _max_vocab_id + 1, self.vocab_pad_id + 1, 1)
        # Mirror data.is_us_vocab_path / word-level flags (avoid importing data → transformers).
        _vbase = os.path.basename(self.vocab_path).lower()
        self._vocab_is_us = "dict_massive" in _vbase
        self._vocab_use_jieba = ("jieba" in _vbase) or ("csmd_word" in _vbase)
        self._vocab_word_level = bool(self._vocab_is_us or self._vocab_use_jieba)

        self.global_parameter = nn.Parameter(torch.randn(10, 1, self.global_feat_dim))
        # Market connect: default Hadamard with day-phase g; mkt_as_ind = concat like industry.
        if self.vin_connect_mode == "mkt_as_ind":
            self.global_mean_proj = None
            self.global_connect_proj = nn.Linear(
                int(price_emb_dim) + int(self.global_feat_dim), int(self.global_feat_dim)
            )
        else:
            self.global_mean_proj = nn.Linear(price_emb_dim, self.global_feat_dim)
            self.global_connect_proj = None
        # Legacy competition_scorer / mlp_virtual_industry removed (PredFusion uses
        # pred_fusion_industry_proj only); stripped on ckpt load.
        self.news_w_off_pos_emb = None
        print(
            f"[VIN] connect_mode={self.vin_connect_mode} "
            f"(default=mkt⊙g_day+ind_cat; mkt_as_ind=mkt_cat(g_day); ind_as_mkt=ind⊙v_k)",
            flush=True,
        )

        # ---- price temporal encoder (paper stacks only) ----
        if True:
            self.rnn = None
            self.price_5_lstm = None
            self.price_rnn_out_dim = int(price_emb_dim)
            self.price_rnn_proj = nn.Identity()
            # day price: Mix / shared 都用小 TF pool → 128；shared 成交量另用 MLP
            _day_d = min(64, int(price_emb_dim))
            _pw = max(1, int(getattr(self, "unitrans_num_days", 5) or 5))
            self.unitrans_price_mlp = None
            self.unitrans_news_mlp = None
            if bool(getattr(self, "include_price_token", True)):
                # diagram: 当天 H/L/C（或 HLCV）各作 1 个 token → 小 TF → 日向量
                # 其它 unitrans: 沿用 lookback 窗 [pw, F] 小 TF
                _n_ch = max(1, int(self.price_input_dim))
                self.unitrans_price_tok = SeqTransformerPool(
                    in_dim=1,
                    d_model=_day_d,
                    nhead=_nhead_for(_day_d, 4),
                    num_layers=1,
                    dropout=0.1,
                    max_len=_n_ch,
                    out_dim=int(price_emb_dim),
                    dim_ff_mult=2,
                )
                self.unitrans_price_as_hlc_tokens = True
            else:
                self.unitrans_price_tok = None
                self.unitrans_price_as_hlc_tokens = False
                print(
                    "[PRICE] price_token=OFF (fixed zero): skip HLC SeqTF; "
                    "Price day tokens = 0",
                    flush=True,
                )
            self.unitrans_news_text_chunk = 64
            self.unitrans_news_tok = None
            self.unitrans_no_news_emb = None
            self.unitrans_pad_price = nn.Parameter(torch.zeros(1, 1, price_emb_dim))
            # Mix：新闻保持 news_emb_dim，price-Q 融成 1 个 NS
            # shared：新闻与股价同维，M 篇直接相接
            _nd = int(self.news_emb_dim)
            _npm = max(1, int(getattr(self, "news_top_k", 5) or 5))
            self.unitrans_news_per_day = _npm
            self.unitrans_pad_news = nn.Parameter(torch.zeros(1, 1, _nd))
            nn.init.trunc_normal_(self.unitrans_pad_price, std=0.02)
            nn.init.trunc_normal_(self.unitrans_pad_news, std=0.02)
            # 仅 interleaved UniTransStock（非 pyr）仍需同维，才升维
            self.unitrans_news_up_proj = None
            if True:
                print(
                    f"[UNITRANS] dims: price/S={int(price_emb_dim)} "
                    f"news_slots={_nd}×M={_npm} → NS={_nd} "
                    f"(news: Q=price K/V=news; PredFusion=h only)",
                    flush=True,
                )

            # no_news：日块只要价格；full：价格+新闻
            _day_include_news = bool(getattr(self, "encoder_include_news", False))
            _day_include_price = bool(getattr(self, "encoder_include_price", True))
            _build_news_text = bool(_day_include_news)

            if not _build_news_text:
                self.unitrans_news_text_tf = None
                self.news_transformer_encoder = None
                self.unitrans_no_news_emb = None
            elif str(self.news_encode_mode).startswith("vocab"):
                # cross_attn only: fixed M news slots / (day,stock). When a slot has no
                # real article, fill with TextTF(placeholder) so K×M memory stays dense
                # (Cross Transformer Q=price, K/V=news slots). CN char vocab →「当日无文本」;
                # EN word vocab (dict_massive) → "no news". partial/full use
                # variable-length news + pad mask — they never read this buffer.
                if self.price_encoder == "cross_attn":
                    _no_news_text = "no news" if self._vocab_is_us else "当日无文本"
                    _unk = int(self.vocab_dict.get("<UNK>", self.vocab_dict.get("<unk>", 1)))
                    _max_nn = 32
                    _pad = int(self.vocab_pad_id)
                    if self._vocab_word_level:
                        if self._vocab_use_jieba:
                            import jieba

                            _toks = [t for t in jieba.lcut(_no_news_text) if t and not t.isspace()]
                        else:
                            _toks = [t for t in _no_news_text.split() if t]
                        _ids = []
                        for _w in _toks:
                            if _w in self.vocab_dict:
                                _ids.append(int(self.vocab_dict[_w]))
                            elif _w.lower() in self.vocab_dict:
                                _ids.append(int(self.vocab_dict[_w.lower()]))
                            else:
                                _ids.append(_unk)
                            if len(_ids) >= _max_nn:
                                break
                    else:
                        _ids = [int(self.vocab_dict.get(ch, _unk)) for ch in _no_news_text][:_max_nn]
                    if len(_ids) < _max_nn:
                        _ids = _ids + [_pad] * (_max_nn - len(_ids))
                    else:
                        _ids = _ids[:_max_nn]
                    self.register_buffer(
                        "unitrans_no_news_vocab_ids",
                        torch.tensor(_ids, dtype=torch.long).view(1, _max_nn),
                        persistent=False,
                    )
                    self._no_news_placeholder_text = _no_news_text
                else:
                    self.unitrans_no_news_vocab_ids = None
                    self._no_news_placeholder_text = None
                _nt_d = min(64, max(32, int(_nd)))
                _nt_layers = max(1, int(news_text_tf_layers))
                # vocab id → random Embedding + TF → news_emb_dim
                self.news_transformer_encoder = NewsTransformerEncoder(
                    vocab_size=int(self.vocab_size),
                    pad_id=int(self.vocab_pad_id),
                    d_model=_nt_d,
                    out_dim=int(_nd),
                    nhead=_nhead_for(_nt_d, 4),
                    num_layers=_nt_layers,
                    dropout=0.1,
                    max_len=64,
                    dim_ff_mult=2,
                )
                self.unitrans_news_text_tf = None
                self.news_text_tf_layers = _nt_layers
                print(
                    f"[NEWS-TF] layers={_nt_layers} d_model={_nt_d} out_dim={int(_nd)} "
                    f"word_emb={getattr(self, 'news_word_emb', None)}",
                    flush=True,
                )
            else:
                raise ValueError(
                    f"unsupported news_encode_mode={self.news_encode_mode!r}; "
                    "paper path requires vocab_*"
                )

            if self.price_encoder == "cross_attn":
                self._setup_cross_attn(
                    price_emb_dim=int(price_emb_dim),
                    news_emb_dim=int(_nd),
                    news_per_day=int(_npm),
                    include_news=_day_include_news,
                    include_price=_day_include_price,
                )
            else:
                # partial_unified / full_unified
                _k = int(self.unitrans_num_days)
                _use_type = _mtvc_env("USE_TYPE_EMB", "1").strip().lower() not in (
                    "0", "false", "off", "no",
                )
                _joint_L = max(1, int(_mtvc_env("TRANSFORMER_LAYERS", "6") or 6))
                _mode = "full" if self.price_encoder == "full_unified" else "partial"
                self.unified_mode = _mode
                self.unitrans = PartialUnifiedModel(
                    d_model=int(price_emb_dim),
                    nhead=_nhead_for(price_emb_dim, 4),
                    num_days=_k,
                    num_layers=_joint_L,
                    dropout=0.1,
                    use_type_emb=_use_type,
                    mode=_mode,
                )
                self.unitrans_joint_proj = None
                self.company_graph_tf = None
                self.company_graph_fuse = None
                # news → d_model
                if int(_nd) != int(price_emb_dim):
                    self.unified_news_proj = nn.Linear(int(_nd), int(price_emb_dim))
                else:
                    self.unified_news_proj = nn.Identity()
                if _mode == "full":
                    self._setup_full_unified(
                        price_emb_dim=int(price_emb_dim),
                        news_emb_dim=int(_nd),
                        joint_L=_joint_L,
                        use_type=_use_type,
                        num_days=_k,
                    )
                else:
                    self.unitrans_price_day_mlp = None
                    # ensure HLC small TF path
                    self.unitrans_price_as_hlc_tokens = True
                    _causal = bool(getattr(self.unitrans, "causal", True))
                    _same = bool(getattr(self.unitrans, "same_day", False))
                    _rule = (
                        "same-day only (j==t)"
                        if _same
                        else ("price/news attend day j<=t" if _causal else "full attn")
                    )
                    print(
                        f"[UNIFIED-PARTIAL] joint Encoder L={_joint_L} type_emb={int(_use_type)} "
                        f"causal={int(_causal)} same_day={int(_same)} gt=0 "
                        f"price=HLC-SeqTF news=TextTF "
                        f"d={int(price_emb_dim)} news_d={int(_nd)} "
                        f"days={_k} no_window PredFusion "
                        f"({_rule}; state=prefix)",
                        flush=True,
                    )

        # ---- 统一预测头：Transformer 融合特征 + Linear(D→1)，无主 MLP ----
        # tokens = [h, (news...), global?, industry?]  —— 不再拼 K 天股价编码
        _vin_head = self.price_encoder in (
            "cross_attn",
            "partial_unified",
            "full_unified",
        )
        self.pred_fusion_num_price_days = 0
        _n_tok = 1  # final h only
        if (not _vin_head) and self.include_news_in_main_mlp:
            _n_tok += 1
        if (not _vin_head) and self.include_news_global_in_main_mlp:
            _n_tok += 1
        if self.include_global_in_main_mlp:
            _n_tok += 1
        if self.use_pred_fusion_global_score:
            _n_tok += 1
        if self.include_industry_in_pred_fusion:
            _n_tok += 1
        self.pred_fusion_num_tokens = int(_n_tok)
        self.use_pred_fusion_head = True
        self.pred_fusion_tf = PredFusionTransformer(
            d_model=int(price_emb_dim),
            nhead=_nhead_for(price_emb_dim, 4),
            num_layers=1,
            num_tokens=self.pred_fusion_num_tokens,
            dropout=0.1,
        )
        self.pred_fusion_global_proj = nn.Linear(self.global_feat_dim, int(price_emb_dim))
        self.pred_fusion_global_score_proj = (
            nn.Linear(1, int(price_emb_dim)) if self.use_pred_fusion_global_score else None
        )
        if self.vin_connect_mode == "ind_as_mkt":
            # mean⊙v_k already in price_emb_dim; keep a light Linear for capacity parity.
            self.pred_fusion_industry_proj = nn.Linear(int(price_emb_dim), int(price_emb_dim))
        else:
            _ind_proj_in = int(price_emb_dim) if self.industry_no_mean else int(price_emb_dim) * 2
            self.pred_fusion_industry_proj = nn.Linear(_ind_proj_in, int(price_emb_dim))
        # news token 投影到 D（news_main_feat_dim 可能更宽）
        self.pred_fusion_news_proj = (
            nn.Identity()
            if int(self.news_main_feat_dim) == int(price_emb_dim)
            else nn.Linear(int(self.news_main_feat_dim), int(price_emb_dim))
        )
        print(
            f"[PRED_HEAD] Transformer fusion tokens={self.pred_fusion_num_tokens} "
            f"(h only, no K-price; vin={_vin_head}, news={self.include_news_in_main_mlp}, "
            f"news_global={self.include_news_global_in_main_mlp}, "
            f"market_global={self.include_global_in_main_mlp}, "
            f"global_score={int(self.use_pred_fusion_global_score)}, "
            f"industry_in_pred={self.include_industry_in_pred_fusion}, "
            f"industry_fuse={self.industry_fuse_site}"
            f"{'+crossTF_Llast' if self.fuse_industry_with_price_news else ''}"
            f"{'+CrossAttnPostFusion' if getattr(self, 'fuse_industry_in_fusion', False) else ''}"
            f", industry_tok={'vi_only' if self.industry_no_mean else 'mean+vi'})",
            flush=True,
        )
        # 多时间步融合：直接将最近 5 天特征拼接后用 MLP 融合
        self.time_fuse_mlp = nn.Sequential(
            nn.Linear(price_emb_dim * 5, price_emb_dim),
            nn.ReLU(),
            nn.Linear(price_emb_dim, price_emb_dim),
        )
        self.time_score = nn.Sequential(
            nn.Linear(price_emb_dim, 1)
        )
        # Aggregate last 5 price embedding steps with LSTM
        # (LSTM path: price_5_lstm created above; TF path uses stage2)
        # 同一天内股票间交互：在 stock 维做交叉注意力
        # virtual_window_emb: day-index fallback for dual_tf virt_tok when vin_window_emb absent
        self.virtual_window_emb = nn.Parameter(torch.randn(self.grid_num_days, price_emb_dim) * 0.1)
        # 辅助任务：目标日成交量相对前日涨跌（二分类）
        # unitrans_h：TE 前 UniTrans h；pred_fusion：与股价同一处（PredFusion pooled → Linear）
        # 主 MLP 输入维：price 固定；global/industry/score/news/aux 按消融动态去掉
        _fusion_feat_dims = [price_emb_dim]
        if self.include_global_in_main_mlp:
            _fusion_feat_dims.extend([self.global_feat_dim, 1])
        if self.include_industry_in_main_mlp:
            _fusion_feat_dims.append(price_emb_dim)
        if self.include_news_in_main_mlp:
            if self.dual_own_prop_news:
                _fusion_feat_dims.append(int(self.news_mlp_slot_dim))
            else:
                _fusion_feat_dims.append(self.news_main_feat_dim)
            if self.include_news_global_in_main_mlp:
                _fusion_feat_dims.append(self.global_feat_dim)  # news_global_feat
        if self.include_aux_in_main_mlp:
            _fusion_feat_dims.append(price_emb_dim)
        _mlp_in = sum(_fusion_feat_dims)
        # 主 MLP 已废弃：预测一律走 pred_fusion_tf + Linear(D→1)
        self.mlp = None
        self._legacy_mlp_in_dim = int(_mlp_in)
        # dual 波及路：Q=MLP(dst股价, dst自有新闻)，K=MLP(src股价, 波及新闻)，V=波及新闻
        # 筛边波动 u：prop 边 MLP(dst股价, src股价)，own 边仍只用 dst
        self.prop_cross_q_mlp = None
        self.prop_cross_k_mlp = None

    def _build_industry_token(
        self,
        p_b: torch.Tensor,
        batch,
        num_days: int,
        num_stocks: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Industry token for Cross-Attention path (CrossAttnPostFusion / last Cross TF): [B,1,D].

        industry_type is 1:1 with price nodes.
        Default: proj([mean(price_window), vi]); industry_no_mean: proj(vi).
        """
        if not bool(getattr(self, "fuse_industry_with_price_news", False)):
            return None
        if self.pred_fusion_industry_proj is None:
            return None
        bsz, _k, _d = p_b.shape
        expect = int(num_days) * int(num_stocks)
        if bsz != expect:
            raise RuntimeError(
                f"industry_tok batch {bsz} != num_days*num_stocks={expect}"
            )
        vi_node = self.virtual_industry_node.squeeze(0)  # [num_industry, D]
        use_industry = (
            "industry_type" in batch.node_types
            and getattr(batch["industry_type"], "x", None) is not None
            and int(batch["industry_type"].x.size(0)) == bsz
        )
        if use_industry:
            ids_flat = (
                batch["industry_type"]
                .x[:, 0]
                .to(device)
                .long()
                .clamp(0, self.num_industry - 1)
            )
        else:
            ids_flat = p_b.new_zeros((bsz,), dtype=torch.long)
        vi_sel = vi_node[ids_flat]
        if self.vin_connect_mode == "ind_as_mkt":
            price_mean = p_b.mean(dim=1)
            tok = self.pred_fusion_industry_proj(price_mean * vi_sel)
        elif bool(getattr(self, "industry_no_mean", False)):
            tok = self.pred_fusion_industry_proj(vi_sel)
        else:
            price_mean = p_b.mean(dim=1)
            tok = self.pred_fusion_industry_proj(torch.cat([price_mean, vi_sel], dim=-1))
        return tok.unsqueeze(1)

    def _encode_news_nodes_to_price_dim(self, batch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode news nodes; UniTrans 路径输出 news_emb_dim（默认 32），其它路径多为 price 维。"""
        _unitrans = self.price_encoder in (
            "cross_attn",
            "partial_unified",
            "full_unified",
        )
        _out_dim = int(self.news_emb_dim) if _unitrans else int(self.price_emb_dim)
        if "news" not in batch.node_types or int(batch["news"].x.size(0)) <= 0:
            z = torch.zeros((0, _out_dim), device=device)
            empty = torch.zeros((0,), dtype=torch.long, device=device)
            return z, empty, empty
        N_news = int(batch["news"].x.size(0))
        # day/comp 始终留在 CPU，避免 encode 后 .cpu() 同步把异步 CUDA 错误报到打包处
        news_day_ids = batch["news"].day_id.detach().long().cpu()
        news_comp_ids = batch["news"].company_id.detach().long().cpu()
        # partial / full_unified / cross_attn: random Embedding → NewsTransformerEncoder
        if (
            _unitrans
            and self.news_transformer_encoder is not None
            and str(self.news_encode_mode).startswith("vocab")
            and hasattr(batch["news"], "vocab_input_ids")
        ):
            vocab_input_ids = batch["news"].vocab_input_ids.to(device)
            if vocab_input_ids.dim() != 2:
                vocab_input_ids = vocab_input_ids.reshape(vocab_input_ids.size(0), -1)
            attn = None
            if hasattr(batch["news"], "vocab_attention_mask"):
                attn = batch["news"].vocab_attention_mask.to(device)
                if attn.dim() != 2:
                    attn = attn.reshape(attn.size(0), -1)
            chunk = int(getattr(self, "unitrans_news_text_chunk", 64) or 0)
            news_512 = self.news_transformer_encoder(
                vocab_input_ids, attention_mask=attn, chunk_size=chunk
            )
            return news_512, news_day_ids, news_comp_ids

        news_512 = torch.zeros((N_news, _out_dim), device=device)
        return news_512, news_day_ids, news_comp_ids

    def _unitrans_no_news_token(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """cross_attn empty-slot filler → [1,1,D].

        Placeholder text is CN「当日无文本」or EN ``no news`` (see init).
        Used only by ``_build_day_stock_news_slot_grid`` (fixed-M packing).
        """
        if (
            self.news_transformer_encoder is not None
            and getattr(self, "unitrans_no_news_vocab_ids", None) is not None
        ):
            ids = self.unitrans_no_news_vocab_ids.to(device=device)
            tok = self.news_transformer_encoder(ids, attention_mask=None, chunk_size=0)
            return tok.view(1, 1, -1).to(dtype=dtype)
        if self.unitrans_no_news_emb is not None:
            return self.unitrans_no_news_emb.to(device=device, dtype=dtype)
        return torch.zeros((1, 1, int(self.news_emb_dim)), device=device, dtype=dtype)

    def _build_day_stock_news_slot_grid(
        self,
        news_512: torch.Tensor,
        news_day_ids: torch.Tensor,
        news_comp_ids: torch.Tensor,
        num_days: int,
        num_stocks: int,
        device: torch.device,
        news_per_day: int,
    ) -> torch.Tensor:
        """cross_attn news packing: dense [D, S, M, dim] grid.

        Each (day, stock) keeps up to M article tokens; missing slots filled with
        ``_unitrans_no_news_token`` (CN「当日无文本」or EN ``no news``). Called from
        ``_unitransstock_price_news_encode`` when ``price_encoder=cross_attn``
        (partial/full return earlier via ``_unified_joint_encode``).

        Packing on CPU then one ``.to(device)`` avoids GPU argsort / sync issues.
        """
        m = max(1, int(news_per_day))
        dim = int(news_512.size(-1)) if news_512.numel() > 0 else int(self.news_emb_dim)
        dtype = news_512.dtype if news_512.numel() else torch.float32
        use_encoder_nonews = (
            str(self.news_encode_mode).startswith("vocab")
            and self.news_transformer_encoder is not None
            and getattr(self, "unitrans_no_news_vocab_ids", None) is not None
        )
        # fill 先在目标 device 算（可能走小 TF），再落到 CPU 做 scatter
        if use_encoder_nonews:
            fill = self._unitrans_no_news_token(device, dtype)
            fill = fill.view(1, 1, 1, dim).expand(num_days, num_stocks, m, dim).contiguous()
        elif self.unitrans_no_news_emb is not None:
            fill = self.unitrans_no_news_emb.view(1, 1, 1, -1).expand(num_days, num_stocks, m, dim).contiguous()
        else:
            fill = torch.zeros((num_days, num_stocks, m, dim), device=device, dtype=dtype)
        if news_512.numel() == 0:
            return fill.to(device=device, dtype=dtype)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        day_cpu = news_day_ids.detach().long().cpu().view(-1)
        comp_cpu = news_comp_ids.detach().long().cpu().view(-1)
        valid_cpu = (day_cpu >= 0) & (day_cpu < num_days) & (comp_cpu >= 0) & (comp_cpu < num_stocks)
        grid_cpu = fill.detach().to("cpu", dtype=torch.float32).clone()
        if not bool(valid_cpu.any()):
            return grid_cpu.to(device=device, dtype=dtype)

        news_cpu = news_512.detach().to("cpu", dtype=torch.float32)
        idx = valid_cpu.nonzero(as_tuple=False).view(-1)
        day_cpu = day_cpu[idx]
        comp_cpu = comp_cpu[idx]
        src = news_cpu[idx]
        flat_idx = day_cpu * int(num_stocks) + comp_cpu
        perm = flat_idx.argsort(stable=True)
        flat_s = flat_idx[perm]
        src_s = src[perm]
        if flat_s.numel() == 0:
            return grid_cpu.to(device=device, dtype=dtype)
        prev = torch.full_like(flat_s, -1)
        prev[1:] = flat_s[:-1]
        new_group = flat_s != prev
        group_id = new_group.cumsum(0) - 1
        group_start = new_group.nonzero(as_tuple=False).view(-1)
        starts = group_start[group_id]
        rank = torch.arange(flat_s.numel()) - starts
        keep = rank < m
        flat_k = flat_s[keep]
        rank_k = rank[keep]
        src_k = src_s[keep]
        d_i = flat_k // int(num_stocks)
        s_i = flat_k % int(num_stocks)
        grid_cpu[d_i, s_i, rank_k] = src_k
        return grid_cpu.to(device=device, dtype=dtype)

    def _setup_cross_attn(
        self,
        price_emb_dim: int,
        news_emb_dim: int,
        news_per_day: int,
        include_news: bool,
        include_price: bool,
    ) -> None:
        """Build ``CrossAttentionEncoder`` for ``--price_encoder cross_attn``."""
        _k = int(self.unitrans_num_days)
        _nw = 0 if self.vin_ablation_mode == "no_window" else _k
        _use_type = _mtvc_env("USE_TYPE_EMB", "1").strip().lower() not in (
            "0", "false", "off", "no",
        )
        _unified = str(_mtvc_env("UNIFIED_LAYERS", "") or "").strip()
        if _unified:
            _L = max(1, int(_unified))
            _news_enc_L = _L
            _cross_L = _L
        else:
            _news_enc_L = max(1, int(_mtvc_env("NEWS_ENCODER_LAYERS", "1") or 1))
            _cross_L = max(1, int(_mtvc_env("TRANSFORMER_LAYERS", "2") or 2))
        _price_enc_L = max(1, int(_mtvc_env("PRICE_ENCODER_LAYERS", "1") or 1))
        self.unitrans = CrossAttentionEncoder(
            d_model=int(price_emb_dim),
            nhead=_nhead_for(price_emb_dim, 4),
            num_days=_k,
            d_news=int(news_emb_dim),
            news_emb_dim=int(news_emb_dim),
            news_per_day=int(news_per_day),
            dropout=0.1,
            price_encoder_layers=_price_enc_L,
            news_encoder_layers=_news_enc_L,
            transformer_layers=_cross_L,
            include_news=bool(include_news),
            include_price=bool(include_price),
            include_virtual=True,
            num_v_window=_nw,
            price_news_align="strict",
            use_type_emb=_use_type,
        )
        self.unitrans_joint_proj = None
        self.company_graph_tf = None
        self.company_graph_fuse = None
        print(
            f"[CROSS-ATTN] price_encoder=cross_attn "
            f"include_news={int(include_news)} include_price={int(include_price)} "
            f"news_ablation={getattr(self, 'news_ablation_mode', 'full')} "
            f"no_price={int(not bool(getattr(self, 'include_price_token', True)))} "
            f"days={_k} d_s={int(price_emb_dim)} news_slots=M={int(news_per_day)} "
            f"d_news={int(news_emb_dim)} "
            f"Enc(price/news)={_price_enc_L}/{_news_enc_L} "
            f"CrossTF(layers)={_cross_L} CrossAttnPostFusion=1 "
            f"mem=expand_KM band=causal type_emb={int(_use_type)} "
            f"align=strict+day_pos vin_window={_nw}; "
            f"HLC×3→SeqTF; within-day news slot TF (no mean-pool); "
            f"causal price←news cross on K*M slots (day j<=i); "
            f"CrossTF → CrossAttnPostFusion → h; PredFusion",
            flush=True,
        )

    # ------------------------------------------------------------------
    # full_unified (price TokenMLP; news / joint shared with partial)
    # ------------------------------------------------------------------
    def _setup_full_unified(
        self,
        price_emb_dim: int,
        news_emb_dim: int,
        joint_L: int,
        use_type: bool,
        num_days: int,
    ) -> None:
        """Wire full-unified-only heads after shared ``PartialUnifiedModel`` is built."""
        self.unitrans_price_day_mlp = TokenMLP(
            in_dim=max(1, int(self.price_input_dim)),
            d_model=int(price_emb_dim),
        )
        self.unitrans_price_as_hlc_tokens = False
        _causal = bool(getattr(self.unitrans, "causal", True))
        _same = bool(getattr(self.unitrans, "same_day", False))
        print(
            f"[UNIFIED-FULL] joint Encoder L={joint_L} type_emb={int(use_type)} "
            f"causal={int(_causal)} same_day={int(_same)} gt=0 "
            f"price=TokenMLP(HLC) news=NewsTransformerEncoder "
            f"d={int(price_emb_dim)} news_d={int(news_emb_dim)} "
            f"days={int(num_days)} no_window PredFusion",
            flush=True,
        )

    def _encode_price_day_token_full_unified(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Full-unified price day token: last-step HLC → TokenMLP → [N, D]."""
        n_ch = max(1, int(self.price_input_dim))
        f_avail = int(x_seq.size(-1))
        last = x_seq[:, -1, : min(n_ch, f_avail)]
        if last.size(-1) < n_ch:
            last = torch.cat(
                [last, last.new_zeros(last.size(0), n_ch - last.size(-1))], dim=-1
            )
        return self.unitrans_price_day_mlp(last)

    def _unified_joint_encode(
        self,
        x_seq: torch.Tensor,
        batch,
        num_days: int,
        num_stocks: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Partial/full unified: price days + variable news → shared Encoder → h."""
        K = int(self.unitrans_num_days)
        D = int(self.price_emb_dim)
        day_ids = torch.arange(num_days, device=device)
        offsets = torch.arange(K, device=device) - (K - 1)
        src = day_ids[:, None] + offsets[None, :]  # [D,K]
        valid = src >= 0
        src_clamped = src.clamp(min=0)
        valid_exp = valid[:, :, None, None]

        # ---- price day tokens [D*S, K, D] ----
        # no_price_token: skip price SeqTF (fixed zeros)
        _pfill = str(getattr(self, "price_token_fill", "real") or "real")
        mode = str(getattr(self, "unified_mode", "partial"))
        if _pfill == "zero" or not bool(getattr(self, "include_price_token", True)):
            p_b = x_seq.new_zeros((num_days * num_stocks, K, D))
        else:
            if mode == "full" and getattr(self, "unitrans_price_day_mlp", None) is not None:
                price_day_tok = self._encode_price_day_token_full_unified(x_seq)
            elif bool(getattr(self, "unitrans_price_as_hlc_tokens", False)):
                n_ch = max(1, int(self.price_input_dim))
                f_avail = int(x_seq.size(-1))
                last = x_seq[:, -1, : min(n_ch, f_avail)]
                if last.size(-1) < n_ch:
                    last = torch.cat(
                        [last, last.new_zeros(last.size(0), n_ch - last.size(-1))], dim=-1
                    )
                tok_in = last.unsqueeze(-1)
                pad_mask = last.abs().eq(0)
                price_day_tok = self.unitrans_price_tok(tok_in, key_padding_mask=pad_mask)
            else:
                pad_mask = x_seq.abs().sum(dim=-1).eq(0)
                price_day_tok = self.unitrans_price_tok(x_seq, key_padding_mask=pad_mask)
            price_grid = price_day_tok.view(num_days, num_stocks, -1)
            p_win = price_grid[src_clamped]
            pad_p = self.unitrans_pad_price.expand(num_days, K, num_stocks, -1)
            p_win = torch.where(valid_exp, p_win, pad_p)
            p_b = p_win.permute(0, 2, 1, 3).reshape(num_days * num_stocks, K, -1)

        # ---- variable news: list then pad ----
        # news_ablation=no_news: skip news tokens entirely
        _use_news = bool(getattr(self, "encoder_include_news", True))
        if _use_news:
            news_feat, news_day_ids, news_comp_ids = self._encode_news_nodes_to_price_dim(
                batch, device
            )
        else:
            news_feat = p_b.new_zeros((0, D))
            news_day_ids = torch.zeros((0,), dtype=torch.long, device=device)
            news_comp_ids = torch.zeros((0,), dtype=torch.long, device=device)
        # project to D
        if _use_news and news_feat.numel() > 0:
            if int(news_feat.size(-1)) != D:
                proj = getattr(self, "unified_news_proj", None)
                if proj is None:
                    raise RuntimeError("unified_news_proj missing for dim mismatch")
                news_feat = proj(news_feat.to(device=device, dtype=p_b.dtype))
            else:
                news_feat = news_feat.to(device=device, dtype=p_b.dtype)
            news_day_ids = news_day_ids.to(device=device)
            news_comp_ids = news_comp_ids.to(device=device)
        B = num_days * num_stocks
        # Cap news tokens / row and chunk joint SA — avoids OOM + CUDA FATAL under packing.
        _max_news = max(1, int(str(_mtvc_env("UNIFIED_MAX_NEWS", "48") or "48").strip() or 48))
        _chunk = max(1, int(str(_mtvc_env("UNIFIED_CHUNK", "256") or "256").strip() or 256))
        # per-row lists
        row_news: list[list[tuple[int, torch.Tensor]]] = [[] for _ in range(B)]
        if _use_news and news_feat.numel() > 0:
            for i in range(int(news_feat.size(0))):
                d_abs = int(news_day_ids[i].item())
                s = int(news_comp_ids[i].item())
                if s < 0 or s >= num_stocks or d_abs < 0 or d_abs >= num_days:
                    continue
                d0 = max(0, d_abs)
                d1 = min(num_days, d_abs + K)
                for d_pred in range(d0, d1):
                    k_off = d_abs - (d_pred - (K - 1))
                    if k_off < 0 or k_off >= K:
                        continue
                    row = d_pred * num_stocks + s
                    row_news[row].append((k_off, news_feat[i]))
        # truncate long rows: keep highest k_off (closest to predict day) then fill
        nmax = 0
        for r, items in enumerate(row_news):
            if len(items) > _max_news:
                items.sort(key=lambda t: int(t[0]), reverse=True)
                row_news[r] = items[:_max_news]
            nmax = max(nmax, len(row_news[r]))
        if nmax == 0:
            news_tok = p_b.new_zeros((B, 0, D))
            news_day_idx = torch.zeros((B, 0), dtype=torch.long, device=device)
            news_pad = torch.zeros((B, 0), dtype=torch.bool, device=device)
        else:
            news_tok = p_b.new_zeros((B, nmax, D))
            news_day_idx = torch.zeros((B, nmax), dtype=torch.long, device=device)
            news_pad = torch.ones((B, nmax), dtype=torch.bool, device=device)
            for r, items in enumerate(row_news):
                for j, (k_off, feat) in enumerate(items):
                    news_tok[r, j] = feat
                    news_day_idx[r, j] = int(k_off)
                    news_pad[r, j] = False

        # SSL pretrain: stash joint inputs and skip joint + PredFusion path.
        if getattr(self, "_ssl_tokens_only", False):
            self._ssl_price_tok = p_b
            self._ssl_news_tok = news_tok
            self._ssl_news_day_idx = news_day_idx
            self._ssl_news_pad = news_pad
            dummy = p_b.new_zeros((p_b.size(0), int(self.price_emb_dim)))
            return dummy, p_b, dummy, None

        if B <= _chunk:
            h = self.unitrans.forward_unified(
                p_b, news_tok=news_tok, news_day_idx=news_day_idx, news_pad=news_pad
            )
        else:
            hs = []
            for i0 in range(0, B, _chunk):
                i1 = min(B, i0 + _chunk)
                hs.append(
                    self.unitrans.forward_unified(
                        p_b[i0:i1],
                        news_tok=news_tok[i0:i1],
                        news_day_idx=news_day_idx[i0:i1],
                        news_pad=news_pad[i0:i1],
                    )
                )
            h = torch.cat(hs, dim=0)
        return h, p_b, h, None


    def ensure_contrast_heads(self, proj_dim: int = 64) -> None:
        """Attach MTVC contrast projection heads (price / news)."""
        import torch.nn as nn

        d = int(self.price_emb_dim)
        pd = max(8, int(proj_dim))
        if getattr(self, "contrast_proj_p", None) is None:
            self.contrast_proj_p = nn.Sequential(
                nn.Linear(d, d),
                nn.GELU(),
                nn.Linear(d, pd),
            )
        if getattr(self, "contrast_proj_n", None) is None:
            self.contrast_proj_n = nn.Sequential(
                nn.Linear(d, d),
                nn.GELU(),
                nn.Linear(d, pd),
            )
        # Legacy attr aliases (old ckpts / call sites).
        self.ssl_proj_p = self.contrast_proj_p
        self.ssl_proj_n = self.contrast_proj_n

    # Back-compat name.
    ensure_ssl_heads = ensure_contrast_heads

    def compute_ssl_losses(self, *args, **kwargs):
        """SSL pretrain removed from the paper package."""
        raise RuntimeError("compute_ssl_losses / SSL pretrain not shipped in MTVC_paper_repro")

    def _unitransstock_price_news_encode(
        self,
        x_seq: torch.Tensor,
        batch,
        num_days: int,
        num_stocks: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Day-serial UniTrans over last K calendar days.
        Day tokens:
          - shared: OneTrans MLP tokenizer on flattened price/news features
          - mix: Transformer over price window steps
          - news (optional): Embedding + Text Transformer, then MLP or Mix NS
        Returns:
          price_emb_flat [N,D], price_emb_5 [N,K,D], price_emb_penultimate [N,D], None
        """
        K = int(self.unitrans_num_days)
        if self.price_encoder in ("partial_unified", "full_unified"):
            return self._unified_joint_encode(x_seq, batch, num_days, num_stocks, device)
        # ---- cross_attn below: fixed K×M slots (CN「当日无文本」/ EN "no news" fillers) ----
        include_news = bool(getattr(self.unitrans, "include_news", True))
        _pfill = str(getattr(self, "price_token_fill", "real") or "real")
        # Real HLC encode only when fill=real; zero still feeds price Enc if encoder_include_price.
        include_price_encode = (_pfill == "real") and bool(
            getattr(self.unitrans, "include_price", True)
        )
        day_ids = torch.arange(num_days, device=device)
        offsets = torch.arange(K, device=device) - (K - 1)  # e.g. [-4,-3,-2,-1,0]
        src = day_ids[:, None] + offsets[None, :]  # [D,K]
        valid = src >= 0
        src_clamped = src.clamp(min=0)
        valid_exp = valid[:, :, None, None]
        if _pfill == "zero" or not bool(getattr(self.unitrans, "include_price", True)):
            # fixed zeros (token ablation or structural news_only)
            p_b = x_seq.new_zeros((num_days * num_stocks, K, int(self.price_emb_dim)))
        else:
            if getattr(self, "unitrans_price_mlp", None) is not None:
                price_day_tok = self.unitrans_price_mlp(x_seq)  # flatten lookback → D
            elif bool(getattr(self, "unitrans_price_as_hlc_tokens", False)):
                # 仅用该日最后一根 bar 的 H/L/C(V)：[N,F] → [N,F,1] 三个(或四个)标量 token
                n_ch = max(1, int(self.price_input_dim))
                f_avail = int(x_seq.size(-1))
                last = x_seq[:, -1, : min(n_ch, f_avail)]
                if last.size(-1) < n_ch:
                    last = torch.cat(
                        [last, last.new_zeros(last.size(0), n_ch - last.size(-1))],
                        dim=-1,
                    )
                tok_in = last.unsqueeze(-1)  # [N, F, 1]
                pad_mask = last.abs().eq(0)  # [N, F] True=pad（全 0 通道）
                price_day_tok = self.unitrans_price_tok(tok_in, key_padding_mask=pad_mask)
            else:
                pad_mask = x_seq.abs().sum(dim=-1).eq(0)  # [N,L]
                price_day_tok = self.unitrans_price_tok(x_seq, key_padding_mask=pad_mask)  # [N, D]
            price_grid = price_day_tok.view(num_days, num_stocks, -1)
            p_win = price_grid[src_clamped]
            pad_p = self.unitrans_pad_price.expand(num_days, K, num_stocks, -1)
            p_win = torch.where(valid_exp, p_win, pad_p)
            p_b = p_win.permute(0, 2, 1, 3).reshape(num_days * num_stocks, K, -1)
        n_b = None
        news_512 = None
        news_day_ids = None
        news_comp_ids = None
        if include_news:
            _npm = int(getattr(self, "unitrans_news_per_day", getattr(self, "news_top_k", 5)) or 5)
            news_512, news_day_ids, news_comp_ids = self._encode_news_nodes_to_price_dim(
                batch, device
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            # Dense [D,S,M]: real articles + no-news placeholder fillers → CrossAttn memory
            news_grid = self._build_day_stock_news_slot_grid(
                news_512,
                news_day_ids,
                news_comp_ids,
                num_days,
                num_stocks,
                device,
                news_per_day=_npm,
            )  # [D,S,M,dim]
            # gather K-day windows: [D,K,S,M,dim]
            n_win = news_grid[src_clamped]
            pad_n = self.unitrans_pad_news.view(1, 1, 1, 1, -1).expand(
                num_days, K, num_stocks, _npm, -1
            )
            # valid_exp: [D,K,1,1,1]
            n_win = torch.where(valid[:, :, None, None, None], n_win, pad_n)
            # [D*S, K, M, dim]
            n_b = n_win.permute(0, 2, 1, 3, 4).reshape(
                num_days * num_stocks, K, _npm, -1
            )
            news_mlp = getattr(self, "unitrans_news_mlp", None)
            if news_mlp is not None:
                bsz_n, kk, mm, dd = n_b.shape
                n_b = news_mlp(n_b.reshape(bsz_n * kk, mm * dd)).view(bsz_n, kk, -1)
            else:
                up = getattr(self, "unitrans_news_up_proj", None)
                if up is not None:
                    n_b = up(n_b.mean(dim=2))

        virt_tok = None
        if (
            self.price_encoder == "cross_attn"
            and getattr(self.unitrans, "include_virtual", False)
        ):
            # dual_tf only: optional virtual window tokens (type slots removed)
            bsz = int(p_b.size(0))
            nw = int(getattr(self.unitrans, "num_v_window", K))
            if nw > 0:
                if getattr(self.unitrans, "vin_window_emb", None) is not None:
                    vw = self.unitrans.vin_window_emb[:nw]
                else:
                    vw = self.virtual_window_emb[:nw]
                    if vw.size(0) < nw:
                        vw = torch.cat([vw, vw.new_zeros((nw - vw.size(0), vw.size(1)))], dim=0)
                virt_tok = vw.unsqueeze(0).expand(bsz, -1, -1)

        if getattr(self.unitrans, "forward_unified", None) is not None:
            n_call = n_b
            _price_arg = p_b
            _ind_tok = None
            if bool(getattr(self, "fuse_industry_with_price_news", False)):
                _ind_tok = self._build_industry_token(
                    p_b, batch, num_days, num_stocks, device
                )
            out = self.unitrans.forward_unified(
                _price_arg,
                n_call,
                virt_tok=virt_tok,
                industry_tok=_ind_tok,
            )
        else:
            # interleaved UniTransStock: always needs [B,K,D] news tokens
            if n_b is None:
                n_b = self.unitrans_pad_news.view(1, 1, -1).expand(p_b.size(0), K, -1)
            elif n_b.dim() == 4:
                n_b = n_b.mean(dim=2)
            out = self.unitrans(p_b, n_b)

        price_emb_5 = p_b
        return out, price_emb_5, out, None

    @staticmethod
    def _infer_price_grid_shape(
        batch, n_price: int, grid_num_days: int, grid_num_stocks: int,
    ) -> tuple[int, int]:
        """Infer (num_days, num_stocks) when segment length varies (tail partial segment).

        Data packs ``N_price = num_days * num_stocks``; prefer ``n_price // grid_num_stocks``.
        Fallback: ``price.day_id`` max+1 when stock count does not divide.
        """
        num_stocks = int(grid_num_stocks)
        if num_stocks > 0 and n_price % num_stocks == 0:
            num_days = n_price // num_stocks
        elif "price" in batch and hasattr(batch["price"], "day_id"):
            day_ids = batch["price"].day_id
            num_days = int(day_ids.max().item()) + 1 if day_ids.numel() > 0 else int(grid_num_days)
            num_days = max(1, num_days)
            if n_price % num_days != 0:
                raise RuntimeError(
                    f"price grid mismatch: N_price={n_price}, num_days={num_days}, "
                    f"num_stocks={num_stocks}"
                )
            num_stocks = n_price // num_days
        else:
            num_days = max(1, int(grid_num_days))
            if n_price % num_days != 0:
                raise RuntimeError(
                    f"price grid mismatch: N_price={n_price}, num_days={num_days}, "
                    f"num_stocks={num_stocks}"
                )
            num_stocks = n_price // num_days
        if n_price != num_days * num_stocks:
            raise RuntimeError(
                f"price grid mismatch: N_price={n_price} != {num_days}*{num_stocks}={num_days * num_stocks}"
            )
        return num_days, num_stocks


    def forward(self, batch):
        """WindowBatch forward → PredFusion score [N_price]."""
        device = next(self.parameters()).device

        price_x = batch["price"].x.to(device)  # [N_price, T, F] 图缓存为 HLC+weekday
        # tokenizer 只吃 H/L/C；绝不把 weekday 当特征。
        if price_x.dim() == 3:
            fdim = int(price_x.size(-1))
            if fdim >= 3:
                price_x = price_x[..., :3]
            else:
                pad = price_x.new_zeros(*price_x.shape[:-1], 3 - fdim)
                price_x = torch.cat([price_x, pad], dim=-1)
        if price_x.numel() == 0:
            return torch.zeros((0,), device=device)
        n_price_raw = int(price_x.size(0))
        num_days, num_stocks = self._infer_price_grid_shape(
            batch, n_price_raw, self.grid_num_days, self.grid_num_stocks,
        )
        x = price_x.view(1, num_days, num_stocks, price_x.size(1), price_x.size(2))
        batch_size, num_days, num_stocks, seq_length, input_dim = x.shape

        x_seq = x.reshape(batch_size * num_days * num_stocks, seq_length, input_dim)
        N_price = int(x_seq.size(0))

        price_emb_from_last5, price_emb_5, price_emb_penultimate, _ = self._unitransstock_price_news_encode(
            x_seq, batch, num_days, num_stocks, device
        )
        price_emb = price_emb_from_last5.view(num_days, num_stocks, -1)
        price_emb_flat = price_emb_from_last5
        price_day_ids = batch["price"].day_id.to(device).long()
        news_feat_for_price = price_emb_flat.new_zeros(
            (N_price, int(self.news_mlp_slot_dim if self.dual_own_prop_news else self.news_main_feat_dim))
        )
        news_global_feat = None
        # 3) global_feat：按日横截面均值投影后与 global_parameter 逐元素相乘，再广播到各股票
        global_feat = None
        _use_pred_fusion = bool(getattr(self, "use_pred_fusion_head", True)) and (
            getattr(self, "pred_fusion_tf", None) is not None
        )
        if self.include_global_in_main_mlp or _use_pred_fusion:
            day_mean = price_emb.mean(dim=1, keepdim=True)  # [num_days,1,D]
            gp = self.global_parameter.repeat(2, 1, 1)[:num_days]  # [num_days,1,Dg]
            if self.vin_connect_mode == "mkt_as_ind" and self.global_connect_proj is not None:
                # Keep day-phase g; connect like industry: Proj([mean ‖ g_day]).
                global_day = self.global_connect_proj(torch.cat([day_mean, gp], dim=-1))
            else:
                global_day = self.global_mean_proj(day_mean) * gp
            global_feat = global_day.expand(num_days, num_stocks, -1).reshape(num_days * num_stocks, -1)

        # 4) Industry token for PredFusion.
        industry_feat_for_price = None
        if self.include_industry_in_pred_fusion:
            industry_feat_for_price = torch.zeros(
                (N_price, price_emb_flat.size(-1)), dtype=torch.float32, device=device
            )
            use_industry = (
                "industry_type" in batch.node_types
                and ("price", "to", "industry_type") in batch.edge_types
            )
            if use_industry and self.pred_fusion_industry_proj is not None:
                industry_ids = batch["industry_type"].x[:, 0].to(device).long().clamp(0, self.num_industry - 1)
                vi_node = self.virtual_industry_node.squeeze(0)  # [num_industry, D]
                if bool(getattr(self, "industry_no_mean", False)) and self.vin_connect_mode != "ind_as_mkt":
                    industry_feat_for_price = self.pred_fusion_industry_proj(vi_node[industry_ids])
                else:
                    days = price_day_ids.to(device=device, dtype=torch.long).view(-1)
                    if days.numel() != N_price:
                        days = batch["price"].day_id.to(device).long().view(-1)
                    d_dim = int(price_emb_flat.size(-1))
                    sum_agg = price_emb_flat.new_zeros((self.num_industry, d_dim))
                    sum_cnt = price_emb_flat.new_zeros((self.num_industry, 1))
                    max_day = int(days.max().item()) if days.numel() else -1
                    for d in range(max_day + 1):
                        mask_d = days == d
                        if not bool(mask_d.any()):
                            continue
                        ids_d = industry_ids[mask_d]
                        emb_d = price_emb_flat[mask_d]
                        sum_agg.index_add_(0, ids_d, emb_d)
                        sum_cnt.index_add_(0, ids_d, emb_d.new_ones((int(mask_d.sum().item()), 1)))
                        mean_sel = (sum_agg / sum_cnt.clamp_min(1.0))[ids_d]
                        vi_sel = vi_node[ids_d]
                        if self.vin_connect_mode == "ind_as_mkt":
                            industry_feat_for_price[mask_d] = self.pred_fusion_industry_proj(
                                mean_sel * vi_sel
                            )
                        else:
                            industry_feat_for_price[mask_d] = self.pred_fusion_industry_proj(
                                torch.cat([mean_sel, vi_sel], dim=-1)
                            )

        # 5) PredFusion → Linear(D→1)
        if industry_feat_for_price is None:
            industry_feat_for_price = price_emb_flat.new_zeros(
                (N_price, price_emb_flat.size(-1))
            )
        if global_feat is None:
            global_feat = price_emb_flat.new_zeros((N_price, self.global_feat_dim))

        _vin_head = self.price_encoder in (
            "cross_attn",
            "partial_unified",
            "full_unified",
        )
        # 最终 h（不再拼 K 天股价编码）
        _tok_list = [price_emb_flat]
        if (not _vin_head) and self.include_news_in_main_mlp:
            _nf = news_feat_for_price
            if _nf is None:
                _nf = price_emb_flat.new_zeros((N_price, self.news_main_feat_dim))
            # dual_own_prop 等宽新闻：取前 D 或投影
            if int(_nf.size(-1)) == int(self.price_emb_dim):
                _tok_list.append(_nf)
            elif hasattr(self, "pred_fusion_news_proj") and self.pred_fusion_news_proj is not None:
                # 若是 2*slot，先 mean 两路再投
                if self.dual_own_prop_news and int(_nf.size(-1)) == int(self.news_mlp_slot_dim):
                    _half = int(self.news_main_feat_dim)
                    _nf = 0.5 * (_nf[:, :_half] + _nf[:, _half:])
                _tok_list.append(self.pred_fusion_news_proj(_nf))
            else:
                _tok_list.append(_nf[:, : self.price_emb_dim])
        if self.include_global_in_main_mlp:
            _tok_list.append(self.pred_fusion_global_proj(global_feat))
            _score_1d = self.reduce_dim(global_feat)
            self._last_global_score_1d = _score_1d.detach()
            if (
                self.use_pred_fusion_global_score
                and self.pred_fusion_global_score_proj is not None
            ):
                _tok_list.append(self.pred_fusion_global_score_proj(_score_1d))
        elif (
            self.use_pred_fusion_global_score
            and self.pred_fusion_global_score_proj is not None
        ):
            _score_1d = self.reduce_dim(global_feat)
            self._last_global_score_1d = _score_1d.detach()
            _tok_list.append(self.pred_fusion_global_score_proj(_score_1d))
        if self.include_industry_in_pred_fusion:
            _tok_list.append(industry_feat_for_price)
        if len(_tok_list) != int(getattr(self, "pred_fusion_num_tokens", len(_tok_list))):
            raise RuntimeError(
                f"pred_fusion token count mismatch: got {len(_tok_list)}, "
                f"expected {self.pred_fusion_num_tokens}"
            )
        tokens = torch.stack(_tok_list, dim=1)
        return self.pred_fusion_tf(tokens)
