"""Price, news, and prediction-head encoders.

``SeqTransformerPool`` pools an H/L/C channel sequence into a day vector.
``NewsTransformerEncoder`` maps vocab ids to a news vector.
``TokenMLP`` maps last-step H/L/C to a day token for full-unified.
``PredFusionTransformer`` mixes the joint state with optional global and industry tokens.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenMLP(nn.Module):
    """OneTrans-style tokenizer: concat raw features → MLP → one token."""

    def __init__(self, in_dim: int, d_model: int, hidden: int | None = None):
        super().__init__()
        self.in_dim = max(1, int(in_dim))
        hid = int(hidden) if hidden is not None else max(int(d_model), self.in_dim)
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, hid), nn.GELU(), nn.Linear(hid, int(d_model))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map raw features to one token. Accepts [N, F] or [N, T, F]."""
        if x.dim() == 3:
            x = x.reshape(x.size(0), -1)
        elif x.dim() != 2:
            raise ValueError(f"TokenMLP expected [N,F] or [N,T,F], got {tuple(x.shape)}")
        (n, f) = x.shape
        if f < self.in_dim:
            x = torch.cat([x, x.new_zeros((n, self.in_dim - f))], dim=-1)
        elif f > self.in_dim:
            x = x[:, : self.in_dim]
        return self.net(x)


class SeqTransformerPool(nn.Module):
    """
    Token / timestep sequence → small TransformerEncoder → masked mean pool → optional out_proj.
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
        soft_prune: bool = False,
        loop_rounds: int = 1,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.max_len = max(1, int(max_len))
        self.out_dim = int(out_dim) if out_dim is not None else self.d_model
        self.soft_prune = bool(soft_prune)
        self.loop_rounds = max(1, int(loop_rounds))
        self.num_layers = max(1, int(num_layers))
        nhead = int(nhead)
        while self.d_model % nhead != 0 and nhead > 1:
            nhead //= 2
        self.nhead = max(1, nhead)
        self.in_proj = (
            nn.Linear(int(in_dim), self.d_model) if int(in_dim) != self.d_model else nn.Identity()
        )
        self.pos_emb = nn.Parameter(torch.zeros(1, self.max_len, self.d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        ff = max(self.d_model, self.d_model * max(1, int(dim_ff_mult)))
        self.layers = None
        self.kw_gates = None
        self.encoder = None
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
        self.last_kw_keep: torch.Tensor | None = None
        self.last_kw_keep_mean: torch.Tensor | None = None
        self.last_kw_keep_layers: list[torch.Tensor] | None = None

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None, chunk_size: int = 0
    ) -> torch.Tensor:
        """
        x: [N, L, Fin]
        key_padding_mask: [N, L] True = pad (ignored), optional
        chunk_size: if >0, encode in chunks of this size (saves activation memory)
        returns: [N, out_dim]
        """
        if x.dim() != 3:
            raise ValueError(f"expected [N,L,F], got {tuple(x.shape)}")
        n = int(x.size(0))
        if chunk_size and chunk_size > 0 and (n > chunk_size):
            outs = []
            keeps = []
            layer_bufs: list[list[torch.Tensor]] | None = None
            for i in range(0, n, chunk_size):
                sl = slice(i, i + chunk_size)
                m = None if key_padding_mask is None else key_padding_mask[sl]
                (o, k, layers_k) = self._forward_one(x[sl], m, return_keep=True)
                outs.append(o)
                if k is not None:
                    keeps.append(k)
                if layers_k:
                    if layer_bufs is None:
                        layer_bufs = [[] for _ in layers_k]
                    for li, lk in enumerate(layers_k):
                        layer_bufs[li].append(lk)
            self.last_kw_keep = torch.cat(keeps, dim=0) if keeps else None
            if self.last_kw_keep is not None:
                self.last_kw_keep_mean = self.last_kw_keep.mean()
            self.last_kw_keep_layers = (
                [torch.cat(bufs, dim=0) for bufs in layer_bufs] if layer_bufs else None
            )
            return torch.cat(outs, dim=0)
        (out, keep, layers_k) = self._forward_one(x, key_padding_mask, return_keep=True)
        self.last_kw_keep = keep
        self.last_kw_keep_mean = None if keep is None else keep.mean()
        self.last_kw_keep_layers = layers_k
        return out

    def _forward_one(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None, return_keep: bool = False
    ):
        """Run one unchunked sequence and masked-mean-pool it."""
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
        keep_acc = None
        layer_keeps: list[torch.Tensor] = []
        h = self.encoder(h, src_key_padding_mask=mask)
        h = self.out_norm(h)
        if mask is None:
            pooled = h.mean(dim=1)
        else:
            keep_m = (~mask).to(dtype=h.dtype).unsqueeze(-1)
            pooled = (h * keep_m).sum(dim=1) / keep_m.sum(dim=1).clamp(min=1.0)
        out = self.out_proj(pooled)
        if return_keep:
            return (out, keep_acc, layer_keeps)
        return out


class NewsTransformerEncoder(nn.Module):
    """
    Token ids → learnable Embedding → small Transformer → doc vector.
    No Sogou/GloVe/RoBERTa backbone weights; suitable for from-scratch MLM pretrain.
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


class PredFusionTransformer(nn.Module):
    """
    Outer task tower:
      tokens = [joint state, optional global, optional industry]
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
            return (logits, pooled)
        return logits


def _make_transformer_encoder(layer: nn.Module, num_layers: int) -> nn.TransformerEncoder:
    """Build TransformerEncoder with nested_tensor disabled (avoids illegal memory access)."""
    try:
        return nn.TransformerEncoder(layer, num_layers=int(num_layers), enable_nested_tensor=False)
    except TypeError:
        return nn.TransformerEncoder(layer, num_layers=int(num_layers))
