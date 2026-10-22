"""Partially / fully unified joint self-attention backbone for price+news.

Sequence (per prediction row):
  [state] + price_day[0..K-1] + news_tokens(variable, same stock, days in window)

- Same calendar day_pos for price day t and all news of day t
- type_emb: 0=price, 1=news (state uses neither / or type 0)
- Causal day mask (MTVC_UNIFIED_CAUSAL, default ON):
    * state may attend to all tokens; all tokens may attend to state
    * price/news at day t may only attend to price/news with day j <= t
- Same-day mask (MTVC_UNIFIED_SAME_DAY=1):
    * state free as above
    * price/news at day t may only attend to tokens with day j == t
      (price↔news cross-day blocked; also no cross-day within modality)
- Readout = state token output
"""

from __future__ import annotations
import os
import torch
import torch.nn as nn


def _nhead_for(d: int, prefer: int = 4) -> int:
    """Largest head count at most ``prefer`` that divides ``d``."""
    h = int(prefer)
    while d % h != 0 and h > 1:
        h //= 2
    return max(1, h)


def _env_causal_default_on() -> bool:
    """MTVC_UNIFIED_CAUSAL, default on."""
    from naming import env_mtvc

    return str(env_mtvc("UNIFIED_CAUSAL", "1")).strip().lower() not in ("0", "false", "off", "no")


def _env_same_day_on() -> bool:
    """MTVC_UNIFIED_SAME_DAY, default off."""
    from naming import env_mtvc

    return str(env_mtvc("UNIFIED_SAME_DAY", "0")).strip().lower() in ("1", "true", "yes", "on")


def _make_encoder(
    d_model: int, nhead: int, num_layers: int, dropout: float
) -> nn.TransformerEncoder:
    """Joint self-attention stack. Nested tensor is off."""
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
            self.type_emb = nn.Embedding(2, self.d_model)
            nn.init.trunc_normal_(self.type_emb.weight, std=0.02)
        else:
            self.type_emb = None
        self.encoder = _make_encoder(self.d_model, nhead, self.num_layers, dropout)
        self.out_norm = nn.LayerNorm(self.d_model)
        self.ssl_mask_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        nn.init.trunc_normal_(self.ssl_mask_token, std=0.02)
        self.news_gate_enabled = False
        self.news_gates: nn.ModuleList | None = None
        self.case_mine_layer: int = 6
        self.last_mtvc_gate_w: float | None = None
        self.last_mtvc_price: torch.Tensor | None = None
        self.last_mtvc_news: torch.Tensor | None = None
        self.soft_prune = False
        self.prune_mode = "off"
        self.last_news_keep = None
        self.last_cross_keep = None
        self.include_news = True
        self.include_price = True

    def _day_layout(
        self, bsz: int, k: int, news_day_idx: torch.Tensor | None, nmax: int, device: torch.device
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
        return (day, is_state_q, is_state_k)

    def _causal_src_mask(
        self, bsz: int, k: int, news_day_idx: torch.Tensor | None, nmax: int, device: torch.device
    ) -> torch.Tensor:
        """Bool attn mask [B*nhead, L, L]: True=block. Day rule: key.day <= query.day."""
        (day, is_state_q, is_state_k) = self._day_layout(bsz, k, news_day_idx, nmax, device)
        q_day = day.unsqueeze(2)
        k_day = day.unsqueeze(1)
        allow = is_state_q | is_state_k | (k_day <= q_day)
        L = day.size(1)
        block = (~allow).unsqueeze(1).expand(bsz, self.nhead, L, L)
        return block.reshape(bsz * self.nhead, L, L).contiguous()

    def _same_day_src_mask(
        self, bsz: int, k: int, news_day_idx: torch.Tensor | None, nmax: int, device: torch.device
    ) -> torch.Tensor:
        """Bool attn mask [B*nhead, L, L]: True=block. Day rule: key.day == query.day.

        Price day t and news of day t can attend each other; cross-day blocked.
        State remains a free prefix (sees all / visible to all).
        """
        (day, is_state_q, is_state_k) = self._day_layout(bsz, k, news_day_idx, nmax, device)
        q_day = day.unsqueeze(2)
        k_day = day.unsqueeze(1)
        allow = is_state_q | is_state_k | (k_day == q_day)
        L = day.size(1)
        block = (~allow).unsqueeze(1).expand(bsz, self.nhead, L, L)
        return block.reshape(bsz * self.nhead, L, L).contiguous()

    def enable_news_gate(self, enabled: bool = True) -> None:
        """Create / toggle per-layer news gates (MTVC contrast)."""
        self.news_gate_enabled = bool(enabled)
        if self.news_gate_enabled and self.news_gates is None:
            n_gate = max(0, self.num_layers - 1)
            self.news_gates = nn.ModuleList([nn.Linear(self.d_model, 1) for _ in range(n_gate)])
            for g in self.news_gates:
                nn.init.zeros_(g.weight)
                nn.init.zeros_(g.bias)
            try:
                dev = next(self.encoder.parameters()).device
                self.news_gates.to(dev)
            except StopIteration:
                pass

    def _apply_news_layer_gate(
        self, h: torch.Tensor, pad: torch.Tensor, k: int, nmax: int, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Scale news tokens by sample-level wi after layer ``layer_idx`` (0-based)."""
        if (
            not self.news_gate_enabled
            or self.news_gates is None
            or nmax <= 0
            or (layer_idx >= len(self.news_gates))
        ):
            return (h, None)
        news_h = h[:, 1 + k : 1 + k + nmax, :]
        news_pad = pad[:, 1 + k : 1 + k + nmax]
        valid = ~news_pad
        w_mask = valid.unsqueeze(-1).to(dtype=news_h.dtype)
        summed = (news_h * w_mask).sum(dim=1)
        denom = w_mask.sum(dim=1).clamp_min(1.0)
        pooled = summed / denom
        has_news = valid.any(dim=1)
        logit = self.news_gates[layer_idx](pooled)
        w = torch.sigmoid(logit).squeeze(-1)
        w = torch.where(has_news, w, torch.ones_like(w))
        news_scaled = news_h * w.view(-1, 1, 1)
        news_scaled = news_scaled.masked_fill(news_pad.unsqueeze(-1), 0.0)
        h = h.clone()
        h[:, 1 + k : 1 + k + nmax, :] = news_scaled
        return (h, w)

    def _run_encoder(
        self, x: torch.Tensor, pad: torch.Tensor, src_mask: torch.Tensor | None, nmax: int, k: int
    ) -> torch.Tensor:
        """Joint SA; optional news-token attn_decay residual halt (enc site)."""
        pad = pad
        try:
            from tool.numeric import sanitize_hidden, sanitize_pad_mask
        except ImportError:
            from tool.numeric import sanitize_hidden, sanitize_pad_mask
        pad = sanitize_pad_mask(pad)
        self.last_mtvc_gate_w = None
        self.last_mtvc_price = None
        self.last_mtvc_news = None
        if self.news_gate_enabled and self.news_gates is not None:
            layers = list(self.encoder.layers)
            h = x
            w_acc: list[torch.Tensor] = []
            for i, layer in enumerate(layers):
                h = layer(h, src_mask=src_mask, src_key_padding_mask=pad)
                h = sanitize_hidden(h)
                _case_l = int(getattr(self, "case_mine_layer", 6) or 6)
                _case_l = max(1, min(_case_l, len(layers)))
                if i == _case_l - 1:
                    price_h = h[:, 1 : 1 + k, :]
                    self.last_mtvc_price = price_h.mean(dim=1)
                    if nmax > 0:
                        news_h = h[:, 1 + k : 1 + k + nmax, :]
                        news_pad = pad[:, 1 + k : 1 + k + nmax]
                        valid = (~news_pad).unsqueeze(-1).to(dtype=news_h.dtype)
                        summed = (news_h * valid).sum(dim=1)
                        denom = valid.sum(dim=1).clamp_min(1.0)
                        self.last_mtvc_news = summed / denom
                    else:
                        self.last_mtvc_news = h.new_zeros(h.size(0), h.size(-1))
                if i < len(layers) - 1 and nmax > 0:
                    (h, w) = self._apply_news_layer_gate(h, pad, k, nmax, i)
                    if w is not None:
                        w_acc.append(w.detach())
            if w_acc:
                self.last_mtvc_gate_w = float(torch.stack(w_acc).mean().item())
            return sanitize_hidden(h)
        return sanitize_hidden(self.encoder(x, mask=src_mask, src_key_padding_mask=pad))

    def _attn_src_mask(
        self, bsz: int, k: int, news_day_idx: torch.Tensor | None, nmax: int, device: torch.device
    ) -> torch.Tensor | None:
        """Causal mask, same-day mask, or no mask."""
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
        (bsz, k, d) = price_tok.shape
        if k != self.num_days or d != self.d_model:
            raise ValueError(
                f"price_tok {tuple(price_tok.shape)} != K={self.num_days} D={self.d_model}"
            )
        device = price_tok.device
        day_p = self.day_pos.weight[:k]
        p = price_tok + day_p.unsqueeze(0)
        if self.type_emb is not None:
            p = p + self.type_emb.weight[0].view(1, 1, -1)
        parts = [self.state_tok.expand(bsz, -1, -1), p]
        pad_parts = [
            torch.zeros(bsz, 1, dtype=torch.bool, device=device),
            torch.zeros(bsz, k, dtype=torch.bool, device=device),
        ]
        nmax = 0
        news_day_for_mask: torch.Tensor | None = None
        if news_tok is not None and news_tok.numel() > 0 and (news_tok.size(1) > 0):
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
        h = self.out_norm(x[:, 0, :])
        return h

    def encode_contrast_towers(
        self,
        price_tok: torch.Tensor,
        news_tok: torch.Tensor | None = None,
        news_day_idx: torch.Tensor | None = None,
        news_pad: torch.Tensor | None = None,
        price_day_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """Joint-encode for contrast helpers; return price-day / news-day pools.

        Runs the same joint encoder as supervised MTVC (encode_contrast_towers).

        Returns:
          price_h [B,K,D], price_tok [B,K,D] (unmasked recon target),
          news_day [B,K,D] or None, news_day_valid [B,K] or None,
          price_day_mask [B,K] or None.
        """
        if price_tok.dim() != 3:
            raise ValueError(f"price_tok expected [B,K,D], got {tuple(price_tok.shape)}")
        (bsz, k, d) = price_tok.shape
        if k != self.num_days or d != self.d_model:
            raise ValueError(
                f"price_tok {tuple(price_tok.shape)} != K={self.num_days} D={self.d_model}"
            )
        device = price_tok.device
        price_tgt = price_tok
        p_in = price_tok
        if price_day_mask is not None:
            if price_day_mask.shape != (bsz, k):
                raise ValueError(f"price_day_mask {tuple(price_day_mask.shape)} != {(bsz, k)}")
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
        if news_tok is not None and news_tok.numel() > 0 and (news_tok.size(1) > 0):
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
        if nmax > 0 and di is not None and (news_pad is not None):
            news_out = x[:, 1 + k :, :]
            valid = ~news_pad
            news_day = price_tok.new_zeros(bsz, k, d)
            idx = di.unsqueeze(-1).expand(-1, -1, d)
            news_day.scatter_add_(1, idx, news_out.masked_fill(~valid.unsqueeze(-1), 0.0))
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
        """Alias of ``forward_unified``."""
        return self.forward_unified(*args, **kwargs)


class FullUnifiedModel(PartialUnifiedModel):
    """Fully unified joint encoder.

    Same self-attention backbone as ``PartialUnifiedModel``. The only extra
    piece is ``price_day_mlp`` (HLC → token), built by ``build_price_day_mlp``
    after the news projection so init order matches the paper run.
    News tokens still come from ``NewsTransformerEncoder``.
    """

    def __init__(self, *args, **kwargs):
        """Same backbone as partial-unified, with mode forced to full."""
        kwargs["mode"] = "full"
        super().__init__(*args, **kwargs)
        self.price_day_mlp: nn.Module | None = None

    def build_price_day_mlp(self, in_dim: int) -> nn.Module:
        """Build the HLC TokenMLP. Call this after the news projection."""
        try:
            from tool.encoders import TokenMLP
        except ImportError:
            from .encoders import TokenMLP
        self.price_day_mlp = TokenMLP(in_dim=max(1, int(in_dim)), d_model=int(self.d_model))
        return self.price_day_mlp

    def encode_price_day(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Last-step H/L/C → TokenMLP → [N, D]."""
        if self.price_day_mlp is None:
            raise RuntimeError("FullUnifiedModel.price_day_mlp is not built")
        n_ch = int(self.price_day_mlp.in_dim)
        f_avail = int(x_seq.size(-1))
        last = x_seq[:, -1, : min(n_ch, f_avail)]
        if last.size(-1) < n_ch:
            last = torch.cat([last, last.new_zeros(last.size(0), n_ch - last.size(-1))], dim=-1)
        return self.price_day_mlp(last)
