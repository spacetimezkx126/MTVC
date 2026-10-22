"""Cross-Attention ablation encoder (--price_encoder cross_attn).

Extracted from code/mtvc/model.py for the transition tree.
Paper name: Cross-Attention. Formal CLI: cross_attn.
"""

from __future__ import annotations
import torch
from torch import nn
from tool.numeric import stable_attn_softmax


class _CrossAttnRMSNorm(nn.Module):
    """RMSNorm used inside ``CrossAttnPostFusion`` (cross_attn only)."""

    def __init__(self, dim: int, eps: float = 1e-06):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale ``x`` by the inverse root-mean-square."""
        inv = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * inv * self.weight


class _CrossAttnDualStreamAttention(nn.Module):
    """Dual-stream MHA for Cross-Attention post-fusion (price side + news side).

    Used only by ``CrossAttnPostFusion`` after the price←news Cross Transformer.
    Not part of ``PartialUnifiedModel`` / full-unified.
    """

    def __init__(self, d_s: int, d_ns: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        (d_s, d_ns, nhead) = (int(d_s), int(d_ns), int(nhead))
        while d_s % nhead != 0 and nhead > 1:
            nhead //= 2
        self.d_s = d_s
        self.d_ns = d_ns
        self.nhead = nhead
        self.head_dim = d_s // nhead
        self.scale = self.head_dim ** (-0.5)
        self.qkv_s = nn.Linear(d_s, 3 * d_s, bias=False)
        self.qkv_ns = nn.Linear(d_ns, 3 * d_s, bias=False)
        self.out_s = nn.Linear(d_s, d_s, bias=False)
        self.out_ns = nn.Linear(d_s, d_ns, bias=False)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self, x_s: torch.Tensor, x_ns: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Attend over the concatenated price and news streams."""
        (bsz, ls, _) = x_s.shape
        qkv_s = self.qkv_s(x_s)
        if x_ns is None or x_ns.numel() == 0 or x_ns.size(1) == 0:
            qkv = qkv_s
            lns = 0
        else:
            qkv = torch.cat([qkv_s, self.qkv_ns(x_ns)], dim=1)
            lns = int(x_ns.size(1))
        length = ls + lns
        (q, k, v) = qkv.chunk(3, dim=-1)
        q = q.view(bsz, length, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(bsz, length, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(bsz, length, self.nhead, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(stable_attn_softmax(scores, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bsz, length, self.d_s)
        out_s = self.out_s(out[:, :ls, :])
        if lns == 0:
            return (out_s, None)
        return (out_s, self.out_ns(out[:, ls:, :]))


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
        """Apply a separate feed-forward network to each stream."""
        y_s = self.ff_s(x_s)
        if x_ns is None or x_ns.numel() == 0 or x_ns.size(1) == 0:
            return (y_s, None)
        return (y_s, self.ff_ns(x_ns))


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
        """One residual dual-stream block per fusion layer."""
        for layer in self.layers:
            hs = layer["norm_s1"](x_s)
            hns = layer["norm_ns1"](x_ns) if x_ns is not None and x_ns.size(1) > 0 else None
            (ds, dns) = layer["attn"](hs, hns)
            x_s = x_s + ds
            if dns is not None and x_ns is not None:
                x_ns = x_ns + dns
            hs = layer["norm_s2"](x_s)
            hns = layer["norm_ns2"](x_ns) if x_ns is not None and x_ns.size(1) > 0 else None
            (ds, dns) = layer["ffn"](hs, hns)
            x_s = x_s + ds
            if dns is not None and x_ns is not None:
                x_ns = x_ns + dns
        return (x_s, x_ns)


def _make_transformer_encoder(layer: nn.Module, num_layers: int) -> nn.TransformerEncoder:
    """Shared TransformerEncoder builder (Cross-Attention Enc + SeqTF / PredFusion)."""
    try:
        return nn.TransformerEncoder(layer, num_layers=int(num_layers), enable_nested_tensor=False)
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
            else self.num_days if self.include_virtual else 0
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
        if self.use_type_emb:
            self.type_emb = nn.Embedding(3, self.d_model)
            nn.init.trunc_normal_(self.type_emb.weight, std=0.02)
        else:
            self.type_emb = None
        self.news_up = (
            nn.Identity() if self.d_news == self.d_model else nn.Linear(self.d_news, self.d_model)
        )
        self.news_down = (
            nn.Identity() if self.d_news == self.d_model else nn.Linear(self.d_model, self.d_news)
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

        self.price_transformer_encoder = _make_transformer_encoder_stack(self.price_encoder_layers)
        self.news_transformer_encoder = _make_transformer_encoder_stack(self.news_encoder_layers)
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
                transformer_layer, num_layers=self.transformer_layers, enable_nested_tensor=False
            )
        except TypeError:
            self.transformer = nn.TransformerDecoder(
                transformer_layer, num_layers=self.transformer_layers
            )
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
        self.prune_mode = "off"
        self.last_mtvc_gate_w: float | None = None
        self.last_mtvc_price: torch.Tensor | None = None
        self.last_mtvc_news: torch.Tensor | None = None
        self.last_mtvc_case_price: torch.Tensor | None = None
        self.last_mtvc_case_news: torch.Tensor | None = None
        self.last_mtvc_gate_w_mean = None

    def _shared_day_pos(self, k: int) -> torch.Tensor:
        """[K, D] calendar-day PE; shared by price day t and news slots of day t."""
        return self.day_pos.weight[:k]

    def _expand_day_pos_to_news_slots(self, day_p: torch.Tensor, m_slots: int) -> torch.Tensor:
        """Repeat each day vector across that day's news slots: [K,D] → [K*M,D]."""
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
    def _causal_bool_mask_with_prefix(
        k: int, device: torch.device, n_prefix: int = 1
    ) -> torch.Tensor:
        """Causal mask for [prefix | K days]. Prefix sees everything; True means block."""
        n_prefix = max(1, int(n_prefix))
        n = n_prefix + int(k)
        m = torch.ones(n, n, device=device, dtype=torch.bool)
        m[:n_prefix, :] = False
        for i in range(int(k)):
            m[n_prefix + i, : n_prefix + i + 1] = False
        return m

    def _causal_memory_mask_with_prefix(
        self, k: int, device: torch.device, n_prefix: int = 1, news_per_day: int = 1
    ) -> torch.Tensor | None:
        """Block future-day news for tgt=[prefix|days] against memory [K*M]. True means block."""
        if self.price_news_align == "none":
            return None
        n_prefix = max(1, int(n_prefix))
        m_slots = max(1, int(news_per_day))
        mem_len = int(k) * m_slots
        m = torch.zeros(n_prefix + int(k), mem_len, device=device, dtype=torch.bool)
        key_day = torch.arange(k, device=device).repeat_interleave(m_slots).view(1, mem_len)
        for i in range(int(k)):
            m[n_prefix + i] = key_day > i
        return m

    def _normalize_industry_tok(self, industry_tok: torch.Tensor, bsz: int, d: int) -> torch.Tensor:
        """Squeeze an industry token to [B, 1, D]."""
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
        (bsz, k, d) = price_h.shape
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
            if i == len(layers) - 1 and industry_tok is not None and (industry_tok.numel() > 0):
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

    def encode_contrast_towers(self, price_tok, news_tok=None, **_ignored):
        """Contrast helper. Price tokens are already day embeddings; news is not re-encoded."""
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
        vol_tok: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Price days cross-attend to news slots, then dual-stream fusion. Returns [B, D]."""
        del vol_tok
        if price_tok is None:
            raise ValueError("CrossAttentionEncoder requires price_tok [B,K,D]")
        if price_tok.dim() != 3:
            raise ValueError(f"price_tok expected [B,K,D], got {tuple(price_tok.shape)}")
        (bsz, k, d) = price_tok.shape
        if k != self.num_days or d != self.d_model:
            raise ValueError(
                f"price_tok {tuple(price_tok.shape)} != K={self.num_days} D={self.d_model}"
            )
        device = price_tok.device
        if news_day_idx is not None and news_tok is not None and (news_tok.dim() == 3):
            (bsz_n, nmax, dn) = news_tok.shape
            src = news_tok
            if dn != int(self.d_news):
                if dn == int(self.d_model):
                    src = self.news_down(news_tok)
                else:
                    raise ValueError(f"mtvc news_tok D={dn} != d_news={self.d_news}")
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
            pad = news_tok.abs().sum(dim=-1) == 0
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
            vw = self.vin_window_emb[: min(self.num_v_window, k)].unsqueeze(0).expand(bsz, -1, -1)
            if te is not None:
                vw = vw + te[2].view(1, 1, -1)
            s_parts.append(vw)
        ph = price_h
        if te is not None:
            ph = ph + te[0].view(1, 1, -1)
        s_parts.append(ph)
        x_s = torch.cat(s_parts, dim=1)
        (x_s, _) = self.price_news_fusion(x_s, news_ns)
        self.last_mtvc_gate_w_mean = None
        self.last_mtvc_gate_w = None
        self.last_mtvc_case_price = price_h.mean(dim=1)
        self.last_mtvc_price = self.last_mtvc_case_price
        if news_memory is not None:
            valid = (
                (~news_pad_flat).unsqueeze(-1).to(dtype=news_memory.dtype)
                if news_pad_flat is not None
                else news_memory.new_ones(news_memory.shape[:2] + (1,))
            )
            self.last_mtvc_case_news = (news_memory * valid).sum(dim=1) / valid.sum(
                dim=1
            ).clamp_min(1.0)
        else:
            self.last_mtvc_case_news = price_h.new_zeros(bsz, d)
        self.last_mtvc_news = self.last_mtvc_case_news
        return self.out_norm(x_s[:, 0, :])

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Alias of ``forward_unified``."""
        return self.forward_unified(*args, **kwargs)

    def forward_with_price_aux(self, *args, **kwargs):
        """Return ``(h, h)`` for callers that also expect a price auxiliary vector."""
        h = self.forward_unified(*args, **kwargs)
        return (h, h)


DiagramDualTF = CrossAttentionEncoder
PriceNewsFusion = CrossAttnPostFusion
