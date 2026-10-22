from __future__ import annotations
import torch
from naming import env_mtvc


class PriceNewsEncodingMixin:

    @staticmethod
    def _remap_vocab_input_ids(self, vocab_input_ids: torch.Tensor) -> torch.Tensor:
        """Map dataset vocab ids onto the frozen word-table rows."""
        vocab_input_ids = vocab_input_ids.clamp(0, self.vocab_size - 1)
        return self.vocab_id_remap[vocab_input_ids]

    def _encode_news_nodes_to_price_dim(
        self, batch, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode news with NewsTransformerEncoder. Output dim is news_emb_dim."""
        _unitrans = self.price_encoder in ("cross_attn", "partial_unified", "full_unified")
        _out_dim = int(self.news_emb_dim) if _unitrans else int(self.price_emb_dim)
        if "news" not in batch.node_types or int(batch["news"].x.size(0)) <= 0:
            z = torch.zeros((0, _out_dim), device=device)
            empty = torch.zeros((0,), dtype=torch.long, device=device)
            return (z, empty, empty)
        N_news = int(batch["news"].x.size(0))
        news_day_ids = batch["news"].day_id.detach().long().cpu()
        news_comp_ids = batch["news"].company_id.detach().long().cpu()
        if (
            _unitrans
            and self.bert_tok_transformer is not None
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
            chunk = 64
            news_512 = self.bert_tok_transformer(
                vocab_input_ids, attention_mask=attn, chunk_size=chunk
            )
            return (news_512, news_day_ids, news_comp_ids)
        raise RuntimeError(
            "news encode fell through: paper path needs vocab_input_ids + NewsTransformerEncoder"
        )

    def _unified_joint_encode(
        self, x_seq: torch.Tensor, batch, num_days: int, num_stocks: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Partial/full unified: price days + variable news → shared Encoder → h."""
        K = int(self.unitrans_num_days)
        D = int(self.price_emb_dim)
        day_ids = torch.arange(num_days, device=device)
        offsets = torch.arange(K, device=device) - (K - 1)
        src = day_ids[:, None] + offsets[None, :]
        valid = src >= 0
        src_clamped = src.clamp(min=0)
        valid_exp = valid[:, :, None, None]
        _pfill = str(getattr(self, "price_token_fill", "real") or "real")
        mode = str(getattr(self, "unified_mode", "partial"))
        if _pfill == "zero" or not bool(getattr(self, "include_price_token", True)):
            p_b = x_seq.new_zeros((num_days * num_stocks, K, D))
        elif _pfill == "pad":
            p_b = (
                self.unitrans_pad_price.view(1, 1, -1)
                .expand(num_days * num_stocks, K, D)
                .contiguous()
            )
        else:
            if mode == "full" and getattr(self.unitrans, "encode_price_day", None) is not None:
                price_day_tok = self.unitrans.encode_price_day(x_seq)
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
        _use_news = bool(getattr(self, "include_news_tower", True)) and bool(
            getattr(self, "include_news_token", True)
        )
        if _use_news:
            (news_feat, news_day_ids, news_comp_ids) = self._encode_news_nodes_to_price_dim(
                batch, device
            )
        else:
            news_feat = p_b.new_zeros((0, D))
            news_day_ids = torch.zeros((0,), dtype=torch.long, device=device)
            news_comp_ids = torch.zeros((0,), dtype=torch.long, device=device)
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
        import os as _os_uj

        _max_news = max(1, int(str(env_mtvc("UNIFIED_MAX_NEWS", "48") or "48").strip() or 48))
        _chunk = max(1, int(str(env_mtvc("UNIFIED_CHUNK", "256") or "256").strip() or 256))
        row_news: list[list[tuple[int, torch.Tensor]]] = [[] for _ in range(B)]
        if _use_news and news_feat.numel() > 0:
            for i in range(int(news_feat.size(0))):
                d_abs = int(news_day_ids[i].item())
                s = int(news_comp_ids[i].item())
                if s < 0 or s >= num_stocks or d_abs < 0 or (d_abs >= num_days):
                    continue
                d0 = max(0, d_abs)
                d1 = min(num_days, d_abs + K)
                for d_pred in range(d0, d1):
                    k_off = d_abs - (d_pred - (K - 1))
                    if k_off < 0 or k_off >= K:
                        continue
                    row = d_pred * num_stocks + s
                    row_news[row].append((k_off, news_feat[i]))
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
        return (h, p_b, h, None)

    def _encode_price_news(
        self, x_seq: torch.Tensor, batch, num_days: int, num_stocks: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Day-serial encode: partial/full joint path or cross_attn slot path."""
        if self.price_encoder in ("partial_unified", "full_unified"):
            return self._unified_joint_encode(x_seq, batch, num_days, num_stocks, device)
        if self.price_encoder == "cross_attn":
            return self._cross_attn_price_news_encode(x_seq, batch, num_days, num_stocks, device)
        raise ValueError(f"unsupported price_encoder={self.price_encoder!r}")

    @staticmethod
    def _infer_price_grid_shape(
        batch, n_price: int, grid_num_days: int, grid_num_stocks: int
    ) -> tuple[int, int]:
        """Infer (num_days, num_stocks) when segment length varies (tail partial segment)."""
        num_stocks = int(grid_num_stocks)
        range_dates = getattr(batch, "range_dates", None)
        if range_dates is None:
            try:
                range_dates = batch["range_dates"] if "range_dates" in batch else None
            except (TypeError, KeyError):
                range_dates = None
        if range_dates is not None:
            num_days = len(range_dates)
        elif hasattr(batch, "node_types") and "virtual_window" in batch.node_types:
            num_days = int(batch["virtual_window"].x.size(0))
        elif "price" in batch and hasattr(batch["price"], "day_id"):
            day_ids = batch["price"].day_id
            num_days = int(day_ids.max().item()) + 1 if day_ids.numel() > 0 else int(grid_num_days)
        else:
            num_days = int(grid_num_days)
        num_days = max(1, num_days)
        if n_price % num_stocks == 0:
            num_days = n_price // num_stocks
        elif n_price % num_days == 0:
            num_stocks = n_price // num_days
        else:
            raise RuntimeError(
                f"price grid mismatch: N_price={n_price}, num_days={num_days}, num_stocks={num_stocks}"
            )
        if n_price != num_days * num_stocks:
            raise RuntimeError(
                f"price grid mismatch: N_price={n_price} != {num_days}*{num_stocks}={num_days * num_stocks}"
            )
        return (num_days, num_stocks)


import torch
from naming import env_mtvc


def _nhead_for(d: int, prefer: int = 4) -> int:
    d = int(d)
    h = min(int(prefer), d)
    while h > 1 and d % h != 0:
        h //= 2
    return max(1, h)


class CrossAttentionMixin:

    def _unitrans_no_news_token(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """cross_attn empty-slot filler → [1,1,D]."""
        news_enc = getattr(self, "bert_tok_transformer", None)
        if news_enc is None:
            news_enc = getattr(self, "news_transformer_encoder", None)
        if news_enc is not None and getattr(self, "unitrans_no_news_vocab_ids", None) is not None:
            ids = self.unitrans_no_news_vocab_ids.to(device=device)
            tok = news_enc(ids, attention_mask=None, chunk_size=0)
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
        """cross_attn news packing: dense [D, S, M, dim] grid."""
        m = max(1, int(news_per_day))
        dim = int(news_512.size(-1)) if news_512.numel() > 0 else int(self.news_emb_dim)
        dtype = news_512.dtype if news_512.numel() else torch.float32
        use_encoder_nonews = (
            str(self.news_encode_mode).startswith("vocab")
            and (
                getattr(self, "bert_tok_transformer", None) is not None
                or getattr(self, "news_transformer_encoder", None) is not None
            )
            and (getattr(self, "unitrans_no_news_vocab_ids", None) is not None)
        )
        if use_encoder_nonews:
            fill = self._unitrans_no_news_token(device, dtype)
            fill = fill.view(1, 1, 1, dim).expand(num_days, num_stocks, m, dim).contiguous()
        elif self.unitrans_no_news_emb is not None:
            fill = (
                self.unitrans_no_news_emb.view(1, 1, 1, -1)
                .expand(num_days, num_stocks, m, dim)
                .contiguous()
            )
        else:
            fill = torch.zeros((num_days, num_stocks, m, dim), device=device, dtype=dtype)
        if news_512.numel() == 0:
            return fill.to(device=device, dtype=dtype)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        day_cpu = news_day_ids.detach().long().cpu().view(-1)
        comp_cpu = news_comp_ids.detach().long().cpu().view(-1)
        valid_cpu = (
            (day_cpu >= 0) & (day_cpu < num_days) & (comp_cpu >= 0) & (comp_cpu < num_stocks)
        )
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
        from tool.cross_attn import CrossAttentionEncoder

        self.unitrans_no_news_emb = None
        _k = int(self.unitrans_num_days)
        _nw = (
            0
            if self.vin_ablation_mode
            in ("no_window", "with_news_type_no_window", "no_news_type_no_window")
            else _k
        )
        _use_type = env_mtvc("USE_TYPE_EMB", "1").strip().lower() not in ("0", "false", "off", "no")
        _unified = str(env_mtvc("UNIFIED_LAYERS", "") or "").strip()
        if _unified:
            _L = max(1, int(_unified))
            _news_enc_L = _L
            _cross_L = _L
        else:
            _news_enc_L = max(1, int(env_mtvc("NEWS_ENCODER_LAYERS", "1") or 1))
            _cross_L = max(1, int(env_mtvc("TRANSFORMER_LAYERS", "2") or 2))
        _price_enc_L = max(1, int(env_mtvc("PRICE_ENCODER_LAYERS", "1") or 1))
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
        self.unified_news_proj = None
        self.unitrans_price_day_mlp = None
        print(
            f"[CROSS_ATTN] dual-tower: price_encoder=cross_attn include_news={int(include_news)} include_price={int(include_price)} news_ablation={getattr(self, 'news_ablation_mode', 'full')} no_price={int(not bool(getattr(self, 'include_price_token', True)))} days={_k} d_s={int(price_emb_dim)} news_slots=M={int(news_per_day)} d_news={int(news_emb_dim)} Enc(price/news)={_price_enc_L}/{_news_enc_L} CrossTF(layers)={_cross_L} CrossAttnPostFusion=1 mem=expand_KM band=causal type_emb={int(_use_type)} align=strict+day_pos vin_window={_nw}; HLC×3→SeqTF; within-day news slot TF (no mean-pool); causal price←news cross on K*M slots (day j<=i); CrossTF → CrossAttnPostFusion → h; PredFusion",
            flush=True,
        )

    def _cross_attn_price_news_encode(
        self, x_seq: torch.Tensor, batch, num_days: int, num_stocks: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """cross_attn: fixed K×M news slots + CrossAttentionEncoder."""
        K = int(self.unitrans_num_days)
        include_news = bool(getattr(self.unitrans, "include_news", True))
        _pfill = str(getattr(self, "price_token_fill", "real") or "real")
        day_ids = torch.arange(num_days, device=device)
        offsets = torch.arange(K, device=device) - (K - 1)
        src = day_ids[:, None] + offsets[None, :]
        valid = src >= 0
        src_clamped = src.clamp(min=0)
        valid_exp = valid[:, :, None, None]
        if _pfill == "zero" or not bool(getattr(self.unitrans, "include_price", True)):
            p_b = x_seq.new_zeros((num_days * num_stocks, K, int(self.price_emb_dim)))
        else:
            if getattr(self, "unitrans_price_mlp", None) is not None:
                price_day_tok = self.unitrans_price_mlp(x_seq)
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
        n_b = None
        if include_news:
            _npm = int(getattr(self, "unitrans_news_per_day", getattr(self, "news_top_k", 5)) or 5)
            (news_512, news_day_ids, news_comp_ids) = self._encode_news_nodes_to_price_dim(
                batch, device
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            news_grid = self._build_day_stock_news_slot_grid(
                news_512,
                news_day_ids,
                news_comp_ids,
                num_days,
                num_stocks,
                device,
                news_per_day=_npm,
            )
            n_win = news_grid[src_clamped]
            pad_n = self.unitrans_pad_news.view(1, 1, 1, 1, -1).expand(
                num_days, K, num_stocks, _npm, -1
            )
            n_win = torch.where(valid[:, :, None, None, None], n_win, pad_n)
            n_b = n_win.permute(0, 2, 1, 3, 4).reshape(num_days * num_stocks, K, _npm, -1)
            news_mlp = getattr(self, "unitrans_news_mlp", None)
            if news_mlp is not None:
                (bsz_n, kk, mm, dd) = n_b.shape
                n_b = news_mlp(n_b.reshape(bsz_n * kk, mm * dd)).view(bsz_n, kk, -1)
            else:
                up = getattr(self, "unitrans_news_up_proj", None)
                if up is not None:
                    n_b = up(n_b.mean(dim=2))
        virt_tok = None
        if getattr(self.unitrans, "include_virtual", False):
            bsz = int(p_b.size(0))
            nw = int(getattr(self.unitrans, "num_v_window", K))
            if nw > 0:
                if getattr(self.unitrans, "vin_window_emb", None) is not None:
                    vw = self.unitrans.vin_window_emb[:nw]
                else:
                    vw = getattr(self, "virtual_window_emb", None)
                    if vw is None:
                        vw = p_b.new_zeros((nw, int(self.price_emb_dim)))
                    else:
                        vw = vw[:nw]
                        if vw.size(0) < nw:
                            vw = torch.cat([vw, vw.new_zeros((nw - vw.size(0), vw.size(1)))], dim=0)
                virt_tok = vw.unsqueeze(0).expand(bsz, -1, -1)
        _price_arg = p_b
        _ind_tok = None
        if bool(getattr(self, "fuse_industry_with_price_news", False)):
            _build_ind = getattr(self, "_build_industry_token", None)
            if _build_ind is not None:
                _ind_tok = _build_ind(p_b, batch, num_days, num_stocks, device)
        out = self.unitrans.forward_unified(
            _price_arg, n_b, virt_tok=virt_tok, industry_tok=_ind_tok
        )
        return (out, p_b, out, None)


import math
import os
import zipfile
from contextlib import nullcontext
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as dist
import pickle as pkl
from naming import env_mtvc, normalize_price_encoder

DEFAULT_NUM_SCORE_NEWS_TYPES = 13
UNIFORM01_MEAN = 0.5
UNIFORM01_STD = 1.0 / math.sqrt(12.0)
MULTISTEP_UNIFORM_TOL_MEAN = 0.04
MULTISTEP_UNIFORM_TOL_STD = 0.04
MULTISTEP_UNIFORM_MIN_EDGES = 16
MULTISTEP_UNIFORM_MIN_STEPS = 1
EDGE_NOISE_CLAMP = 0.0001


def _nhead_for(d: int, prefer: int = 4) -> int:
    """Pick attention heads dividing d (prefer small for memory)."""
    d = int(d)
    h = min(int(prefer), d)
    while h > 1 and d % h != 0:
        h //= 2
    return max(1, h)


def _vocab_conv_channels(news_emb_dim: int) -> int:
    """Output channels of vocab_convs (kernels 3/4/5)."""
    return int(max(32, min(64, int(news_emb_dim))))


_MODEL_DIR = os.environ.get("DUAL_TF_MODEL_DIR", os.path.dirname(os.path.abspath(__file__)))
_NEWS_TEXT_INIT_N_WORDS = 4762
_NEWS_TEXT_INIT_EMB_DIM = 300


def _build_vocab_id_remap(
    vocab_size: int, n_pretrain: int, pad_id: int
) -> tuple[torch.Tensor, int]:
    """dict id -> stub embedding row; rows 0..n-1 kept, <pad> maps to extra padding row."""
    pad_row = n_pretrain
    last_pretrain = max(n_pretrain - 1, 0)
    remap = torch.arange(vocab_size, dtype=torch.long)
    if n_pretrain < vocab_size:
        remap[n_pretrain:] = last_pretrain
    remap[pad_id] = pad_row
    return (remap, pad_row)


class Model(PriceNewsEncodingMixin, CrossAttentionMixin, nn.Module):
    """Price window plus vocab news, fused to a next-day direction logit."""

    def __init__(
        self,
        price_emb_dim: int = 128,
        news_emb_dim: int = 128,
        hidden_dim: int = 128,
        news_encode_mode: str = "vocab_padding",
        news_window_agg_mode: str = "false",
        news_top_k: int = 5,
        grid_num_days: int = 20,
        grid_num_stocks: int = 50,
        industry_window_count: int = 20,
        num_industry: int = 27,
        vocab_path: str | None = None,
        price_input_dim: int = 3,
        no_price_token: bool = False,
        no_news_token: bool = False,
        no_news_token_pad: bool = False,
        no_price_token_pad: bool = False,
        modality_ablation: str = "full",
        industry_no_mean: bool = False,
        num_score_windows: int | None = None,
        num_score_news_types: int = DEFAULT_NUM_SCORE_NEWS_TYPES,
        news_ablation_mode: str = "full",
        main_mlp_ablation_mode: str = "full",
        vin_ablation_mode: str = "full",
        industry_fuse_site: str = "pred_fusion",
        news_word_emb: str = "random",
        news_text_tf_layers: int = 1,
        news_process_mode: str = "default",
        price_encoder: str = "partial_unified",
        vin_connect_mode: str = "default",
        init_news_vocab_size: int = _NEWS_TEXT_INIT_N_WORDS,
        init_news_embedding_dim: int = _NEWS_TEXT_INIT_EMB_DIM,
    ):
        """Build the news encoder, joint encoder, PredFusion head, and frozen companion modules."""
        super().__init__()
        self.industry_window_count = int(industry_window_count)
        self.num_industry = int(num_industry)
        self.num_rule_trend = 8
        self.include_vol_in_price = False
        self.price_encoder = normalize_price_encoder(price_encoder)
        _vcm = str(vin_connect_mode or "default").lower().strip()
        if _vcm not in ("default", "mkt_as_ind", "ind_as_mkt"):
            raise ValueError(
                f"vin_connect_mode must be default|mkt_as_ind|ind_as_mkt, got {vin_connect_mode!r}"
            )
        self.vin_connect_mode = _vcm
        self.unitrans_num_days = 5
        self.use_aux_posterior = False
        self.use_vol_aux = False
        self.vol_aux_from = "unitrans_h"
        if bool(no_price_token_pad) and bool(no_price_token):
            raise ValueError("use only one of --no_price_token / --no_price_token_pad")
        if bool(no_news_token_pad) and bool(no_news_token):
            raise ValueError("use only one of --no_news_token / --no_news_token_pad")
        self.price_token_fill = (
            "pad" if bool(no_price_token_pad) else "zero" if bool(no_price_token) else "real"
        )
        self.news_token_fill = (
            "pad" if bool(no_news_token_pad) else "zero" if bool(no_news_token) else "real"
        )
        self.include_price_token = self.price_token_fill == "real"
        self.include_news_token = self.news_token_fill == "real"
        _mod = str(modality_ablation or "full").strip().lower()
        if _mod not in ("full", "price_only", "news_only"):
            _mod = "full"
        self.modality_ablation = _mod
        if self.modality_ablation == "price_only":
            self.include_price_token = True
            self.price_token_fill = "real"
        elif self.modality_ablation == "news_only":
            self.include_price_token = False
            self.price_token_fill = "zero"
        self.industry_no_mean = bool(industry_no_mean)
        self.include_vol_token = False
        self.price_input_dim = 3
        self.dual_own_prop_news = False
        self.own_news_only = False
        self.prop_same_trend_only = False
        self.companies = []
        self._company2id = {}
        self.num_score_windows = (
            max(1, int(num_score_windows)) if num_score_windows is not None else 5
        )
        self.virtual_industry_node = nn.Parameter(
            torch.randn(1, self.num_industry, price_emb_dim) * 0.1
        )
        if not str(news_encode_mode).startswith("vocab"):
            raise ValueError(f"only vocab news encoding is supported, got {news_encode_mode!r}")
        self.news_encode_mode = news_encode_mode
        self.bert_tok_transformer = None
        _ablation = str(news_ablation_mode or "full").lower()
        if _ablation in ("no_news_global",):
            _ablation = "no_news"
        if _ablation not in ("full", "no_news"):
            raise ValueError(f"news_ablation_mode must be full|no_news, got {news_ablation_mode!r}")
        self.news_ablation_mode = _ablation
        self.include_news_in_main_mlp = self.news_ablation_mode != "no_news"
        if self.modality_ablation in ("price_only", "news_only"):
            self.include_news_in_main_mlp = False
        if self.modality_ablation == "price_only":
            self.include_news_tower = False
            self.include_price_tower = True
        elif self.modality_ablation == "news_only":
            self.include_news_tower = True
            self.include_price_tower = False
        else:
            self.include_news_tower = self.price_encoder != "tf" and bool(
                self.include_news_in_main_mlp
            )
            self.include_price_tower = True
        self.include_news_global_in_main_mlp = False
        self.use_aux_branch = bool(self.use_aux_posterior) and self.include_news_in_main_mlp
        self.include_aux_in_main_mlp = self.use_aux_branch
        _mlp_ab = str(main_mlp_ablation_mode or "full").lower()
        if _mlp_ab not in ("full", "no_global", "no_industry", "no_global_industry"):
            _mlp_ab = "full"
        self.main_mlp_ablation_mode = _mlp_ab
        _vin_ab = str(vin_ablation_mode or "full").lower().strip()
        _vin_ok = ("full", "with_news_type", "no_window", "with_news_type_no_window")
        if _vin_ab not in _vin_ok:
            raise ValueError(
                f"vin_ablation_mode must be full|with_news_type|no_window|with_news_type_no_window; got {vin_ablation_mode!r}"
            )
        self.vin_ablation_mode = _vin_ab
        _nwe = str(news_word_emb or "random").lower().strip()
        if _nwe == "pretrained":
            pass
        elif _nwe != "random":
            _nwe = "random"
        self.news_word_emb = _nwe
        self.include_global_in_main_mlp = _mlp_ab not in ("no_global", "no_global_industry")
        self.include_industry_in_main_mlp = _mlp_ab not in ("no_industry", "no_global_industry")
        _ifs = str(industry_fuse_site or "pred_fusion").lower().strip()
        if _ifs not in ("cross", "fusion", "pred_fusion"):
            raise ValueError(
                f"industry_fuse_site must be pred_fusion|cross|fusion; got {industry_fuse_site!r}"
            )
        _joint_enc = price_encoder in ("cross_attn", "partial_unified", "full_unified")
        self.industry_fuse_site = _ifs
        self.fuse_industry_with_price_news = bool(
            _joint_enc and _ifs == "cross" and self.include_industry_in_main_mlp
        )
        self.fuse_industry_in_fusion = bool(
            _joint_enc and _ifs == "fusion" and self.include_industry_in_main_mlp
        )
        self.include_industry_in_pred_fusion = bool(
            self.include_industry_in_main_mlp and _ifs == "pred_fusion"
        )
        self.price_emb_dim = int(price_emb_dim)
        self.news_emb_dim = max(8, int(news_emb_dim))
        self.global_feat_dim = 128
        self.news_main_feat_dim = self.price_emb_dim
        if self.include_global_in_main_mlp:
            self.news_main_feat_dim += self.price_emb_dim
        if self.include_industry_in_main_mlp:
            self.news_main_feat_dim += self.price_emb_dim
        self.news_mlp_slot_dim = int(self.news_main_feat_dim) * (
            2 if self.dual_own_prop_news else 1
        )
        _proc = str(news_process_mode or "default").lower()
        if _proc not in ("default", "", "none"):
            raise ValueError(f"news_process_mode={news_process_mode!r} removed; only default")
        self.news_process_mode = "default"
        self.news_top_k = int(news_top_k)
        self.grid_num_days = max(1, int(grid_num_days))
        self.grid_num_stocks = int(grid_num_stocks)
        self.market_global_score = nn.Linear(self.global_feat_dim, 1)
        self.market_global_score.requires_grad_(False)
        _candidates = ([vocab_path] if vocab_path else []) + [
            os.path.join(_MODEL_DIR, "../dict/dict_cn.pkl"),
            "dict/dict_cn.pkl",
            "dict/dict_cn.pkl",
            "dict/dict_us.pkl",
            "dict/dict_csmd.pkl",
        ]
        _loaded_path = None
        for _p in _candidates:
            if _p and os.path.isfile(_p):
                self.vocab_dict = pkl.load(open(_p, "rb"))
                _loaded_path = _p
                break
        if _loaded_path is None:
            raise FileNotFoundError(f"vocab not found, tried: {_candidates}")
        self.vocab_pad_id = int(self.vocab_dict.get("<pad>", 0))
        _vocab_vals = [int(v) for v in self.vocab_dict.values() if isinstance(v, (int, np.integer))]
        _max_vocab_id = max(_vocab_vals) if _vocab_vals else 0
        self.vocab_size = max(len(self.vocab_dict), _max_vocab_id + 1, self.vocab_pad_id + 1, 1)
        _vbase = os.path.basename(str(_loaded_path or "")).lower()
        self._vocab_is_us = "dict_massive" in _vbase
        self._vocab_use_jieba = "jieba" in _vbase or "csmd_word" in _vbase
        self._vocab_word_level = bool(self._vocab_is_us or self._vocab_use_jieba)
        (_n_word, word_emb_dim) = (int(init_news_vocab_size), int(init_news_embedding_dim))
        (_vocab_remap, _word_pad_idx) = _build_vocab_id_remap(
            self.vocab_size, _n_word, self.vocab_pad_id
        )
        self.word_pad_idx = int(_word_pad_idx)
        self.register_buffer("vocab_id_remap", _vocab_remap, persistent=False)
        self.news_word_emb_dim = word_emb_dim
        _ch = _vocab_conv_channels(news_emb_dim)
        self.vocab_emb = nn.Embedding(int(_n_word) + 1, int(word_emb_dim))
        self.vocab_convs = nn.ModuleList(
            [nn.Conv2d(1, _ch, (int(_k), int(word_emb_dim))) for _k in (3, 4, 5)]
        )
        self.vocab_proj = nn.Linear(_ch * 3, int(news_emb_dim))
        self.vocab_emb.requires_grad_(False)
        self.vocab_convs.requires_grad_(False)
        self.vocab_proj.requires_grad_(False)
        self.word_embedding = None
        self.global_parameter = nn.Parameter(torch.randn(10, 1, self.global_feat_dim))
        if self.vin_connect_mode == "mkt_as_ind":
            self.global_mean_proj = None
            self.global_connect_proj = nn.Linear(
                int(price_emb_dim) + int(self.global_feat_dim), int(self.global_feat_dim)
            )
        else:
            self.global_mean_proj = nn.Linear(price_emb_dim, self.global_feat_dim)
            self.global_connect_proj = None
        self.news_global_emb = nn.Parameter(torch.randn(1, 1, self.global_feat_dim))
        self.news_global_emb.requires_grad_(False)
        self.news_global_mlp = nn.Sequential(
            nn.Linear(price_emb_dim + self.global_feat_dim, self.global_feat_dim),
            nn.ReLU(),
            nn.Linear(self.global_feat_dim, self.global_feat_dim),
        )
        self.news_global_mlp.requires_grad_(False)
        self.industry_pair_score = nn.Linear(int(price_emb_dim), 1)
        if self.industry_pair_score is not None:
            self.industry_pair_score.requires_grad_(False)
        _vi_mlp_in = price_emb_dim * (2 if self.industry_no_mean else 3)
        self.industry_peer_mix = nn.Sequential(
            nn.Linear(_vi_mlp_in, price_emb_dim),
            nn.ReLU(),
            nn.Linear(price_emb_dim, price_emb_dim),
            nn.ReLU(),
        )
        self.industry_peer_mix.requires_grad_(False)
        _d = int(price_emb_dim)
        self.industry_rule_score = nn.Linear(_d, 1)
        self.industry_rule_score.requires_grad_(False)
        self.industry_rule_emb = nn.Parameter(torch.randn(int(self.num_rule_trend), _d) * 0.1)
        self.industry_rule_emb.requires_grad_(False)
        self.industry_rule_mlp = nn.Sequential(nn.Linear(_d * 3 + 1, _d), nn.Linear(_d, 1))
        self.industry_rule_mlp.requires_grad_(False)
        _allowed_pe = {"cross_attn", "partial_unified", "full_unified"}
        if self.price_encoder not in _allowed_pe:
            raise ValueError(
                f"unsupported price_encoder={self.price_encoder!r}; allowed={sorted(_allowed_pe)}"
            )
        if self.price_encoder in ("cross_attn", "partial_unified", "full_unified"):
            from tool.encoders import (
                SeqTransformerPool,
                TokenMLP,
                NewsTransformerEncoder,
                PredFusionTransformer,
            )

            _day_d = min(64, int(price_emb_dim))
            _pw = max(1, int(getattr(self, "unitrans_num_days", 5) or 5))
            if bool(getattr(self, "include_price_token", True)):
                if self.price_encoder in ("cross_attn", "partial_unified", "full_unified"):
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
            self.unitrans_pad_price = nn.Parameter(torch.zeros(1, 1, price_emb_dim))
            _nd = int(self.news_emb_dim)
            _npm = max(1, int(getattr(self, "news_top_k", 5) or 5))
            self.unitrans_news_per_day = _npm
            self.unitrans_pad_news = nn.Parameter(torch.zeros(1, 1, _nd))
            nn.init.trunc_normal_(self.unitrans_pad_price, std=0.02)
            nn.init.trunc_normal_(self.unitrans_pad_news, std=0.02)
            _day_include_news = bool(getattr(self, "include_news_tower", False))
            _day_include_price = bool(getattr(self, "include_price_tower", True))
            _build_news_text = _day_include_news and bool(getattr(self, "include_news_token", True))
            if self.price_encoder == "tf" or not _build_news_text:
                self.unitrans_news_text_tf = None
                self.bert_tok_transformer = None
            elif str(self.news_encode_mode).startswith("vocab"):
                if self.price_encoder == "cross_attn":
                    _no_news_text = "no news" if self._vocab_is_us else "当日无文本"
                    _unk = int(self.vocab_dict.get("<UNK>", self.vocab_dict.get("<unk>", 1)))
                    _max_nn = 32
                    _pad = int(self.vocab_pad_id)
                    if self._vocab_word_level:
                        if self._vocab_use_jieba:
                            import jieba

                            _toks = [
                                t for t in jieba.lcut(_no_news_text) if t and (not t.isspace())
                            ]
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
                        _ids = [int(self.vocab_dict.get(ch, _unk)) for ch in _no_news_text][
                            :_max_nn
                        ]
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
                self.bert_tok_transformer = NewsTransformerEncoder(
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
            else:
                raise ValueError(
                    f"news_encode_mode must start with vocab, got {self.news_encode_mode!r}"
                )
            if self.price_encoder in ("partial_unified", "full_unified"):
                try:
                    from tool.unified_joint import FullUnifiedModel, PartialUnifiedModel
                except ImportError:
                    from .unified_joint import FullUnifiedModel, PartialUnifiedModel
                _k = int(self.unitrans_num_days)
                _use_type = str(env_mtvc("USE_TYPE_EMB", "1")).strip().lower() not in (
                    "0",
                    "false",
                    "off",
                    "no",
                )
                _joint_L = max(1, int(env_mtvc("TRANSFORMER_LAYERS", "6") or 6))
                _mode = "full" if self.price_encoder == "full_unified" else "partial"
                self.unified_mode = _mode
                _joint_kw = dict(
                    d_model=int(price_emb_dim),
                    nhead=_nhead_for(price_emb_dim, 4),
                    num_days=_k,
                    num_layers=_joint_L,
                    dropout=0.1,
                    use_type_emb=_use_type,
                )
                if _mode == "full":
                    self.unitrans = FullUnifiedModel(**_joint_kw)
                else:
                    self.unitrans = PartialUnifiedModel(**_joint_kw, mode="partial")
                if int(_nd) != int(price_emb_dim):
                    self.unified_news_proj = nn.Linear(int(_nd), int(price_emb_dim))
                else:
                    self.unified_news_proj = nn.Identity()
                if _mode == "full":
                    self.unitrans_price_day_mlp = self.unitrans.build_price_day_mlp(
                        int(self.price_input_dim)
                    )
                    self.unitrans_price_as_hlc_tokens = False
                else:
                    self.unitrans_price_day_mlp = None
                    self.unitrans_price_as_hlc_tokens = True
            elif self.price_encoder == "cross_attn":
                self._setup_cross_attn(
                    price_emb_dim=int(price_emb_dim),
                    news_emb_dim=int(_nd),
                    news_per_day=int(_npm),
                    include_news=_day_include_news,
                    include_price=_day_include_price,
                )
        _vin_head = self.price_encoder in ("cross_attn", "partial_unified", "full_unified")
        self.pred_fusion_num_price_days = 0
        _n_tok = 1
        if self.include_global_in_main_mlp:
            _n_tok += 1
        if self.include_industry_in_pred_fusion:
            _n_tok += 1
        self.pred_fusion_num_tokens = int(_n_tok)
        self.use_pred_fusion_head = True
        try:
            from tool.encoders import PredFusionTransformer as _PFT
        except ImportError:
            from .encoders import PredFusionTransformer as _PFT
        self.pred_fusion_tf = _PFT(
            d_model=int(price_emb_dim),
            nhead=_nhead_for(price_emb_dim, 4),
            num_layers=1,
            num_tokens=self.pred_fusion_num_tokens,
            dropout=0.1,
        )
        self.pred_fusion_global_proj = nn.Linear(self.global_feat_dim, int(price_emb_dim))
        if self.vin_connect_mode == "ind_as_mkt":
            self.pred_fusion_industry_proj = nn.Linear(int(price_emb_dim), int(price_emb_dim))
        else:
            _ind_proj_in = int(price_emb_dim) if self.industry_no_mean else int(price_emb_dim) * 2
            self.pred_fusion_industry_proj = nn.Linear(_ind_proj_in, int(price_emb_dim))
        if int(self.news_main_feat_dim) == int(price_emb_dim):
            self.pred_fusion_news_proj = nn.Identity()
        else:
            self.pred_fusion_news_proj = nn.Linear(int(self.news_main_feat_dim), int(price_emb_dim))
            self.pred_fusion_news_proj.requires_grad_(False)
        self.pred_fusion_news_global_proj = nn.Linear(self.global_feat_dim, int(price_emb_dim))
        self.pred_fusion_news_global_proj.requires_grad_(False)
        self.pred_fusion_day_mix = nn.Sequential(
            nn.Linear(price_emb_dim * 5, price_emb_dim),
            nn.ReLU(),
            nn.Linear(price_emb_dim, price_emb_dim),
        )
        self.pred_fusion_day_mix.requires_grad_(False)
        self.pred_fusion_day_score = nn.Sequential(nn.Linear(price_emb_dim, 1))
        self.pred_fusion_day_score.requires_grad_(False)
        self.pred_fusion_stock_attn = nn.MultiheadAttention(
            embed_dim=price_emb_dim, num_heads=2, batch_first=True, dropout=0.1
        )
        self.pred_fusion_stock_attn.requires_grad_(False)
        self.pred_fusion_stock_norm = nn.LayerNorm(price_emb_dim)
        self.pred_fusion_stock_norm.requires_grad_(False)
        self.num_score_news_types = max(3, int(num_score_news_types))
        _d = int(price_emb_dim)
        self.news_window_score = nn.Sequential(nn.Linear(_d * 2, 256), nn.Linear(256, 1))
        self.news_type_score = nn.Sequential(nn.Linear(_d * 2, 256), nn.Linear(256, 1))
        self.news_window_score.requires_grad_(False)
        self.news_type_score.requires_grad_(False)
        self.news_gate_proj = nn.Linear(news_emb_dim, price_emb_dim)
        self.news_gate_proj.requires_grad_(False)
        _d = int(price_emb_dim)
        self.virtual_window_emb = nn.Parameter(torch.randn(int(self.grid_num_days), _d) * 0.1)
        self.virtual_price_type_emb = nn.Parameter(torch.randn(10, _d) * 0.1)
        self.virtual_news_type_emb = nn.Parameter(
            torch.randn(int(self.num_score_news_types), _d) * 0.1
        )
        for _p in (
            self.virtual_window_emb,
            self.virtual_price_type_emb,
            self.virtual_news_type_emb,
        ):
            _p.requires_grad_(False)
        self.window_msg_mlp = nn.Sequential(nn.Linear(_d * 3, _d), nn.Linear(_d, _d))
        self.news_type_msg_mlp = nn.Sequential(nn.Linear(_d * 3, _d), nn.Linear(_d, _d))
        self.window_msg_mlp.requires_grad_(False)
        self.news_type_msg_mlp.requires_grad_(False)
        self.news_day_attn = nn.MultiheadAttention(
            embed_dim=price_emb_dim, num_heads=8, batch_first=True, dropout=0.1
        )
        self.news_day_attn.requires_grad_(False)
        self.news_day_norm = nn.LayerNorm(price_emb_dim)
        self.news_day_norm.requires_grad_(False)
        self.news_day_weight = nn.Sequential(
            nn.Linear(price_emb_dim, price_emb_dim // 2),
            nn.ReLU(),
            nn.Linear(price_emb_dim // 2, 1),
        )
        self.news_day_weight.requires_grad_(False)
        self.news_type_mix = nn.Sequential(
            nn.Linear(price_emb_dim * 2, price_emb_dim),
            nn.GELU(),
            nn.Linear(price_emb_dim, price_emb_dim),
        )
        self.news_type_mix.requires_grad_(False)
        self.news_window_mix = nn.Sequential(
            nn.Linear(price_emb_dim * 2, price_emb_dim),
            nn.GELU(),
            nn.Linear(price_emb_dim, price_emb_dim),
        )
        self.news_window_mix.requires_grad_(False)
        self.news_attn_mix = nn.Sequential(
            nn.Linear(price_emb_dim * 2, price_emb_dim),
            nn.ReLU(),
            nn.Linear(price_emb_dim, price_emb_dim),
        )
        self.news_attn_mix.requires_grad_(False)
        self.news_attn_score = nn.Sequential(
            nn.Linear(price_emb_dim * 2, price_emb_dim), nn.ReLU(), nn.Linear(price_emb_dim, 2)
        )
        self.news_attn_score.requires_grad_(False)
        self.news_gate_reduce = nn.Sequential(
            nn.Linear(price_emb_dim * 2, self.news_main_feat_dim), nn.GELU()
        )
        self.news_gate_reduce.requires_grad_(False)
        if self.news_main_feat_dim != price_emb_dim:
            self.news_main_proj = nn.Sequential(
                nn.Linear(price_emb_dim, self.news_main_feat_dim), nn.GELU()
            )
            self.news_main_proj.requires_grad_(False)
        else:
            self.news_main_proj = nn.Identity()
        self.news_edge_reliab = nn.ModuleList(
            [nn.Sequential(nn.Linear(1, 64), nn.Linear(64, 1)) for _ in range(3)]
        )
        self.news_edge_reliab.requires_grad_(False)
        self.news_edge_mix = nn.Sequential(
            nn.Linear(price_emb_dim * 3, price_emb_dim),
            nn.ReLU(),
            nn.Linear(price_emb_dim, 3),
            nn.Sigmoid(),
        )
        self.news_edge_mix.requires_grad_(False)
        _d = int(price_emb_dim)
        self.pred_fusion_trend = nn.ModuleList(
            [nn.Sequential(nn.Linear(_d * 2, _d), nn.Linear(_d, _d)) for _ in range(3)]
        )
        self.pred_fusion_trend.requires_grad_(False)

    def train(self, mode: bool = True):
        """Keep the standard ``nn.Module`` train/eval switch."""
        return super().train(mode)

    def ensure_contrast_heads(self, proj_dim: int = 64) -> None:
        """Price/news contrast projections, then two frozen companion projections."""
        import torch.nn as nn

        d = int(self.price_emb_dim)
        pd = max(8, int(proj_dim))
        if getattr(self, "contrast_proj_p", None) is None:
            self.contrast_proj_p = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, pd))
        if getattr(self, "contrast_proj_n", None) is None:
            self.contrast_proj_n = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, pd))
        if getattr(self, "contrast_state_proj", None) is None:
            self.contrast_state_proj = nn.Linear(d, d)
            self.contrast_state_proj.requires_grad_(False)
        if getattr(self, "contrast_pair_proj", None) is None:
            self.contrast_pair_proj = nn.Linear(d, d)
            self.contrast_pair_proj.requires_grad_(False)

    def forward(self, batch):
        """One window. Returns a logit per price row."""
        device = next(self.parameters()).device
        _nam = str(getattr(self, "news_ablation_mode", "full") or "full").lower()
        if _nam not in ("full", "no_news", "no_news_global"):
            raise ValueError(f"news_ablation_mode={_nam!r} removed")
        price_x = batch["price"].x.to(device)
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
        (num_days, num_stocks) = self._infer_price_grid_shape(
            batch, n_price_raw, self.grid_num_days, self.grid_num_stocks
        )
        x = price_x.view(1, num_days, num_stocks, price_x.size(1), price_x.size(2))
        (batch_size, num_days, num_stocks, seq_length, input_dim) = x.shape
        x_seq = x.reshape(batch_size * num_days * num_stocks, seq_length, input_dim)
        N_price = int(x_seq.size(0))
        if self.price_encoder not in ("cross_attn", "partial_unified", "full_unified"):
            raise ValueError(
                f"unsupported price_encoder={self.price_encoder!r}; use cross_attn|partial_unified|full_unified"
            )
        (price_emb_from_last5, price_emb_5, price_emb_penultimate, _news_vol_aux_feat) = (
            self._encode_price_news(x_seq, batch, num_days, num_stocks, device)
        )
        del _news_vol_aux_feat
        price_emb = price_emb_from_last5.view(num_days, num_stocks, -1)
        price_emb_flat = price_emb_from_last5
        price_day_ids = batch["price"].day_id.to(device).long()
        global_feat = None
        if self.include_global_in_main_mlp:
            day_mean = price_emb.mean(dim=1, keepdim=True)
            gp = self.global_parameter.repeat(2, 1, 1)[:num_days]
            if self.vin_connect_mode == "mkt_as_ind" and self.global_connect_proj is not None:
                global_day = self.global_connect_proj(torch.cat([day_mean, gp], dim=-1))
            else:
                global_day = self.global_mean_proj(day_mean) * gp
            global_feat = global_day.expand(num_days, num_stocks, -1).reshape(
                num_days * num_stocks, -1
            )
        industry_feat_for_price = None
        if self.include_industry_in_pred_fusion or (
            self.include_industry_in_main_mlp and (not self.fuse_industry_with_price_news)
        ):
            industry_feat_for_price = torch.zeros(
                (N_price, price_emb_flat.size(-1)), dtype=torch.float32, device=device
            )
            use_industry = (
                "industry_type" in batch.node_types
                and ("price", "to", "industry_type") in batch.edge_types
            )
            if use_industry:
                industry_ids = (
                    batch["industry_type"].x[:, 0].to(device).long().clamp(0, self.num_industry - 1)
                )
                vi_node = self.virtual_industry_node.squeeze(0)
                if (
                    bool(getattr(self, "industry_no_mean", False))
                    and self.vin_connect_mode != "ind_as_mkt"
                ):
                    vi_all = vi_node[industry_ids]
                    industry_feat_for_price = self.pred_fusion_industry_proj(vi_all)
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
                        mean_agg = sum_agg / sum_cnt.clamp_min(1.0)
                        mean_sel = mean_agg[ids_d]
                        vi_sel = vi_node[ids_d]
                        if self.vin_connect_mode == "ind_as_mkt":
                            industry_feat_for_price[mask_d] = self.pred_fusion_industry_proj(
                                mean_sel * vi_sel
                            )
                        else:
                            industry_feat_for_price[mask_d] = self.pred_fusion_industry_proj(
                                torch.cat([mean_sel, vi_sel], dim=-1)
                            )
        if industry_feat_for_price is None:
            industry_feat_for_price = price_emb_flat.new_zeros((N_price, price_emb_flat.size(-1)))
        if global_feat is None:
            global_feat = price_emb_flat.new_zeros((N_price, self.global_feat_dim))
        _tok_list = [price_emb_flat]
        if self.include_global_in_main_mlp:
            _tok_list.append(self.pred_fusion_global_proj(global_feat))
        if self.include_industry_in_pred_fusion:
            _tok_list.append(industry_feat_for_price)
        if len(_tok_list) != int(getattr(self, "pred_fusion_num_tokens", len(_tok_list))):
            raise RuntimeError(
                f"pred_fusion token count mismatch: got {len(_tok_list)}, expected {self.pred_fusion_num_tokens}"
            )
        tokens = torch.stack(_tok_list, dim=1)
        return self.pred_fusion_tf(tokens)
