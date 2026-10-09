#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Contrast auxiliary branch (alongside ``model.py``): L_pair + case mining.

Role
----
- Mine in-batch contrast packs (``ClipPack`` / ``mine_contrast_pack``).
- Reuse ``Model`` encoders (``unitrans_price_tok`` / news TF) + ``unitrans.forward_unified``
  from ``Model.unitrans`` (PartialUnifiedModel in model.py §2) — does **not** run the BCE ``forward``.
- ``enable_contrast_gates`` + ``compute_contrast_aux`` → λ_pair * L_pair (added in ``train.py``).
- Peer scope: label (default) / news-similar / price-similar.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# In-batch contrast packs (formerly dataset_mine.py)
# ---------------------------------------------------------------------------


@dataclass
class ContrastPack:
    """One mini-batch of contrastive anchors mined from a market window."""

    price: torch.Tensor          # [B, K, F]
    pos_ids: torch.Tensor        # [N_pos, L]
    pos_attn: torch.Tensor       # [N_pos, L]
    pos_ptr: torch.Tensor        # [B+1] CSR into pos news
    neg_ids: torch.Tensor        # [N_neg, L]
    neg_attn: torch.Tensor       # [N_neg, L]
    neg_ptr: torch.Tensor        # [B*K_neg + 1] CSR into neg news (flattened B×K)
    neg_valid: torch.Tensor      # [B, K_neg] bool
    n_neg: int


@dataclass
class ClipPack:
    """CLIP-style pairs: same-company price window + in-window news (batch negs)."""

    price: torch.Tensor          # [B, K, F]
    news_ids: torch.Tensor       # [N_news, L]
    news_attn: torch.Tensor      # [N_news, L]
    news_ptr: torch.Tensor       # [B+1] CSR into news bags
    company_id: torch.Tensor     # [B] for diagnostics
    label: torch.Tensor | None = None          # [B] optional 涨跌
    label_valid: torch.Tensor | None = None    # [B] optional
    day_id: torch.Tensor | None = None         # [B] optional day slot in window


def _label_at(batch, pidx: int) -> tuple[float, bool]:
    if "label" not in batch or not hasattr(batch["label"], "x"):
        return 0.0, False
    lab = batch["label"].x.view(-1)
    if pidx < 0 or pidx >= int(lab.numel()):
        return 0.0, False
    y = float(lab[pidx].item())
    valid = True
    if hasattr(batch["label"], "valid_mask") and batch["label"].valid_mask is not None:
        vm = batch["label"].valid_mask.view(-1).bool()
        if pidx < int(vm.numel()):
            valid = bool(vm[pidx].item())
    return y, valid


def _price_feat(price_x: torch.Tensor) -> torch.Tensor:
    """Take HLC (first 3 dims) from [..., F] price feature."""
    if price_x.size(-1) >= 3:
        return price_x[..., :3]
    pad = price_x.new_zeros(*price_x.shape[:-1], 3 - price_x.size(-1))
    return torch.cat([price_x, pad], dim=-1)


def _sim_matrix(price_flat: torch.Tensor) -> torch.Tensor:
    """Negative L2 distance on flattened price windows. [N,N], higher=more similar."""
    # price_flat: [N, D]
    # use cosine on z-scored returns-like vectors for scale robustness
    x = price_flat - price_flat.mean(dim=-1, keepdim=True)
    x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return x @ x.T


def mine_hardneg_pack(
    batch,
    *,
    num_neg: int = 4,
    max_anchors: int = 64,
    max_news_per_bag: int = 8,
    day_mode: str = "last",
    rng: torch.Generator | None = None,
) -> ContrastPack | None:
    """Build contrastive pack from one MarketWindowDataset item.

    Positive: same company 5-day price + its news on that day.
    Negatives: news of other companies whose price window is most similar
    (cosine on centered HLC flatten), same calendar day slot.
    """
    price = batch["price"]
    news = batch["news"]
    if not hasattr(price, "x") or not hasattr(news, "vocab_input_ids"):
        return None
    price_x = price.x  # [N_price, K, F]
    valid = price.valid_mask.bool() if hasattr(price, "valid_mask") else None
    if valid is not None and valid.dim() > 1:
        valid = valid.view(-1)
    if valid is None or int(valid.sum().item()) == 0:
        return None
    if not hasattr(news, "vocab_input_ids") or news.vocab_input_ids.numel() == 0:
        return None

    company_id = price.company_id.long().view(-1)
    day_id = price.day_id.long().view(-1)
    n_price = int(price_x.size(0))
    k_days = int(price_x.size(1))
    # infer grid
    num_days = int(day_id.max().item()) + 1 if n_price > 0 else 1
    num_stocks = max(1, n_price // max(1, num_days))

    news_comp = news.company_id.long()
    news_day = news.day_id.long()
    vocab_ids = news.vocab_input_ids
    vocab_attn = news.vocab_attention_mask
    n_news = int(vocab_ids.size(0))

    # index news by (day, company) → list of news row indices
    news_buckets: dict[tuple[int, int], list[int]] = {}
    for ni in range(n_news):
        key = (int(news_day[ni].item()), int(news_comp[ni].item()))
        news_buckets.setdefault(key, []).append(ni)

    # candidate days
    if day_mode == "last":
        day_list = [num_days - 1]
    elif day_mode == "all":
        day_list = list(range(num_days))
    else:
        day_list = [num_days - 1]

    anchor_pidx: list[int] = []
    for d in day_list:
        for s in range(num_stocks):
            pidx = d * num_stocks + s
            if pidx >= n_price or not bool(valid[pidx].item()):
                continue
            cid = int(company_id[pidx].item())
            if (d, cid) not in news_buckets:
                continue
            anchor_pidx.append(pidx)

    if not anchor_pidx:
        return None

    # subsample anchors
    if len(anchor_pidx) > max_anchors:
        if rng is None:
            perm = torch.randperm(len(anchor_pidx))[:max_anchors]
        else:
            perm = torch.randperm(len(anchor_pidx), generator=rng)[:max_anchors]
        anchor_pidx = [anchor_pidx[int(i)] for i in perm.tolist()]

    # group by day for hard-neg mining
    by_day: dict[int, list[int]] = {}
    for pidx in anchor_pidx:
        d = int(day_id[pidx].item())
        by_day.setdefault(d, []).append(pidx)

    # also need non-anchor companies with news for hard-neg candidates
    # use all valid price rows on that day that have news
    day_candidates: dict[int, list[int]] = {}
    for d in by_day:
        cands = []
        for s in range(num_stocks):
            pidx = d * num_stocks + s
            if pidx >= n_price or not bool(valid[pidx].item()):
                continue
            cid = int(company_id[pidx].item())
            if (d, cid) not in news_buckets:
                continue
            cands.append(pidx)
        day_candidates[d] = cands

    prices: list[torch.Tensor] = []
    pos_ids_list: list[torch.Tensor] = []
    pos_attn_list: list[torch.Tensor] = []
    pos_ptr = [0]
    neg_ids_list: list[torch.Tensor] = []
    neg_attn_list: list[torch.Tensor] = []
    neg_ptr = [0]
    neg_valid_rows: list[list[bool]] = []

    feat_all = _price_feat(price_x)  # [N, K, 3]

    for d, anchors in by_day.items():
        cands = day_candidates.get(d, [])
        if len(cands) < 2:
            continue
        cand_idx = torch.tensor(cands, dtype=torch.long)
        flat = feat_all[cand_idx].reshape(len(cands), -1)  # [Nc, K*3]
        sim = _sim_matrix(flat)  # [Nc, Nc]
        # map pidx → local row
        local = {int(cands[i]): i for i in range(len(cands))}

        for pidx in anchors:
            if pidx not in local:
                continue
            li = local[pidx]
            cid = int(company_id[pidx].item())
            # hard neg: highest sim among other companies
            scores = sim[li].clone()
            scores[li] = float("-inf")
            # also drop same company if somehow duplicated
            for j, cj in enumerate(cands):
                if int(company_id[cj].item()) == cid:
                    scores[j] = float("-inf")
            k_take = min(int(num_neg), int((scores > float("-inf")).sum().item()))
            if k_take <= 0:
                continue
            topk = torch.topk(scores, k=k_take).indices.tolist()

            # positive news bag
            pos_rows = news_buckets[(d, cid)][:max_news_per_bag]
            if not pos_rows:
                continue

            prices.append(feat_all[pidx])
            for r in pos_rows:
                pos_ids_list.append(vocab_ids[r])
                pos_attn_list.append(vocab_attn[r])
            pos_ptr.append(pos_ptr[-1] + len(pos_rows))

            row_valid: list[bool] = []
            for t in range(int(num_neg)):
                if t < len(topk):
                    pj = cands[int(topk[t])]
                    ncid = int(company_id[pj].item())
                    neg_rows = news_buckets[(d, ncid)][:max_news_per_bag]
                    if not neg_rows:
                        # empty slot
                        # still need a placeholder news row? use pad zeros
                        L = int(vocab_ids.size(1))
                        neg_ids_list.append(vocab_ids.new_zeros(L))
                        neg_attn_list.append(vocab_attn.new_zeros(L))
                        neg_ptr.append(neg_ptr[-1] + 1)
                        row_valid.append(False)
                    else:
                        for r in neg_rows:
                            neg_ids_list.append(vocab_ids[r])
                            neg_attn_list.append(vocab_attn[r])
                        neg_ptr.append(neg_ptr[-1] + len(neg_rows))
                        row_valid.append(True)
                else:
                    L = int(vocab_ids.size(1))
                    neg_ids_list.append(vocab_ids.new_zeros(L))
                    neg_attn_list.append(vocab_attn.new_zeros(L))
                    neg_ptr.append(neg_ptr[-1] + 1)
                    row_valid.append(False)
            neg_valid_rows.append(row_valid)

    if not prices:
        return None

    price_t = torch.stack(prices, dim=0)
    pos_ids_t = torch.stack(pos_ids_list, dim=0)
    pos_attn_t = torch.stack(pos_attn_list, dim=0)
    neg_ids_t = torch.stack(neg_ids_list, dim=0)
    neg_attn_t = torch.stack(neg_attn_list, dim=0)
    return ContrastPack(
        price=price_t,
        pos_ids=pos_ids_t,
        pos_attn=pos_attn_t,
        pos_ptr=torch.tensor(pos_ptr, dtype=torch.long),
        neg_ids=neg_ids_t,
        neg_attn=neg_attn_t,
        neg_ptr=torch.tensor(neg_ptr, dtype=torch.long),
        neg_valid=torch.tensor(neg_valid_rows, dtype=torch.bool),
        n_neg=int(num_neg),
    )


def _append_news_bag(
    rows: list[int],
    vocab_ids: torch.Tensor,
    vocab_attn: torch.Tensor,
    ids_list: list[torch.Tensor],
    attn_list: list[torch.Tensor],
    ptr: list[int],
) -> bool:
    """Append one news bag; return False if empty (writes a pad placeholder)."""
    L = int(vocab_ids.size(1))
    if not rows:
        ids_list.append(vocab_ids.new_zeros(L))
        attn_list.append(vocab_attn.new_zeros(L))
        ptr.append(ptr[-1] + 1)
        return False
    for r in rows:
        ids_list.append(vocab_ids[r])
        attn_list.append(vocab_attn[r])
    ptr.append(ptr[-1] + len(rows))
    return True


def mine_temporal_pack(
    batch,
    *,
    num_neg: int = 4,
    max_anchors: int = 64,
    max_news_per_bag: int = 8,
    day_mode: str = "all",
    rng: torch.Generator | None = None,
    random_fill: bool = True,
) -> ContrastPack | None:
    """Same-company temporal contrastive pack.

    Positive: company c price window at day d + news of (c, d).
    Negatives: news of same company on other days d'≠d in this market window;
               if not enough, optionally fill with random other companies' news.
    """
    price = batch["price"]
    news = batch["news"]
    if not hasattr(price, "x") or not hasattr(news, "vocab_input_ids"):
        return None
    price_x = price.x
    valid = price.valid_mask.bool() if hasattr(price, "valid_mask") else None
    if valid is not None and valid.dim() > 1:
        valid = valid.view(-1)
    if valid is None or int(valid.sum().item()) == 0:
        return None
    if news.vocab_input_ids.numel() == 0:
        return None

    company_id = price.company_id.long().view(-1)
    day_id = price.day_id.long().view(-1)
    n_price = int(price_x.size(0))
    num_days = int(day_id.max().item()) + 1 if n_price > 0 else 1
    num_stocks = max(1, n_price // max(1, num_days))

    vocab_ids = news.vocab_input_ids
    vocab_attn = news.vocab_attention_mask
    news_comp = news.company_id.long()
    news_day = news.day_id.long()
    n_news = int(vocab_ids.size(0))

    news_buckets: dict[tuple[int, int], list[int]] = {}
    # company → days that have news
    company_news_days: dict[int, list[int]] = {}
    all_news_keys: list[tuple[int, int]] = []
    for ni in range(n_news):
        d = int(news_day[ni].item())
        c = int(news_comp[ni].item())
        key = (d, c)
        if key not in news_buckets:
            news_buckets[key] = []
            all_news_keys.append(key)
            company_news_days.setdefault(c, []).append(d)
        news_buckets[key].append(ni)

    if day_mode == "last":
        day_list = [num_days - 1]
    else:
        day_list = list(range(num_days))

    anchor_pidx: list[int] = []
    for d in day_list:
        for s in range(num_stocks):
            pidx = d * num_stocks + s
            if pidx >= n_price or not bool(valid[pidx].item()):
                continue
            cid = int(company_id[pidx].item())
            if (d, cid) not in news_buckets:
                continue
            # need at least one temporal neg day OR random fill
            other_days = [dd for dd in company_news_days.get(cid, []) if dd != d]
            if not other_days and not random_fill:
                continue
            if not other_days and random_fill and len(all_news_keys) < 2:
                continue
            anchor_pidx.append(pidx)

    if not anchor_pidx:
        return None

    if len(anchor_pidx) > max_anchors:
        if rng is None:
            perm = torch.randperm(len(anchor_pidx))[:max_anchors]
        else:
            perm = torch.randperm(len(anchor_pidx), generator=rng)[:max_anchors]
        anchor_pidx = [anchor_pidx[int(i)] for i in perm.tolist()]

    feat_all = _price_feat(price_x)
    prices: list[torch.Tensor] = []
    pos_ids_list: list[torch.Tensor] = []
    pos_attn_list: list[torch.Tensor] = []
    pos_ptr = [0]
    neg_ids_list: list[torch.Tensor] = []
    neg_attn_list: list[torch.Tensor] = []
    neg_ptr = [0]
    neg_valid_rows: list[list[bool]] = []

    def _rand_int(n: int) -> int:
        if n <= 0:
            return 0
        if rng is None:
            return int(torch.randint(0, n, (1,)).item())
        return int(torch.randint(0, n, (1,), generator=rng).item())

    for pidx in anchor_pidx:
        d = int(day_id[pidx].item())
        cid = int(company_id[pidx].item())
        pos_rows = news_buckets[(d, cid)][:max_news_per_bag]
        if not pos_rows:
            continue

        # temporal neg days for same company (prefer farther days first)
        other_days = sorted(
            [dd for dd in company_news_days.get(cid, []) if dd != d],
            key=lambda dd: abs(dd - d),
            reverse=True,
        )
        # shuffle among equal distance via random sample without replacement
        if len(other_days) > 1:
            if rng is None:
                order = torch.randperm(len(other_days)).tolist()
            else:
                order = torch.randperm(len(other_days), generator=rng).tolist()
            # keep distance preference: sort by |dd-d| desc, break ties by shuffle key
            keyed = sorted(
                enumerate(other_days),
                key=lambda iv: (-abs(iv[1] - d), order[iv[0]]),
            )
            other_days = [dd for _, dd in keyed]

        neg_day_slots: list[tuple[int, int] | None] = []  # (day, company) or None
        for dd in other_days:
            if len(neg_day_slots) >= int(num_neg):
                break
            neg_day_slots.append((dd, cid))

        # fill remaining with random other-company news keys
        if random_fill:
            tries = 0
            while len(neg_day_slots) < int(num_neg) and tries < max(20, 5 * int(num_neg)):
                tries += 1
                key = all_news_keys[_rand_int(len(all_news_keys))]
                if key[1] == cid and key[0] == d:
                    continue
                if key in neg_day_slots:
                    continue
                neg_day_slots.append(key)

        if not neg_day_slots:
            continue

        prices.append(feat_all[pidx])
        _append_news_bag(
            pos_rows, vocab_ids, vocab_attn, pos_ids_list, pos_attn_list, pos_ptr
        )

        row_valid: list[bool] = []
        for t in range(int(num_neg)):
            if t < len(neg_day_slots) and neg_day_slots[t] is not None:
                nd, nc = neg_day_slots[t]  # type: ignore[misc]
                neg_rows = news_buckets[(nd, nc)][:max_news_per_bag]
                ok = _append_news_bag(
                    neg_rows, vocab_ids, vocab_attn, neg_ids_list, neg_attn_list, neg_ptr
                )
                row_valid.append(ok)
            else:
                _append_news_bag(
                    [], vocab_ids, vocab_attn, neg_ids_list, neg_attn_list, neg_ptr
                )
                row_valid.append(False)
        neg_valid_rows.append(row_valid)

    if not prices:
        return None

    return ContrastPack(
        price=torch.stack(prices, dim=0),
        pos_ids=torch.stack(pos_ids_list, dim=0),
        pos_attn=torch.stack(pos_attn_list, dim=0),
        pos_ptr=torch.tensor(pos_ptr, dtype=torch.long),
        neg_ids=torch.stack(neg_ids_list, dim=0),
        neg_attn=torch.stack(neg_attn_list, dim=0),
        neg_ptr=torch.tensor(neg_ptr, dtype=torch.long),
        neg_valid=torch.tensor(neg_valid_rows, dtype=torch.bool),
        n_neg=int(num_neg),
    )


def mine_clip_pack(
    batch,
    *,
    max_anchors: int = 64,
    max_news_per_bag: int = 8,
    day_mode: str = "last",
    rng: torch.Generator | None = None,
    **_ignored,
) -> ClipPack | None:
    """CLIP pairs from one MarketWindow.

    Positive: company c's 5-day price window + all news of c within the window.
    Negatives: other companies in the same mined batch (handled by clip_info_nce).

    Price anchors default to the last day of the window (full 5-day lookback).
    """
    price = batch["price"]
    news = batch["news"]
    if not hasattr(price, "x") or not hasattr(news, "vocab_input_ids"):
        return None
    price_x = price.x
    valid = price.valid_mask.bool() if hasattr(price, "valid_mask") else None
    if valid is not None and valid.dim() > 1:
        valid = valid.view(-1)
    if valid is None or int(valid.sum().item()) == 0:
        return None
    if news.vocab_input_ids.numel() == 0:
        return None

    company_id = price.company_id.long().view(-1)
    day_id = price.day_id.long().view(-1)
    n_price = int(price_x.size(0))
    num_days = int(day_id.max().item()) + 1 if n_price > 0 else 1
    num_stocks = max(1, n_price // max(1, num_days))

    vocab_ids = news.vocab_input_ids
    vocab_attn = news.vocab_attention_mask
    news_comp = news.company_id.long()
    news_day = news.day_id.long()
    n_news = int(vocab_ids.size(0))

    # company → news rows whose day is inside this window
    company_news_rows: dict[int, list[int]] = {}
    for ni in range(n_news):
        d = int(news_day[ni].item())
        if d < 0 or d >= num_days:
            continue
        c = int(news_comp[ni].item())
        company_news_rows.setdefault(c, []).append(ni)

    if not company_news_rows:
        return None

    # which price day(s) to take as the window endpoint
    if str(day_mode).strip().lower() == "all":
        day_list = list(range(num_days))
    else:
        day_list = [num_days - 1]

    # unique companies only (CLIP diagonal must be 1:1)
    seen_cid: set[int] = set()
    anchor_pidx: list[int] = []
    for d in day_list:
        for s in range(num_stocks):
            pidx = d * num_stocks + s
            if pidx >= n_price or not bool(valid[pidx].item()):
                continue
            cid = int(company_id[pidx].item())
            if cid in seen_cid:
                continue
            rows = company_news_rows.get(cid)
            if not rows:
                continue
            seen_cid.add(cid)
            anchor_pidx.append(pidx)

    if len(anchor_pidx) < 2:
        return None

    if len(anchor_pidx) > max_anchors:
        if rng is None:
            perm = torch.randperm(len(anchor_pidx))[:max_anchors]
        else:
            perm = torch.randperm(len(anchor_pidx), generator=rng)[:max_anchors]
        anchor_pidx = [anchor_pidx[int(i)] for i in perm.tolist()]
        if len(anchor_pidx) < 2:
            return None

    feat_all = _price_feat(price_x)
    prices: list[torch.Tensor] = []
    cids: list[int] = []
    news_ids_list: list[torch.Tensor] = []
    news_attn_list: list[torch.Tensor] = []
    news_ptr = [0]

    for pidx in anchor_pidx:
        cid = int(company_id[pidx].item())
        rows = company_news_rows[cid][:max_news_per_bag]
        if not rows:
            continue
        prices.append(feat_all[pidx])
        cids.append(cid)
        _append_news_bag(
            rows, vocab_ids, vocab_attn, news_ids_list, news_attn_list, news_ptr
        )

    if len(prices) < 2:
        return None

    return ClipPack(
        price=torch.stack(prices, dim=0),
        news_ids=torch.stack(news_ids_list, dim=0),
        news_attn=torch.stack(news_attn_list, dim=0),
        news_ptr=torch.tensor(news_ptr, dtype=torch.long),
        company_id=torch.tensor(cids, dtype=torch.long),
    )


def mine_bag_pack(
    batch,
    *,
    max_anchors: int = 64,
    max_news_per_bag: int = 8,
    day_mode: str = "all",
    rng: torch.Generator | None = None,
    **_ignored,
) -> ClipPack | None:
    """All (day, company) bags with news+label for label-sim joint contrast."""
    price = batch["price"]
    news = batch["news"]
    if not hasattr(price, "x") or not hasattr(news, "vocab_input_ids"):
        return None
    price_x = price.x
    valid = price.valid_mask.bool() if hasattr(price, "valid_mask") else None
    if valid is not None and valid.dim() > 1:
        valid = valid.view(-1)
    if valid is None or int(valid.sum().item()) == 0:
        return None
    if news.vocab_input_ids.numel() == 0:
        return None

    company_id = price.company_id.long().view(-1)
    day_id = price.day_id.long().view(-1)
    n_price = int(price_x.size(0))
    num_days = int(day_id.max().item()) + 1 if n_price > 0 else 1
    num_stocks = max(1, n_price // max(1, num_days))

    vocab_ids = news.vocab_input_ids
    vocab_attn = news.vocab_attention_mask
    news_comp = news.company_id.long()
    news_day = news.day_id.long()
    n_news = int(vocab_ids.size(0))

    news_buckets: dict[tuple[int, int], list[int]] = {}
    for ni in range(n_news):
        key = (int(news_day[ni].item()), int(news_comp[ni].item()))
        news_buckets.setdefault(key, []).append(ni)

    if str(day_mode).strip().lower() == "all":
        day_list = list(range(num_days))
    else:
        day_list = [num_days - 1]

    anchor_pidx: list[int] = []
    for d in day_list:
        for s in range(num_stocks):
            pidx = d * num_stocks + s
            if pidx >= n_price or not bool(valid[pidx].item()):
                continue
            cid = int(company_id[pidx].item())
            if not news_buckets.get((d, cid)):
                continue
            _y, yv = _label_at(batch, pidx)
            if not yv:
                continue
            anchor_pidx.append(pidx)

    if len(anchor_pidx) < 2:
        return None
    if len(anchor_pidx) > max_anchors:
        if rng is None:
            perm = torch.randperm(len(anchor_pidx))[:max_anchors]
        else:
            perm = torch.randperm(len(anchor_pidx), generator=rng)[:max_anchors]
        anchor_pidx = [anchor_pidx[int(i)] for i in perm.tolist()]
        if len(anchor_pidx) < 2:
            return None

    feat_all = _price_feat(price_x)
    prices: list[torch.Tensor] = []
    labels: list[float] = []
    label_valids: list[bool] = []
    cids: list[int] = []
    days: list[int] = []
    news_ids_list: list[torch.Tensor] = []
    news_attn_list: list[torch.Tensor] = []
    news_ptr = [0]

    for pidx in anchor_pidx:
        cid = int(company_id[pidx].item())
        d = int(day_id[pidx].item())
        rows = news_buckets[(d, cid)][:max_news_per_bag]
        prices.append(feat_all[pidx])
        y, yv = _label_at(batch, pidx)
        labels.append(y)
        label_valids.append(yv)
        cids.append(cid)
        days.append(d)
        _append_news_bag(
            rows, vocab_ids, vocab_attn, news_ids_list, news_attn_list, news_ptr
        )

    if len(prices) < 2:
        return None

    return ClipPack(
        price=torch.stack(prices, dim=0),
        news_ids=torch.stack(news_ids_list, dim=0),
        news_attn=torch.stack(news_attn_list, dim=0),
        news_ptr=torch.tensor(news_ptr, dtype=torch.long),
        company_id=torch.tensor(cids, dtype=torch.long),
        label=torch.tensor(labels, dtype=torch.float32),
        label_valid=torch.tensor(label_valids, dtype=torch.bool),
        day_id=torch.tensor(days, dtype=torch.long),
    )


def mine_contrast_pack(
    batch,
    *,
    neg_mode: str = "temporal",
    **kwargs,
) -> ContrastPack | ClipPack | None:
    mode = str(neg_mode or "temporal").strip().lower()
    if mode in ("bag", "bags", "label_sim", "sim_label", "peer", "jsim"):
        return mine_bag_pack(batch, **kwargs)
    if mode in ("clip", "batch", "infonce", "nce", "ssl"):
        return mine_clip_pack(batch, **kwargs)
    if mode in ("temporal", "time", "same_company", "shift"):
        return mine_temporal_pack(batch, **kwargs)
    if mode in ("hardneg", "hard", "price_sim", "price"):
        return mine_hardneg_pack(batch, **kwargs)
    raise ValueError(f"unknown neg_mode={neg_mode!r}; use clip|temporal|hardneg|bag")

# ---------------------------------------------------------------------------
# Contrast-path token encode (formerly encode_tokens.py)
# ---------------------------------------------------------------------------

def _encode_price_day_tokens(model, price: torch.Tensor) -> torch.Tensor:
    """[B,K,3] → day tokens [B,K,D] from the price SeqTF only (no joint)."""
    price_tok_mod = getattr(model, "unitrans_price_tok", None)
    unitrans = getattr(model, "unitrans", None)
    if price_tok_mod is None or unitrans is None:
        raise RuntimeError("Model missing unitrans_price_tok / unitrans")
    bsz, k, f = price.shape
    n_ch = min(3, f)
    hlc = price[..., :n_ch]
    if n_ch < 3:
        hlc = torch.cat([hlc, hlc.new_zeros(bsz, k, 3 - n_ch)], dim=-1)
    tok_in = hlc.reshape(bsz * k, 3, 1)
    pad_mask = hlc.reshape(bsz * k, 3).abs().eq(0)
    day = price_tok_mod(tok_in, key_padding_mask=pad_mask).view(bsz, k, -1)
    num_days = int(getattr(unitrans, "num_days", k) or k)
    if day.size(1) != num_days:
        if day.size(1) < num_days:
            day = torch.cat(
                [day.new_zeros(bsz, num_days - day.size(1), day.size(-1)), day], dim=1
            )
        else:
            day = day[:, -num_days:, :]
    return day

def _encode_news_row_feat(model, ids: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
    """Encode each news row to price_dim (before contrast_proj). [N, D]."""
    news_tf = getattr(model, "news_transformer_encoder", None) or getattr(
        model, "bert_tok_transformer", None
    )
    if news_tf is None:
        raise RuntimeError("Model missing news_transformer_encoder")
    h = news_tf(ids, attention_mask=attn)
    proj = getattr(model, "unified_news_proj", None)
    if proj is not None:
        h = proj(h)
    return h

def _encode_news_tokens_feat(
    model, ids: torch.Tensor, attn: torch.Tensor, ptr: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad per-news tokens to [B, Nmax, D] with True=pad mask, matching main-path joint input."""
    bsz = int(ptr.numel()) - 1
    d = int(getattr(model, "price_emb_dim", 128))
    if bsz <= 0:
        z = ids.new_zeros((0, 0, d), dtype=torch.float32)
        return z, ids.new_zeros((0, 0), dtype=torch.bool)
    if ids.numel() == 0:
        z = ids.new_zeros((bsz, 0, d), dtype=torch.float32)
        return z, torch.zeros((bsz, 0), dtype=torch.bool, device=ids.device)
    h = _encode_news_row_feat(model, ids, attn)
    ptr_cpu = ptr.detach().to("cpu").tolist()
    nmax = 0
    for i in range(bsz):
        nmax = max(nmax, int(ptr_cpu[i + 1]) - int(ptr_cpu[i]))
    if nmax <= 0:
        z = h.new_zeros((bsz, 0, h.size(-1)))
        return z, torch.zeros((bsz, 0), dtype=torch.bool, device=h.device)
    news_tok = h.new_zeros((bsz, nmax, h.size(-1)))
    news_pad = torch.ones((bsz, nmax), dtype=torch.bool, device=h.device)
    for i in range(bsz):
        a, b = int(ptr_cpu[i]), int(ptr_cpu[i + 1])
        n = b - a
        if n > 0:
            news_tok[i, :n] = h[a:b]
            news_pad[i, :n] = False
    return news_tok, news_pad

# ---------------------------------------------------------------------------
# Gates + L_pair
# ---------------------------------------------------------------------------

def enable_contrast_gates(model, case_mine_layer: int | None = None) -> None:
    """Turn on per-layer news gates on PartialUnifiedModel."""
    unitrans = getattr(model, "unitrans", None)
    if unitrans is None:
        return
    if hasattr(unitrans, "enable_news_gate"):
        unitrans.enable_news_gate(True)
    if case_mine_layer is not None and hasattr(unitrans, "case_mine_layer"):
        unitrans.case_mine_layer = int(case_mine_layer)


def _peer_thr(sim: torch.Tensor, quantile: float) -> float:
    bsz = int(sim.size(0))
    eye = torch.eye(bsz, device=sim.device, dtype=torch.bool)
    off = sim[~eye]
    if off.numel() == 0:
        return 0.0
    return float(torch.quantile(off.float(), float(quantile)).item())


def _encode_contrast_pack(
    model,
    price: torch.Tensor,
    news_ids: torch.Tensor,
    news_attn: torch.Tensor,
    ptr: torch.Tensor,
    *,
    enable_gate: bool = True,
    case_mine_layer: int = 6,
    contrast_news_mode: str = "nbag",
    max_news_per_bag: int = 6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float | None]:
    """Return (z_fusion_proj, zs_case, zt_case, w_mean).

    zs_case / zt_case: joint-layer price/news pools for case mining (layer =
    ``case_mine_layer``, 1-indexed; default 6 (official full)).
    z_fusion_proj: contrast_proj_p(joint last-layer state) for L_pair.

    contrast_news_mode:
      - nbag (default): mean-pool news → 1 token on last day (paper full path)
      - multinews: keep per-news tokens on last day (token-level L_pair ablation)
    """
    if getattr(model, "contrast_proj_p", getattr(model, "ssl_proj_p", None)) is None:
        model.ensure_contrast_heads()
    unitrans = getattr(model, "unitrans", None)
    if unitrans is None or not hasattr(unitrans, "forward_unified"):
        raise RuntimeError("Model.unitrans.forward_unified required for MTVC")
    _mode = str(contrast_news_mode or "nbag").strip().lower()
    if _mode not in ("nbag", "multinews"):
        _mode = "nbag"
    if enable_gate:
        enable_contrast_gates(model, case_mine_layer=int(case_mine_layer))
    elif hasattr(unitrans, "case_mine_layer"):
        unitrans.case_mine_layer = int(case_mine_layer)

    day = _encode_price_day_tokens(model, price)
    news_tok, news_pad = _encode_news_tokens_feat(model, news_ids, news_attn, ptr)
    bsz = int(day.size(0))
    if int(news_tok.size(0)) != bsz:
        raise ValueError(f"news tokens {int(news_tok.size(0))} != price {bsz}")
    nmax = int(news_tok.size(1))
    _cap = max(0, int(max_news_per_bag))
    if _cap > 0 and nmax > _cap:
        news_tok = news_tok[:, :_cap]
        news_pad = news_pad[:, :_cap]
        nmax = _cap
    if nmax > 0:
        valid = (~news_pad).unsqueeze(-1).to(dtype=news_tok.dtype)
        nbag_vec = (news_tok * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
    else:
        nbag_vec = day.new_zeros((bsz, day.size(-1)))

    num_days = int(day.size(1))
    if nmax > 0:
        if _mode == "multinews":
            joint_news = news_tok
            joint_pad = news_pad.clone()
            has_news = (~news_pad).any(dim=1)
            joint_pad = joint_pad | (~has_news).unsqueeze(1)
            news_day = torch.full(
                (bsz, nmax), num_days - 1, dtype=torch.long, device=day.device
            )
        else:
            joint_news = nbag_vec.unsqueeze(1)
            joint_pad = torch.zeros(bsz, 1, dtype=torch.bool, device=day.device)
            has_news = (~news_pad).any(dim=1)
            joint_pad = joint_pad | (~has_news).unsqueeze(1)
            news_day = torch.full(
                (bsz, 1), num_days - 1, dtype=torch.long, device=day.device
            )
        h_j = unitrans.forward_unified(
            day, news_tok=joint_news, news_day_idx=news_day, news_pad=joint_pad
        )
    else:
        empty_news = day.new_zeros((bsz, 0, day.size(-1)))
        empty_pad = torch.zeros((bsz, 0), dtype=torch.bool, device=day.device)
        empty_day = torch.zeros((bsz, 0), dtype=torch.long, device=day.device)
        h_j = unitrans.forward_unified(
            day, news_tok=empty_news, news_day_idx=empty_day, news_pad=empty_pad
        )
    w_mean = getattr(unitrans, "last_mtvc_gate_w_mean", None)
    zs = getattr(unitrans, "last_mtvc_case_price", None)
    zt = getattr(unitrans, "last_mtvc_case_news", None)
    if zs is None or zt is None:
        zs = day.mean(dim=1)
        zt = nbag_vec
    return model.contrast_proj_p(h_j), zs, zt, w_mean


def compute_contrast_aux(
    model,
    batch,
    *,
    temperature_pair: float = 0.07,
    max_anchors: int = 48,
    max_news_per_bag: int = 6,
    day_mode: str = "all",
    sim_quantile: float = 0.75,
    tau_y: float = 0.5,
    beta_hard: float = 0.5,
    lambda_pair: float = 0.2,
    pair_peer: str = "label",
    device: torch.device | None = None,
    enable_gate: bool = True,
    case_mine_layer: int = 6,
    contrast_news_mode: str = "nbag",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return (λ1 L_clip + λ2 L_pair, stats). L_clip only if use_clip=True.

    pair_peer=label|news|price controls which peers enter L_pair.
    enable_gate=False: L_pair only (no per-layer news weighting).
    case_mine_layer: 1-indexed joint layer for zs/zt case features (default 6).
    contrast_news_mode: nbag (default) or multinews (token-level L_pair).
    """
    _peer = str(pair_peer or "label").strip().lower()
    if _peer in ("jsim_news", "news_sim", "similar_news"):
        _peer = "news"
    elif _peer in ("jsim_price", "price_sim", "similar_price", "price_peer"):
        _peer = "price"
    elif _peer not in ("label", "news", "price"):
        _peer = "label"
    use_clip = False  # paper MTVC: L_pair only (no L_clip)
    if enable_gate:
        enable_contrast_gates(model, case_mine_layer=int(case_mine_layer))
    elif hasattr(getattr(model, "unitrans", None), "case_mine_layer"):
        model.unitrans.case_mine_layer = int(case_mine_layer)
    pack = mine_contrast_pack(
        batch,
        neg_mode="bag",
        max_anchors=int(max_anchors),
        max_news_per_bag=int(max_news_per_bag),
        day_mode=str(day_mode),
    )
    if pack is None or not isinstance(pack, ClipPack):
        ref = batch["price"].x
        return ref.new_zeros(()), {"loss": 0.0, "skipped": 1.0, "bsz": 0.0}
    if pack.label is None or pack.label_valid is None:
        ref = batch["price"].x
        return ref.new_zeros(()), {"loss": 0.0, "skipped": 1.0, "bsz": 0.0}

    if device is None:
        device = batch["price"].x.device
    if getattr(model, "contrast_proj_p", getattr(model, "ssl_proj_p", None)) is None:
        model.ensure_contrast_heads()
    if use_clip and getattr(model, "contrast_proj_n", getattr(model, "ssl_proj_n", None)) is None:
        model.ensure_contrast_heads()

    z_f, zs_raw, zt_raw, w_mean = _encode_contrast_pack(
        model,
        pack.price.to(device),
        pack.news_ids.to(device),
        pack.news_attn.to(device),
        pack.news_ptr,
        enable_gate=bool(enable_gate),
        case_mine_layer=int(case_mine_layer),
        contrast_news_mode=str(contrast_news_mode or "nbag"),
        max_news_per_bag=int(max_news_per_bag),
    )
    labels = pack.label.to(device).view(-1).float()
    valid = pack.label_valid.to(device).view(-1).bool()
    bsz = int(z_f.size(0))
    if bsz < 2:
        return z_f.new_zeros(()), {"loss": 0.0, "skipped": 1.0, "bsz": float(bsz)}

    # Case mining on pre-joint modality pools (price SeqTF / news bag)
    zs_n = F.normalize(zs_raw, dim=-1)
    zt_n = F.normalize(zt_raw, dim=-1)
    zf_n = F.normalize(z_f, dim=-1)

    sim_s = zs_n @ zs_n.T
    sim_t = zt_n @ zt_n.T
    thr_s = _peer_thr(sim_s, float(sim_quantile))
    thr_t = _peer_thr(sim_t, float(sim_quantile))
    eye = torch.eye(bsz, device=device, dtype=torch.bool)

    price_sim = (sim_s > thr_s) & ~eye
    news_sim = (sim_t > thr_t) & ~eye
    price_dis = (~(sim_s > thr_s)) & ~eye
    news_dis = (~(sim_t > thr_t)) & ~eye

    dy = (labels.unsqueeze(0) - labels.unsqueeze(1)).abs()
    both_valid = valid.unsqueeze(0) & valid.unsqueeze(1) & ~eye
    same_y = (dy < float(tau_y)) & both_valid
    diff_y = (dy > float(tau_y)) & both_valid

    case1 = price_sim & news_dis & diff_y
    case2 = price_sim & news_dis & same_y
    case3 = price_dis & news_sim & diff_y
    case4 = price_dis & news_sim & same_y

    beta = torch.ones(bsz, bsz, device=device, dtype=torch.float32)
    if _peer == "news":
        # hard pairs only among news-similar / price-dissimilar (case4)
        beta = beta.masked_fill(case4, float(beta_hard))
    elif _peer == "price":
        # hard pairs only among price-similar / news-dissimilar (case2)
        beta = beta.masked_fill(case2, float(beta_hard))
    else:
        beta = beta.masked_fill(case2 | case4, float(beta_hard))

    # peer mask for L_pair (jsim-style when news/price)
    if _peer == "news":
        peer = news_sim & both_valid
    elif _peer == "price":
        peer = price_sim & both_valid
    else:
        peer = both_valid

    L_clip = z_f.new_zeros(())
    lam1 = 0.0
    gamma = torch.ones(bsz, device=device, dtype=torch.float32)
    alpha = torch.ones(bsz, device=device, dtype=torch.float32)
    pos_sim = (zs_n * zt_n).sum(dim=-1)


    tau_p = float(temperature_pair)
    logits_f = (zf_n @ zf_n.T) / tau_p
    pair_pos = same_y & peer
    pair_neg = diff_y & peer
    pair_losses = []
    for i in range(bsz):
        if not bool(valid[i]):
            continue
        pos_js = pair_pos[i].nonzero(as_tuple=False).view(-1)
        if _peer in ("news", "price"):
            # SupCon among peers: denom = all peers (pos+neg)
            peer_js = peer[i].nonzero(as_tuple=False).view(-1)
            if pos_js.numel() == 0 or peer_js.numel() == 0:
                continue
            for j in pos_js.tolist():
                num = logits_f[i, j]
                peer_logits = logits_f[i, peer_js]
                denom = torch.logsumexp(peer_logits, dim=0)
                lp = -(num - denom)
                wij = beta[i, j]
                if use_clip:
                    wij = wij * gamma[i] * gamma[j]
                pair_losses.append(wij * lp)
        else:
            neg_js = pair_neg[i].nonzero(as_tuple=False).view(-1)
            if pos_js.numel() == 0 or neg_js.numel() == 0:
                continue
            for j in pos_js.tolist():
                num = logits_f[i, j]
                neg_logits = logits_f[i, neg_js]
                denom = torch.logsumexp(torch.cat([num.view(1), neg_logits], dim=0), dim=0)
                lp = -(num - denom)
                wij = beta[i, j]
                if use_clip:
                    wij = wij * gamma[i] * gamma[j]
                pair_losses.append(wij * lp)
    if pair_losses:
        L_pair = torch.stack(pair_losses).mean()
    else:
        L_pair = z_f.new_zeros(())

    lam2 = float(lambda_pair)
    loss = lam1 * L_clip + lam2 * L_pair

    with torch.no_grad():
        stats = {
            "loss": float(loss.detach().item()),
            "L_pair": float(L_pair.detach().item()),
            "L_clip": float(L_clip.detach().item()) if use_clip else 0.0,
            "lam1": float(lam1),
            "lam2": lam2,
            "use_clip": 1.0 if use_clip else 0.0,
            "pair_peer": {"label": 0.0, "news": 1.0, "price": 2.0}.get(_peer, 0.0),
            "bsz": float(bsz),
            "skipped": 0.0,
            "pos_sim": float((zf_n @ zf_n.T)[pair_pos].mean().item()) if bool(pair_pos.any()) else float(pos_sim.mean().item()),
            "neg_sim": float((zf_n @ zf_n.T)[pair_neg].mean().item()) if bool(pair_neg.any()) else 0.0,
            "n_peer": float(peer.sum().item()),
            "n_case1": float(case1.sum().item()),
            "n_case2": float(case2.sum().item()),
            "n_case3": float(case3.sum().item()),
            "n_case4": float(case4.sum().item()),
            "thr_s": thr_s,
            "thr_t": thr_t,
            "w_mean": float(w_mean) if w_mean is not None else 0.5,
            "gamma_mean": float(gamma.mean().item()),
            "frac_bad_alpha": float((alpha < 0.999).float().mean().item()) if use_clip else 0.0,
        }
    return loss, stats
