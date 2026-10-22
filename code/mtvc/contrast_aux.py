"""Merged contrast stack (): dataset_mine + losses + method1/13_aux + contrast_aux."""

from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass
class ContrastPack:
    """One mini-batch of contrastive anchors mined from a market window."""

    price: torch.Tensor
    pos_ids: torch.Tensor
    pos_attn: torch.Tensor
    pos_ptr: torch.Tensor
    neg_ids: torch.Tensor
    neg_attn: torch.Tensor
    neg_ptr: torch.Tensor
    neg_valid: torch.Tensor
    n_neg: int


@dataclass
class ClipPack:
    """CLIP-style pairs: same-company price window + in-window news (batch negs)."""

    price: torch.Tensor
    news_ids: torch.Tensor
    news_attn: torch.Tensor
    news_ptr: torch.Tensor
    company_id: torch.Tensor
    label: torch.Tensor | None = None
    label_valid: torch.Tensor | None = None
    day_id: torch.Tensor | None = None


def _label_at(batch, pidx: int) -> tuple[float, bool]:
    if "label" not in batch or not hasattr(batch["label"], "x"):
        return (0.0, False)
    lab = batch["label"].x.view(-1)
    if pidx < 0 or pidx >= int(lab.numel()):
        return (0.0, False)
    y = float(lab[pidx].item())
    valid = True
    if hasattr(batch["label"], "valid_mask") and batch["label"].valid_mask is not None:
        vm = batch["label"].valid_mask.view(-1).bool()
        if pidx < int(vm.numel()):
            valid = bool(vm[pidx].item())
    return (y, valid)


def _price_feat(price_x: torch.Tensor) -> torch.Tensor:
    """Take HLC (first 3 dims) from [..., F] price feature."""
    if price_x.size(-1) >= 3:
        return price_x[..., :3]
    pad = price_x.new_zeros(*price_x.shape[:-1], 3 - price_x.size(-1))
    return torch.cat([price_x, pad], dim=-1)


def _sim_matrix(price_flat: torch.Tensor) -> torch.Tensor:
    """Negative L2 distance on flattened price windows. [N,N], higher=more similar."""
    x = price_flat - price_flat.mean(dim=-1, keepdim=True)
    x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-06)
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
    price_x = price.x
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
    num_days = int(day_id.max().item()) + 1 if n_price > 0 else 1
    num_stocks = max(1, n_price // max(1, num_days))
    news_comp = news.company_id.long()
    news_day = news.day_id.long()
    vocab_ids = news.vocab_input_ids
    vocab_attn = news.vocab_attention_mask
    n_news = int(vocab_ids.size(0))
    news_buckets: dict[tuple[int, int], list[int]] = {}
    for ni in range(n_news):
        key = (int(news_day[ni].item()), int(news_comp[ni].item()))
        news_buckets.setdefault(key, []).append(ni)
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
    if len(anchor_pidx) > max_anchors:
        if rng is None:
            perm = torch.randperm(len(anchor_pidx))[:max_anchors]
        else:
            perm = torch.randperm(len(anchor_pidx), generator=rng)[:max_anchors]
        anchor_pidx = [anchor_pidx[int(i)] for i in perm.tolist()]
    by_day: dict[int, list[int]] = {}
    for pidx in anchor_pidx:
        d = int(day_id[pidx].item())
        by_day.setdefault(d, []).append(pidx)
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
    feat_all = _price_feat(price_x)
    for d, anchors in by_day.items():
        cands = day_candidates.get(d, [])
        if len(cands) < 2:
            continue
        cand_idx = torch.tensor(cands, dtype=torch.long)
        flat = feat_all[cand_idx].reshape(len(cands), -1)
        sim = _sim_matrix(flat)
        local = {int(cands[i]): i for i in range(len(cands))}
        for pidx in anchors:
            if pidx not in local:
                continue
            li = local[pidx]
            cid = int(company_id[pidx].item())
            scores = sim[li].clone()
            scores[li] = float("-inf")
            for j, cj in enumerate(cands):
                if int(company_id[cj].item()) == cid:
                    scores[j] = float("-inf")
            k_take = min(int(num_neg), int((scores > float("-inf")).sum().item()))
            if k_take <= 0:
                continue
            topk = torch.topk(scores, k=k_take).indices.tolist()
            pos_rows = news_buckets[d, cid][:max_news_per_bag]
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
                    neg_rows = news_buckets[d, ncid][:max_news_per_bag]
                    if not neg_rows:
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
            other_days = [dd for dd in company_news_days.get(cid, []) if dd != d]
            if not other_days and (not random_fill):
                continue
            if not other_days and random_fill and (len(all_news_keys) < 2):
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
        pos_rows = news_buckets[d, cid][:max_news_per_bag]
        if not pos_rows:
            continue
        other_days = sorted(
            [dd for dd in company_news_days.get(cid, []) if dd != d],
            key=lambda dd: abs(dd - d),
            reverse=True,
        )
        if len(other_days) > 1:
            if rng is None:
                order = torch.randperm(len(other_days)).tolist()
            else:
                order = torch.randperm(len(other_days), generator=rng).tolist()
            keyed = sorted(enumerate(other_days), key=lambda iv: (-abs(iv[1] - d), order[iv[0]]))
            other_days = [dd for (_, dd) in keyed]
        neg_day_slots: list[tuple[int, int] | None] = []
        for dd in other_days:
            if len(neg_day_slots) >= int(num_neg):
                break
            neg_day_slots.append((dd, cid))
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
        _append_news_bag(pos_rows, vocab_ids, vocab_attn, pos_ids_list, pos_attn_list, pos_ptr)
        row_valid: list[bool] = []
        for t in range(int(num_neg)):
            if t < len(neg_day_slots) and neg_day_slots[t] is not None:
                (nd, nc) = neg_day_slots[t]
                neg_rows = news_buckets[nd, nc][:max_news_per_bag]
                ok = _append_news_bag(
                    neg_rows, vocab_ids, vocab_attn, neg_ids_list, neg_attn_list, neg_ptr
                )
                row_valid.append(ok)
            else:
                _append_news_bag([], vocab_ids, vocab_attn, neg_ids_list, neg_attn_list, neg_ptr)
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
    company_news_rows: dict[int, list[int]] = {}
    for ni in range(n_news):
        d = int(news_day[ni].item())
        if d < 0 or d >= num_days:
            continue
        c = int(news_comp[ni].item())
        company_news_rows.setdefault(c, []).append(ni)
    if not company_news_rows:
        return None
    if str(day_mode).strip().lower() == "all":
        day_list = list(range(num_days))
    else:
        day_list = [num_days - 1]
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
        _append_news_bag(rows, vocab_ids, vocab_attn, news_ids_list, news_attn_list, news_ptr)
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
            (_y, yv) = _label_at(batch, pidx)
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
        rows = news_buckets[d, cid][:max_news_per_bag]
        prices.append(feat_all[pidx])
        (y, yv) = _label_at(batch, pidx)
        labels.append(y)
        label_valids.append(yv)
        cids.append(cid)
        days.append(d)
        _append_news_bag(rows, vocab_ids, vocab_attn, news_ids_list, news_attn_list, news_ptr)
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
    batch, *, neg_mode: str = "temporal", **kwargs
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


import torch
import torch.nn.functional as F


def clip_info_nce(
    z_price: torch.Tensor, z_news: torch.Tensor, *, temperature: float = 0.07
) -> torch.Tensor:
    """Symmetric CLIP InfoNCE; positives are diagonal (same company).

    z_price / z_news: [B, D]. Off-diagonal pairs are cross-company negatives.
    Requires B >= 2.
    """
    b = int(z_price.size(0))
    if b < 2:
        return z_price.new_zeros(())
    a = F.normalize(z_price, dim=-1)
    n = F.normalize(z_news, dim=-1)
    logits = a @ n.T / float(temperature)
    labels = torch.arange(b, device=a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def hardneg_info_nce(
    z_anchor: torch.Tensor,
    z_pos: torch.Tensor,
    z_neg: torch.Tensor,
    *,
    temperature: float = 0.07,
    neg_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """z_anchor/z_pos: [B, D]; z_neg: [B, K, D].

    For each row i: positive = pos[i], negatives = neg[i, :].
    Optional neg_valid [B, K] bool drops padded hard-neg slots.
    """
    if z_anchor.numel() == 0:
        return z_anchor.new_zeros(())
    bsz = int(z_anchor.size(0))
    if bsz == 0:
        return z_anchor.new_zeros(())
    a = F.normalize(z_anchor, dim=-1)
    p = F.normalize(z_pos, dim=-1)
    n = F.normalize(z_neg, dim=-1)
    pos_logit = (a * p).sum(dim=-1) / float(temperature)
    neg_logit = (a.unsqueeze(1) * n).sum(dim=-1) / float(temperature)
    if neg_valid is not None:
        mask = neg_valid.bool()
        neg_logit = neg_logit.masked_fill(~mask, float("-inf"))
        has_neg = mask.any(dim=-1)
    else:
        has_neg = torch.ones(bsz, dtype=torch.bool, device=a.device)
    logits = torch.cat([pos_logit.unsqueeze(1), neg_logit], dim=1)
    labels = torch.zeros(bsz, dtype=torch.long, device=a.device)
    if not bool(has_neg.all()):
        pass
    loss = F.cross_entropy(logits, labels, reduction="none")
    if not bool(has_neg.all()):
        loss = loss[has_neg]
        if loss.numel() == 0:
            return a.new_zeros(())
    return loss.mean()


def peer_mask_from_sim(sim: torch.Tensor, *, quantile: float = 0.75) -> torch.Tensor:
    bsz = int(sim.size(0))
    if bsz < 2:
        return torch.zeros_like(sim, dtype=torch.bool)
    eye = torch.eye(bsz, device=sim.device, dtype=torch.bool)
    off = sim[~eye]
    if off.numel() == 0:
        return torch.zeros_like(sim, dtype=torch.bool)
    thr = torch.quantile(off.float(), float(quantile))
    return (sim >= thr) & ~eye


def peer_label_supcon(
    z: torch.Tensor,
    labels: torch.Tensor,
    peer: torch.Tensor,
    *,
    temperature: float = 0.07,
    label_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """Among peers: same label = pos, different = neg (SupCon on joint state)."""
    bsz = int(z.size(0))
    if bsz < 2:
        return (z.new_zeros(()), {"pos_sim": 0.0, "neg_sim": 0.0, "n_anchor": 0})
    a = F.normalize(z, dim=-1)
    y = labels.view(-1).float()
    valid = (
        torch.ones(bsz, dtype=torch.bool, device=a.device)
        if label_valid is None
        else label_valid.view(-1).bool()
    )
    peer = peer.bool().clone()
    peer &= valid.unsqueeze(0) & valid.unsqueeze(1)
    peer.fill_diagonal_(False)
    same = (y.unsqueeze(0) == y.unsqueeze(1)) & peer
    diff = (y.unsqueeze(0) != y.unsqueeze(1)) & peer
    logits = a @ a.T / float(temperature)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    losses = []
    for i in range(bsz):
        if not bool(valid[i]):
            continue
        pos_i = same[i]
        peer_i = peer[i]
        if not bool(pos_i.any()) or not bool(peer_i.any()):
            continue
        log_prob = logits[i] - torch.logsumexp(logits[i].masked_fill(~peer_i, float("-inf")), dim=0)
        losses.append(-log_prob[pos_i].mean())
    loss = a.new_zeros(()) if not losses else torch.stack(losses).mean()
    with torch.no_grad():
        cos = a @ a.T
        pos_sim = float(cos[same].mean().item()) if bool(same.any()) else 0.0
        neg_sim = float(cos[diff].mean().item()) if bool(diff.any()) else 0.0
    return (loss, {"pos_sim": pos_sim, "neg_sim": neg_sim, "n_anchor": int(len(losses))})


import sys
from collections import defaultdict, deque
from pathlib import Path
import torch
import torch.nn.functional as F


class ContrastPairState:
    """Sliding pos_sim history for dropping weak positives."""

    def __init__(self, window_k: int = 3, tau_pos: float = 0.3) -> None:
        self.window_k = max(1, int(window_k))
        self.tau_pos = float(tau_pos)
        self.hist: dict[tuple[int, int], deque] = defaultdict(lambda: deque(maxlen=self.window_k))
        self.epoch = 0
        self._epoch_buf: dict[tuple[int, int], list[float]] = defaultdict(list)

    def set_epoch(self, epoch: int) -> None:
        ep = int(epoch)
        if ep != self.epoch and self._epoch_buf:
            for key, vals in self._epoch_buf.items():
                if vals:
                    self.hist[key].append(sum(vals) / len(vals))
            self._epoch_buf = defaultdict(list)
        self.epoch = ep

    def observe(
        self, company_id: torch.Tensor, day_id: torch.Tensor | None, pos_sim: torch.Tensor
    ) -> None:
        cid = company_id.detach().view(-1).tolist()
        if day_id is None:
            did = [0] * len(cid)
        else:
            did = day_id.detach().view(-1).tolist()
        ps = pos_sim.detach().view(-1).tolist()
        for c, d, s in zip(cid, did, ps):
            self._epoch_buf[int(c), int(d)].append(float(s))

    def alpha(
        self,
        company_id: torch.Tensor,
        day_id: torch.Tensor | None,
        *,
        alpha_bad: float = 0.3,
        device: torch.device,
    ) -> torch.Tensor:
        """α_i = α_bad if K-window avg pos_sim < τ_pos and window full, else 1."""
        bsz = int(company_id.numel())
        out = torch.ones(bsz, device=device, dtype=torch.float32)
        cid = company_id.detach().view(-1).tolist()
        if day_id is None:
            did = [0] * bsz
        else:
            did = day_id.detach().view(-1).tolist()
        for i, (c, d) in enumerate(zip(cid, did)):
            h = self.hist.get((int(c), int(d)))
            if h is not None and len(h) >= self.window_k:
                avg = sum(h) / len(h)
                if avg < self.tau_pos:
                    out[i] = float(alpha_bad)
        return out


def _peer_thr(sim: torch.Tensor, quantile: float) -> float:
    bsz = int(sim.size(0))
    eye = torch.eye(bsz, device=sim.device, dtype=torch.bool)
    off = sim[~eye]
    if off.numel() == 0:
        return 0.0
    return float(torch.quantile(off.float(), float(quantile)).item())


def _lambda_clip_anneal(epoch: int, max_epochs: int, lam_start: float, lam_end: float) -> float:
    """Linear anneal λ1: large early → small late."""
    t = 0.0 if max_epochs <= 1 else min(1.0, max(0.0, (epoch - 1) / (max_epochs - 1)))
    return float(lam_start) * (1.0 - t) + float(lam_end) * t


def compute_method1_aux(
    model,
    batch,
    *,
    state: ContrastPairState,
    epoch: int = 1,
    max_epochs: int = 50,
    temperature: float = 0.07,
    temperature_pair: float | None = None,
    max_anchors: int = 48,
    max_news_per_bag: int = 6,
    day_mode: str = "all",
    sim_quantile: float = 0.75,
    tau_y: float = 0.5,
    beta_hard: float = 0.5,
    alpha_bad: float = 0.3,
    lambda_clip_start: float = 0.3,
    lambda_clip_end: float = 0.05,
    lambda_pair: float = 0.2,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return (λ1 L_clip + λ2 L_pair, stats). L_pred stays outside (unweighted)."""
    pack = mine_contrast_pack(
        batch,
        neg_mode="bag",
        max_anchors=int(max_anchors),
        max_news_per_bag=int(max_news_per_bag),
        day_mode=str(day_mode),
    )
    if pack is None or not isinstance(pack, ClipPack):
        ref = batch["price"].x
        return (ref.new_zeros(()), {"loss": 0.0, "skipped": 1.0, "bsz": 0.0})
    if pack.label is None or pack.label_valid is None:
        ref = batch["price"].x
        return (ref.new_zeros(()), {"loss": 0.0, "skipped": 1.0, "bsz": 0.0})
    if device is None:
        device = batch["price"].x.device
    if getattr(model, "contrast_proj_p", None) is None:
        model.ensure_contrast_heads()
    state.set_epoch(int(epoch))
    tau_p = float(temperature_pair if temperature_pair is not None else temperature)
    price = pack.price.to(device)
    (z_j, h_p, h_n) = _encode_joint_state_proj(
        model, price, pack.news_ids.to(device), pack.news_attn.to(device), pack.news_ptr
    )
    z_s = model.contrast_proj_p(h_p)
    z_t = model.contrast_proj_n(h_n)
    z_f = z_j
    labels = pack.label.to(device).view(-1).float()
    valid = pack.label_valid.to(device).view(-1).bool()
    bsz = int(z_s.size(0))
    if bsz < 2:
        return (z_s.new_zeros(()), {"loss": 0.0, "skipped": 1.0, "bsz": float(bsz)})
    zs_n = F.normalize(z_s, dim=-1)
    zt_n = F.normalize(z_t, dim=-1)
    zf_n = F.normalize(z_f, dim=-1)
    pos_sim = (zs_n * zt_n).sum(dim=-1)
    day_id = getattr(pack, "day_id", None)
    if day_id is not None:
        day_id = day_id.to(device)
    state.observe(pack.company_id.to(device), day_id, pos_sim)
    alpha = state.alpha(
        pack.company_id.to(device), day_id, alpha_bad=float(alpha_bad), device=device
    )
    sim_s = zs_n @ zs_n.T
    sim_t = zt_n @ zt_n.T
    thr_s = _peer_thr(sim_s, float(sim_quantile))
    thr_t = _peer_thr(sim_t, float(sim_quantile))
    eye = torch.eye(bsz, device=device, dtype=torch.bool)
    price_sim = sim_s > thr_s
    news_sim = sim_t > thr_t
    price_dis = ~price_sim & ~eye
    news_dis = ~news_sim & ~eye
    price_sim = price_sim & ~eye
    news_sim = news_sim & ~eye
    dy = (labels.unsqueeze(0) - labels.unsqueeze(1)).abs()
    same_y = (dy < float(tau_y)) & ~eye
    diff_y = (dy > float(tau_y)) & ~eye
    both_valid = valid.unsqueeze(0) & valid.unsqueeze(1) & ~eye
    same_y = same_y & both_valid
    diff_y = diff_y & both_valid
    case1 = price_sim & news_dis & diff_y
    case2 = price_sim & news_dis & same_y
    case3 = price_dis & news_sim & diff_y
    case4 = price_dis & news_sim & same_y
    beta = torch.ones(bsz, bsz, device=device, dtype=torch.float32)
    beta = beta.masked_fill(case2 | case4, float(beta_hard))
    gamma = torch.ones(bsz, device=device, dtype=torch.float32)
    for a in range(bsz):
        na_mask = news_dis[a] & both_valid[a]
        na = int(na_mask.sum().item())
        na2 = int((na_mask & case2[a]).sum().item())
        nb_mask = news_sim[a] & both_valid[a]
        nb = int(nb_mask.sum().item())
        nb4 = int((nb_mask & case4[a]).sum().item())
        rA = na2 / na if na > 0 else 0.0
        rB = nb4 / nb if nb > 0 else 0.0
        gamma[a] = 1.0 - 0.5 * (rA + rB)
    gamma = gamma.clamp(0.0, 1.0)
    logits_clip = zs_n @ zt_n.T / float(temperature)
    log_prob = logits_clip - torch.logsumexp(logits_clip, dim=1, keepdim=True)
    nll = -log_prob.diag()
    w_clip = (alpha * gamma).clamp_min(0.0)
    if bool(valid.any()):
        L_clip = (w_clip[valid] * nll[valid]).sum() / w_clip[valid].sum().clamp_min(1e-06)
    else:
        L_clip = z_s.new_zeros(())
    logits_f = zf_n @ zf_n.T / float(tau_p)
    pair_losses = []
    for i in range(bsz):
        if not bool(valid[i]):
            continue
        pos_js = same_y[i].nonzero(as_tuple=False).view(-1)
        neg_js = diff_y[i].nonzero(as_tuple=False).view(-1)
        if pos_js.numel() == 0 or neg_js.numel() == 0:
            continue
        for j in pos_js.tolist():
            num = logits_f[i, j]
            neg_logits = logits_f[i, neg_js]
            denom = torch.logsumexp(torch.cat([num.view(1), neg_logits], dim=0), dim=0)
            lp = -(num - denom)
            bij = beta[i, j]
            wij = bij * gamma[i] * gamma[j]
            pair_losses.append(wij * lp)
    if pair_losses:
        L_pair = torch.stack(pair_losses).mean()
    else:
        L_pair = z_s.new_zeros(())
    lam1 = _lambda_clip_anneal(
        int(epoch), int(max_epochs), float(lambda_clip_start), float(lambda_clip_end)
    )
    lam2 = float(lambda_pair)
    loss = lam1 * L_clip + lam2 * L_pair
    with torch.no_grad():
        n_bad = float((alpha < 0.999).float().mean().item())
        stats = {
            "loss": float(loss.detach().item()),
            "L_clip": float(L_clip.detach().item()),
            "L_pair": float(L_pair.detach().item()),
            "lam1": lam1,
            "lam2": lam2,
            "bsz": float(bsz),
            "skipped": 0.0,
            "pos_sim": float(pos_sim.mean().item()),
            "neg_sim": float((zf_n @ zf_n.T)[diff_y].mean().item() if bool(diff_y.any()) else 0.0),
            "frac_bad_alpha": n_bad,
            "gamma_mean": float(gamma.mean().item()),
            "n_case1": float(case1.sum().item()),
            "n_case2": float(case2.sum().item()),
            "n_case3": float(case3.sum().item()),
            "n_case4": float(case4.sum().item()),
            "thr_s": thr_s,
            "thr_t": thr_t,
        }
    return (loss, stats)


import sys
from pathlib import Path
import torch
import torch.nn.functional as F


def enable_contrast_gates(model, case_mine_layer: int | None = None) -> None:
    """Turn on per-layer news gates on PartialUnifiedModel."""
    unitrans = getattr(model, "unitrans", None)
    if unitrans is None:
        return
    if hasattr(unitrans, "enable_news_gate"):
        unitrans.enable_news_gate(True)
    elif hasattr(unitrans, "enable_news_gate"):
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


def _encode_mtvc_pack(
    model,
    price: torch.Tensor,
    news_ids: torch.Tensor,
    news_attn: torch.Tensor,
    ptr: torch.Tensor,
    *,
    case_feat: str = "joint",
    case_mine_layer: int = 6,
    enable_gate: bool = True,
    contrast_news_mode: str = "nbag",
    max_news_per_bag: int = 6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float | None]:
    """Return (z_fusion_proj, zs_case, zt_case, w_mean).

    case_feat:
      - joint: mid-joint layer price/news pools (default paper path)
      - encoder: pre-joint modality encoder pools (price SeqTF / news bag)
    L_pair still uses joint fusion state z_fusion_proj.
    """
    if getattr(model, "contrast_proj_p", None) is None:
        model.ensure_contrast_heads()
    unitrans = getattr(model, "unitrans", None)
    if unitrans is None or not hasattr(unitrans, "forward_unified"):
        raise RuntimeError("Model.unitrans.forward_unified required for MTVC contrast")
    _feat = str(case_feat or "joint").strip().lower()
    if _feat not in ("joint", "encoder"):
        _feat = "joint"
    if enable_gate:
        enable_contrast_gates(model, case_mine_layer=int(case_mine_layer))
    elif hasattr(unitrans, "case_mine_layer"):
        unitrans.case_mine_layer = int(case_mine_layer)
    (h_p, day) = _encode_price_day_feat(model, price)
    bsz = int(day.size(0))
    num_days = int(day.size(1))
    _news_mode = str(contrast_news_mode or "nbag").strip().lower()
    if _news_mode == "multinews":
        (news_tok, news_pad, h_n) = _encode_news_token_pad(
            model, news_ids, news_attn, ptr, max_news=int(max_news_per_bag)
        )
        if int(news_tok.size(0)) != bsz:
            raise ValueError(f"news tokens {int(news_tok.size(0))} != price {bsz}")
        news_day = torch.full(
            (bsz, int(news_tok.size(1))), num_days - 1, dtype=torch.long, device=day.device
        )
    else:
        h_n = _encode_news_bag_feat(model, news_ids, news_attn, ptr)
        if int(h_n.size(0)) != bsz:
            raise ValueError(f"news bags {int(h_n.size(0))} != price {bsz}")
        news_tok = h_n.unsqueeze(1)
        news_day = torch.full((bsz, 1), num_days - 1, dtype=torch.long, device=day.device)
        news_pad = torch.zeros(bsz, 1, dtype=torch.bool, device=day.device)
    h_j = unitrans.forward_unified(day, news_tok=news_tok, news_day_idx=news_day, news_pad=news_pad)
    if _feat == "encoder":
        (zs, zt) = (h_p, h_n)
    else:
        zs = getattr(unitrans, "last_mtvc_price", None)
        zt = getattr(unitrans, "last_mtvc_news", None)
        if zs is None or zt is None:
            zs = h_p
            zt = h_n
    w_mean = getattr(unitrans, "last_mtvc_gate_w", None)
    return (model.contrast_proj_p(h_j), zs, zt, w_mean)


def compute_mtvc_aux(
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
    use_clip: bool = False,
    state: ContrastPairState | None = None,
    epoch: int = 1,
    max_epochs: int = 50,
    temperature: float = 0.07,
    alpha_bad: float = 0.3,
    lambda_clip_start: float = 0.3,
    lambda_clip_end: float = 0.05,
    device: torch.device | None = None,
    enable_gate: bool = True,
    case_mine_layer: int = 6,
    case_feat: str = "joint",
    contrast_news_mode: str = "nbag",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return (λ1 L_clip + λ2 L_pair, stats). L_clip only if use_clip=True.

    pair_peer=label|news|price controls which peers enter L_pair.
    case_feat=joint|encoder: features used to mine hard/easy cases (sim thr).
    """
    _peer = str(pair_peer or "label").strip().lower()
    if _peer in ("jsim_news", "news_sim", "similar_news"):
        _peer = "news"
    elif _peer in ("jsim_price", "price_sim", "similar_price", "price_peer"):
        _peer = "price"
    elif _peer not in ("label", "news", "price"):
        _peer = "label"
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
        return (ref.new_zeros(()), {"loss": 0.0, "skipped": 1.0, "bsz": 0.0})
    if pack.label is None or pack.label_valid is None:
        ref = batch["price"].x
        return (ref.new_zeros(()), {"loss": 0.0, "skipped": 1.0, "bsz": 0.0})
    if device is None:
        device = batch["price"].x.device
    if getattr(model, "contrast_proj_p", None) is None:
        model.ensure_contrast_heads()
    if use_clip and getattr(model, "contrast_proj_n", None) is None:
        model.ensure_contrast_heads()
    (z_f, zs_raw, zt_raw, w_mean) = _encode_mtvc_pack(
        model,
        pack.price.to(device),
        pack.news_ids.to(device),
        pack.news_attn.to(device),
        pack.news_ptr,
        case_feat=str(case_feat or "joint"),
        case_mine_layer=int(case_mine_layer),
        enable_gate=bool(enable_gate),
        contrast_news_mode=str(contrast_news_mode or "nbag"),
        max_news_per_bag=int(max_news_per_bag),
    )
    labels = pack.label.to(device).view(-1).float()
    valid = pack.label_valid.to(device).view(-1).bool()
    bsz = int(z_f.size(0))
    if bsz < 2:
        return (z_f.new_zeros(()), {"loss": 0.0, "skipped": 1.0, "bsz": float(bsz)})
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
    price_dis = ~(sim_s > thr_s) & ~eye
    news_dis = ~(sim_t > thr_t) & ~eye
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
        beta = beta.masked_fill(case4, float(beta_hard))
    elif _peer == "price":
        beta = beta.masked_fill(case2, float(beta_hard))
    else:
        beta = beta.masked_fill(case2 | case4, float(beta_hard))
    if _peer == "news":
        peer = news_sim & both_valid
    elif _peer == "price":
        peer = price_sim & both_valid
    else:
        peer = both_valid
    gamma = torch.ones(bsz, device=device, dtype=torch.float32)
    alpha = torch.ones(bsz, device=device, dtype=torch.float32)
    pos_sim = (zs_n * zt_n).sum(dim=-1)
    if use_clip:
        for a in range(bsz):
            na_mask = news_dis[a] & both_valid[a]
            na = int(na_mask.sum().item())
            na2 = int((na_mask & case2[a]).sum().item())
            nb_mask = news_sim[a] & both_valid[a]
            nb = int(nb_mask.sum().item())
            nb4 = int((nb_mask & case4[a]).sum().item())
            rA = na2 / na if na > 0 else 0.0
            rB = nb4 / nb if nb > 0 else 0.0
            gamma[a] = 1.0 - 0.5 * (rA + rB)
        gamma = gamma.clamp(0.0, 1.0)
        if state is not None:
            state.set_epoch(int(epoch))
            day_id = getattr(pack, "day_id", None)
            if day_id is not None:
                day_id = day_id.to(device)
            cid = pack.company_id.to(device)
            state.observe(cid, day_id, pos_sim)
            alpha = state.alpha(cid, day_id, alpha_bad=float(alpha_bad), device=device)
        z_s = model.contrast_proj_p(zs_raw)
        z_t = model.contrast_proj_n(zt_raw)
        zs_c = F.normalize(z_s, dim=-1)
        zt_c = F.normalize(z_t, dim=-1)
        pos_sim = (zs_c * zt_c).sum(dim=-1)
        logits_clip = zs_c @ zt_c.T / float(temperature)
        log_prob = logits_clip - torch.logsumexp(logits_clip, dim=1, keepdim=True)
        nll = -log_prob.diag()
        w_clip = (alpha * gamma).clamp_min(0.0)
        if bool(valid.any()):
            L_clip = (w_clip[valid] * nll[valid]).sum() / w_clip[valid].sum().clamp_min(1e-06)
        else:
            L_clip = z_f.new_zeros(())
        lam1 = _lambda_clip_anneal(
            int(epoch), int(max_epochs), float(lambda_clip_start), float(lambda_clip_end)
        )
    else:
        L_clip = z_f.new_zeros(())
        lam1 = 0.0
    tau_p = float(temperature_pair)
    logits_f = zf_n @ zf_n.T / tau_p
    pair_pos = same_y & peer
    pair_neg = diff_y & peer
    pair_losses = []
    for i in range(bsz):
        if not bool(valid[i]):
            continue
        pos_js = pair_pos[i].nonzero(as_tuple=False).view(-1)
        if _peer in ("news", "price"):
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
            "pos_sim": (
                float((zf_n @ zf_n.T)[pair_pos].mean().item())
                if bool(pair_pos.any())
                else float(pos_sim.mean().item())
            ),
            "neg_sim": (
                float((zf_n @ zf_n.T)[pair_neg].mean().item()) if bool(pair_neg.any()) else 0.0
            ),
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
    return (loss, stats)


import sys
from pathlib import Path
import torch
import torch.nn.functional as F


def _price_hlc(price_x: torch.Tensor) -> torch.Tensor:
    if price_x.size(-1) >= 3:
        return price_x[..., :3]
    pad = price_x.new_zeros(*price_x.shape[:-1], 3 - price_x.size(-1))
    return torch.cat([price_x, pad], dim=-1)


def _encode_price_proj(model, price: torch.Tensor) -> torch.Tensor:
    """[B,K,3] → contrast_proj_p(last-day price_h). Uses Model.unitrans_* path."""
    if getattr(model, "contrast_proj_p", None) is None:
        model.ensure_contrast_heads()
    price_tok_mod = getattr(model, "unitrans_price_tok", None)
    unitrans = getattr(model, "unitrans", None)
    if price_tok_mod is None or unitrans is None:
        raise RuntimeError("Model missing unitrans_price_tok / unitrans for contrast aux")
    (bsz, k, f) = price.shape
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
            day = torch.cat([day.new_zeros(bsz, num_days - day.size(1), day.size(-1)), day], dim=1)
        else:
            day = day[:, -num_days:, :]
    towers = unitrans.encode_contrast_towers(day, news_tok=None)
    return model.contrast_proj_p(towers["price_h"][:, -1, :])


def _encode_news_rows(model, ids: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
    """Encode news token ids → contrast_proj_n. Supports BertTok (CSMD char) and TextTF+glove (massive)."""
    if getattr(model, "contrast_proj_n", None) is None:
        model.ensure_contrast_heads()
    bert = getattr(model, "bert_tok_transformer", None)
    text_tf = getattr(model, "unitrans_news_text_tf", None)
    wemb = getattr(model, "word_embedding", None)
    if bert is not None:
        h = bert(ids, attention_mask=attn)
    elif text_tf is not None and wemb is not None:
        remap = getattr(model, "_remap_vocab_input_ids", None)
        remapped = remap(ids) if callable(remap) else ids
        tok_emb = wemb(remapped)
        pad_idx = int(getattr(model, "word_pad_idx", 0) or 0)
        pad_mask = remapped.eq(pad_idx)
        max_l = int(getattr(text_tf, "max_len", tok_emb.size(1)))
        if tok_emb.size(1) > max_l:
            tok_emb = tok_emb[:, :max_l, :]
            pad_mask = pad_mask[:, :max_l]
        apply_tf = getattr(model, "_apply_news_text_tf", None)
        if callable(apply_tf):
            h = apply_tf(tok_emb, key_padding_mask=pad_mask, chunk_size=0)
        else:
            h = text_tf(tok_emb, key_padding_mask=pad_mask)
    else:
        raise RuntimeError(
            "Model missing news text encoder for contrast aux (need bert_tok_transformer or unitrans_news_text_tf+word_embedding)"
        )
    proj = getattr(model, "unified_news_proj", None)
    if proj is not None:
        h = proj(h)
    return model.contrast_proj_n(h)


def _encode_news_bag(
    model, ids: torch.Tensor, attn: torch.Tensor, ptr: torch.Tensor
) -> torch.Tensor:
    bsz = int(ptr.numel()) - 1
    if bsz <= 0:
        last = model.contrast_proj_n[-1]
        pd = int(getattr(last, "out_features", 64))
        return ids.new_zeros((0, pd), dtype=torch.float32)
    z = _encode_news_rows(model, ids, attn)
    ptr_cpu = ptr.detach().to("cpu").tolist()
    n_news = int(z.size(0))
    bag_ids = torch.empty(n_news, dtype=torch.long, device=z.device)
    for i in range(bsz):
        (a, b) = (int(ptr_cpu[i]), int(ptr_cpu[i + 1]))
        if b > a:
            bag_ids[a:b] = i
    out = z.new_zeros((bsz, z.size(-1)))
    counts = z.new_zeros((bsz, 1))
    out.index_add_(0, bag_ids, z)
    counts.index_add_(0, bag_ids, z.new_ones((n_news, 1)))
    return out / counts.clamp_min(1.0)


def _encode_price_day_feat(model, price: torch.Tensor):
    """[B,K,3] → (last-day price_h [B,D], day_tok [B,K,D]) via encode_contrast_towers."""
    price_tok_mod = getattr(model, "unitrans_price_tok", None)
    unitrans = getattr(model, "unitrans", None)
    if price_tok_mod is None or unitrans is None:
        raise RuntimeError("Model missing unitrans_price_tok / unitrans")
    (bsz, k, f) = price.shape
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
            day = torch.cat([day.new_zeros(bsz, num_days - day.size(1), day.size(-1)), day], dim=1)
        else:
            day = day[:, -num_days:, :]
    towers = unitrans.encode_contrast_towers(day, news_tok=None)
    return (towers["price_h"][:, -1, :], day)


def _encode_news_bag_feat(
    model, ids: torch.Tensor, attn: torch.Tensor, ptr: torch.Tensor
) -> torch.Tensor:
    """News bag mean-pool in price_dim (before contrast_proj)."""
    bsz = int(ptr.numel()) - 1
    if bsz <= 0:
        d = int(getattr(model, "price_emb_dim", 128))
        return ids.new_zeros((0, d), dtype=torch.float32)
    bert = getattr(model, "bert_tok_transformer", None)
    text_tf = getattr(model, "unitrans_news_text_tf", None)
    wemb = getattr(model, "word_embedding", None)
    if bert is not None:
        h = bert(ids, attention_mask=attn)
    elif text_tf is not None and wemb is not None:
        remap = getattr(model, "_remap_vocab_input_ids", None)
        remapped = remap(ids) if callable(remap) else ids
        tok_emb = wemb(remapped)
        pad_idx = int(getattr(model, "word_pad_idx", 0) or 0)
        pad_mask = remapped.eq(pad_idx)
        max_l = int(getattr(text_tf, "max_len", tok_emb.size(1)))
        if tok_emb.size(1) > max_l:
            tok_emb = tok_emb[:, :max_l, :]
            pad_mask = pad_mask[:, :max_l]
        apply_tf = getattr(model, "_apply_news_text_tf", None)
        if callable(apply_tf):
            h = apply_tf(tok_emb, key_padding_mask=pad_mask, chunk_size=0)
        else:
            h = text_tf(tok_emb, key_padding_mask=pad_mask)
    else:
        raise RuntimeError("Model missing news text encoder")
    proj = getattr(model, "unified_news_proj", None)
    if proj is not None:
        h = proj(h)
    ptr_cpu = ptr.detach().to("cpu").tolist()
    n_news = int(h.size(0))
    bag_ids = torch.empty(n_news, dtype=torch.long, device=h.device)
    for i in range(bsz):
        (a, b) = (int(ptr_cpu[i]), int(ptr_cpu[i + 1]))
        if b > a:
            bag_ids[a:b] = i
    out = h.new_zeros((bsz, h.size(-1)))
    counts = h.new_zeros((bsz, 1))
    out.index_add_(0, bag_ids, h)
    counts.index_add_(0, bag_ids, h.new_ones((n_news, 1)))
    return out / counts.clamp_min(1.0)


def _encode_news_token_pad(
    model, ids: torch.Tensor, attn: torch.Tensor, ptr: torch.Tensor, max_news: int = 6
):
    """Per-news tokens for multinews_casel6. Returns (tok [B,M,D], pad [B,M], bag [B,D])."""
    bsz = int(ptr.numel()) - 1
    d = int(getattr(model, "price_emb_dim", 128))
    if bsz <= 0:
        empty = ids.new_zeros((0, 0, d), dtype=torch.float32)
        pad = torch.zeros((0, 0), dtype=torch.bool)
        return (empty, pad, ids.new_zeros((0, d), dtype=torch.float32))
    bert = getattr(model, "bert_tok_transformer", None)
    text_tf = getattr(model, "unitrans_news_text_tf", None)
    wemb = getattr(model, "word_embedding", None)
    if bert is not None:
        h = bert(ids, attention_mask=attn)
    elif text_tf is not None and wemb is not None:
        remap = getattr(model, "_remap_vocab_input_ids", None)
        remapped = remap(ids) if callable(remap) else ids
        tok_emb = wemb(remapped)
        pad_idx = int(getattr(model, "word_pad_idx", 0) or 0)
        pad_mask = remapped.eq(pad_idx)
        max_l = int(getattr(text_tf, "max_len", tok_emb.size(1)))
        if tok_emb.size(1) > max_l:
            tok_emb = tok_emb[:, :max_l, :]
            pad_mask = pad_mask[:, :max_l]
        apply_tf = getattr(model, "_apply_news_text_tf", None)
        if callable(apply_tf):
            h = apply_tf(tok_emb, key_padding_mask=pad_mask, chunk_size=0)
        else:
            h = text_tf(tok_emb, key_padding_mask=pad_mask)
    else:
        raise RuntimeError("Model missing news text encoder")
    proj = getattr(model, "unified_news_proj", None)
    if proj is not None:
        h = proj(h)
    ptr_cpu = ptr.detach().to("cpu").tolist()
    cap = int(max_news)
    lengths = [max(0, int(ptr_cpu[i + 1]) - int(ptr_cpu[i])) for i in range(bsz)]
    if cap > 0:
        lengths = [min(n, cap) for n in lengths]
    m = max(lengths) if lengths else 0
    tok = h.new_zeros((bsz, m, h.size(-1)))
    pad = torch.ones((bsz, m), dtype=torch.bool, device=h.device)
    for i in range(bsz):
        a = int(ptr_cpu[i])
        n = lengths[i]
        if n > 0:
            tok[i, :n] = h[a : a + n]
            pad[i, :n] = False
    if m == 0:
        bag = h.new_zeros((bsz, h.size(-1)))
    else:
        valid = (~pad).unsqueeze(-1).to(dtype=tok.dtype)
        bag = (tok * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
    return (tok, pad, bag)


def _encode_joint_state_proj(
    model, price: torch.Tensor, news_ids: torch.Tensor, news_attn: torch.Tensor, ptr: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (z_joint, h_price_last, h_news_bag).

    z_joint = contrast_proj_p(forward_unified state) — deepest UnifiedJoint layer.
    """
    if getattr(model, "contrast_proj_p", None) is None:
        model.ensure_contrast_heads()
    unitrans = getattr(model, "unitrans", None)
    if unitrans is None or not hasattr(unitrans, "forward_unified"):
        raise RuntimeError("Model.unitrans.forward_unified required for jsim contrast")
    (h_p, day) = _encode_price_day_feat(model, price)
    h_n = _encode_news_bag_feat(model, news_ids, news_attn, ptr)
    bsz = int(day.size(0))
    if int(h_n.size(0)) != bsz:
        raise ValueError(f"news bags {int(h_n.size(0))} != price {bsz}")
    news_tok = h_n.unsqueeze(1)
    num_days = int(day.size(1))
    news_day = torch.full((bsz, 1), num_days - 1, dtype=torch.long, device=day.device)
    news_pad = torch.zeros(bsz, 1, dtype=torch.bool, device=day.device)
    h_j = unitrans.forward_unified(day, news_tok=news_tok, news_day_idx=news_day, news_pad=news_pad)
    return (model.contrast_proj_p(h_j), h_p, h_n)


def compute_contrast_aux_loss(
    model,
    batch,
    *,
    mode: str = "temporal",
    temperature: float = 0.07,
    num_neg: int = 4,
    max_anchors: int = 48,
    max_news_per_bag: int = 6,
    day_mode: str = "all",
    sim_quantile: float = 0.75,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return (loss, stats). loss is 0-scalar if pack unavailable."""
    mode = str(mode or "temporal").strip().lower()
    if mode in ("nce", "ssl_nce", "day_nce"):
        z = batch["price"].x
        return (z.new_zeros(()), {"loss": 0.0, "bsz": 0.0, "skipped": 1.0})
    if mode in (
        "jsim_news",
        "jsim_price",
        "jsim_both",
        "label_sim_news",
        "label_sim_price",
        "label_sim_both",
        "news",
        "price_peer",
        "both_peer",
    ):
        if mode in ("news", "jsim_news", "label_sim_news"):
            peer_modes = ("news",)
        elif mode in ("price_peer", "jsim_price", "label_sim_price"):
            peer_modes = ("price",)
        else:
            peer_modes = ("news", "price")
        pack = mine_contrast_pack(
            batch,
            neg_mode="bag",
            max_anchors=int(max_anchors),
            max_news_per_bag=int(max_news_per_bag),
            day_mode=str(day_mode),
        )
        if pack is None or not isinstance(pack, ClipPack):
            ref = batch["price"].x
            return (ref.new_zeros(()), {"loss": 0.0, "bsz": 0.0, "skipped": 1.0})
        if pack.label is None or pack.label_valid is None:
            ref = batch["price"].x
            return (ref.new_zeros(()), {"loss": 0.0, "bsz": 0.0, "skipped": 1.0})
        if device is None:
            device = batch["price"].x.device
        (z_j, h_p, h_n) = _encode_joint_state_proj(
            model,
            pack.price.to(device),
            pack.news_ids.to(device),
            pack.news_attn.to(device),
            pack.news_ptr,
        )
        labels = pack.label.to(device)
        label_valid = pack.label_valid.to(device)
        with torch.no_grad():
            sn = F.normalize(h_n.detach(), dim=-1)
            sp = F.normalize(h_p.detach(), dim=-1)
            news_sim = sn @ sn.T
            price_sim = sp @ sp.T
        q = float(sim_quantile)
        losses = []
        (pos_sims, neg_sims) = ([], [])
        for pm in peer_modes:
            peer = peer_mask_from_sim(news_sim if pm == "news" else price_sim, quantile=q)
            (ln, st) = peer_label_supcon(
                z_j, labels, peer, temperature=float(temperature), label_valid=label_valid
            )
            losses.append(ln)
            pos_sims.append(st["pos_sim"])
            neg_sims.append(st["neg_sim"])
        loss = torch.stack(losses).mean() if losses else z_j.new_zeros(())
        return (
            loss,
            {
                "loss": float(loss.detach().item()),
                "bsz": float(z_j.size(0)),
                "skipped": 0.0,
                "pos_sim": float(sum(pos_sims) / max(1, len(pos_sims))),
                "neg_sim": float(sum(neg_sims) / max(1, len(neg_sims))),
            },
        )
    pack = mine_contrast_pack(
        batch,
        neg_mode="clip" if mode in ("clip", "batch") else "temporal",
        num_neg=int(num_neg),
        max_anchors=int(max_anchors),
        max_news_per_bag=int(max_news_per_bag),
        day_mode=str(day_mode),
        random_fill=True,
    )
    if pack is None:
        ref = batch["price"].x
        return (ref.new_zeros(()), {"loss": 0.0, "bsz": 0.0, "skipped": 1.0})
    if device is None:
        device = batch["price"].x.device
    if isinstance(pack, ClipPack):
        z_p = _encode_price_proj(model, pack.price.to(device))
        z_n = _encode_news_bag(
            model, pack.news_ids.to(device), pack.news_attn.to(device), pack.news_ptr
        )
        loss = clip_info_nce(z_p, z_n, temperature=float(temperature))
        with torch.no_grad():
            a = F.normalize(z_p, dim=-1)
            n = F.normalize(z_n, dim=-1)
            sim = a @ n.T
            bsz = int(sim.size(0))
            pos_sim = float(sim.diag().mean().item()) if bsz else 0.0
            if bsz > 1:
                off = sim - torch.diag_embed(sim.diag())
                neg_sim = float(off.sum().item() / (bsz * (bsz - 1)))
            else:
                neg_sim = 0.0
        return (
            loss,
            {
                "loss": float(loss.detach().item()),
                "bsz": float(bsz),
                "skipped": 0.0,
                "pos_sim": pos_sim,
                "neg_sim": neg_sim,
            },
        )
    assert isinstance(pack, ContrastPack)
    z_p = _encode_price_proj(model, pack.price.to(device))
    z_pos = _encode_news_bag(model, pack.pos_ids.to(device), pack.pos_attn.to(device), pack.pos_ptr)
    bsz = int(z_p.size(0))
    z_neg_flat = _encode_news_bag(
        model, pack.neg_ids.to(device), pack.neg_attn.to(device), pack.neg_ptr
    )
    z_neg = z_neg_flat.view(bsz, int(pack.n_neg), -1)
    loss = hardneg_info_nce(
        z_p, z_pos, z_neg, temperature=float(temperature), neg_valid=pack.neg_valid.to(device)
    )
    with torch.no_grad():
        a = F.normalize(z_p, dim=-1)
        p = F.normalize(z_pos, dim=-1)
        n = F.normalize(z_neg, dim=-1)
        pos_sim = float((a * p).sum(-1).mean().item())
        nv = pack.neg_valid.to(device).float()
        neg_sim = float(
            ((a.unsqueeze(1) * n).sum(-1) * nv).sum().item() / nv.sum().clamp_min(1.0).item()
        )
    return (
        loss,
        {
            "loss": float(loss.detach().item()),
            "bsz": float(bsz),
            "skipped": 0.0,
            "pos_sim": pos_sim,
            "neg_sim": neg_sim,
        },
    )
