"""Canonical public names for the open-source MTVC stack."""

from __future__ import annotations
import os

PRICE_ENCODER_CANONICAL = ("cross_attn", "partial_unified", "full_unified")
CONTRAST_MODE_CANONICAL = ("mtvc", "mtvc_nogate", "nocontrast")
VIN_CONNECT_CANONICAL = ("default", "mkt_as_ind", "ind_as_mkt")
NEWS_PRICE_ABLATION_CANONICAL = ("full", "no_news", "no_price")


def normalize_price_encoder(name: str | None) -> str:
    """Return cross_attn, partial_unified, or full_unified."""
    key = str(name or "partial_unified").strip().lower()
    if key not in PRICE_ENCODER_CANONICAL:
        raise ValueError(
            f"price_encoder must be one of {list(PRICE_ENCODER_CANONICAL)}; got {name!r}"
        )
    return key


def normalize_contrast_mode(name: str | None) -> str:
    """Map a contrast-mode string onto the paper loss name."""
    key = str(name or "nocontrast").strip().lower()
    if key in ("off", "0", ""):
        key = "nocontrast"
    if key not in CONTRAST_MODE_CANONICAL:
        raise ValueError(
            f"contrast_mode must be one of {list(CONTRAST_MODE_CANONICAL)}; got {name!r}"
        )
    return key


def is_mtvc_contrast(mode: str | None) -> bool:
    """True when contrast_mode is mtvc or mtvc_nogate."""
    return normalize_contrast_mode(mode) != "nocontrast"


def mtvc_pair_peer(mode: str | None) -> str:
    """Official runs pair on label peers."""
    normalize_contrast_mode(mode)
    return "label"


def mtvc_enable_gate(mode: str | None) -> bool:
    """News gate is on only for mtvc. mtvc_nogate and nocontrast leave it off."""
    return normalize_contrast_mode(mode) == "mtvc"


def normalize_vin_connect(name: str | None) -> str:
    """Return default, mkt_as_ind, or ind_as_mkt."""
    key = str(name or "default").strip().lower()
    if key not in VIN_CONNECT_CANONICAL:
        raise ValueError(
            f"vin_connect_mode must be one of {list(VIN_CONNECT_CANONICAL)}; got {name!r}"
        )
    return key


def normalize_news_price_ablation(news_ablation: str | None, no_price: bool) -> str:
    """Return full, no_news (abl_nonews), or no_price (abl_noprice)."""
    news = str(news_ablation or "full").strip().lower()
    if bool(no_price) and news == "no_news":
        raise ValueError("abl_nonews and abl_noprice are separate runs")
    if bool(no_price):
        key = "no_price"
    else:
        key = news
    if key not in NEWS_PRICE_ABLATION_CANONICAL:
        raise ValueError(
            f"news/price ablation must be one of {list(NEWS_PRICE_ABLATION_CANONICAL)}; got {news_ablation!r}"
        )
    return key


def env_mtvc(key: str, default: str = "") -> str:
    """Read ``MTVC_<KEY>``."""
    k = str(key).removeprefix("MTVC_")
    v = os.environ.get(f"MTVC_{k}")
    if v is None:
        return default
    return str(v)
