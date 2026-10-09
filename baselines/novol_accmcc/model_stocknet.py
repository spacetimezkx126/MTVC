"""
PyTorch reimplementation of StockNet MIE path from PEN Model.py (# stocknet 分支).

MEL -> corpus_embed -> concat(price) -> VMD -> TDA -> generative ATA.
（PEN 的 # pen 分支走 MSIN，见 model_pen.py）
"""
from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def _xavier_init_(tensor: torch.Tensor, uniform: bool = True) -> torch.Tensor:
    if uniform:
        nn.init.xavier_uniform_(tensor)
    else:
        nn.init.xavier_normal_(tensor)
    return tensor


class PenLinear(nn.Module):
    """TF Model._linear: concat args, matmul weight, optional bias + activation."""

    def __init__(
        self,
        in_size: int,
        out_size: int,
        activation: str | None = None,
        use_bias: bool = True,
    ):
        super().__init__()
        self.out_size = out_size
        self.activation = activation
        self.fc = nn.Linear(in_size, out_size, bias=use_bias)
        _xavier_init_(self.fc.weight)
        if use_bias:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc(x)
        if self.activation == "tanh":
            y = torch.tanh(y)
        elif self.activation == "sigmoid":
            y = torch.sigmoid(y)
        elif self.activation == "relu":
            y = F.relu(y)
        elif self.activation == "softmax":
            y = F.softmax(y, dim=-1)
        return y


class DiagonalGaussianZ(nn.Module):
    """TF Model._z: Gaussian with mean/logvar heads; prior uses z=mean."""

    def __init__(self, z_dim: int):
        super().__init__()
        self.mu = nn.Linear(z_dim, z_dim)
        self.lv = nn.Linear(z_dim, z_dim)
        for m in (self.mu, self.lv):
            _xavier_init_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, context: torch.Tensor, is_prior: bool):
        mu = self.mu(context)
        logvar = self.lv(context)
        std = torch.exp(0.5 * logvar).clamp_min(1e-6)
        if is_prior:
            z = mu
        else:
            z = mu + std * torch.randn_like(std)
        pdf = dist.Normal(mu, std)
        return z, pdf


class MessageEmbedLayer(nn.Module):
    """
    TF Model._create_msg_embed_layer: per-day BiRNN over words, gather at ss_index, (h_f+h_b)/2.
    """

    def __init__(
        self,
        vocab_size: int,
        word_embed_size: int,
        mel_h_size: int,
        pad_id: int = 0,
        cell_type: Literal["gru", "basic"] = "gru",
        dropout_in: float = 0.3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.mel_h_size = mel_h_size
        self.pad_id = pad_id
        self.word_embed = nn.Embedding(vocab_size, word_embed_size, padding_idx=pad_id)
        _xavier_init_(self.word_embed.weight)
        if cell_type == "gru":
            self.rnn_f = nn.GRU(
                word_embed_size, mel_h_size, batch_first=True, bidirectional=False
            )
            self.rnn_b = nn.GRU(
                word_embed_size, mel_h_size, batch_first=True, bidirectional=False
            )
        else:
            self.rnn_f = nn.RNN(
                word_embed_size, mel_h_size, batch_first=True, bidirectional=False
            )
            self.rnn_b = nn.RNN(
                word_embed_size, mel_h_size, batch_first=True, bidirectional=False
            )
        self.drop_in = nn.Dropout(dropout_in)
        self.drop = nn.Dropout(dropout)

    def _run_one_direction(
        self, rnn: nn.Module, emb: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        packed = pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        return out

    def forward(
        self,
        word_ids: torch.Tensor,
        n_words: torch.Tensor,
        ss_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        word_ids: [B, L, M, W]
        n_words: [B, L, M]
        ss_index: [B, L, M]
        -> msg_embed [B, L, M, mel_h_size]
        """
        b, l, m, w = word_ids.shape
        flat_ids = word_ids.reshape(b * l * m, w)
        flat_n = n_words.reshape(b * l * m).clamp(min=1)
        flat_ss = ss_index.reshape(b * l * m).clamp(0, w - 1)

        emb = self.word_embed(flat_ids.clamp(min=0))
        emb = self.drop_in(emb)

        out_f = self._run_one_direction(self.rnn_f, emb, flat_n)
        rev_emb = torch.flip(emb, dims=[1])
        rev_len = flat_n.clone()
        out_b_rev = self._run_one_direction(self.rnn_b, rev_emb, rev_len)
        out_b = torch.flip(out_b_rev, dims=[1])

        idx = torch.arange(flat_ids.size(0), device=word_ids.device)
        h_f = out_f[idx, flat_ss]
        h_b = out_b[idx, flat_ss]
        msg = (h_f + h_b) * 0.5
        msg = self.drop(msg)
        return msg.view(b, l, m, self.mel_h_size)


class CorpusEmbedLayer(nn.Module):
    """TF Model._create_corpus_embed."""

    def __init__(self, mel_dim: int, dropout: float = 0.0):
        super().__init__()
        self.proj_u = nn.Linear(mel_dim, mel_dim, bias=False)
        self.w_u = nn.Parameter(torch.empty(mel_dim, 1))
        _xavier_init_(self.w_u)
        self.drop = nn.Dropout(dropout)

    def forward(self, msg_embed: torch.Tensor, n_msgs: torch.Tensor) -> torch.Tensor:
        b, l, m, d = msg_embed.shape
        proj = torch.tanh(self.proj_u(msg_embed))
        u = torch.matmul(proj, self.w_u).squeeze(-1)
        idx = torch.arange(m, device=n_msgs.device).view(1, 1, m).expand(b, l, m)
        mask = idx < n_msgs.clamp(min=0).unsqueeze(-1)
        u = u.masked_fill(~mask, float("-inf"))
        u = F.softmax(u, dim=-1)
        u = torch.nan_to_num(u, nan=0.0)
        corpus = torch.sum(u.unsqueeze(-1) * msg_embed, dim=2)
        return self.drop(corpus)


class VMDZhRec(nn.Module):
    """TF Model._create_vmd_with_zh_rec."""

    def __init__(
        self,
        x_size: int,
        h_size: int,
        z_size: int,
        g_size: int,
        y_size: int = 2,
        cell_type: Literal["gru", "ln-lstm"] = "gru",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.h_size = h_size
        self.z_size = z_size
        self.g_size = g_size
        self.y_size = y_size
        self.drop = nn.Dropout(dropout)
        if cell_type == "gru":
            self.seq_cell = nn.GRU(x_size, h_size, batch_first=True)
        else:
            self.seq_cell = nn.LSTM(x_size, h_size, batch_first=True)

        z_in_prior = x_size + h_size + z_size
        z_in_post = x_size + h_size + y_size + z_size
        self.enc_prior = nn.Sequential(nn.Linear(z_in_prior, z_size), nn.Tanh())
        self.enc_post = nn.Sequential(nn.Linear(z_in_post, z_size), nn.Tanh())
        self.prior_z = DiagonalGaussianZ(z_size)
        self.post_z = DiagonalGaussianZ(z_size)
        self.g_lin = PenLinear(h_size + z_size, g_size, activation="tanh")
        self.y_lin = PenLinear(g_size, y_size, activation="softmax")

    def forward(
        self,
        x: torch.Tensor,
        y_onehot: torch.Tensor,
        seq_len: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        x = self.drop(x)
        b, l, _ = x.shape
        if isinstance(self.seq_cell, nn.GRU):
            h_s, _ = self.seq_cell(x)
        else:
            h_s, _ = self.seq_cell(x)

        g_list, y_prob_list, kl_list = [], [], []
        z_prior_list, z_post_list = [], []
        z_prev = torch.randn(b, self.z_size, device=x.device, dtype=x.dtype)

        for t in range(l):
            x_t = x[:, t, :]
            h_t = h_s[:, t, :]
            y_t = y_onehot[:, t, :]

            h_z_p = self.enc_prior(torch.cat([x_t, h_t, z_prev], dim=-1))
            z_p, dist_p = self.prior_z(h_z_p, is_prior=True)

            h_z_q = self.enc_post(torch.cat([x_t, h_t, y_t, z_prev], dim=-1))
            z_q, dist_q = self.post_z(h_z_q, is_prior=False)

            kl_t = dist.kl_divergence(dist_q, dist_p).sum(-1)
            kl_list.append(kl_t)

            g_t = self.g_lin(torch.cat([h_t, z_q], dim=-1))
            y_t_prob = self.y_lin(g_t)

            g_list.append(g_t)
            y_prob_list.append(y_t_prob)
            z_prior_list.append(z_p)
            z_post_list.append(z_q)
            z_prev = z_q

        return {
            "g": torch.stack(g_list, dim=1),
            "y_prob": torch.stack(y_prob_list, dim=1),
            "kl": torch.stack(kl_list, dim=1),
            "z_prior": torch.stack(z_prior_list, dim=1),
            "z_post": torch.stack(z_post_list, dim=1),
            "h_seq": h_s,
            "x": x,
        }


class VMDHRec(nn.Module):
    """TF Model._create_vmd_with_h_rec."""

    def __init__(
        self,
        x_size: int,
        h_size: int,
        z_size: int,
        g_size: int,
        y_size: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.x_size = x_size
        self.h_size = h_size
        self.z_size = z_size
        self.g_size = g_size
        self.y_size = y_size
        self.drop = nn.Dropout(dropout)

        xz = x_size + h_size + z_size
        self.gru_r = nn.Linear(xz, h_size)
        self.gru_u = nn.Linear(xz, h_size)
        self.gru_h = nn.Linear(x_size + h_size + z_size, h_size)

        self.enc_prior = nn.Sequential(nn.Linear(x_size + h_size, z_size), nn.Tanh())
        self.enc_post = nn.Sequential(nn.Linear(x_size + h_size + y_size, z_size), nn.Tanh())
        self.prior_z = DiagonalGaussianZ(z_size)
        self.post_z = DiagonalGaussianZ(z_size)
        self.g_lin = PenLinear(x_size + h_size + z_size, g_size, activation="tanh")
        self.y_lin = PenLinear(g_size, y_size, activation="softmax")

    def _init_state(self, batch: int, device, dtype):
        h = torch.tanh(torch.randn(batch, self.h_size, device=device, dtype=dtype))
        h_z = torch.tanh(torch.randn(batch, self.z_size, device=device, dtype=dtype))
        z, _ = self.post_z(h_z, is_prior=False)
        return h, z

    def forward(self, x: torch.Tensor, y_onehot: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.drop(x)
        b, l, _ = x.shape
        device, dtype = x.device, x.dtype
        h, z = self._init_state(b, device, dtype)

        g_list, y_prob_list, kl_list = [], [], []
        z_prior_list, z_post_list, h_list = [], [], []

        for t in range(l):
            x_t = x[:, t, :]
            y_t = y_onehot[:, t, :]
            gate_in = torch.cat([x_t, h, z], dim=-1)
            r = torch.sigmoid(self.gru_r(gate_in))
            u = torch.sigmoid(self.gru_u(gate_in))
            h_tilde = torch.tanh(self.gru_h(torch.cat([x_t, r * h, z], dim=-1)))
            h = (1.0 - u) * h + u * h_tilde

            h_z_p = self.enc_prior(torch.cat([x_t, h], dim=-1))
            z_p, dist_p = self.prior_z(h_z_p, is_prior=True)

            h_z_q = self.enc_post(torch.cat([x_t, h, y_t], dim=-1))
            z_q, dist_q = self.post_z(h_z_q, is_prior=False)

            kl_t = dist.kl_divergence(dist_q, dist_p).sum(-1)
            kl_list.append(kl_t)

            g_t = self.g_lin(torch.cat([x_t, h, z_q], dim=-1))
            y_t_prob = self.y_lin(g_t)

            g_list.append(g_t)
            y_prob_list.append(y_t_prob)
            z_prior_list.append(z_p)
            z_post_list.append(z_q)
            h_list.append(h)

        return {
            "g": torch.stack(g_list, dim=1),
            "y_prob": torch.stack(y_prob_list, dim=1),
            "kl": torch.stack(kl_list, dim=1),
            "z_prior": torch.stack(z_prior_list, dim=1),
            "z_post": torch.stack(z_post_list, dim=1),
            "h_seq": torch.stack(h_list, dim=1),
            "x": x,
        }


class TemporalAttnDecoder(nn.Module):
    """TF Model._build_temporal_att."""

    def __init__(self, g_size: int, y_size: int = 2, daily_att: str = "y"):
        super().__init__()
        assert daily_att in ("y", "g")
        self.daily_att = daily_att
        self.g_size = g_size
        self.ctx_size = y_size if daily_att == "y" else g_size
        self.proj_i = nn.Linear(g_size, g_size, bias=False)
        self.w_i = nn.Parameter(torch.empty(g_size))
        nn.init.normal_(self.w_i, std=0.02)
        self.proj_d = nn.Linear(g_size, g_size, bias=False)
        self.y_T_lin = PenLinear(self.ctx_size + g_size, y_size, activation="softmax")

    def forward(
        self,
        g: torch.Tensor,
        g_T: torch.Tensor,
        y_prob: torch.Tensor,
        mask_aux: torch.Tensor,
    ):
        proj_i = torch.tanh(self.proj_i(g))
        v_i = (proj_i * self.w_i).sum(dim=-1)

        proj_d = torch.tanh(self.proj_d(g))
        v_d = (proj_d * g_T.unsqueeze(1)).sum(dim=-1)

        aux_score = v_i * v_d
        aux_score = aux_score.masked_fill(~mask_aux, float("-inf"))
        v_stared = F.softmax(aux_score, dim=-1)
        v_stared = torch.nan_to_num(v_stared, nan=0.0)

        ctx = g if self.daily_att == "g" else y_prob
        att_c = torch.sum(v_stared.unsqueeze(-1) * ctx, dim=1)
        y_T = self.y_T_lin(torch.cat([att_c, g_T], dim=-1))
        return y_T, v_stared


def mask_auxiliary_days(batch: int, seq_len: int, t_idx: torch.Tensor, device) -> torch.Tensor:
    """TF sequence_mask(T-1, max_n_days) per sample."""
    day_idx = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, seq_len)
    return day_idx < (t_idx.unsqueeze(1) - 1).clamp(min=0)


def generative_ata_loss(
    y_prob: torch.Tensor,
    y_T_prob: torch.Tensor,
    y_onehot: torch.Tensor,
    kl: torch.Tensor,
    v_stared: torch.Tensor,
    alpha: float,
    kl_lambda: float,
    t_idx: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """TF Model._create_generative_ata."""
    log_y = torch.log(y_prob + eps)
    likelihood_aux = (y_onehot * log_y).sum(dim=-1)
    obj_aux = likelihood_aux - kl_lambda * kl
    v_aux = alpha * v_stared

    b = y_onehot.size(0)
    batch_idx = torch.arange(b, device=y_onehot.device)
    target_idx = (t_idx - 1).clamp(min=0)
    y_T_true = y_onehot[batch_idx, target_idx, :]

    log_y_T = torch.log(y_T_prob + eps)
    likelihood_T = (y_T_true * log_y_T).sum(dim=-1, keepdim=True)
    kl_T = kl[batch_idx, target_idx].unsqueeze(1)
    obj_T = likelihood_T - kl_lambda * kl_T

    obj = obj_T + (obj_aux * v_aux).sum(dim=1, keepdim=True)
    return (-obj).mean()


class StockNetModel(nn.Module):
    """
    StockNet MIE（Model.py # stocknet）：MEL + corpus_embed + price concat -> VMD -> TDA。
    """

    def __init__(
        self,
        vocab_size: int,
        pad_id: int = 0,
        word_embed_size: int = 50,
        mel_h_size: int = 100,
        h_size: int = 150,
        g_size: int = 50,
        z_size: int | None = None,
        y_size: int = 2,
        max_n_days: int = 5,
        max_n_msgs: int = 20,
        max_n_words: int = 100,
        variant_type: str = "hedge",
        vmd_rec: str = "zh",
        mel_cell_type: str = "gru",
        vmd_cell_type: str = "gru",
        daily_att: str = "y",
        alpha: float = 0.5,
        kl_lambda: float = 0.0,
        dropout_mel_in: float = 0.3,
        dropout_mel: float = 0.0,
        dropout_ce: float = 0.0,
        dropout_vmd_in: float = 0.3,
        dropout_vmd: float = 0.0,
        price_input_size: int = 3,
    ):
        super().__init__()
        assert variant_type in ("hedge", "fund", "tech")
        assert vmd_rec in ("zh", "h")
        assert daily_att in ("y", "g")
        self.variant_type = variant_type
        self.vmd_rec = vmd_rec
        self.y_size = y_size
        self.max_n_days = max_n_days
        self.max_n_msgs = max_n_msgs
        self.max_n_words = max_n_words
        self.alpha = alpha
        self.kl_lambda = kl_lambda
        self.daily_att = daily_att
        self.pad_id = pad_id
        z_size = z_size or h_size

        self.mel = MessageEmbedLayer(
            vocab_size=vocab_size,
            word_embed_size=word_embed_size,
            mel_h_size=mel_h_size,
            pad_id=pad_id,
            cell_type="gru" if mel_cell_type == "gru" else "basic",
            dropout_in=dropout_mel_in,
            dropout=dropout_mel,
        )
        self.corpus = CorpusEmbedLayer(mel_h_size, dropout=dropout_ce)

        self.price_input_size = int(price_input_size)
        if variant_type == "tech":
            self.x_size = self.price_input_size
        elif variant_type == "fund":
            self.x_size = mel_h_size
        else:
            self.x_size = mel_h_size + self.price_input_size

        if vmd_rec == "zh":
            self.vmd = VMDZhRec(
                self.x_size,
                h_size,
                z_size,
                g_size,
                y_size=y_size,
                cell_type="gru" if vmd_cell_type == "gru" else "ln-lstm",
                dropout=dropout_vmd,
            )
        else:
            self.vmd = VMDHRec(
                self.x_size, h_size, z_size, g_size, y_size=y_size, dropout=dropout_vmd
            )
        self.tda = TemporalAttnDecoder(g_size, y_size=y_size, daily_att=daily_att)
        self.dropout_vmd_in = nn.Dropout(dropout_vmd_in)

    def _build_x(
        self,
        word_ids: torch.Tensor,
        n_words: torch.Tensor,
        ss_index: torch.Tensor,
        n_msgs: torch.Tensor,
        price: torch.Tensor,
    ) -> torch.Tensor:
        if self.variant_type == "tech":
            return price
        msg = self.mel(word_ids, n_words, ss_index)
        corpus = self.corpus(msg, n_msgs)
        if self.variant_type == "fund":
            return corpus
        return torch.cat([corpus, price], dim=-1)

    def _g_T_infer_gen(self, x: torch.Tensor, vmd_out: dict, t_idx: torch.Tensor):
        g = vmd_out["g"]
        b = g.size(0)
        batch_idx = torch.arange(b, device=g.device)
        target_idx = (t_idx - 1).clamp(min=0)
        if self.training:
            return g[batch_idx, target_idx, :]
        x_T = x[batch_idx, target_idx, :]
        if self.vmd_rec == "zh":
            h_T = vmd_out["h_seq"][batch_idx, target_idx, :]
        else:
            h_T = vmd_out["h_seq"][batch_idx, target_idx, :]
        z_p_T = vmd_out["z_prior"][batch_idx, target_idx, :]
        if self.vmd_rec == "zh":
            g_T = self.vmd.g_lin(torch.cat([h_T, z_p_T], dim=-1))
        else:
            g_T = self.vmd.g_lin(torch.cat([x_T, h_T, z_p_T], dim=-1))
        return g_T

    def forward(
        self,
        word_ids: torch.Tensor,
        n_words: torch.Tensor,
        ss_index: torch.Tensor,
        n_msgs: torch.Tensor,
        price: torch.Tensor,
        y_onehot: torch.Tensor,
        t_idx: torch.Tensor,
        compute_loss: bool = True,
    ):
        x = self._build_x(word_ids, n_words, ss_index, n_msgs, price)
        x = self.dropout_vmd_in(x)

        if self.vmd_rec == "zh":
            vmd_out = self.vmd(x, y_onehot, t_idx)
        else:
            vmd_out = self.vmd(x, y_onehot)

        mask_aux = mask_auxiliary_days(x.size(0), x.size(1), t_idx, x.device)
        g_T = self._g_T_infer_gen(x, vmd_out, t_idx)
        y_T, v_stared = self.tda(vmd_out["g"], g_T, vmd_out["y_prob"], mask_aux)

        loss = None
        if compute_loss:
            loss = generative_ata_loss(
                vmd_out["y_prob"],
                y_T,
                y_onehot,
                vmd_out["kl"],
                v_stared,
                self.alpha,
                self.kl_lambda,
                t_idx,
            )

        b = y_onehot.size(0)
        batch_idx = torch.arange(b, device=y_onehot.device)
        target_idx = (t_idx - 1).clamp(min=0)
        y_true = y_onehot[batch_idx, target_idx, :]
        y_pred = y_T
        return {
            "loss": loss,
            "y_T": y_T,
            "y_true": y_true,
            "y_pred_class": y_pred.argmax(dim=-1),
            "y_true_class": y_true.argmax(dim=-1),
            "vmd_out": vmd_out,
        }

    def set_kl_lambda(self, value: float) -> None:
        self.kl_lambda = float(value)
