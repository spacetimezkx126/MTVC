"""
PyTorch reimplementation of PEN (# pen 分支) from
/home/zhaokx/Pattern/compared_model/PEN/PEN_python2/PEN-main/src/Model.py

MEL -> MSIN(price, msg_embed) -> VMD -> TDA -> generative / discriminative ATA.
StockNet 路径见 model_stocknet.py（MEL -> corpus_embed -> concat price）。
"""
from __future__ import annotations

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
    def __init__(
        self,
        in_size: int,
        out_size: int,
        activation: str | None = None,
        use_bias: bool = True,
    ):
        super().__init__()
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
        z = mu if is_prior else mu + std * torch.randn_like(std)
        return z, dist.Normal(mu, std)


class MessageEmbedLayer(nn.Module):
    """TF _create_msg_embed_layer."""

    def __init__(
        self,
        vocab_size: int,
        word_embed_size: int,
        mel_h_size: int,
        pad_id: int = 0,
        cell_type: Literal["gru", "basic", "ln-lstm"] = "gru",
        dropout_in: float = 0.3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.mel_h_size = mel_h_size
        self.word_embed = nn.Embedding(vocab_size, word_embed_size, padding_idx=pad_id)
        _xavier_init_(self.word_embed.weight)
        rnn_cls = nn.GRU if cell_type == "gru" else nn.RNN
        self.rnn_f = rnn_cls(word_embed_size, mel_h_size, batch_first=True)
        self.rnn_b = rnn_cls(word_embed_size, mel_h_size, batch_first=True)
        self.drop_in = nn.Dropout(dropout_in)
        self.drop = nn.Dropout(dropout)

    def _run_one_direction(self, rnn, emb, lengths):
        packed = pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        return out

    def forward(self, word_ids, n_words, ss_index):
        b, l, m, w = word_ids.shape
        flat_ids = word_ids.reshape(b * l * m, w)
        flat_n = n_words.reshape(b * l * m).clamp(min=1)
        flat_ss = ss_index.reshape(b * l * m).clamp(0, w - 1)
        emb = self.drop_in(self.word_embed(flat_ids.clamp(min=0)))
        out_f = self._run_one_direction(self.rnn_f, emb, flat_n)
        out_b = torch.flip(
            self._run_one_direction(self.rnn_b, torch.flip(emb, dims=[1]), flat_n),
            dims=[1],
        )
        idx = torch.arange(flat_ids.size(0), device=word_ids.device)
        msg = (out_f[idx, flat_ss] + out_b[idx, flat_ss]) * 0.5
        return self.drop(msg).view(b, l, m, self.mel_h_size)


class CorpusEmbedLayer(nn.Module):
    """TF Model._create_corpus_embed — StockNet MIE 路径专用。"""

    def __init__(self, mel_dim: int, dropout: float = 0.0):
        super().__init__()
        self.proj_u = nn.Linear(mel_dim, mel_dim, bias=False)
        self.w_u = nn.Parameter(torch.empty(mel_dim, 1))
        _xavier_init_(self.w_u)
        self.drop = nn.Dropout(dropout)

    def forward(self, msg_embed, n_msgs):
        b, l, m, _ = msg_embed.shape
        u = torch.matmul(torch.tanh(self.proj_u(msg_embed)), self.w_u).squeeze(-1)
        idx = torch.arange(m, device=n_msgs.device).view(1, 1, m).expand(b, l, m)
        u = u.masked_fill(~(idx < n_msgs.clamp(min=0).unsqueeze(-1)), float("-inf"))
        u = torch.nan_to_num(F.softmax(u, dim=-1), nan=0.0)
        return self.drop(torch.sum(u.unsqueeze(-1) * msg_embed, dim=2))


class MSINCell(nn.Module):
    """TF MSINModule.MSINCell — PEN MIE：逐日融合 price 与 msg_embed。"""

    def __init__(self, input_size: int, num_units: int, v_size: int):
        super().__init__()
        self.num_units = num_units
        self.v_size = v_size
        self.W_sa = nn.Linear(v_size, num_units, bias=False)
        self.W_ha = nn.Linear(num_units, num_units, bias=False)
        self.b_a = nn.Parameter(torch.zeros(num_units))
        self.v_a = nn.Parameter(torch.zeros(num_units))
        self.W_f = nn.Linear(v_size, num_units, bias=False)
        self.W_hf = nn.Linear(num_units, num_units, bias=False)
        self.b_f = nn.Parameter(torch.zeros(num_units))
        self.W_o = nn.Linear(v_size, num_units, bias=False)
        self.W_ho = nn.Linear(num_units, num_units, bias=False)
        self.b_o = nn.Parameter(torch.zeros(num_units))
        self.W_t = nn.Linear(v_size, num_units, bias=False)
        self.W_ht = nn.Linear(num_units, num_units, bias=False)
        self.b_t = nn.Parameter(torch.zeros(num_units))
        self.W_k = nn.Linear(input_size, num_units, bias=False)
        self.W_hk = nn.Linear(num_units, num_units, bias=False)
        self.W_vk = nn.Linear(v_size, num_units, bias=False)
        self.b_k = nn.Parameter(torch.zeros(num_units))
        self.W_s = nn.Linear(input_size, num_units, bias=False)
        self.W_hs = nn.Linear(num_units, num_units, bias=False)
        self.b_s = nn.Parameter(torch.zeros(num_units))
        self.W_v = nn.Linear(v_size, num_units, bias=False)
        self.W_hv = nn.Linear(num_units, num_units, bias=False)
        self.b_v = nn.Parameter(torch.zeros(num_units))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(
        self,
        price_t: torch.Tensor,
        msg_t: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
        msg_mask: torch.Tensor | None = None,
    ):
        h, v = state
        attn = torch.tanh(self.W_sa(msg_t) + (self.W_ha(h) + self.b_a).unsqueeze(1))
        scores = torch.einsum("bmd,d->bm", attn, self.v_a)
        if msg_mask is not None:
            scores = scores.masked_fill(~msg_mask, float("-inf"))
        p = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0)
        text = torch.einsum("bm,bmd->bd", p, msg_t)
        gate_f = torch.sigmoid(self.W_f(text) + self.W_hf(h) + self.b_f)
        gate_o = torch.sigmoid(self.W_o(text) + self.W_ho(h) + self.b_o)
        text_new = torch.tanh(self.W_t(text) + self.W_ht(h) + self.b_t)
        v_new = gate_f * v + gate_o * text_new
        k = torch.sigmoid(self.W_k(price_t) + self.W_hk(h) + self.W_vk(v_new) + self.b_k)
        hx = torch.tanh(self.W_s(price_t) + self.W_hs(h) + self.b_s)
        hv = torch.tanh(self.W_hv(h) + self.W_v(v_new) + self.b_v)
        h_new = (1.0 - k) * hv + k * hx
        return h_new, p, (h_new, v_new)


class MSINEncoder(nn.Module):
    """TF MSIN.dynamic_msin — 对 [B,L] 序列逐日调用 MSINCell。"""

    def __init__(self, input_size: int, num_units: int, v_size: int, max_n_msgs: int):
        super().__init__()
        self.max_n_msgs = max_n_msgs
        self.cell = MSINCell(input_size, num_units, v_size)

    def forward(
        self,
        price: torch.Tensor,
        msg_embed: torch.Tensor,
        n_msgs: torch.Tensor,
        msg_keep_mask: torch.Tensor | None = None,
    ):
        b, l, m, _ = msg_embed.shape
        h = msg_embed.new_zeros(b, self.cell.num_units)
        v = msg_embed.new_zeros(b, self.cell.v_size)
        xs, ps = [], []
        idx = torch.arange(m, device=msg_embed.device).view(1, 1, m).expand(b, l, m)
        for t in range(l):
            mask = idx[:, t, :] < n_msgs[:, t].clamp(min=0).unsqueeze(-1)
            if msg_keep_mask is not None:
                mask = mask & msg_keep_mask[:, t, :]
            h, p, (h, v) = self.cell(price[:, t, :], msg_embed[:, t, :, :], (h, v), mask)
            xs.append(h)
            ps.append(p)
        return torch.stack(xs, dim=1), torch.stack(ps, dim=1)


class VMDZhRec(nn.Module):
    """TF _create_vmd_with_zh_rec."""

    def __init__(
        self, x_size, h_size, z_size, g_size, y_size=2,
        cell_type: Literal["gru", "ln-lstm"] = "gru", dropout=0.0,
    ):
        super().__init__()
        self.h_size = h_size
        self.z_size = z_size
        self.drop = nn.Dropout(dropout)
        self.seq_cell = nn.GRU(x_size, h_size, batch_first=True) if cell_type == "gru" else nn.LSTM(x_size, h_size, batch_first=True)
        z_in_prior = x_size + h_size + z_size
        z_in_post = x_size + h_size + y_size + z_size
        self.enc_prior = nn.Sequential(nn.Linear(z_in_prior, z_size), nn.Tanh())
        self.enc_post = nn.Sequential(nn.Linear(z_in_post, z_size), nn.Tanh())
        self.prior_z = DiagonalGaussianZ(z_size)
        self.post_z = DiagonalGaussianZ(z_size)
        self.g_lin = PenLinear(h_size + z_size, g_size, activation="tanh")
        self.y_lin = PenLinear(g_size, y_size, activation="softmax")

    def forward(self, x, y_onehot, t_idx):
        del t_idx
        x = self.drop(x)
        b, l, _ = x.shape
        h_s, _ = self.seq_cell(x)
        g_list, y_list, kl_list = [], [], []
        z_prior_list, z_post_list = [], []
        z_prev = torch.randn(b, self.z_size, device=x.device, dtype=x.dtype)
        for t in range(l):
            x_t, h_t, y_t = x[:, t], h_s[:, t], y_onehot[:, t]
            h_z_p = self.enc_prior(torch.cat([x_t, h_t, z_prev], dim=-1))
            z_p, dist_p = self.prior_z(h_z_p, is_prior=True)
            h_z_q = self.enc_post(torch.cat([x_t, h_t, y_t, z_prev], dim=-1))
            z_q, dist_q = self.post_z(h_z_q, is_prior=False)
            kl_list.append(dist.kl_divergence(dist_q, dist_p).sum(-1))
            g_t = self.g_lin(torch.cat([h_t, z_q], dim=-1))
            y_list.append(self.y_lin(g_t))
            g_list.append(g_t)
            z_prior_list.append(z_p)
            z_post_list.append(z_q)
            z_prev = z_q
        return {
            "g": torch.stack(g_list, 1),
            "y_prob": torch.stack(y_list, 1),
            "kl": torch.stack(kl_list, 1),
            "z_prior": torch.stack(z_prior_list, 1),
            "z_post": torch.stack(z_post_list, 1),
            "h_seq": h_s,
            "x": x,
        }


class VMDHRec(nn.Module):
    """TF _create_vmd_with_h_rec."""

    def __init__(self, x_size, h_size, z_size, g_size, y_size=2, dropout=0.0):
        super().__init__()
        self.x_size, self.h_size, self.z_size = x_size, h_size, z_size
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

    def forward(self, x, y_onehot):
        x = self.drop(x)
        b, l, _ = x.shape
        h = torch.tanh(torch.randn(b, self.h_size, device=x.device, dtype=x.dtype))
        h_z = torch.tanh(torch.randn(b, self.z_size, device=x.device, dtype=x.dtype))
        z, _ = self.post_z(h_z, is_prior=False)
        g_list, y_list, kl_list = [], [], []
        z_prior_list, z_post_list, h_list = [], [], []
        for t in range(l):
            x_t, y_t = x[:, t], y_onehot[:, t]
            gate = torch.cat([x_t, h, z], dim=-1)
            r, u = torch.sigmoid(self.gru_r(gate)), torch.sigmoid(self.gru_u(gate))
            h = (1 - u) * h + u * torch.tanh(self.gru_h(torch.cat([x_t, r * h, z], dim=-1)))
            z_p, dist_p = self.prior_z(self.enc_prior(torch.cat([x_t, h], -1)), True)
            z_q, dist_q = self.post_z(self.enc_post(torch.cat([x_t, h, y_t], -1)), False)
            kl_list.append(dist.kl_divergence(dist_q, dist_p).sum(-1))
            g_t = self.g_lin(torch.cat([x_t, h, z_q], -1))
            g_list.append(g_t)
            y_list.append(self.y_lin(g_t))
            z_prior_list.append(z_p)
            z_post_list.append(z_q)
            h_list.append(h)
        return {
            "g": torch.stack(g_list, 1),
            "y_prob": torch.stack(y_list, 1),
            "kl": torch.stack(kl_list, 1),
            "z_prior": torch.stack(z_prior_list, 1),
            "z_post": torch.stack(z_post_list, 1),
            "h_seq": torch.stack(h_list, 1),
            "x": x,
        }


class VMDDiscriminative(nn.Module):
    """TF _create_discriminative_vmd."""

    def __init__(self, x_size, h_size, z_size, g_size, y_size=2,
                 cell_type: Literal["gru", "ln-lstm"] = "gru", dropout=0.0):
        super().__init__()
        self.z_size = z_size
        self.drop = nn.Dropout(dropout)
        cell = nn.GRU if cell_type == "gru" else nn.LSTM
        self.seq_cell = cell(x_size, h_size, batch_first=True)
        self.h_z = PenLinear(x_size + h_size + z_size, z_size, activation="tanh")
        self.z_lin = PenLinear(z_size, z_size, activation="tanh")
        self.g_lin = PenLinear(h_size + z_size, g_size, activation="tanh")
        self.y_lin = PenLinear(g_size, y_size, activation="softmax")

    def forward(self, x, t_idx):
        del t_idx
        x = self.drop(x)
        b, l, _ = x.shape
        h_s, _ = self.seq_cell(x)
        z_prev = torch.randn(b, self.z_size, device=x.device, dtype=x.dtype)
        g_list, y_list, z_list = [], [], []
        for t in range(l):
            h_z = self.h_z(torch.cat([x[:, t], h_s[:, t], z_prev], -1))
            z_t = self.z_lin(h_z)
            g_t = self.g_lin(torch.cat([h_s[:, t], z_t], -1))
            g_list.append(g_t)
            y_list.append(self.y_lin(g_t))
            z_list.append(z_t)
            z_prev = z_t
        z = torch.stack(z_list, 1)
        kl = torch.zeros(b, l, device=x.device, dtype=x.dtype)
        return {
            "g": torch.stack(g_list, 1),
            "y_prob": torch.stack(y_list, 1),
            "kl": kl,
            "z_prior": z,
            "z_post": z,
            "h_seq": h_s,
            "x": x,
        }


class TemporalAttnDecoder(nn.Module):
    """TF _build_temporal_att."""

    def __init__(self, g_size: int, y_size: int = 2, daily_att: str = "y"):
        super().__init__()
        assert daily_att in ("y", "g")
        self.daily_att = daily_att
        self.ctx_size = y_size if daily_att == "y" else g_size
        self.proj_i = nn.Linear(g_size, g_size, bias=False)
        self.w_i = nn.Parameter(torch.empty(g_size))
        nn.init.normal_(self.w_i, std=0.02)
        self.proj_d = nn.Linear(g_size, g_size, bias=False)
        self.y_T_lin = PenLinear(self.ctx_size + g_size, y_size, activation="softmax")

    def forward(self, g, g_T, y_prob, mask_aux):
        v_i = (torch.tanh(self.proj_i(g)) * self.w_i).sum(-1)
        v_d = (torch.tanh(self.proj_d(g)) * g_T.unsqueeze(1)).sum(-1)
        aux_score = (v_i * v_d).masked_fill(~mask_aux, float("-inf"))
        v_stared = torch.nan_to_num(F.softmax(aux_score, dim=-1), nan=0.0)
        ctx = y_prob if self.daily_att == "y" else g
        att_c = torch.sum(v_stared.unsqueeze(-1) * ctx, dim=1)
        return self.y_T_lin(torch.cat([att_c, g_T], dim=-1)), v_stared


def mask_auxiliary_days(batch, seq_len, t_idx, device):
    day_idx = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, seq_len)
    return day_idx < (t_idx.unsqueeze(1) - 1).clamp(min=0)


def _target_index(t_idx):
    return (t_idx - 1).clamp(min=0)


def generative_ata_loss(y_prob, y_T_prob, y_onehot, kl, v_stared, alpha, kl_lambda, t_idx, eps=0.0):
    """TF _create_generative_ata (minor=0.0 in original)."""
    log_y = torch.log(y_prob + eps)
    obj_aux = (y_onehot * log_y).sum(-1) - kl_lambda * kl
    b = y_onehot.size(0)
    bi = torch.arange(b, device=y_onehot.device)
    ti = _target_index(t_idx)
    y_T_true = y_onehot[bi, ti]
    obj_T = (y_T_true * torch.log(y_T_prob + eps)).sum(-1, keepdim=True) - kl_lambda * kl[bi, ti].unsqueeze(1)
    obj = obj_T + (obj_aux * (alpha * v_stared)).sum(1, keepdim=True)
    return (-obj).mean()


def discriminative_ata_loss(y_prob, y_T_prob, y_onehot, v_stared, alpha, t_idx, P=None, eps=1e-7):
    """TF _create_discriminative_ata（含 MSIN 消息注意力正则 P_obj）。"""
    log_y = torch.log(y_prob + eps)
    obj_aux = (y_onehot * log_y).sum(-1)
    b = y_onehot.size(0)
    bi = torch.arange(b, device=y_onehot.device)
    ti = _target_index(t_idx)
    y_T_true = y_onehot[bi, ti]
    obj_T = (y_T_true * torch.log(y_T_prob + eps)).sum(-1, keepdim=True)
    obj = obj_T + (obj_aux * (alpha * v_stared)).sum(1, keepdim=True)
    loss = (-obj).mean()
    if P is not None:
        new_p = P.clamp(min=1e-8, max=1.0)
        p_obj = (new_p * torch.log(new_p)).sum(dim=-1).mean()
        loss = loss + (-p_obj)
    return loss


class PENModel(nn.Module):
    """
    PEN assemble_graph（Model.py 中 # pen 分支）：
      MEL -> MSIN(price, msg_embed) -> VMD -> TDA -> ATA
    与 StockNet（# stocknet: MEL -> corpus_embed -> concat price）不同。
    """

    def __init__(
        self,
        vocab_size: int,
        pad_id: int = 0,
        word_embed_size: int = 50,
        mel_h_size: int = 100,
        msin_h_size: int = 100,
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
        dropout_vmd_in: float = 0.3,
        dropout_vmd: float = 0.0,
        price_input_size: int = 3,
    ):
        super().__init__()
        assert variant_type in ("hedge", "fund", "tech", "discriminative")
        assert vmd_rec in ("zh", "h")
        self.variant_type = variant_type
        self.vmd_rec = vmd_rec
        self.y_size = y_size
        self.max_n_msgs = max_n_msgs
        self.alpha = alpha
        self.kl_lambda = kl_lambda
        self.use_tda_y = daily_att in ("y", "g")
        self.daily_att = daily_att if self.use_tda_y else "y"
        self.last_p = None
        z_size = z_size or h_size

        mel_cell = "gru" if mel_cell_type == "gru" else "basic"
        self.mel = MessageEmbedLayer(
            vocab_size, word_embed_size, mel_h_size, pad_id,
            cell_type=mel_cell, dropout_in=dropout_mel_in, dropout=dropout_mel,
        )
        self.price_input_size = int(price_input_size)
        if variant_type == "tech":
            self.msin = None
            self.x_size = self.price_input_size
        else:
            self.msin = MSINEncoder(self.price_input_size, msin_h_size, mel_h_size, max_n_msgs)
            self.x_size = msin_h_size

        vmd_cell = "gru" if vmd_cell_type == "gru" else "ln-lstm"
        if variant_type == "discriminative":
            self.vmd = VMDDiscriminative(
                self.x_size, h_size, z_size, g_size, y_size, vmd_cell, dropout_vmd
            )
        elif vmd_rec == "zh":
            self.vmd = VMDZhRec(self.x_size, h_size, z_size, g_size, y_size, vmd_cell, dropout_vmd)
        else:
            self.vmd = VMDHRec(self.x_size, h_size, z_size, g_size, y_size, dropout_vmd)
        self.tda = TemporalAttnDecoder(g_size, y_size, daily_att=self.daily_att)
        self.dropout_vmd_in = nn.Dropout(dropout_vmd_in)

    def _build_x(self, word_ids, n_words, ss_index, n_msgs, price):
        if self.variant_type == "tech":
            self.last_p = None
            return price
        msg = self.mel(word_ids, n_words, ss_index)
        x, p = self.msin(price, msg, n_msgs)
        self.last_p = p
        return x

    def _gather_target(self, tensor, t_idx):
        b = tensor.size(0)
        bi = torch.arange(b, device=tensor.device)
        return tensor[bi, _target_index(t_idx)]

    def _g_T_infer_gen(self, x, vmd_out, t_idx):
        if self.training:
            return self._gather_target(vmd_out["g"], t_idx)
        b = x.size(0)
        bi = torch.arange(b, device=x.device)
        ti = _target_index(t_idx)
        x_T = x[bi, ti]
        h_T = vmd_out["h_seq"][bi, ti]
        z_p = vmd_out["z_prior"][bi, ti]
        if self.variant_type == "discriminative":
            return vmd_out["g"][bi, ti]
        if self.vmd_rec == "zh":
            return self.vmd.g_lin(torch.cat([h_T, z_p], dim=-1))
        return self.vmd.g_lin(torch.cat([x_T, h_T, z_p], dim=-1))

    def _y_T_infer_gen(self, vmd_out, t_idx):
        y_all = vmd_out["y_prob"]
        if self.training:
            return self._gather_target(y_all, t_idx)
        b = y_all.size(0)
        bi = torch.arange(b, device=y_all.device)
        ti = _target_index(t_idx)
        if self.variant_type == "discriminative":
            return y_all[bi, ti]
        x = vmd_out["x"]
        h_T = vmd_out["h_seq"][bi, ti]
        z_p = vmd_out["z_prior"][bi, ti]
        if self.vmd_rec == "zh":
            g_T = self.vmd.g_lin(torch.cat([h_T, z_p], dim=-1))
        else:
            g_T = self.vmd.g_lin(torch.cat([x[bi, ti], h_T, z_p], dim=-1))
        return self.vmd.y_lin(g_T)

    def forward(
        self,
        word_ids,
        n_words,
        ss_index,
        n_msgs,
        price,
        y_onehot,
        t_idx,
        compute_loss=True,
    ):
        x = self.dropout_vmd_in(self._build_x(word_ids, n_words, ss_index, n_msgs, price))
        if self.variant_type == "discriminative":
            vmd_out = self.vmd(x, t_idx)
        elif self.vmd_rec == "zh":
            vmd_out = self.vmd(x, y_onehot, t_idx)
        else:
            vmd_out = self.vmd(x, y_onehot)

        mask_aux = mask_auxiliary_days(x.size(0), x.size(1), t_idx, x.device)
        g_T = self._g_T_infer_gen(x, vmd_out, t_idx)
        if self.use_tda_y:
            y_T, v_stared = self.tda(vmd_out["g"], g_T, vmd_out["y_prob"], mask_aux)
        else:
            v_stared = self.tda(vmd_out["g"], g_T, vmd_out["y_prob"], mask_aux)[1]
            y_T = self._y_T_infer_gen(vmd_out, t_idx)

        loss = None
        if compute_loss:
            if self.variant_type == "discriminative":
                loss = discriminative_ata_loss(
                    vmd_out["y_prob"], y_T, y_onehot, v_stared, self.alpha, t_idx, P=self.last_p
                )
            else:
                loss = generative_ata_loss(
                    vmd_out["y_prob"], y_T, y_onehot, vmd_out["kl"],
                    v_stared, self.alpha, self.kl_lambda, t_idx,
                )

        y_true = self._gather_target(y_onehot, t_idx)
        return {
            "loss": loss,
            "y_T": y_T,
            "y_true": y_true,
            "y_pred_class": y_T.argmax(dim=-1),
            "y_true_class": y_true.argmax(dim=-1),
            "vmd_out": vmd_out,
        }

    def set_kl_lambda(self, value: float) -> None:
        self.kl_lambda = float(value)


PEN1_DEFAULTS = {
    "max_n_days": 5,
    "max_n_msgs": 20,
    "max_n_words": 30,
    "word_embed_size": 50,
    "mel_h_size": 100,
    "msin_h_size": 100,
    "h_size": 150,
    "g_size": 50,
    "variant_type": "hedge",
    "vmd_rec": "zh",
    "mel_cell_type": "gru",
    "vmd_cell_type": "gru",
    "daily_att": "y",
    "alpha": 0.5,
    "kl_lambda": 0.0,
    "dropout_mel_in": 0.3,
    "dropout_mel": 0.0,
    "dropout_vmd_in": 0.3,
    "dropout_vmd": 0.0,
    "y_size": 2,
    "pad_id": 0,
}


def build_pen_model(vocab_size: int, **overrides) -> PENModel:
    cfg = dict(PEN1_DEFAULTS)
    cfg.update(overrides)
    return PENModel(vocab_size=vocab_size, **cfg)


build_pen1_model = build_pen_model
